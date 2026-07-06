from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .model import AttentionPool2d
from .qat_attention import QATMultiheadAttention
from .qat_attention_pool import QATAttentionPool2d
from .qat_layers import QATConv2d, QATLinear, QATWeightOnly, disable_qat_observers, enable_fake_quant


class BakedWeightOnlyLinear(nn.Module):
    """
    Linear-like module used where the original CLIP code reads .weight directly
    and passes it to functional attention. It intentionally does not subclass
    nn.Linear, so dynamic quantization will not replace it with a module whose
    packed weight is no longer a plain Tensor.
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


@torch.no_grad()
def _fake_quant_weight(fake_quant, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.detach()
    try:
        return fake_quant(weight).detach().clone()
    except RuntimeError as exc:
        if "dimensions of scale and zero-point" not in str(exc):
            raise

    observer = getattr(fake_quant, "activation_post_process", None)
    if observer is not None:
        try:
            scale, zero_point = observer.calculate_qparams()
        except Exception:
            observer(weight)
            scale, zero_point = observer.calculate_qparams()

        fake_quant.scale.resize_(scale.shape)
        fake_quant.scale.copy_(scale.to(fake_quant.scale.device))
        fake_quant.zero_point.resize_(zero_point.shape)
        fake_quant.zero_point.copy_(zero_point.to(fake_quant.zero_point.device))

    try:
        return fake_quant(weight).detach().clone()
    except RuntimeError as exc:
        if "dimensions of scale and zero-point" not in str(exc):
            raise

    quant_min = int(getattr(fake_quant, "quant_min", -128))
    quant_max = int(getattr(fake_quant, "quant_max", 127))
    ch_axis = int(getattr(fake_quant, "ch_axis", 0))

    if weight.dim() > 1:
        reduce_dims = tuple(dim for dim in range(weight.dim()) if dim != ch_axis)
        max_abs = weight.abs().amax(dim=reduce_dims)
        scale = (max_abs / float(max(abs(quant_min), abs(quant_max)))).clamp(min=torch.finfo(weight.dtype).eps)
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
        return torch.fake_quantize_per_channel_affine(
            weight,
            scale.to(weight.device),
            zero_point.to(weight.device),
            ch_axis,
            quant_min,
            quant_max,
        ).detach().clone()

    max_abs = weight.abs().max()
    scale = (max_abs / float(max(abs(quant_min), abs(quant_max)))).clamp(min=torch.finfo(weight.dtype).eps)
    return torch.fake_quantize_per_tensor_affine(
        weight,
        float(scale.item()),
        0,
        quant_min,
        quant_max,
    ).detach().clone()


@torch.no_grad()
def _linear_from_qat(module: QATLinear) -> nn.Linear:
    linear = nn.Linear(
        module.linear.in_features,
        module.linear.out_features,
        bias=module.linear.bias is not None,
    )
    linear.weight.copy_(_fake_quant_weight(module.weight_fake_quant, module.linear.weight))
    if module.linear.bias is not None:
        linear.bias.copy_(module.linear.bias.detach())
    return linear


@torch.no_grad()
def _conv_from_qat(module: QATConv2d) -> nn.Conv2d:
    conv = nn.Conv2d(
        module.conv.in_channels,
        module.conv.out_channels,
        module.conv.kernel_size,
        stride=module.conv.stride,
        padding=module.conv.padding,
        dilation=module.conv.dilation,
        groups=module.conv.groups,
        bias=module.conv.bias is not None,
        padding_mode=module.conv.padding_mode,
    )
    conv.weight.copy_(_fake_quant_weight(module.weight_fake_quant, module.conv.weight))
    if module.conv.bias is not None:
        conv.bias.copy_(module.conv.bias.detach())
    return conv


@torch.no_grad()
def _linear_from_weight_only(module: QATWeightOnly) -> BakedWeightOnlyLinear:
    return BakedWeightOnlyLinear(
        _fake_quant_weight(module.weight_fake_quant, module.linear.weight),
        module.linear.bias,
    )


@torch.no_grad()
def _mha_from_qat(module: QATMultiheadAttention) -> nn.MultiheadAttention:
    attn = nn.MultiheadAttention(module.embed_dim, module.num_heads, bias=True, batch_first=False)
    q_weight = _fake_quant_weight(module.q_weight_fq, module.q_proj.weight)
    k_weight = _fake_quant_weight(module.k_weight_fq, module.k_proj.weight)
    v_weight = _fake_quant_weight(module.v_weight_fq, module.v_proj.weight)
    attn.in_proj_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
    attn.in_proj_bias.copy_(
        torch.cat(
            [
                module.q_proj.bias.detach(),
                module.k_proj.bias.detach(),
                module.v_proj.bias.detach(),
            ],
            dim=0,
        )
    )
    attn.out_proj.weight.copy_(_fake_quant_weight(module.out_weight_fq, module.out_proj.weight))
    attn.out_proj.bias.copy_(module.out_proj.bias.detach())
    return attn


@torch.no_grad()
def _attnpool_from_qat(module: QATAttentionPool2d) -> AttentionPool2d:
    """
    Bake QATAttentionPool2d ve AttentionPool2d (model.py) voi weight da fake-
    quantized, dung BakedWeightOnlyLinear cho ca 4 projection - GIONG cach
    attnpool weight-only (QATWeightOnly) da tung duoc bake truoc day.

    QUAN TRONG: KHONG duoc dung nn.Linear thuong o day. AttentionPool2d.forward()
    (model.py) doc truc tiep .weight/.bias roi dua vao F.multi_head_attention_forward,
    KHONG goi q_proj(x). Neu q_proj/k_proj/v_proj/c_proj la nn.Linear thuong, buoc
    convert_linear_to_dynamic_int8() phia sau se quet trung va thay bang
    nn.quantized.dynamic.Linear - class nay KHONG co attribute .weight/.bias
    truc tiep (chi co method .weight()/.bias()) -> AttentionPool2d.forward() se
    crash ngay khi chay. BakedWeightOnlyLinear tranh duoc loi nay vi no khong
    subclass nn.Linear nen bi convert_linear_to_dynamic_int8() bo qua, dung
    y het BakedWeightOnlyLinear dang dung cho duong QATWeightOnly mac dinh.

    He qua: viec bat quantize_attention_internals=True chi cai thien do chinh
    xac cua qua trinh MO PHONG quantize luc train QAT (activation + Q@K^T +
    attn_weights@V duoc thay thuc trong forward), con o buoc deploy INT8 cuoi
    cung, phan compute cua attnpool VAN la FP32 - giong het duong weight-only
    mac dinh. Muon attnpool thuc su chay INT8 dynamic tren CPU se can viet lai
    forward() de goi q_proj(x)/k_proj(x)/... nhu Linear binh thuong thay vi
    doc .weight/.bias truc tiep - ngoai pham vi thay doi nay.
    """
    pool = AttentionPool2d(
        spacial_dim=module.positional_embedding.shape[0] - 1,
        embed_dim=module.embed_dim,
        num_heads=module.num_heads,
        output_dim=module.c_proj.out_features,
    )
    pool.positional_embedding.data.copy_(module.positional_embedding.detach())

    pool.q_proj = BakedWeightOnlyLinear(
        _fake_quant_weight(module.q_weight_fq, module.q_proj.weight), module.q_proj.bias
    )
    pool.k_proj = BakedWeightOnlyLinear(
        _fake_quant_weight(module.k_weight_fq, module.k_proj.weight), module.k_proj.bias
    )
    pool.v_proj = BakedWeightOnlyLinear(
        _fake_quant_weight(module.v_weight_fq, module.v_proj.weight), module.v_proj.bias
    )
    pool.c_proj = BakedWeightOnlyLinear(
        _fake_quant_weight(module.out_weight_fq, module.c_proj.weight), module.c_proj.bias
    )

    return pool


def bake_qat_fake_quant_weights(module: nn.Module) -> nn.Module:
    """
    Replace custom QAT wrappers with normal PyTorch modules whose weights have
    already been fake-quantized using the learned QAT observer scale/zero-point.

    After this step:
      - QATLinear/QATWeightOnly become nn.Linear with baked quantized weights.
      - QATConv2d becomes nn.Conv2d with baked quantized weights.
      - QATMultiheadAttention becomes nn.MultiheadAttention with baked weights.
      - QATAttentionPool2d becomes AttentionPool2d (q/k/v/c_proj as
        BakedWeightOnlyLinear so it stays safe for the raw-weight-read forward()).

    Conv2d remains FP32 compute after baking. Linear layers can then be converted
    to dynamic INT8 for CPU inference with torch.quantization.quantize_dynamic().
    """
    disable_qat_observers(module)
    enable_fake_quant(module)

    for name, child in list(module.named_children()):
        if isinstance(child, QATLinear):
            setattr(module, name, _linear_from_qat(child))
        elif isinstance(child, QATConv2d):
            setattr(module, name, _conv_from_qat(child))
        elif isinstance(child, QATWeightOnly):
            setattr(module, name, _linear_from_weight_only(child))
        elif isinstance(child, QATMultiheadAttention):
            setattr(module, name, _mha_from_qat(child))
        elif isinstance(child, QATAttentionPool2d):
            setattr(module, name, _attnpool_from_qat(child))
        else:
            bake_qat_fake_quant_weights(child)

    return module


def convert_linear_to_dynamic_int8(
    model: nn.Module,
    engine: str = "fbgemm",
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """
    Convert eligible nn.Linear modules to dynamic INT8 modules.

    This targets CPU inference. CUDA kernels do not generally run PyTorch dynamic
    quantized Linear modules.
    """
    if engine:
        supported_engines = list(getattr(torch.backends.quantized, "supported_engines", []))
        if engine in supported_engines:
            torch.backends.quantized.engine = engine
        else:
            fallback = next((item for item in ("qnnpack", "x86", "fbgemm", "onednn") if item in supported_engines), None)
            if fallback is None:
                raise RuntimeError("No supported quantized engine found. Supported engines: {}".format(supported_engines))
            print(
                "Quantized engine {} is not supported; using {} instead. Supported engines: {}".format(
                    engine, fallback, supported_engines
                )
            )
            torch.backends.quantized.engine = fallback
    model.cpu().eval()
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)