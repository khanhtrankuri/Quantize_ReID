"""
apply_qat_clip.py

Ham chinh: apply_qat_to_clip(model, ...).

CHI goi ham nay SAU KHI build_model() (trong model.py) da load checkpoint
pretrained thanh cong. Ham nay PATCH submodule (thay nn.Conv2d/nn.Linear bang
QATConv2d/QATLinear, giu nguyen weight da pretrained), KHONG tao lai model tu dau.

Chien luoc quantize ap dung:
  - Bottleneck (ResNet): quantize Conv1/2/3 + downsample conv (giu BN tach rieng,
    KHONG fuse luc training - chi fuse sau khi train QAT xong, truoc convert)
  - Stem 3 conv dau ModifiedResNet: quantize tuong tu Bottleneck
  - AttentionPool2d:
      * quantize_attention_internals=False (mac dinh): q/k/v/c_proj -> weight-only
        (vi forward doc truc tiep .weight/.bias roi dua vao
        F.multi_head_attention_forward)
      * quantize_attention_internals=True: THAY THE HOAN TOAN bang
        QATAttentionPool2d (qat_attention_pool.py) - quantize ca activation
        va Q@K^T / attn_weights@V matmul, khong chi weight
  - ResidualAttentionBlock.mlp (c_fc, c_proj): full QATLinear - uu tien cao nhat
  - ResidualAttentionBlock.attn: GIU NGUYEN nn.MultiheadAttention (mac dinh),
    chi thay bang QATMultiheadAttention neu quantize_attention_internals=True
  - VisionTransformer.conv1 (patch embedding): full QATConv2d
  - KHONG dong vao: LayerNorm, embedding, positional_embedding, logit_scale,
    phan cosine similarity cuoi cung trong CLIP.forward()
"""

from __future__ import annotations

import fnmatch
import json
import torch.nn as nn

from .model import Bottleneck, AttentionPool2d, ModifiedResNet, VisionTransformer, ResidualAttentionBlock, CLIP
from .qat_layers import QATConv2d, QATLinear, QATWeightOnly


# ---------------------------------------------------------------------------
# 1. Bottleneck (ResNet block trong ModifiedResNet)
# ---------------------------------------------------------------------------

def patch_bottleneck(bottleneck: Bottleneck) -> Bottleneck:
    """
    Thay conv1/conv2/conv3 (va downsample conv neu co) bang QATConv2d, INPLACE.
    KHONG fuse BatchNorm o day - BN van tach rieng, chay binh thuong sau Conv
    da fake-quantized. Chi fuse BN sau khi train QAT xong (dung fuse_bottleneck()
    trong qat_layers.py truoc khi convert sang INT8 thuc).
    """
    bottleneck.conv1 = QATConv2d.from_float(bottleneck.conv1)
    bottleneck.conv2 = QATConv2d.from_float(bottleneck.conv2)
    bottleneck.conv3 = QATConv2d.from_float(bottleneck.conv3)

    if bottleneck.downsample is not None:
        # downsample = nn.Sequential(OrderedDict([("-1", AvgPool2d), ("0", Conv2d), ("1", BatchNorm2d)]))
        bottleneck.downsample[1] = QATConv2d.from_float(bottleneck.downsample[1])

    return bottleneck


# ---------------------------------------------------------------------------
# 2. AttentionPool2d
# ---------------------------------------------------------------------------

def patch_attention_pool2d(pool: AttentionPool2d, quantize_attention_internals: bool = False):
    """
    Mac dinh (quantize_attention_internals=False): q_proj/k_proj/v_proj/c_proj
    deu duoc AttentionPool2d.forward() doc truc tiep qua .weight/.bias roi dua
    vao F.multi_head_attention_forward (q_proj_weight=, k_proj_weight=,
    v_proj_weight=, out_proj_weight=, out_proj_bias=) - KHONG mot module nao
    trong 4 module nay duoc goi qua forward() binh thuong cua Linear. Vi vay
    CA 4 deu dung QATWeightOnly (chi fake-quantize .weight khi truy cap, khong
    tu chay F.linear).

    Neu quantize_attention_internals=True: THAY THE HOAN TOAN AttentionPool2d
    bang QATAttentionPool2d (qat_attention_pool.py), reimplement lai dung phep
    toan cua forward() goc nhung chen duoc FakeQuantize vao ca activation lan
    Q@K^T / attn_weights@V matmul - dung khi ban muon quantize "tu Conv2d den
    attention" trọn ven cho backbone RN50. Rui ro accuracy cao hon, nen luon
    danh gia lai (eval_before/sau finetune) khi bat co nay.
    """
    if quantize_attention_internals:
        from .qat_attention_pool import replace_attention_pool2d_with_qat
        return replace_attention_pool2d_with_qat(pool)

    pool.q_proj = QATWeightOnly.from_float(pool.q_proj)
    pool.k_proj = QATWeightOnly.from_float(pool.k_proj)
    pool.v_proj = QATWeightOnly.from_float(pool.v_proj)
    pool.c_proj = QATWeightOnly.from_float(pool.c_proj)
    return pool


# ---------------------------------------------------------------------------
# 3. ModifiedResNet (RN50 visual backbone)
# ---------------------------------------------------------------------------

def apply_qat_to_modified_resnet(
    visual: ModifiedResNet, quantize_attention_internals: bool = False
) -> ModifiedResNet:
    """
    Patch INPLACE: visual van la instance ModifiedResNet (khong doi class),
    chi cac Conv2d/Linear con ben trong duoc thay bang ban QAT (rieng attnpool
    co the doi han sang QATAttentionPool2d neu quantize_attention_internals=True,
    van gan lai vao visual.attnpool nen ModifiedResNet.forward() khong can sua
    gi - no chi goi self.attnpool(x4) nho tinh da hinh cua nn.Module).
    """
    # 3 stem convolutions - giu BN tach rieng, chi quantize Conv
    visual.conv1 = QATConv2d.from_float(visual.conv1)
    visual.conv2 = QATConv2d.from_float(visual.conv2)
    visual.conv3 = QATConv2d.from_float(visual.conv3)

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(visual, layer_name)
        for i, bottleneck in enumerate(layer):
            layer[i] = patch_bottleneck(bottleneck)

    visual.attnpool = patch_attention_pool2d(
        visual.attnpool, quantize_attention_internals=quantize_attention_internals
    )

    return visual


# ---------------------------------------------------------------------------
# 4. ResidualAttentionBlock - dung chung cho Vision Transformer va Text Transformer
# ---------------------------------------------------------------------------

def _qat_kwargs(options):
    return {
        "weight_bits": int(getattr(options, "WEIGHT_BITS", 8)),
        "activation_bits": int(getattr(options, "ACTIVATION_BITS", 8)),
        "activation_observer": str(getattr(options, "ACTIVATION_OBSERVER", "moving_average")),
        "activation_percentile": float(getattr(options, "ACTIVATION_PERCENTILE", 99.99)),
    }


def _excluded(name, options):
    patterns = list(getattr(options, "FP32_MODULE_PATTERNS", ()))
    sensitivity_path = getattr(options, "SENSITIVITY_JSON", "")
    if sensitivity_path:
        try:
            with open(sensitivity_path, "r") as handle:
                data = json.load(handle)
            threshold = float(getattr(options, "MAX_MAP_DROP_PER_MODULE", 0.005))
            patterns.extend(key for key, result in data.items() if result.get("delta_map", 0.0) < -threshold)
        except (OSError, ValueError, TypeError):
            pass
    return any(fnmatch.fnmatch(name, pattern) or pattern in name for pattern in patterns)


def patch_residual_attention_block(block: ResidualAttentionBlock, options=None, block_index=0, total_blocks=0) -> ResidualAttentionBlock:
    """
    - mlp.c_fc, mlp.c_proj -> QATLinear (uu tien cao nhat, chiem FLOPs lon nhat)
    - attn -> mac dinh GIU NGUYEN nn.MultiheadAttention (F.multi_head_attention_forward
      la black-box, khong chen FakeQuantize vao giua duoc). Neu
      quantize_attention_internals=True, THAY THE HOAN TOAN bang QATMultiheadAttention
      (object moi, copy weight tu attn cu - xem qat_attention.py).
    - ln_1, ln_2 -> KHONG quantize (LayerNorm)
    - QuickGELU -> KHONG quantize (activation function)
    """
    options = options or _LegacyQATOptions()
    excluded = (
        block_index < int(getattr(options, "EXCLUDE_FIRST_N_BLOCKS", 0))
        or block_index >= total_blocks - int(getattr(options, "EXCLUDE_LAST_N_BLOCKS", 0))
        or _excluded("image_encoder.transformer.resblocks.{}".format(block_index), options)
    )
    kwargs = _qat_kwargs(options)
    if not excluded and bool(getattr(options, "QUANTIZE_MLP", True)):
        block.mlp.c_fc = QATLinear.from_float(block.mlp.c_fc, name="block_{}.mlp.c_fc".format(block_index), **kwargs)
        block.mlp.c_proj = QATLinear.from_float(block.mlp.c_proj, name="block_{}.mlp.c_proj".format(block_index), **kwargs)

    if not excluded and (bool(getattr(options, "QUANTIZE_QKV_PROJ", True)) or bool(getattr(options, "QUANTIZE_ATTN_OUT_PROJ", True))):
        from .qat_attention import replace_multihead_attention_with_qat
        block.attn = replace_multihead_attention_with_qat(
            block.attn,
            quantize_qkv=bool(getattr(options, "QUANTIZE_QKV_PROJ", True)),
            quantize_out=bool(getattr(options, "QUANTIZE_ATTN_OUT_PROJ", True)),
            quantize_qk_matmul=bool(getattr(options, "QUANTIZE_QK_MATMUL", False)),
            quantize_av_matmul=bool(getattr(options, "QUANTIZE_AV_MATMUL", False)),
            qat_kwargs=kwargs,
        )

    return block


# ---------------------------------------------------------------------------
# 5. VisionTransformer
# ---------------------------------------------------------------------------

def apply_qat_to_vision_transformer(visual: VisionTransformer, options=None, quantize_attention_internals=False) -> VisionTransformer:
    """Quantize patch-embedding conv1, va patch toan bo resblocks ben trong."""
    options = options or _LegacyQATOptions(quantize_attention_internals=quantize_attention_internals)
    if bool(getattr(options, "QUANTIZE_PATCH_EMBED", False)):
        visual.conv1 = QATConv2d.from_float(visual.conv1)

    total_blocks = len(visual.transformer.resblocks)
    for i, block in enumerate(visual.transformer.resblocks):
        visual.transformer.resblocks[i] = patch_residual_attention_block(
            block, options=options, block_index=i, total_blocks=total_blocks
        )

    # ln_pre, ln_post: KHONG quantize (LayerNorm)
    # class_embedding, positional_embedding, proj: KHONG quantize (Parameter nho, nhay cam)
    return visual


# ---------------------------------------------------------------------------
# 6. Entry point chinh - ap dung cho toan bo CLIP model
# ---------------------------------------------------------------------------

class _LegacyQATOptions:
    """Safe options for callers that still use the old boolean API."""
    def __init__(self, quantize_attention_internals=False):
        self.WEIGHT_BITS = 8; self.ACTIVATION_BITS = 8
        self.ACTIVATION_OBSERVER = "moving_average"; self.ACTIVATION_PERCENTILE = 99.99
        self.QUANTIZE_PATCH_EMBED = False; self.QUANTIZE_MLP = True
        self.QUANTIZE_QKV_PROJ = bool(quantize_attention_internals)
        self.QUANTIZE_ATTN_OUT_PROJ = bool(quantize_attention_internals)
        self.QUANTIZE_QK_MATMUL = False; self.QUANTIZE_AV_MATMUL = False
        self.EXCLUDE_FIRST_N_BLOCKS = 0; self.EXCLUDE_LAST_N_BLOCKS = 0
        self.FP32_MODULE_PATTERNS = (); self.SENSITIVITY_JSON = ""


def apply_qat_to_clip(model: CLIP, options=None, quantize_attention_internals: bool = False) -> CLIP:
    """
    Ap dung QAT cho toan bo model CLIP - tu dong nhan dien visual backbone la
    ModifiedResNet (RN50) hay VisionTransformer (ViT) va xu ly tuong ung.

    CHI goi ham nay SAU KHI model da duoc tao boi build_model(state_dict, ...)
    (trong model.py) va da load weight pretrained day du - vi cac ham
    QATConv2d.from_float()/QATLinear.from_float() can weight thuc de copy vao,
    khong phai random init.

    Args:
        model: instance CLIP da load pretrained weight (output cua build_model()).
        quantize_attention_internals: neu True, quantize toan bo phan attention:
            - ResidualAttentionBlock.attn (ca visual ViT resblocks lan text
              transformer resblocks) -> QATMultiheadAttention
            - ModifiedResNet.attnpool (RN50) -> QATAttentionPool2d
            Mac dinh False - chi quantize Conv/Linear (+ weight-only cho
            attnpool), an toan hon, du dung cho hau het truong hop.

    Returns:
        model da duoc patch INPLACE (cung instance, cac submodule con ben trong
        bi thay the bang ban QAT).

    KHONG dong vao:
        - LayerNorm (ln_1, ln_2, ln_pre, ln_post, ln_final)
        - class_embedding, positional_embedding, text_projection, proj
        - logit_scale va toan bo cosine similarity trong CLIP.forward()
        - token_embedding (nn.Embedding)
    """
    # --- Visual backbone ---
    options = options or _LegacyQATOptions(quantize_attention_internals)
    if isinstance(model.visual, ModifiedResNet):
        apply_qat_to_modified_resnet(
            model.visual, quantize_attention_internals=quantize_attention_internals
        )
    elif isinstance(model.visual, VisionTransformer):
        apply_qat_to_vision_transformer(model.visual, options=options)
    else:
        raise TypeError(f"Khong nhan dien duoc loai visual backbone: {type(model.visual)}")

    # Text is not needed by image retrieval deployment.  Keep it FP32 unless an
    # experiment explicitly opts in; this also protects cached teacher text features.
    if not bool(getattr(options, "VISUAL_ONLY", True)) and bool(getattr(options, "QUANTIZE_TEXT_ENCODER", False)):
        total_blocks = len(model.transformer.resblocks)
        for i, block in enumerate(model.transformer.resblocks):
            model.transformer.resblocks[i] = patch_residual_attention_block(block, options, i, total_blocks)

    return model
