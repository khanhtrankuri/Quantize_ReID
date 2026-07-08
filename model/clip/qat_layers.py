"""
qat_layers.py

Cac building block QAT co ban, dung torch.ao.quantization.FakeQuantize (API chinh thuc).
Tat ca deu co .from_float() de tao tu module FP32 da co weight (KHONG random init).

Day la cac "lop nen" - duoc dung boi apply_qat_clip.py de patch model SAU KHI
build_model() (trong model.py) da load checkpoint pretrained thanh cong.
"""

from __future__ import annotations
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.ao.quantization import (
        FakeQuantize,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
    )
except ImportError:
    from torch.quantization import (
        FakeQuantize,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
    )


# ---------------------------------------------------------------------------
# Factory functions cho FakeQuantize
# ---------------------------------------------------------------------------

def make_activation_fake_quant(
    dtype: torch.dtype = torch.quint8,
    qscheme: torch.qscheme = torch.per_tensor_affine,
    averaging_constant: float = 0.01,
) -> FakeQuantize:
    """
    FakeQuantize cho activation (per-tensor).
    dtype=quint8 (range [0,255])  -> activation sau ReLU (luon >= 0)
    dtype=qint8  (range [-128,127]) -> activation co the am (Q/K trong attention, truoc ReLU)
    """
    if dtype == torch.quint8:
        quant_min, quant_max = 0, 255
    else:
        quant_min, quant_max = -128, 127

    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=quant_min,
        quant_max=quant_max,
        dtype=dtype,
        qscheme=qscheme,
        averaging_constant=averaging_constant,
    )


def make_weight_fake_quant_per_channel(ch_axis: int = 0) -> FakeQuantize:
    """FakeQuantize cho weight, per-channel symmetric - chuan cho Conv2d/Linear."""
    return FakeQuantize(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
    )


# ---------------------------------------------------------------------------
# QATLinear
# ---------------------------------------------------------------------------

class QATLinear(nn.Module):
    """Wrap nn.Linear: fake-quantize input activation (per-tensor) va weight (per-channel)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_fake_quant = make_weight_fake_quant_per_channel(ch_axis=0)
        self.input_fake_quant = make_activation_fake_quant(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_fake_quant(x)
        w = self.weight_fake_quant(self.linear.weight)
        return F.linear(x, w, self.linear.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_fake_quant(self.linear.weight)

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    @classmethod
    def from_float(cls, mod: nn.Linear) -> "QATLinear":
        module = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        module.linear.weight.data = mod.weight.data.clone()
        if mod.bias is not None:
            module.linear.bias.data = mod.bias.data.clone()
        return module

    def extra_repr(self) -> str:
        return f"in_features={self.linear.in_features}, out_features={self.linear.out_features}, bias={self.linear.bias is not None}"


# ---------------------------------------------------------------------------
# QATConv2d
# ---------------------------------------------------------------------------

class QATConv2d(nn.Module):
    """Wrap nn.Conv2d: fake-quantize input activation va weight."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, bias: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.weight_fake_quant = make_weight_fake_quant_per_channel(ch_axis=0)
        self.input_fake_quant = make_activation_fake_quant(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_fake_quant(x)
        w = self.weight_fake_quant(self.conv.weight)
        return F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    # Pass-through cac attribute cua nn.Conv2d ben trong - can thiet de
    # fuse_conv_bn()/fuse_bottleneck() (ben duoi trong file nay) hoat dong dung
    # tren Bottleneck da duoc patch QAT. Neu thieu cac property nay,
    # fuse_conv_bn(bottleneck.conv1, ...) se nem AttributeError vi QATConv2d
    # (nn.Module) khong tu dong forward attribute tu self.conv ra ngoai.
    @property
    def in_channels(self) -> int:
        return self.conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def stride(self):
        return self.conv.stride

    @property
    def padding(self):
        return self.conv.padding

    @property
    def dilation(self):
        return self.conv.dilation

    @property
    def groups(self) -> int:
        return self.conv.groups

    @property
    def padding_mode(self) -> str:
        return self.conv.padding_mode

    @classmethod
    def from_float(cls, mod: nn.Conv2d) -> "QATConv2d":
        module = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            bias=mod.bias is not None,
        )
        module.conv.weight.data = mod.weight.data.clone()
        if mod.bias is not None:
            module.conv.bias.data = mod.bias.data.clone()
        return module

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.conv.in_channels}, "
            f"out_channels={self.conv.out_channels}, "
            f"kernel_size={self.conv.kernel_size}"
        )


# ---------------------------------------------------------------------------
# Fuse Conv + BatchNorm (chi dung SAU KHI train QAT xong, truoc convert INT8)
# ---------------------------------------------------------------------------

@torch.no_grad()
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse Conv2d + BatchNorm2d thanh 1 Conv2d duy nhat.
    CHI goi ham nay SAU KHI model da train xong (eval mode, BN stats da on dinh),
    KHONG goi luc dang QAT training (vi BN can tiep tuc cap nhat running_mean/var).
    """
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )

    bn_eps = bn.eps
    bn_std = torch.sqrt(bn.running_var + bn_eps)
    scale_factor = bn.weight / bn_std  # shape [out_channels]

    # weight Conv2d co shape [out_channels, in_channels, kH, kW]
    # -> reshape scale_factor ve [out_channels, 1, 1, 1] de broadcast dung chieu
    fused_conv.weight.data = conv.weight.data * scale_factor.reshape(-1, 1, 1, 1)

    conv_bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused_conv.bias.data = (conv_bias - bn.running_mean) * scale_factor + bn.bias.data

    return fused_conv


def fuse_bottleneck(bottleneck) -> None:
    """
    Fuse Conv+BN trong 1 Bottleneck (ModifiedResNet), INPLACE.
    Goi ham nay SAU KHI da QAT-train xong, truoc khi convert sang INT8 thuc.
    """
    bottleneck.conv1 = fuse_conv_bn(bottleneck.conv1, bottleneck.bn1)
    bottleneck.bn1 = nn.Identity()

    bottleneck.conv2 = fuse_conv_bn(bottleneck.conv2, bottleneck.bn2)
    bottleneck.bn2 = nn.Identity()

    bottleneck.conv3 = fuse_conv_bn(bottleneck.conv3, bottleneck.bn3)
    bottleneck.bn3 = nn.Identity()

    if bottleneck.downsample is not None:
        avgpool = bottleneck.downsample[0]
        conv = bottleneck.downsample[1]
        bn = bottleneck.downsample[2]
        fused_conv = fuse_conv_bn(conv, bn)
        bottleneck.downsample = nn.Sequential(OrderedDict([
            ("-1", avgpool),
            ("0", fused_conv),
            ("1", nn.Identity()),
        ]))


# ---------------------------------------------------------------------------
# QATWeightOnly - dung khi weight duoc lay ra dung rieng (AttentionPool2d)
# ---------------------------------------------------------------------------

class QATWeightOnly(nn.Module):
    """
    Chi fake-quantize weight, khong tu chay F.linear mac dinh - dung khi code goc
    lay .weight cua nn.Linear ra de dua vao ham khac (F.multi_head_attention_forward).
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.weight_fake_quant = make_weight_fake_quant_per_channel(ch_axis=0)

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_fake_quant(self.linear.weight)

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_float(cls, mod: nn.Linear) -> "QATWeightOnly":
        return cls(mod)

    def extra_repr(self) -> str:
        return f"in_features={self.linear.in_features}, out_features={self.linear.out_features}, bias={self.linear.bias is not None}"


# ---------------------------------------------------------------------------
# QATMatMulSelective - tuy chon dung cho attention matmul (Q@K^T, attn@V)
# ---------------------------------------------------------------------------

class QATMatMulSelective(nn.Module):
    """
    Wrap torch.matmul(a, b), cho phep chon fake-quantize input nao.
    Mac dinh CA 2 deu quantize (an toan, bat nguoi dung phai chu dong tat).
    """

    def __init__(self, quantize_a: bool = True, quantize_b: bool = True):
        super().__init__()
        self.quantize_a = quantize_a
        self.quantize_b = quantize_b
        if quantize_a:
            self.fq_a = make_activation_fake_quant(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        if quantize_b:
            self.fq_b = make_activation_fake_quant(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_q = self.fq_a(a) if self.quantize_a else a
        b_q = self.fq_b(b) if self.quantize_b else b
        return torch.matmul(a_q, b_q)


# ---------------------------------------------------------------------------
# Helper enable/disable observer va fake-quant
# ---------------------------------------------------------------------------

def enable_qat_observers(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, FakeQuantize):
            m.enable_observer()


def disable_qat_observers(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, FakeQuantize):
            m.disable_observer()


def enable_fake_quant(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, FakeQuantize):
            m.enable_fake_quant()


def disable_fake_quant(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, FakeQuantize):
            m.disable_fake_quant()