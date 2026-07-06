"""
qat_attention_pool.py

QAT-aware AttentionPool2d cho ModifiedResNet (RN50 CLIP visual backbone).

AttentionPool2d goc (trong model.py) KHONG the dung QATMultiheadAttention
(qat_attention.py) vi no khong phai instance nn.MultiheadAttention - no tu goi
F.multi_head_attention_forward truc tiep trong forward() cua chinh no, doc
q_proj.weight / k_proj.weight / v_proj.weight / c_proj.weight / c_proj.bias
truc tiep thay vi goi q_proj(x) nhu mot Linear binh thuong. Do do
patch_attention_pool2d (apply_qat_clip.py) truoc day chi co the dung
QATWeightOnly (chi quantize weight, KHONG quantize activation, KHONG cham toi
Q@K^T / softmax / attn@V).

Class nay REIMPLEMENT lai dung phep toan cua AttentionPool2d.forward() (NCHW ->
(HW)NC, cat mean-token, cong positional_embedding, roi multi-head attention)
nhung thu cong nhu QATMultiheadAttention trong qat_attention.py, de co the chen
FakeQuantize vao ca activation lan Q@K^T / attn_weights@V matmul - hoan tat
"Conv2d den attention" cho RN50.

CHI dung khi quantize_attention_internals=True khi goi apply_qat_to_clip().
Mac dinh (False) van dung QATWeightOnly - nhe hon, an toan hon, it rui ro accuracy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qat_layers import (
    QATMatMulSelective,
    make_activation_fake_quant,
    make_weight_fake_quant_per_channel,
)


class QATAttentionPool2d(nn.Module):
    """
    Ban thay the QAT-aware cho AttentionPool2d.

    Giu nguyen: positional_embedding (Parameter, KHONG quantize - nhay cam,
    giong class_embedding/positional_embedding cua VisionTransformer).

    Quantize:
      - weight cua q_proj/k_proj/v_proj/c_proj (per-channel symmetric qint8)
      - activation dau vao cua tung projection (per-tensor affine quint8, dung
        CHUNG 1 observer cho q/k/v vi ca 3 deu doc tu CUNG mot tensor x -
        AttentionPool2d la self-attention thuan, query=key=value=x)
      - Q@K^T matmul (qk_matmul)
      - attn_weights@V matmul: CHI quantize V, giu attn_weights FP32 (phan
        phoi xac suat tu softmax, nhay cam - giong qat_attention.py)

    KHONG quantize: buoc reshape NCHW->(HW)NC, mean-token, positional_embedding,
    softmax.
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
        quantize_qk_matmul: bool = True,
        quantize_attn_weights_before_v_matmul: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim phai chia het cho num_heads"
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        output_dim = output_dim or embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)

        self.q_weight_fq = make_weight_fake_quant_per_channel(ch_axis=0)
        self.k_weight_fq = make_weight_fake_quant_per_channel(ch_axis=0)
        self.v_weight_fq = make_weight_fake_quant_per_channel(ch_axis=0)
        self.out_weight_fq = make_weight_fake_quant_per_channel(ch_axis=0)

        # q/k/v deu doc tu CUNG mot tensor x (self-attention thuan) -> dung
        # chung 1 observer, giong pattern trong qat_attention.py.
        self.input_fake_quant = make_activation_fake_quant(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        self.out_input_fake_quant = make_activation_fake_quant(dtype=torch.quint8, qscheme=torch.per_tensor_affine)

        self.qk_matmul = QATMatMulSelective(quantize_a=quantize_qk_matmul, quantize_b=quantize_qk_matmul)
        self.av_matmul = QATMatMulSelective(quantize_a=quantize_attn_weights_before_v_matmul, quantize_b=True)

    def _split_heads(self, x: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        # x: [seq_len, batch_size, embed_dim]
        x = x.view(seq_len, batch_size, self.num_heads, self.head_dim)
        return x.permute(1, 2, 0, 3)  # -> [batch_size, num_heads, seq_len, head_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW -> (HW)NC, giong het AttentionPool2d.forward() goc. KHONG
        # quantize buoc nay - chi la reshape/pooling, chua vao Linear nao.
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)

        seq_len, batch_size, _ = x.shape

        q_in = self.input_fake_quant(x)
        k_in = self.input_fake_quant(x)
        v_in = self.input_fake_quant(x)

        q_w = self.q_weight_fq(self.q_proj.weight)
        k_w = self.k_weight_fq(self.k_proj.weight)
        v_w = self.v_weight_fq(self.v_proj.weight)

        q = F.linear(q_in, q_w, self.q_proj.bias)
        k = F.linear(k_in, k_w, self.k_proj.bias)
        v = F.linear(v_in, v_w, self.v_proj.bias)

        q = self._split_heads(q, seq_len, batch_size)  # [batch, heads, seq, head_dim]
        k = self._split_heads(k, seq_len, batch_size)
        v = self._split_heads(v, seq_len, batch_size)

        scores = self.qk_matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)  # FP32, khong fake-quantize

        context = self.av_matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
        context = context.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.embed_dim)

        context_in = self.out_input_fake_quant(context)
        out_w = self.out_weight_fq(self.c_proj.weight)
        out = F.linear(context_in, out_w, self.c_proj.bias)

        # Giu nguyen full sequence (HW+1, N, output_dim), giong AttentionPool2d
        # goc trong model.py (KHONG lay x[0] - fork nay dung ca patch token cho
        # downstream re-id, khong chi class/mean token).
        return out

    @classmethod
    def from_float(
        cls,
        pool,  # model.clip.model.AttentionPool2d
        quantize_qk_matmul: bool = True,
        quantize_attn_weights_before_v_matmul: bool = False,
    ) -> "QATAttentionPool2d":
        spacial_dim = pool.positional_embedding.shape[0] - 1
        embed_dim = pool.q_proj.in_features
        output_dim = pool.c_proj.out_features

        module = cls(
            spacial_dim=spacial_dim,
            embed_dim=embed_dim,
            num_heads=pool.num_heads,
            output_dim=output_dim,
            quantize_qk_matmul=quantize_qk_matmul,
            quantize_attn_weights_before_v_matmul=quantize_attn_weights_before_v_matmul,
        )

        module.positional_embedding.data = pool.positional_embedding.data.clone()

        module.q_proj.weight.data = pool.q_proj.weight.data.clone()
        module.q_proj.bias.data = pool.q_proj.bias.data.clone()
        module.k_proj.weight.data = pool.k_proj.weight.data.clone()
        module.k_proj.bias.data = pool.k_proj.bias.data.clone()
        module.v_proj.weight.data = pool.v_proj.weight.data.clone()
        module.v_proj.bias.data = pool.v_proj.bias.data.clone()
        module.c_proj.weight.data = pool.c_proj.weight.data.clone()
        module.c_proj.bias.data = pool.c_proj.bias.data.clone()

        return module


def replace_attention_pool2d_with_qat(
    pool,
    quantize_qk_matmul: bool = True,
    quantize_attn_weights_before_v_matmul: bool = False,
) -> QATAttentionPool2d:
    """
    Tao QATAttentionPool2d MOI, copy weight tu AttentionPool2d da pretrained.
    THAY THE HOAN TOAN (object moi) - object `pool` cu se bi garbage-collect
    sau khi ham nay return, tru khi ban tu giu bien tham chieu rieng truoc khi
    goi ham nay (giong luu y trong qat_attention.py).
    """
    return QATAttentionPool2d.from_float(
        pool,
        quantize_qk_matmul=quantize_qk_matmul,
        quantize_attn_weights_before_v_matmul=quantize_attn_weights_before_v_matmul,
    )
