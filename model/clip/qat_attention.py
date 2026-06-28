"""
qat_attention.py

OPTIONAL / NANG CAO: QATMultiheadAttention thay the cho nn.MultiheadAttention,
cho phep quantize ca Q@K^T va attn_weights@V matmul (khong chi Linear).

Mac dinh apply_qat_clip.py KHONG dung file nay (quantize_attention_internals=False).
Chi kich hoat khi ban da do duoc loi ich tu Linear+Conv quantize chua du, va
muon di xa hon - chap nhan rui ro accuracy cao hon va do phuc tap debug cao hon.

QUAN TRONG: replace_multihead_attention_with_qat() THAY THE HOAN TOAN object
nn.MultiheadAttention cu bang object QATMultiheadAttention moi (khong sua tai cho).
Object cu se bi garbage-collect sau khi ham return, tru khi ban tu giu lai bien
tham chieu toi no truoc khi goi ham nay.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.ao.quantization import FakeQuantize, MovingAveragePerChannelMinMaxObserver
except ImportError:
    from torch.quantization import FakeQuantize, MovingAveragePerChannelMinMaxObserver

from .qat_layers import QATMatMulSelective, make_activation_fake_quant


class QATMultiheadAttention(nn.Module):
    """
    Ban thay the cho nn.MultiheadAttention, ho tro:
      - Q/K/V/out projection: fake-quantize weight (per-channel) + input activation
      - Q@K^T matmul: tuy chon quantize (quantize_qk_matmul)
      - attn_weights@V matmul: mac dinh CHI quantize V, giu attn_weights FP32
        (attention weights la phan phoi xac suat tu softmax, rat nhay cam)

    KHONG quantize: softmax, attn_mask (gia tri -inf can giu FP32 nguyen ven).

    Chu y: chi ho tro tap tinh nang ma ResidualAttentionBlock cua CLIP dung
    (attn_mask dang additive float, batch_first=False, need_weights=False).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        quantize_qk_matmul: bool = True,
        quantize_attn_weights_before_v_matmul: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim phai chia het cho num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # CLIP goc dung nn.MultiheadAttention voi bias=True (mac dinh) cho ca
        # in_proj_bias va out_proj.bias -> giu bias=True o day de tuong thich
        # khi copy weight tu checkpoint pretrained (xem replace_multihead_attention_with_qat).
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        def _weight_fq():
            return FakeQuantize(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric, ch_axis=0,
            )

        self.q_weight_fq = _weight_fq()
        self.k_weight_fq = _weight_fq()
        self.v_weight_fq = _weight_fq()
        self.out_weight_fq = _weight_fq()

        self.input_fake_quant = make_activation_fake_quant(dtype=torch.quint8, qscheme=torch.per_tensor_affine)

        # Q@K^T: ca Q va K co the am -> qint8 symmetric
        self.qk_matmul = QATMatMulSelective(quantize_a=quantize_qk_matmul, quantize_b=quantize_qk_matmul)

        # attn_weights@V: mac dinh KHONG quantize attn_weights (an toan hon vi day
        # la phan phoi xac suat tu softmax), CHI quantize V
        self.av_matmul = QATMatMulSelective(quantize_a=quantize_attn_weights_before_v_matmul, quantize_b=True)

        self.out_input_fake_quant = make_activation_fake_quant(dtype=torch.quint8, qscheme=torch.per_tensor_affine)

    def _split_heads(self, x: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        # x: [seq_len, batch_size, embed_dim] (batch_first=False, giong nn.MultiheadAttention mac dinh)
        x = x.view(seq_len, batch_size, self.num_heads, self.head_dim)
        return x.permute(1, 2, 0, 3)  # -> [batch_size, num_heads, seq_len, head_dim]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
        attn_mask: torch.Tensor = None,
    ):
        """
        Giu nguyen ten/thu tu tham so giong nn.MultiheadAttention.forward() de
        tuong thich voi ResidualAttentionBlock.attention():
            self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        Tra ve tuple (output, None) de giu nguyen unpacking [0] cua code goc.
        """
        seq_len, batch_size, _ = query.shape

        q_in = self.input_fake_quant(query)
        k_in = self.input_fake_quant(key)
        v_in = self.input_fake_quant(value)

        q_w = self.q_weight_fq(self.q_proj.weight)
        k_w = self.k_weight_fq(self.k_proj.weight)
        v_w = self.v_weight_fq(self.v_proj.weight)

        q = F.linear(q_in, q_w, self.q_proj.bias)
        k = F.linear(k_in, k_w, self.k_proj.bias)
        v = F.linear(v_in, v_w, self.v_proj.bias)

        q = self._split_heads(q, seq_len, batch_size)  # [batch, num_heads, seq_len, head_dim]
        k = self._split_heads(k, seq_len, batch_size)
        v = self._split_heads(v, seq_len, batch_size)

        scores = self.qk_matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # attn_mask la additive mask (gia tri -inf) - giu FP32, khong fake-quantize
            scores = scores + attn_mask

        attn_weights = torch.softmax(scores, dim=-1)  # FP32, khong fake-quantize

        context = self.av_matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # QUAN TRONG: giu dung 3 chieu [seq_len, batch_size, embed_dim] - KHONG gop
        # seq_len*batch_size thanh 1 chieu, vi ResidualAttentionBlock can shape nay
        # de cong residual (x = x + self.attention(...)).
        context = context.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.embed_dim)

        context_in = self.out_input_fake_quant(context)
        out_w = self.out_weight_fq(self.out_proj.weight)
        out = F.linear(context_in, out_w, self.out_proj.bias)

        return out, None


def replace_multihead_attention_with_qat(
    attn: nn.MultiheadAttention,
    quantize_qk_matmul: bool = True,
    quantize_attn_weights_before_v_matmul: bool = False,
) -> QATMultiheadAttention:
    """
    Tao QATMultiheadAttention MOI, copy weight tu nn.MultiheadAttention da pretrained.
    Day la THAY THE HOAN TOAN (object moi), khong phai sua tai cho object cu.
    Object `attn` truyen vao se khong con duoc tham chieu sau khi ham nay return
    (tru khi ban tu giu bien rieng truoc khi goi), va se bi Python garbage-collect.
    """
    embed_dim = attn.embed_dim
    num_heads = attn.num_heads

    qat_attn = QATMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        quantize_qk_matmul=quantize_qk_matmul,
        quantize_attn_weights_before_v_matmul=quantize_attn_weights_before_v_matmul,
    )

    if attn.in_proj_weight is not None:
        q_w, k_w, v_w = attn.in_proj_weight.data.chunk(3, dim=0)
    else:
        # Truong hop hiem: dung q_proj_weight/k_proj_weight/v_proj_weight rieng
        q_w, k_w, v_w = attn.q_proj_weight.data, attn.k_proj_weight.data, attn.v_proj_weight.data

    if attn.in_proj_bias is not None:
        q_b, k_b, v_b = attn.in_proj_bias.data.chunk(3, dim=0)
    else:
        q_b = k_b = v_b = None

    qat_attn.q_proj.weight.data = q_w.clone()
    qat_attn.k_proj.weight.data = k_w.clone()
    qat_attn.v_proj.weight.data = v_w.clone()

    if q_b is not None:
        qat_attn.q_proj.bias.data = q_b.clone()
        qat_attn.k_proj.bias.data = k_b.clone()
        qat_attn.v_proj.bias.data = v_b.clone()

    qat_attn.out_proj.weight.data = attn.out_proj.weight.data.clone()
    if attn.out_proj.bias is not None:
        qat_attn.out_proj.bias.data = attn.out_proj.bias.data.clone()

    return qat_attn
