"""Selective attention for CLIP QAT and CPU INT8 deployment.

Only the four independent projections are QAT/INT8 candidates.  Scores,
additive masks, softmax and attention-value matmul deliberately remain FP32.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .qat_layers import QATLinear, QATMatMulSelective


class SelectiveQuantMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, *, quantize_qkv=True,
                 quantize_out=True, quantize_qk_matmul=False,
                 quantize_av_matmul=False, qat_kwargs=None):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim, self.scale = embed_dim // num_heads, (embed_dim // num_heads) ** -0.5
        kwargs = qat_kwargs or {}
        linear = (lambda name: QATLinear(embed_dim, embed_dim, True, name=name, **kwargs))
        self.q_proj = linear("attention.q_proj") if quantize_qkv else nn.Linear(embed_dim, embed_dim)
        self.k_proj = linear("attention.k_proj") if quantize_qkv else nn.Linear(embed_dim, embed_dim)
        self.v_proj = linear("attention.v_proj") if quantize_qkv else nn.Linear(embed_dim, embed_dim)
        self.out_proj = linear("attention.out_proj") if quantize_out else nn.Linear(embed_dim, embed_dim)
        # These are opt-in simulation modules only; they never exist by default.
        self.qk_matmul = QATMatMulSelective(True, True) if quantize_qk_matmul else None
        self.av_matmul = QATMatMulSelective(False, True) if quantize_av_matmul else None

    def _split_heads(self, x):
        seq, batch, _ = x.shape
        return x.reshape(seq, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

    def forward(self, query, key, value, need_weights=False, attn_mask=None, **_):
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("SelectiveQuantMultiheadAttention supports batch_first=False only")
        q, k, v = self._split_heads(self.q_proj(query)), self._split_heads(self.k_proj(key)), self._split_heads(self.v_proj(value))
        # Attention core is intentionally float even under autocast/fake quant.
        scores = (self.qk_matmul(q.float(), k.float().transpose(-1, -2)) if self.qk_matmul else torch.matmul(q.float(), k.float().transpose(-1, -2))) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask.float()
        probabilities = torch.softmax(scores, dim=-1)
        context = self.av_matmul(probabilities, v.float()) if self.av_matmul else torch.matmul(probabilities, v.float())
        seq, batch = query.shape[:2]
        context = context.permute(2, 0, 1, 3).reshape(seq, batch, self.embed_dim).to(query.dtype)
        output = self.out_proj(context)
        return output, (probabilities.mean(dim=1) if need_weights else None)


# Backward-compatible import name used by old QAT checkpoints/scripts.
QATMultiheadAttention = SelectiveQuantMultiheadAttention


@torch.no_grad()
def replace_multihead_attention_with_qat(attn: nn.MultiheadAttention, **kwargs):
    module = SelectiveQuantMultiheadAttention(attn.embed_dim, attn.num_heads, **kwargs)
    q_w, k_w, v_w = (attn.in_proj_weight if attn.in_proj_weight is not None else torch.cat(
        [attn.q_proj_weight, attn.k_proj_weight, attn.v_proj_weight])).chunk(3, dim=0)
    q_b, k_b, v_b = (attn.in_proj_bias.chunk(3, dim=0) if attn.in_proj_bias is not None else (None, None, None))
    for dst, weight, bias in ((module.q_proj, q_w, q_b), (module.k_proj, k_w, k_b), (module.v_proj, v_w, v_b), (module.out_proj, attn.out_proj.weight, attn.out_proj.bias)):
        target = dst.linear if isinstance(dst, QATLinear) else dst
        target.weight.copy_(weight)
        if bias is not None and target.bias is not None:
            target.bias.copy_(bias)
    return module
