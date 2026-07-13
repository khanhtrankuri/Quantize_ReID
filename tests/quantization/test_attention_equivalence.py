import unittest
import torch
from model.clip.qat_attention import replace_multihead_attention_with_qat
from model.clip.qat_layers import disable_fake_quant


class AttentionEquivalenceTest(unittest.TestCase):
    def test_fp32_attention_matches_torch(self):
        torch.manual_seed(7)
        source = torch.nn.MultiheadAttention(32, 4).eval()
        candidate = replace_multihead_attention_with_qat(source).eval()
        disable_fake_quant(candidate)
        x = torch.randn(9, 2, 32)
        mask = torch.empty(9, 9).fill_(float("-inf")).triu_(1)
        expected = source(x, x, x, need_weights=False, attn_mask=mask)[0]
        actual = candidate(x, x, x, need_weights=False, attn_mask=mask)[0]
        error = (expected - actual).abs()
        self.assertLess(error.max().item(), 1e-5)
        self.assertLess(error.mean().item(), 1e-6)
