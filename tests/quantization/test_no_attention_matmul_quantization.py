import unittest
from model.clip.qat_attention import SelectiveQuantMultiheadAttention


class NoAttentionMatmulQuantizationTest(unittest.TestCase):
    def test_default_has_no_qk_or_av_fake_quant(self):
        attention = SelectiveQuantMultiheadAttention(16, 4)
        self.assertIsNone(attention.qk_matmul)
        self.assertIsNone(attention.av_matmul)
