import unittest
import torch
from model.clip.model import VisionTransformer, Transformer
from model.clip.apply_qat_clip import apply_qat_to_clip
from model.clip.qat_layers import QATLinear
from model.clip.qat_attention import SelectiveQuantMultiheadAttention


class VisualOnlyPatchTest(unittest.TestCase):
    def test_text_transformer_is_unchanged(self):
        class Options:
            VISUAL_ONLY=True; QUANTIZE_TEXT_ENCODER=False; QUANTIZE_PATCH_EMBED=False
            QUANTIZE_MLP=True; QUANTIZE_QKV_PROJ=True; QUANTIZE_ATTN_OUT_PROJ=True
            QUANTIZE_QK_MATMUL=False; QUANTIZE_AV_MATMUL=False; EXCLUDE_FIRST_N_BLOCKS=0; EXCLUDE_LAST_N_BLOCKS=0
            WEIGHT_BITS=8; ACTIVATION_BITS=8; ACTIVATION_OBSERVER="moving_average"; ACTIVATION_PERCENTILE=99.99
            FP32_MODULE_PATTERNS=(); SENSITIVITY_JSON=""
        class ClipLike(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.visual=VisionTransformer(2, 2, 2, 2, 8, 1, 1, 4); self.transformer=Transformer(8, 1, 1)
        model = apply_qat_to_clip(ClipLike(), options=Options())
        self.assertIsInstance(model.visual.transformer.resblocks[0].mlp.c_fc, QATLinear)
        self.assertIsInstance(model.visual.transformer.resblocks[0].attn, SelectiveQuantMultiheadAttention)
        self.assertIsInstance(model.transformer.resblocks[0].attn, torch.nn.MultiheadAttention)
