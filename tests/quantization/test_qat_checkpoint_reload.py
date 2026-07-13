import unittest
import torch
from model.clip.qat_layers import QATLinear


class QATCheckpointReloadTest(unittest.TestCase):
    def test_qparams_are_in_state_dict(self):
        first = QATLinear(4, 4); first(torch.randn(3, 4))
        state = first.state_dict(); second = QATLinear(4, 4); second.load_state_dict(state)
        self.assertIn("weight_fake_quant.scale", state)
        self.assertTrue(torch.equal(first.weight_fake_quant.scale, second.weight_fake_quant.scale))
