import unittest
import torch
from model.clip.qat_layers import QATLinear


class FP32CheckpointToQATTest(unittest.TestCase):
    def test_from_float_preserves_weights(self):
        source = torch.nn.Linear(7, 9)
        qat = QATLinear.from_float(source)
        self.assertTrue(torch.equal(source.weight, qat.linear.weight))
        self.assertTrue(torch.equal(source.bias, qat.linear.bias))
