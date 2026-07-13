import unittest
import torch
from model.clip.qat_layers import QATLinear, disable_fake_quant


class QATLinearTest(unittest.TestCase):
    def test_fp32_disable_fake_quant(self):
        source = torch.nn.Linear(8, 5)
        qat = QATLinear.from_float(source, name="unit.linear")
        disable_fake_quant(qat)
        x = torch.randn(4, 8)
        self.assertTrue(torch.allclose(source(x), qat(x), atol=1e-6))
        self.assertEqual(qat.weight_fake_quant.dtype, torch.qint8)
        self.assertEqual(qat.input_fake_quant.dtype, torch.qint8)
