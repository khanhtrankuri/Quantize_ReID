import unittest
import torch
from model.clip.qat_layers import QATLinear
from model.clip.int8_convert import convert_linear_to_dynamic_int8


class Int8ConversionTest(unittest.TestCase):
    def test_qat_linear_is_packed_to_int8(self):
        model = torch.nn.Sequential(QATLinear(8, 4)); model(torch.randn(5, 8))
        converted = convert_linear_to_dynamic_int8(model)
        self.assertIsInstance(converted[0], torch.nn.quantized.dynamic.Linear)
        self.assertEqual(converted(torch.randn(2, 8)).shape, (2, 4))
