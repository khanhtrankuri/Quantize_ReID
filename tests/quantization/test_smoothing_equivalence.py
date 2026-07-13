import unittest
import torch
from model.clip.smoothing import fold_smoothing_into_pair


class SmoothingEquivalenceTest(unittest.TestCase):
    def test_adjacent_linear_fold_preserves_output(self):
        first, second = torch.nn.Linear(6, 8), torch.nn.Linear(8, 3)
        x = torch.randn(4, 6); expected = second(first(x))
        fold_smoothing_into_pair(first, second, torch.rand(8) + 0.1)
        self.assertTrue(torch.allclose(expected, second(first(x)), atol=1e-6))
