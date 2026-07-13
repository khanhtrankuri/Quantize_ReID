import unittest
import torch
from processor.processor_clipreid_stage2 import _distillation_loss


class DistillationLossTest(unittest.TestCase):
    def test_teacher_receives_no_gradient(self):
        class D: FEATURE_WEIGHT=1.; RELATION_WEIGHT=5.; DISTANCE_WEIGHT=0.
        class Q: DISTILLATION=D()
        class M: QAT=Q()
        class C: MODEL=M()
        teacher = torch.randn(3, 6, requires_grad=True); student = torch.randn(3, 6, requires_grad=True)
        loss, *_ = _distillation_loss(student, teacher, C()); loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
