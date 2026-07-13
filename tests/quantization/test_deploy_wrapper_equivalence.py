import unittest
import torch
import torch.nn as nn
from model.clip.deploy import CLIPReIDImageDeploy


class _Encoder(nn.Module):
    def forward(self, x, cv=None):
        feature = x.mean((2, 3)).unsqueeze(1)
        return feature, feature, feature * 2


class DeployWrapperEquivalenceTest(unittest.TestCase):
    def test_vit_deploy_matches_full_inference_contract(self):
        class Full(nn.Module):
            def __init__(self):
                super().__init__(); self.image_encoder=_Encoder(); self.bottleneck=nn.BatchNorm1d(3); self.bottleneck_proj=nn.BatchNorm1d(3)
                self.model_name="ViT-B-16"; self.neck_feat="after"; self.sie_coe=1.; self.view_num=1
            def forward(self, x, cam_label=None, view_label=None):
                _, f, p=self.image_encoder(x); return torch.cat([self.bottleneck(f[:,0]),self.bottleneck_proj(p[:,0])],1)
        full=Full().eval(); deploy=CLIPReIDImageDeploy.from_full_model(full).eval(); image=torch.randn(2,3,4,4)
        error=(full(image)-deploy(image)).abs().max().item()
        self.assertLess(error,1e-6)
