"""Image-only deployment wrapper for CLIP-ReID."""
from __future__ import annotations
import torch
import torch.nn as nn


class CLIPReIDImageDeploy(nn.Module):
    """The minimal inference graph: visual encoder, SIE and BN necks only."""
    def __init__(self, model):
        super().__init__()
        self.image_encoder = model.image_encoder
        self.bottleneck = model.bottleneck
        self.bottleneck_proj = model.bottleneck_proj
        self.model_name = model.model_name
        self.neck_feat = model.neck_feat
        self.sie_coe = model.sie_coe
        self.view_num = model.view_num
        if hasattr(model, "cv_embed"):
            self.cv_embed = model.cv_embed

    def forward(self, images, cam_label=None, view_label=None):
        if self.model_name == "RN50":
            last, features, projected = self.image_encoder(images)
            feature = nn.functional.avg_pool2d(features, features.shape[2:]).flatten(1)
            feature_proj = projected[0]
        else:
            cv_embed = None
            if hasattr(self, "cv_embed"):
                if cam_label is not None and view_label is not None:
                    cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
                elif cam_label is not None:
                    cv_embed = self.sie_coe * self.cv_embed[cam_label]
                elif view_label is not None:
                    cv_embed = self.sie_coe * self.cv_embed[view_label]
            _, features, projected = self.image_encoder(images, cv_embed)
            feature, feature_proj = features[:, 0], projected[:, 0]
        if self.neck_feat == "after":
            feature, feature_proj = self.bottleneck(feature), self.bottleneck_proj(feature_proj)
        return torch.cat([feature, feature_proj], dim=1)

    @classmethod
    def from_full_model(cls, model):
        return cls(model)
