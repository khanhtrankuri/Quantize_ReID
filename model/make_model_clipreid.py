from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .clip.apply_qat_clip import (
    apply_qat_to_clip,
    apply_qat_to_modified_resnet,
    apply_qat_to_vision_transformer,
    patch_residual_attention_block,
)
from .clip.model import ModifiedResNet, VisionTransformer

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(
            self.model_name,
            self.h_resolution,
            self.w_resolution,
            self.vision_stride_size,
            cfg.MODEL.PRETRAIN_PATH,
        )
        if cfg.MODEL.QAT.ENABLED:
            print("QAT enabled: patching CLIP visual/text modules with fake quantization")
            clip_model = apply_qat_to_clip(
                clip_model,
                quantize_attention_internals=cfg.MODEL.QAT.QUANTIZE_ATTENTION_INTERNALS,
            )
        clip_model.to(cfg.MODEL.DEVICE)

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(
            num_classes,
            dataset_name,
            clip_model.dtype,
            clip_model.token_embedding,
            device=cfg.MODEL.DEVICE,
        )
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text == True:
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location="cpu")
        param_dict = _extract_checkpoint_state_dict(param_dict)
        model_state = self.state_dict()
        loaded_keys = []
        skipped_keys = []

        for i, value in param_dict.items():
            clean_key = i.replace('module.', '')
            if torch.is_tensor(value) and clean_key in model_state and model_state[clean_key].shape == value.shape:
                model_state[clean_key].copy_(value)
                loaded_keys.append(clean_key)
            else:
                skipped_keys.append(clean_key)

        if not loaded_keys:
            raise RuntimeError(
                "No tensors were loaded from {}. Check whether MODEL.QAT.ENABLED matches the checkpoint type "
                "(FP32 checkpoint -> MODEL.QAT.ENABLED False, QAT checkpoint -> MODEL.QAT.ENABLED True). "
                "First skipped keys: {}".format(trained_path, skipped_keys[:10])
            )

        print('Loading pretrained model from {}'.format(trained_path))
        if skipped_keys:
            print(
                "Loaded {} tensors; skipped {} unmatched tensors. First skipped keys: {}".format(
                    len(loaded_keys), len(skipped_keys), skipped_keys[:10]
                )
            )

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


def apply_qat_to_clipreid_model(model, quantize_attention_internals=False):
    """
    Patch QAT vao wrapper CLIP-ReID SAU KHI da load checkpoint FP32.

    Dung cho luong finetune tu checkpoint ReID da train xong:
      1. build model voi MODEL.QAT.ENABLED = False
      2. load checkpoint FP32
      3. goi helper nay de thay Conv/Linear CLIP bang module QAT

    Cach nay tranh lech key state_dict giua checkpoint FP32 va module QAT.
    """
    if isinstance(model.image_encoder, ModifiedResNet):
        apply_qat_to_modified_resnet(
            model.image_encoder,
            quantize_attention_internals=quantize_attention_internals,
        )
    elif isinstance(model.image_encoder, VisionTransformer):
        apply_qat_to_vision_transformer(
            model.image_encoder,
            quantize_attention_internals=quantize_attention_internals,
        )
    else:
        raise TypeError(f"Khong nhan dien duoc image_encoder: {type(model.image_encoder)}")

    for i, block in enumerate(model.text_encoder.transformer.resblocks):
        model.text_encoder.transformer.resblocks[i] = patch_residual_attention_block(
            block,
            quantize_attention_internals=quantize_attention_internals,
        )

    return model


from .clip import clip

def _extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for state_key in ("state_dict", "model", "model_state_dict"):
            if state_key in checkpoint and isinstance(checkpoint[state_key], dict):
                return checkpoint[state_key]
    return checkpoint


def _convert_clipreid_state_dict(state_dict):
    if any(key.startswith("visual.") for key in state_dict):
        return state_dict

    converted = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[len("module."):] if key.startswith("module.") else key
        if clean_key.startswith("image_encoder."):
            converted["visual." + clean_key[len("image_encoder."):]] = value
        elif clean_key.startswith("text_encoder."):
            converted[clean_key[len("text_encoder."):]] = value

    if not converted:
        return state_dict

    if "token_embedding.weight" not in converted:
        text_width = converted["positional_embedding"].shape[-1]
        converted["token_embedding.weight"] = torch.empty(
            49408,
            text_width,
            dtype=converted["positional_embedding"].dtype,
        )
        nn.init.normal_(converted["token_embedding.weight"], std=0.02)
    if "logit_scale" not in converted:
        converted["logit_scale"] = torch.ones([], dtype=converted["positional_embedding"].dtype) * np.log(1 / 0.07)

    print("Converted CLIP-ReID checkpoint keys to CLIP backbone state_dict.")
    return converted


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size, pretrained_path=""):
    if pretrained_path:
        print("Loading CLIP backbone from MODEL.PRETRAIN_PATH: {}".format(pretrained_path))
        try:
            model = torch.jit.load(pretrained_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = _extract_checkpoint_state_dict(checkpoint)
            state_dict = _convert_clipreid_state_dict(state_dict)
        return clip.build_model(state_dict, h_resolution, w_resolution, vision_stride_size)

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding, device="cuda"):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).to(device)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 
