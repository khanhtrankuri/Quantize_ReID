"""
apply_qat_clip.py

Ham chinh: apply_qat_to_clip(model, ...).

CHI goi ham nay SAU KHI build_model() (trong model.py) da load checkpoint
pretrained thanh cong. Ham nay PATCH submodule (thay nn.Conv2d/nn.Linear bang
QATConv2d/QATLinear, giu nguyen weight da pretrained), KHONG tao lai model tu dau.

Chien luoc quantize ap dung:
  - Bottleneck (ResNet): quantize Conv1/2/3 + downsample conv (giu BN tach rieng,
    KHONG fuse luc training - chi fuse sau khi train QAT xong, truoc convert)
  - Stem 3 conv dau ModifiedResNet: quantize tuong tu Bottleneck
  - AttentionPool2d: q/k/v_proj/c_proj -> weight-only (vi forward doc truc tiep
    .weight/.bias roi dua vao F.multi_head_attention_forward)
  - ResidualAttentionBlock.mlp (c_fc, c_proj): full QATLinear - uu tien cao nhat
  - ResidualAttentionBlock.attn: GIU NGUYEN nn.MultiheadAttention (mac dinh),
    chi thay bang QATMultiheadAttention neu quantize_attention_internals=True
  - VisionTransformer.conv1 (patch embedding): full QATConv2d
  - KHONG dong vao: LayerNorm, embedding, positional_embedding, logit_scale,
    phan cosine similarity cuoi cung trong CLIP.forward()
"""

from __future__ import annotations

import torch.nn as nn

from .model import Bottleneck, AttentionPool2d, ModifiedResNet, VisionTransformer, ResidualAttentionBlock, CLIP
from .qat_layers import QATConv2d, QATLinear, QATWeightOnly


# ---------------------------------------------------------------------------
# 1. Bottleneck (ResNet block trong ModifiedResNet)
# ---------------------------------------------------------------------------

def patch_bottleneck(bottleneck: Bottleneck) -> Bottleneck:
    """
    Thay conv1/conv2/conv3 (va downsample conv neu co) bang QATConv2d, INPLACE.
    KHONG fuse BatchNorm o day - BN van tach rieng, chay binh thuong sau Conv
    da fake-quantized. Chi fuse BN sau khi train QAT xong (dung fuse_bottleneck()
    trong qat_layers.py truoc khi convert sang INT8 thuc).
    """
    bottleneck.conv1 = QATConv2d.from_float(bottleneck.conv1)
    bottleneck.conv2 = QATConv2d.from_float(bottleneck.conv2)
    bottleneck.conv3 = QATConv2d.from_float(bottleneck.conv3)

    if bottleneck.downsample is not None:
        # downsample = nn.Sequential(OrderedDict([("-1", AvgPool2d), ("0", Conv2d), ("1", BatchNorm2d)]))
        bottleneck.downsample[1] = QATConv2d.from_float(bottleneck.downsample[1])

    return bottleneck


# ---------------------------------------------------------------------------
# 2. AttentionPool2d
# ---------------------------------------------------------------------------

def patch_attention_pool2d(pool: AttentionPool2d) -> AttentionPool2d:
    """
    q_proj/k_proj/v_proj/c_proj deu duoc AttentionPool2d.forward() doc truc tiep
    qua .weight/.bias roi dua vao F.multi_head_attention_forward (q_proj_weight=,
    k_proj_weight=, v_proj_weight=, out_proj_weight=, out_proj_bias=) - KHONG mot
    module nao trong 4 module nay duoc goi qua forward() binh thuong cua Linear.

    Vi vay CA 4 deu phai dung QATWeightOnly (chi fake-quantize .weight khi truy
    cap, khong tu chay F.linear). Truoc day c_proj dung QATLinear voi ky vong
    "full quant" (ca input lan weight), nhung vi forward() khong bao gio goi
    c_proj(x) ma chi lay c_proj.weight/c_proj.bias, nen input_fake_quant cua
    QATLinear la dead code (observer khong bao gio duoc cap nhat) va hanh vi
    thuc te da la weight-only tu dau - doi sang QATWeightOnly de code phan anh
    dung nhung gi thuc su dang xay ra.
    """
    pool.q_proj = QATWeightOnly.from_float(pool.q_proj)
    pool.k_proj = QATWeightOnly.from_float(pool.k_proj)
    pool.v_proj = QATWeightOnly.from_float(pool.v_proj)
    pool.c_proj = QATWeightOnly.from_float(pool.c_proj)
    return pool


# ---------------------------------------------------------------------------
# 3. ModifiedResNet (RN50 visual backbone)
# ---------------------------------------------------------------------------

def apply_qat_to_modified_resnet(visual: ModifiedResNet) -> ModifiedResNet:
    """
    Patch INPLACE: visual van la instance ModifiedResNet (khong doi class),
    chi cac Conv2d/Linear con ben trong duoc thay bang ban QAT. Vi
    ModifiedResNet.forward() goi self.conv1(x) (khong quan tam conv1 la
    nn.Conv2d hay QATConv2d nho tinh da hinh cua nn.Module), KHONG can sua
    forward() cua ModifiedResNet/Bottleneck chut nao.
    """
    # 3 stem convolutions - giu BN tach rieng, chi quantize Conv
    visual.conv1 = QATConv2d.from_float(visual.conv1)
    visual.conv2 = QATConv2d.from_float(visual.conv2)
    visual.conv3 = QATConv2d.from_float(visual.conv3)

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(visual, layer_name)
        for i, bottleneck in enumerate(layer):
            layer[i] = patch_bottleneck(bottleneck)

    visual.attnpool = patch_attention_pool2d(visual.attnpool)

    return visual


# ---------------------------------------------------------------------------
# 4. ResidualAttentionBlock - dung chung cho Vision Transformer va Text Transformer
# ---------------------------------------------------------------------------

def patch_residual_attention_block(
    block: ResidualAttentionBlock, quantize_attention_internals: bool = False
) -> ResidualAttentionBlock:
    """
    - mlp.c_fc, mlp.c_proj -> QATLinear (uu tien cao nhat, chiem FLOPs lon nhat)
    - attn -> mac dinh GIU NGUYEN nn.MultiheadAttention (F.multi_head_attention_forward
      la black-box, khong chen FakeQuantize vao giua duoc). Neu
      quantize_attention_internals=True, THAY THE HOAN TOAN bang QATMultiheadAttention
      (object moi, copy weight tu attn cu - xem qat_attention.py).
    - ln_1, ln_2 -> KHONG quantize (LayerNorm)
    - QuickGELU -> KHONG quantize (activation function)
    """
    block.mlp.c_fc = QATLinear.from_float(block.mlp.c_fc)
    block.mlp.c_proj = QATLinear.from_float(block.mlp.c_proj)

    if quantize_attention_internals:
        from .qat_attention import replace_multihead_attention_with_qat
        block.attn = replace_multihead_attention_with_qat(block.attn)
    # else: block.attn giu nguyen nn.MultiheadAttention, khong doi

    return block


# ---------------------------------------------------------------------------
# 5. VisionTransformer
# ---------------------------------------------------------------------------

def apply_qat_to_vision_transformer(
    visual: VisionTransformer, quantize_attention_internals: bool = False
) -> VisionTransformer:
    """Quantize patch-embedding conv1, va patch toan bo resblocks ben trong."""
    visual.conv1 = QATConv2d.from_float(visual.conv1)

    for i, block in enumerate(visual.transformer.resblocks):
        visual.transformer.resblocks[i] = patch_residual_attention_block(
            block, quantize_attention_internals=quantize_attention_internals
        )

    # ln_pre, ln_post: KHONG quantize (LayerNorm)
    # class_embedding, positional_embedding, proj: KHONG quantize (Parameter nho, nhay cam)
    return visual


# ---------------------------------------------------------------------------
# 6. Entry point chinh - ap dung cho toan bo CLIP model
# ---------------------------------------------------------------------------

def apply_qat_to_clip(model: CLIP, quantize_attention_internals: bool = False) -> CLIP:
    """
    Ap dung QAT cho toan bo model CLIP - tu dong nhan dien visual backbone la
    ModifiedResNet (RN50) hay VisionTransformer (ViT) va xu ly tuong ung.

    CHI goi ham nay SAU KHI model da duoc tao boi build_model(state_dict, ...)
    (trong model.py) va da load weight pretrained day du - vi cac ham
    QATConv2d.from_float()/QATLinear.from_float() can weight thuc de copy vao,
    khong phai random init.

    Args:
        model: instance CLIP da load pretrained weight (output cua build_model()).
        quantize_attention_internals: neu True, thay nn.MultiheadAttention bang
            QATMultiheadAttention (quantize ca Q@K^T va attn@V matmul). Mac dinh
            False - chi quantize Conv/Linear, an toan hon, du dung cho hau het truong hop.

    Returns:
        model da duoc patch INPLACE (cung instance, cac submodule con ben trong
        bi thay the bang ban QAT).

    KHONG dong vao:
        - LayerNorm (ln_1, ln_2, ln_pre, ln_post, ln_final)
        - class_embedding, positional_embedding, text_projection, proj
        - logit_scale va toan bo cosine similarity trong CLIP.forward()
        - token_embedding (nn.Embedding)
    """
    # --- Visual backbone ---
    if isinstance(model.visual, ModifiedResNet):
        apply_qat_to_modified_resnet(model.visual)
    elif isinstance(model.visual, VisionTransformer):
        apply_qat_to_vision_transformer(model.visual, quantize_attention_internals=quantize_attention_internals)
    else:
        raise TypeError(f"Khong nhan dien duoc loai visual backbone: {type(model.visual)}")

    # --- Text Transformer (luon ton tai, dung chung ResidualAttentionBlock) ---
    for i, block in enumerate(model.transformer.resblocks):
        model.transformer.resblocks[i] = patch_residual_attention_block(
            block, quantize_attention_internals=quantize_attention_internals
        )

    return model
