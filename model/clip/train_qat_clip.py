"""
train_qat_clip.py

Vi du training loop QAT hoan chinh cho CLIP. Thay phan load data bang code
thuc te cua ban.

Quy trinh:
  1. build_model() - load checkpoint pretrained, model van la FP32 thuan
  2. apply_qat_to_clip() - patch QATConv2d/QATLinear vao, giu weight cu
  3. Calibration - chay vai batch, CHI observer hoc scale/zero_point, CHUA fake-quant
  4. QAT training - bat ca observer + fake-quant, finetune voi LR rat nho
  5. Freeze observer - chot scale, finetune them vai buoc cuoi cho on dinh
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import build_model
from .apply_qat_clip import apply_qat_to_clip
from .qat_layers import (
    enable_qat_observers,
    disable_qat_observers,
    enable_fake_quant,
    disable_fake_quant,
)


def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Standard CLIP contrastive loss - moi anh/text trong batch la positive pair voi chinh no."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def run_qat_pipeline(
    checkpoint_path: str,
    calibration_loader,
    train_loader,
    h_resolution: int = 7,
    w_resolution: int = 7,
    vision_stride_size: int = 32,
    num_calibration_steps: int = 50,
    num_qat_epochs: int = 2,
    lr: float = 1e-6,
    quantize_attention_internals: bool = False,
    device: str = None,
):
    """
    Args:
        checkpoint_path: duong dan file .pt checkpoint CLIP pretrained.
        calibration_loader: DataLoader tra ve dict co key "image", "text".
        train_loader: DataLoader tuong tu, dung de QAT finetune.
        h_resolution, w_resolution, vision_stride_size: tham so cho build_model().
        num_calibration_steps: so batch dung de calibrate observer.
        num_qat_epochs: so epoch QAT training.
        lr: learning rate - PHAI rat nho (1e-6 ~ 1e-5) vi day la finetune tu
            pretrained, chi can thich nghi voi nhieu luong tu hoa.
        quantize_attention_internals: xem apply_qat_to_clip().
        device: "cuda" hoac "cpu", tu dong chon neu None.

    Returns:
        model da QAT-train xong, san sang de fuse BN + convert sang INT8 thuc.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Buoc 1: Load checkpoint pretrained, build model FP32 thuan ---
    print(f"Loading checkpoint tu {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(state_dict, h_resolution, w_resolution, vision_stride_size)
    # Luc nay: model.transformer.resblocks[i].attn van la nn.MultiheadAttention,
    # toan bo weight da load dung tu checkpoint.

    # --- Buoc 2: Patch QAT - chen FakeQuantize vao Conv2d/Linear ---
    print("Patching model voi FakeQuantize ...")
    model = apply_qat_to_clip(model, quantize_attention_internals=quantize_attention_internals)
    model = model.to(device)

    # --- Buoc 3: Calibration (chi observer, chua fake-quant) ---
    print(f"Calibration: thu thap min/max qua {num_calibration_steps} batch ...")
    model.eval()
    enable_qat_observers(model)
    disable_fake_quant(model)  # chi quan sat, chua lam nhieu gradient

    with torch.no_grad():
        for step, batch in enumerate(calibration_loader):
            if step >= num_calibration_steps:
                break
            images = batch["image"].to(device)
            texts = batch["text"].to(device)
            model.encode_image(images)
            model.encode_text(texts)

    # --- Buoc 4: QAT training (ca observer + fake-quant) ---
    print(f"QAT training: {num_qat_epochs} epoch, lr={lr} ...")
    model.train()
    enable_qat_observers(model)
    enable_fake_quant(model)

    # LR PHAI rat nho - day la finetune tu pretrained, LR lon se pha hong weight cu
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_qat_epochs):
        total_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            texts = batch["text"].to(device)

            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, texts)
            loss = clip_contrastive_loss(logits_per_image, logits_per_text)
            loss.backward()  # STE (built-in trong FakeQuantize) tu xu ly gradient qua round()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if step % 50 == 0:
                print(f"  Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch} hoan thanh - Avg loss: {avg_loss:.4f}")

    # --- Buoc 5: Freeze observer, finetune cuoi voi scale co dinh ---
    print("Freeze observer (chot scale), finetune them de on dinh ...")
    disable_qat_observers(model)  # scale/zero_point khong doi nua
    enable_fake_quant(model)      # van fake-quantize, voi scale da chot

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1)
    for step, batch in enumerate(train_loader):
        if step >= 100:  # vai buoc cuoi de on dinh, khong can train lau
            break
        images = batch["image"].to(device)
        texts = batch["text"].to(device)

        optimizer.zero_grad()
        logits_per_image, logits_per_text = model(images, texts)
        loss = clip_contrastive_loss(logits_per_image, logits_per_text)
        loss.backward()
        optimizer.step()

    print("QAT pipeline hoan tat.")
    print("Buoc tiep theo: fuse Conv+BN (dung fuse_bottleneck() trong qat_layers.py)")
    print("roi convert sang INT8 thuc (vi du dung torchao) truoc khi deploy.")
    return model


if __name__ == "__main__":
    # Vi du goi - thay bang DataLoader thuc te cua ban
    raise NotImplementedError(
        "Thay phan nay bang calibration_loader/train_loader thuc te, "
        "roi goi run_qat_pipeline(checkpoint_path='RN50.pt', "
        "calibration_loader=..., train_loader=...)"
    )
