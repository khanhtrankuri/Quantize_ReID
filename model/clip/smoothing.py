"""SmoothQuant utilities for adjacent transformer linear layers."""
from __future__ import annotations
import torch
import torch.nn as nn


@torch.no_grad()
def activation_channel_max(batches, module, max_batches=50):
    """Collect max-absolute per-input-channel activations with a temporary hook."""
    maximum = None
    def hook(_, inputs):
        nonlocal maximum
        value = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).abs().amax(0).cpu()
        maximum = value if maximum is None else torch.maximum(maximum, value)
    handle = module.register_forward_pre_hook(hook)
    try:
        for index, run in enumerate(batches):
            if index >= max_batches:
                break
            run()
    finally:
        handle.remove()
    if maximum is None:
        raise RuntimeError("No calibration activation reached the requested module")
    return maximum


@torch.no_grad()
def smoothing_scale(activation_max, weight, alpha=0.5, eps=1e-8):
    weight_max = weight.detach().float().abs().amax(dim=0).cpu()
    scale = activation_max.clamp_min(eps).pow(alpha) / weight_max.clamp_min(eps).pow(1.0 - alpha)
    return scale.clamp_min(eps)


@torch.no_grad()
def fold_smoothing_into_pair(previous: nn.Linear, following: nn.Linear, scale: torch.Tensor):
    """Exact FP32 fold for an adjacent linear pair: prev output / s, next W * s."""
    if previous.out_features != following.in_features or scale.numel() != previous.out_features:
        raise ValueError("Smoothing scale does not match adjacent linear channels")
    scale = scale.to(previous.weight.device, previous.weight.dtype)
    previous.weight.div_(scale[:, None])
    if previous.bias is not None:
        previous.bias.div_(scale)
    following.weight.mul_(scale[None, :].to(following.weight.device, following.weight.dtype))
    return scale


@torch.no_grad()
def fold_smoothing_into_layernorm(layer_norm, linears, alpha=0.5, eps=1e-8):
    """Fold an exact SmoothQuant transform into LayerNorm and its consumers.

    The LayerNorm affine output is divided by `s`; all consumer input columns
    are multiplied by `s`.  This has no inference-time multiply and preserves
    FP32 output exactly before fake quantization.
    """
    linears = [item for item in linears if hasattr(item, "linear")]
    if not linears or any(bool(item.smoothing_applied.item()) for item in linears):
        return None
    activation = torch.stack([item.input_channel_absmax for item in linears]).amax(0)
    weight = torch.stack([item.linear.weight.detach().float().abs().amax(0).cpu() for item in linears]).amax(0)
    scale = smoothing_scale(activation.cpu(), weight.cpu(), alpha, eps)
    device, dtype = layer_norm.weight.device, layer_norm.weight.dtype
    layer_norm.weight.div_(scale.to(device, dtype))
    if layer_norm.bias is not None:
        layer_norm.bias.div_(scale.to(layer_norm.bias.device, layer_norm.bias.dtype))
    for item in linears:
        item.linear.weight.mul_(scale.to(item.linear.weight.device, item.linear.weight.dtype)[None, :])
        item.smoothing_scale.copy_(scale.to(item.smoothing_scale))
        item.smoothing_applied.fill_(True)
    return scale


@torch.no_grad()
def smooth_transformer_blocks(visual, alpha=0.5):
    """Fold smoothing for ViT LN2->MLP and LN1->Q/K/V projection fan-outs."""
    from .qat_layers import QATLinear
    applied = 0
    for block in visual.transformer.resblocks:
        c_fc = getattr(block.mlp, "c_fc", None)
        if isinstance(c_fc, QATLinear) and fold_smoothing_into_layernorm(block.ln_2, [c_fc], alpha) is not None:
            applied += 1
        attention = getattr(block, "attn", None)
        projections = [getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj")]
        if all(isinstance(item, QATLinear) for item in projections):
            if fold_smoothing_into_layernorm(block.ln_1, projections, alpha) is not None:
                applied += 1
    return applied
