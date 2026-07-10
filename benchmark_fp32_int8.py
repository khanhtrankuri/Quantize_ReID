import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.clip.int8_convert import bake_qat_fake_quant_weights, convert_linear_to_dynamic_int8
from model.make_model_clipreid import make_model


def parse_counts(text):
    counts = []
    for item in text.split(","):
        item = item.strip()
        if item:
            counts.append(int(item))
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("--counts must contain positive integers, for example 1,100,1000")
    return tuple(counts)


def clone_cfg(source_cfg, device, qat_enabled=None):
    cloned = source_cfg.clone()
    cloned.defrost()
    cloned.MODEL.DEVICE = device
    if qat_enabled is not None:
        cloned.MODEL.QAT.ENABLED = qat_enabled
    cloned.freeze()
    return cloned


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_fp32_model(model_cfg, weight_path, num_classes, camera_num, view_num):
    model = make_model(model_cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(weight_path)
    model.cpu().float().eval()
    return model


def load_int8_model(model_cfg, int8_model_path, int8_qat_weight, num_classes, camera_num, view_num):
    if int8_qat_weight:
        qat_cfg = clone_cfg(model_cfg, "cpu", qat_enabled=True)
        model = make_model(qat_cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        model.load_param(int8_qat_weight)
        model.cpu().float().eval()
        model = bake_qat_fake_quant_weights(model)
        return convert_linear_to_dynamic_int8(model)

    if int8_model_path:
        package = load_torch(int8_model_path)
        model = package["model"] if isinstance(package, dict) and "model" in package else package
        if not isinstance(model, nn.Module):
            raise TypeError("{} does not contain a serialized nn.Module.".format(int8_model_path))
        model.cpu().eval()
        return model

    raise ValueError("Pass either --int8_model or --int8_qat_weight.")


def collect_batches(val_loader, max_images):
    batches = []
    total_images = 0
    for img, _, _, camids, target_view, _ in val_loader:
        remaining = max_images - total_images
        if remaining <= 0:
            break
        if img.size(0) > remaining:
            img = img[:remaining]
            camids = camids[:remaining]
            target_view = target_view[:remaining]
        batches.append((img.cpu(), camids.cpu(), target_view.cpu()))
        total_images += img.size(0)

    if not batches:
        raise RuntimeError("No validation images were available.")
    return batches, total_images


def prepare_batch(model_cfg, img, camids, target_view):
    device = model_cfg.MODEL.DEVICE
    img = img.to(device)
    camids = camids.to(device) if model_cfg.MODEL.SIE_CAMERA else None
    target_view = target_view.to(device) if model_cfg.MODEL.SIE_VIEW else None
    return img, camids, target_view


def forward_features(model_cfg, model, cached_batches, max_images):
    device = model_cfg.MODEL.DEVICE
    use_cuda = str(device).startswith("cuda")
    model.to(device)
    model.eval()

    features = []
    collected = 0
    with torch.no_grad():
        for img, camids, target_view in cached_batches:
            remaining = max_images - collected
            if remaining <= 0:
                break
            if img.size(0) > remaining:
                img = img[:remaining]
                camids = camids[:remaining]
                target_view = target_view[:remaining]

            img, camids, target_view = prepare_batch(model_cfg, img, camids, target_view)
            if use_cuda:
                torch.cuda.synchronize()
            feat = model(img, cam_label=camids, view_label=target_view)
            if use_cuda:
                torch.cuda.synchronize()
            if model_cfg.TEST.FEAT_NORM == "yes":
                feat = F.normalize(feat, dim=1)
            features.append(feat.cpu())
            collected += feat.size(0)

    return torch.cat(features, dim=0)


def measure_speed(model_cfg, model, cached_batches, warmup_batches, label):
    device = model_cfg.MODEL.DEVICE
    use_cuda = str(device).startswith("cuda")
    model.to(device)
    model.eval()

    forward_time = 0.0
    timed_batches = 0
    timed_images = 0
    total_images = 0

    with torch.no_grad():
        for batch_idx, (img, camids, target_view) in enumerate(cached_batches):
            batch_size = img.size(0)
            img, camids, target_view = prepare_batch(model_cfg, img, camids, target_view)

            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(img, cam_label=camids, view_label=target_view)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if batch_idx >= warmup_batches:
                forward_time += elapsed
                timed_batches += 1
                timed_images += batch_size
            total_images += batch_size

    if timed_images == 0 or forward_time <= 0:
        return {
            "label": label,
            "total_images": total_images,
            "timed_images": timed_images,
            "timed_batches": timed_batches,
            "forward_time": forward_time,
            "latency_ms": None,
            "throughput": None,
        }

    return {
        "label": label,
        "total_images": total_images,
        "timed_images": timed_images,
        "timed_batches": timed_batches,
        "forward_time": forward_time,
        "latency_ms": 1000.0 * forward_time / timed_images,
        "throughput": timed_images / forward_time,
    }


def cosine_table(fp32_features, int8_features, counts):
    available = min(fp32_features.size(0), int8_features.size(0))
    rows = []
    for requested in counts:
        count = min(requested, available)
        fp32 = F.normalize(fp32_features[:count].float(), dim=1)
        int8 = F.normalize(int8_features[:count].float(), dim=1)
        cosine = F.cosine_similarity(fp32, int8, dim=1)
        rows.append(
            {
                "requested": requested,
                "count": count,
                "mean": cosine.mean().item(),
                "min": cosine.min().item(),
                "max": cosine.max().item(),
                "std": cosine.std(unbiased=False).item() if count > 1 else 0.0,
            }
        )
    return rows


def print_speed(speed):
    if speed["throughput"] is None:
        print("{} speed: not enough timed images".format(speed["label"]))
        return
    print(
        "{} speed: {:.3f} ms/image, {:.2f} images/s, forward {:.4f}s over {} images ({} timed batches)".format(
            speed["label"],
            speed["latency_ms"],
            speed["throughput"],
            speed["forward_time"],
            speed["timed_images"],
            speed["timed_batches"],
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs INT8 CLIP-ReID cosine and inference speed")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str)
    parser.add_argument("--fp32_weight", default="output/it_qat/ViT-B-16_fp32_40.pth", type=str)
    parser.add_argument("--int8_model", default="output/it_qat_from_fp32_40/ViT-B-16_int8.pt", type=str)
    parser.add_argument("--int8_qat_weight", default="", type=str, help="optional QAT .pth to convert to INT8 in memory")
    parser.add_argument("--counts", default="1,100,1000", type=str)
    parser.add_argument("--warmup_batches", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    counts = parse_counts(args.counts)
    max_images = max(counts)

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.DEVICE = "cpu"
    cfg.TEST.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    _, _, val_loader, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    cached_batches, cached_images = collect_batches(val_loader, max_images)

    fp32_cfg = clone_cfg(cfg, "cpu", qat_enabled=False)
    int8_cfg = clone_cfg(cfg, "cpu")
    fp32_model = load_fp32_model(fp32_cfg, args.fp32_weight, num_classes, camera_num, view_num)
    int8_model = load_int8_model(int8_cfg, args.int8_model, args.int8_qat_weight, num_classes, camera_num, view_num)

    print("Cached validation images: {}".format(cached_images))
    print("FP32 weight: {}".format(args.fp32_weight))
    print("INT8 source: {}".format(args.int8_qat_weight or args.int8_model))
    print("")

    fp32_speed = measure_speed(fp32_cfg, fp32_model, cached_batches, args.warmup_batches, "FP32")
    int8_speed = measure_speed(int8_cfg, int8_model, cached_batches, args.warmup_batches, "INT8")
    print_speed(fp32_speed)
    print_speed(int8_speed)

    if fp32_speed["throughput"] and int8_speed["throughput"]:
        throughput_ratio = int8_speed["throughput"] / fp32_speed["throughput"]
        latency_ratio = fp32_speed["latency_ms"] / int8_speed["latency_ms"]
        print("INT8 / FP32 throughput ratio: {:.3f}x".format(throughput_ratio))
        print("INT8 / FP32 latency ratio: {:.3f}x".format(latency_ratio))

    print("")
    fp32_features = forward_features(fp32_cfg, fp32_model, cached_batches, max_images)
    int8_features = forward_features(int8_cfg, int8_model, cached_batches, max_images)
    print("Cosine similarity between FP32 and INT8 features")
    for row in cosine_table(fp32_features, int8_features, counts):
        print(
            "images {:4d} compared {:4d}: mean {:.6f}, min {:.6f}, max {:.6f}, std {:.6f}".format(
                row["requested"],
                row["count"],
                row["mean"],
                row["min"],
                row["max"],
                row["std"],
            )
        )


if __name__ == "__main__":
    main()
