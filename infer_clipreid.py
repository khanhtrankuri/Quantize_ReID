import argparse
from collections import OrderedDict
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.clip.int8_convert import bake_qat_fake_quant_weights, convert_linear_to_dynamic_int8
from model.make_model_clipreid import make_model
from utils.metrics import R1_mAP_eval
from utils.logger import setup_logger


def extract_features(cfg, model, val_loader):
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()

    features = []
    pids = []
    camids = []
    paths = []

    with torch.no_grad():
        for img, pid, camid, camids_batch, target_view, imgpath in val_loader:
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids_batch = camids_batch.to(device)
            else:
                camids_batch = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            feat = model(img, cam_label=camids_batch, view_label=target_view)
            if cfg.TEST.FEAT_NORM == "yes":
                feat = F.normalize(feat, dim=1)

            features.append(feat.cpu())
            pids.extend(pid)
            camids.extend(camid)
            paths.extend(imgpath)

    features = torch.cat(features, dim=0)
    return features, np.asarray(pids), np.asarray(camids), np.asarray(paths)


def save_rankings(features, num_query, paths, output_path, topk):
    query_features = features[:num_query]
    gallery_features = features[num_query:]
    gallery_paths = paths[num_query:]
    topk = min(topk, gallery_features.size(0))
    if topk <= 0:
        raise ValueError("Gallery split is empty; cannot save top-k rankings.")

    chunk_size = int(os.environ.get("REID_RANKING_QUERY_CHUNK_SIZE", "64"))
    chunk_size = max(1, chunk_size)

    rankings = []
    query_features = query_features.float()
    gallery_features = gallery_features.float()
    with torch.no_grad():
        for start in range(0, query_features.size(0), chunk_size):
            end = min(start + chunk_size, query_features.size(0))
            dist = torch.cdist(query_features[start:end], gallery_features, p=2)
            distances, indices = torch.topk(dist, k=topk, dim=1, largest=False, sorted=True)
            distances = distances.cpu().numpy()
            indices = indices.cpu().numpy()

            for row_idx, gallery_indices in enumerate(indices):
                query_idx = start + row_idx
                rankings.append(
                    {
                        "query": str(paths[query_idx]),
                        "gallery": [str(gallery_paths[i]) for i in gallery_indices],
                        "distance": [float(distances[row_idx, i]) for i in range(len(gallery_indices))],
                    }
                )

    np.savez_compressed(
        output_path,
        features=features.numpy(),
        paths=paths,
        rankings=np.asarray(rankings, dtype=object),
    )


def clone_cfg_with_device(source_cfg, device, qat_enabled=None):
    cloned = source_cfg.clone()
    cloned.defrost()
    cloned.MODEL.DEVICE = device
    if qat_enabled is not None:
        cloned.MODEL.QAT.ENABLED = qat_enabled
    cloned.freeze()
    return cloned


def load_torch_package(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(package):
    if isinstance(package, OrderedDict):
        return package
    if isinstance(package, dict):
        if "model" in package and not isinstance(package["model"], nn.Module):
            return extract_state_dict(package["model"])
        if "state_dict" in package:
            return package["state_dict"]
        if "model_state_dict" in package:
            return package["model_state_dict"]
        if package and all(torch.is_tensor(value) for value in package.values()):
            return package
    return None


def checkpoint_looks_qat(state_dict):
    qat_tokens = ("fake_quant", "activation_post_process", ".linear.", ".conv.")
    return any(any(token in key for token in qat_tokens) for key in state_dict.keys())


def load_state_dict_flexible(model, state_dict, weight_path):
    model_state = model.state_dict()
    loaded = 0
    skipped = []
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            model_state[clean_key].copy_(value)
            loaded += 1
        else:
            skipped.append(key)

    if loaded == 0:
        raise RuntimeError("No tensors were loaded from {}".format(weight_path))

    logger = logging.getLogger("transreid.test")
    logger.info("Loaded {} tensors from {}".format(loaded, weight_path))
    if skipped:
        logger.info("Skipped {} tensors because key/shape did not match.".format(len(skipped)))


def build_dynamic_int8_from_checkpoint(source_cfg, int8_model_path, num_classes, camera_num, view_num, package):
    state_dict = extract_state_dict(package)
    if state_dict is None:
        raise TypeError("Unsupported INT8 model package type: {}".format(type(package).__name__))

    is_qat_checkpoint = checkpoint_looks_qat(state_dict)
    model_cfg = clone_cfg_with_device(source_cfg, "cpu", qat_enabled=is_qat_checkpoint)
    model = make_model(model_cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_state_dict_flexible(model, state_dict, int8_model_path)
    model.cpu().float().eval()

    if is_qat_checkpoint:
        model = bake_qat_fake_quant_weights(model)

    logging.getLogger("transreid.test").info(
        "{} is a {} checkpoint; converting eligible Linear layers to dynamic INT8 on CPU.".format(
            int8_model_path, "QAT" if is_qat_checkpoint else "FP32"
        )
    )
    return convert_linear_to_dynamic_int8(model)


def load_int8_model(int8_model_path, source_cfg=None, num_classes=None, camera_num=None, view_num=None):
    package = load_torch_package(int8_model_path)
    model = package["model"] if isinstance(package, dict) and "model" in package else package
    if isinstance(model, nn.Module):
        model.cpu().eval()
        return model

    if source_cfg is not None and num_classes is not None and camera_num is not None and view_num is not None:
        return build_dynamic_int8_from_checkpoint(source_cfg, int8_model_path, num_classes, camera_num, view_num, package)

    raise TypeError(
        "{} does not contain a serialized nn.Module. It looks like a checkpoint/state_dict; "
        "run with --config_file so the script can rebuild and convert it, or create a .pt file with convert_int8.py.".format(
            int8_model_path
        )
    )


def prepare_batch_for_device(cfg, img, camids, target_view, device):
    img = img.to(device)
    if cfg.MODEL.SIE_CAMERA:
        camids = camids.to(device)
    else:
        camids = None
    if cfg.MODEL.SIE_VIEW:
        target_view = target_view.to(device)
    else:
        target_view = None
    return img, camids, target_view


def summarize_speed(forward_time, timed_batches, timed_images):
    if timed_images <= 0 or forward_time <= 0:
        return {
            "forward_time": forward_time,
            "timed_batches": timed_batches,
            "timed_images": timed_images,
            "latency_ms_per_image": None,
            "batch_ms": None,
            "throughput_img_s": None,
        }

    return {
        "forward_time": forward_time,
        "timed_batches": timed_batches,
        "timed_images": timed_images,
        "latency_ms_per_image": 1000.0 * forward_time / timed_images,
        "batch_ms": 1000.0 * forward_time / timed_batches,
        "throughput_img_s": timed_images / forward_time,
    }


def do_inference_with_speed(cfg, model, val_loader, num_query, warmup_batches):
    device = cfg.MODEL.DEVICE
    use_cuda = str(device).startswith("cuda")
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        num_samples=len(val_loader.dataset),
    )
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    timed_batches = 0
    timed_images = 0
    forward_time = 0.0
    total_start = time.perf_counter()

    with torch.no_grad():
        for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            feat = model(img, cam_label=camids, view_label=target_view)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if n_iter >= warmup_batches:
                forward_time += elapsed
                timed_batches += 1
                timed_images += img.size(0)

            evaluator.update((feat, pid, camid))

    total_time = time.perf_counter() - total_start
    cmc, mAP, _, _, _, _, _ = evaluator.compute()

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    if timed_images > 0 and forward_time > 0:
        speed = summarize_speed(forward_time, timed_batches, timed_images)
        logger.info(
            "Inference speed - warmup batches: {}, timed batches: {}, timed images: {}".format(
                warmup_batches, timed_batches, timed_images
            )
        )
        logger.info(
            "Forward only - total: {:.4f}s, latency: {:.3f} ms/image, batch time: {:.3f} ms/batch, throughput: {:.2f} images/s".format(
                forward_time,
                speed["latency_ms_per_image"],
                speed["batch_ms"],
                speed["throughput_img_s"],
            )
        )
    else:
        speed = summarize_speed(forward_time, timed_batches, timed_images)
        logger.info("Inference speed - not enough batches after warmup to report forward timing")
    logger.info("End-to-end inference pass time: {:.4f}s".format(total_time))

    return {
        "rank1": cmc[0],
        "rank5": cmc[4],
        "mAP": mAP,
        "total_time": total_time,
        "speed": speed,
    }


def measure_inference_speed(cfg, model, val_loader, warmup_batches, max_images=1000, label="model"):
    device = cfg.MODEL.DEVICE
    use_cuda = str(device).startswith("cuda")
    logger = logging.getLogger("transreid.test")

    if device:
        model.to(device)
    model.eval()

    timed_batches = 0
    timed_images = 0
    forward_time = 0.0
    total_images = 0
    total_start = time.perf_counter()

    with torch.no_grad():
        for n_iter, (img, _, _, camids, target_view, _) in enumerate(val_loader):
            if max_images > 0:
                remaining = max_images - total_images
                if remaining <= 0:
                    break
                if img.size(0) > remaining:
                    img = img[:remaining]
                    camids = camids[:remaining]
                    target_view = target_view[:remaining]

            batch_size = img.size(0)
            img, camids, target_view = prepare_batch_for_device(cfg, img, camids, target_view, device)

            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(img, cam_label=camids, view_label=target_view)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if n_iter >= warmup_batches:
                forward_time += elapsed
                timed_batches += 1
                timed_images += batch_size

            total_images += batch_size

    total_time = time.perf_counter() - total_start
    speed = summarize_speed(forward_time, timed_batches, timed_images)
    logger.info(
        "{} speed - max images: {}, processed images: {}, warmup batches: {}, timed batches: {}, timed images: {}".format(
            label, max_images if max_images > 0 else "all", total_images, warmup_batches, timed_batches, timed_images
        )
    )
    if speed["throughput_img_s"]:
        logger.info(
            "{} forward only - total: {:.4f}s, latency: {:.3f} ms/image, batch time: {:.3f} ms/batch, throughput: {:.2f} images/s".format(
                label,
                speed["forward_time"],
                speed["latency_ms_per_image"],
                speed["batch_ms"],
                speed["throughput_img_s"],
            )
        )
    else:
        logger.info("{} speed - not enough batches after warmup to report forward timing".format(label))
    logger.info("{} end-to-end measured pass time: {:.4f}s".format(label, total_time))

    return {
        "rank1": None,
        "rank5": None,
        "mAP": None,
        "total_time": total_time,
        "speed": speed,
    }


def collect_validation_batches(val_loader, max_images):
    cached_batches = []
    total_images = 0
    for img, _, _, camids, target_view, _ in val_loader:
        if max_images > 0:
            remaining = max_images - total_images
            if remaining <= 0:
                break
            if img.size(0) > remaining:
                img = img[:remaining]
                camids = camids[:remaining]
                target_view = target_view[:remaining]

        cached_batches.append((img.cpu(), camids.cpu(), target_view.cpu()))
        total_images += img.size(0)

    if not cached_batches:
        raise RuntimeError("No validation images were available for inference comparison.")

    return cached_batches, total_images


def measure_cached_inference_speed(cfg, model, cached_batches, warmup_batches, label="model"):
    device = cfg.MODEL.DEVICE
    use_cuda = str(device).startswith("cuda")
    logger = logging.getLogger("transreid.test")

    if device:
        model.to(device)
    model.eval()

    timed_batches = 0
    timed_images = 0
    forward_time = 0.0
    total_images = 0
    total_start = time.perf_counter()

    with torch.no_grad():
        for n_iter, (img, camids, target_view) in enumerate(cached_batches):
            batch_size = img.size(0)
            img, camids, target_view = prepare_batch_for_device(cfg, img, camids, target_view, device)

            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(img, cam_label=camids, view_label=target_view)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if n_iter >= warmup_batches:
                forward_time += elapsed
                timed_batches += 1
                timed_images += batch_size

            total_images += batch_size

    total_time = time.perf_counter() - total_start
    speed = summarize_speed(forward_time, timed_batches, timed_images)
    logger.info(
        "{} speed - cached images: {}, warmup batches: {}, timed batches: {}, timed images: {}".format(
            label, total_images, warmup_batches, timed_batches, timed_images
        )
    )
    if speed["throughput_img_s"]:
        logger.info(
            "{} forward only - total: {:.4f}s, latency: {:.3f} ms/image, batch time: {:.3f} ms/batch, throughput: {:.2f} images/s".format(
                label,
                speed["forward_time"],
                speed["latency_ms_per_image"],
                speed["batch_ms"],
                speed["throughput_img_s"],
            )
        )
    else:
        logger.info("{} speed - not enough batches after warmup to report forward timing".format(label))
    logger.info("{} cached measured pass time: {:.4f}s".format(label, total_time))

    return {
        "rank1": None,
        "rank5": None,
        "mAP": None,
        "total_time": total_time,
        "speed": speed,
    }


def extract_feature_prefix(cfg, model, val_loader, max_images):
    device = cfg.MODEL.DEVICE
    use_cuda = str(device).startswith("cuda")
    model.to(device)
    model.eval()

    features = []
    collected = 0

    with torch.no_grad():
        for img, _, _, camids, target_view, _ in val_loader:
            remaining = max_images - collected
            if remaining <= 0:
                break
            if img.size(0) > remaining:
                img = img[:remaining]
                camids = camids[:remaining]
                target_view = target_view[:remaining]

            img, camids, target_view = prepare_batch_for_device(cfg, img, camids, target_view, device)
            if use_cuda:
                torch.cuda.synchronize()
            feat = model(img, cam_label=camids, view_label=target_view)
            if use_cuda:
                torch.cuda.synchronize()
            if cfg.TEST.FEAT_NORM == "yes":
                feat = F.normalize(feat, dim=1)

            features.append(feat.cpu())
            collected += feat.size(0)

    if not features:
        raise RuntimeError("No validation images were available for cosine comparison.")

    return torch.cat(features, dim=0)


def extract_cached_features(cfg, model, cached_batches, max_images):
    device = cfg.MODEL.DEVICE
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

            img, camids, target_view = prepare_batch_for_device(cfg, img, camids, target_view, device)
            if use_cuda:
                torch.cuda.synchronize()
            feat = model(img, cam_label=camids, view_label=target_view)
            if use_cuda:
                torch.cuda.synchronize()
            if cfg.TEST.FEAT_NORM == "yes":
                feat = F.normalize(feat, dim=1)

            features.append(feat.cpu())
            collected += feat.size(0)

    if not features:
        raise RuntimeError("No cached validation images were available for cosine comparison.")

    return torch.cat(features, dim=0)


def compare_cached_cosine_prefixes(fp32_cfg, fp32_model, int8_cfg, int8_model, cached_batches, sample_counts):
    max_images = max(sample_counts)
    fp32_features = extract_cached_features(fp32_cfg, fp32_model, cached_batches, max_images)
    int8_features = extract_cached_features(int8_cfg, int8_model, cached_batches, max_images)
    available = min(fp32_features.size(0), int8_features.size(0))

    results = []
    for requested in sample_counts:
        count = min(requested, available)
        fp32_slice = F.normalize(fp32_features[:count].float(), dim=1)
        int8_slice = F.normalize(int8_features[:count].float(), dim=1)
        cosine = F.cosine_similarity(fp32_slice, int8_slice, dim=1)
        results.append(
            {
                "requested": requested,
                "count": count,
                "mean": cosine.mean().item(),
                "min": cosine.min().item(),
                "max": cosine.max().item(),
                "std": cosine.std(unbiased=False).item() if count > 1 else 0.0,
            }
        )

    return results


def compare_cosine_prefixes(fp32_cfg, fp32_model, int8_cfg, int8_model, val_loader, sample_counts):
    max_images = max(sample_counts)
    fp32_features = extract_feature_prefix(fp32_cfg, fp32_model, val_loader, max_images)
    int8_features = extract_feature_prefix(int8_cfg, int8_model, val_loader, max_images)
    available = min(fp32_features.size(0), int8_features.size(0))

    results = []
    for requested in sample_counts:
        count = min(requested, available)
        fp32_slice = F.normalize(fp32_features[:count].float(), dim=1)
        int8_slice = F.normalize(int8_features[:count].float(), dim=1)
        cosine = F.cosine_similarity(fp32_slice, int8_slice, dim=1)
        results.append(
            {
                "requested": requested,
                "count": count,
                "mean": cosine.mean().item(),
                "min": cosine.min().item(),
                "max": cosine.max().item(),
                "std": cosine.std(unbiased=False).item() if count > 1 else 0.0,
            }
        )

    return results


def log_speed_comparison(fp32_result, int8_result):
    logger = logging.getLogger("transreid.test")
    fp32_speed = fp32_result["speed"]
    int8_speed = int8_result["speed"]

    if fp32_speed["throughput_img_s"] and int8_speed["throughput_img_s"]:
        throughput_ratio = int8_speed["throughput_img_s"] / fp32_speed["throughput_img_s"]
        latency_ratio = fp32_speed["latency_ms_per_image"] / int8_speed["latency_ms_per_image"]
        logger.info(
            "INT8 vs FP32 speed ratio - throughput: {:.3f}x, latency: {:.3f}x".format(
                throughput_ratio, latency_ratio
            )
        )
        if throughput_ratio >= 1.0:
            logger.info("INT8 is {:.3f}x faster than FP32 by throughput.".format(throughput_ratio))
        else:
            logger.info("INT8 is {:.3f}x slower than FP32 by throughput.".format(1.0 / throughput_ratio))
    else:
        logger.info("INT8 vs FP32 speed ratio unavailable because one run had no timed images.")


def log_cosine_comparison(cosine_results):
    logger = logging.getLogger("transreid.test")
    logger.info("Cosine similarity between FP32 and INT8 feature vectors")
    for item in cosine_results:
        logger.info(
            "Images requested: {:4d}, compared: {:4d}, cosine mean: {:.6f}, min: {:.6f}, max: {:.6f}, std: {:.6f}".format(
                item["requested"],
                item["count"],
                item["mean"],
                item["min"],
                item["max"],
                item["std"],
            )
        )


def parse_sample_counts(text):
    counts = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        count = int(item)
        if count <= 0:
            raise ValueError("Cosine sample counts must be positive integers: {}".format(text))
        counts.append(count)
    if not counts:
        raise ValueError("At least one cosine sample count is required.")
    return tuple(counts)


def main():
    parser = argparse.ArgumentParser(description="CLIP-ReID inference with optional QAT model")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str)
    parser.add_argument("--weight", default="", type=str, help="checkpoint path; overrides TEST.WEIGHT")
    parser.add_argument("--int8_model", default="", type=str, help="converted INT8 .pt model from convert_int8.py")
    parser.add_argument("--save_features", default="", type=str, help="optional .npz output for features/rankings")
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--speed_warmup", default=5, type=int, help="number of initial batches excluded from speed stats")
    parser.add_argument("--speed_images", default=1000, type=int, help="number of validation images used for FP32/INT8 speed comparison; <=0 means all")
    parser.add_argument("--cosine_counts", default="1,100,1000", type=str, help="comma-separated image counts for FP32/INT8 cosine comparison")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cosine_counts = parse_sample_counts(args.cosine_counts)

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.weight:
        cfg.TEST.WEIGHT = args.weight
    compare_fp32_int8 = bool(args.int8_model and cfg.TEST.WEIGHT)
    if args.int8_model:
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=False)
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    if compare_fp32_int8:
        logger.info("Running FP32 and INT8 comparison on CPU")
        fp32_cfg = clone_cfg_with_device(cfg, "cpu", qat_enabled=False)
        int8_cfg = clone_cfg_with_device(cfg, "cpu")

        fp32_model = make_model(fp32_cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        fp32_model.load_param(fp32_cfg.TEST.WEIGHT)
        int8_model = load_int8_model(args.int8_model, int8_cfg, num_classes, camera_num, view_num)
        cache_images = 0 if args.speed_images <= 0 else max(args.speed_images, max(cosine_counts))
        cached_batches, cached_images = collect_validation_batches(val_loader, cache_images)
        logger.info("Cached {} validation images for FP32/INT8 speed and cosine comparison".format(cached_images))

        logger.info("===== FP32 inference speed =====")
        fp32_result = measure_cached_inference_speed(fp32_cfg, fp32_model, cached_batches, args.speed_warmup, label="FP32")

        logger.info("===== INT8 inference speed =====")
        int8_result = measure_cached_inference_speed(int8_cfg, int8_model, cached_batches, args.speed_warmup, label="INT8")

        log_speed_comparison(fp32_result, int8_result)
        cosine_results = compare_cached_cosine_prefixes(
            fp32_cfg,
            fp32_model,
            int8_cfg,
            int8_model,
            cached_batches,
            sample_counts=cosine_counts,
        )
        log_cosine_comparison(cosine_results)

        if args.save_features:
            features, _, _, paths = extract_features(int8_cfg, int8_model, val_loader)
            save_rankings(features, num_query, paths, args.save_features, args.topk)
            logger.info("Saved INT8 features and top-{} rankings to {}".format(args.topk, args.save_features))
        return

    if args.int8_model:
        model = load_int8_model(args.int8_model, cfg, num_classes, camera_num, view_num)
    else:
        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        model.load_param(cfg.TEST.WEIGHT)

    result = do_inference_with_speed(cfg, model, val_loader, num_query, args.speed_warmup)
    logger.info("Rank-1: {:.1%}, Rank-5: {:.1%}, mAP: {:.1%}".format(result["rank1"], result["rank5"], result["mAP"]))

    if args.save_features:
        features, _, _, paths = extract_features(cfg, model, val_loader)
        save_rankings(features, num_query, paths, args.save_features, args.topk)
        logger.info("Saved features and top-{} rankings to {}".format(args.topk, args.save_features))


if __name__ == "__main__":
    main()
