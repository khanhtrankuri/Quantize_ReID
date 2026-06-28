import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
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

    dist = torch.cdist(query_features, gallery_features, p=2).numpy()
    indices = np.argsort(dist, axis=1)[:, :topk]

    rankings = []
    for query_idx, gallery_indices in enumerate(indices):
        rankings.append(
            {
                "query": str(paths[query_idx]),
                "gallery": [str(gallery_paths[i]) for i in gallery_indices],
                "distance": [float(dist[query_idx, i]) for i in gallery_indices],
            }
        )

    np.savez_compressed(
        output_path,
        features=features.numpy(),
        paths=paths,
        rankings=np.asarray(rankings, dtype=object),
    )


def do_inference_with_speed(cfg, model, val_loader, num_query, warmup_batches):
    device = cfg.MODEL.DEVICE
    use_cuda = str(device).startswith("cuda")
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
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
        logger.info(
            "Inference speed - warmup batches: {}, timed batches: {}, timed images: {}".format(
                warmup_batches, timed_batches, timed_images
            )
        )
        logger.info(
            "Forward only - total: {:.4f}s, latency: {:.3f} ms/image, batch time: {:.3f} ms/batch, throughput: {:.2f} images/s".format(
                forward_time,
                1000.0 * forward_time / timed_images,
                1000.0 * forward_time / timed_batches,
                timed_images / forward_time,
            )
        )
    else:
        logger.info("Inference speed - not enough batches after warmup to report forward timing")
    logger.info("End-to-end inference pass time: {:.4f}s".format(total_time))

    return cmc[0], cmc[4], mAP


def main():
    parser = argparse.ArgumentParser(description="CLIP-ReID inference with optional QAT model")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str)
    parser.add_argument("--weight", default="", type=str, help="checkpoint path; overrides TEST.WEIGHT")
    parser.add_argument("--save_features", default="", type=str, help="optional .npz output for features/rankings")
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--speed_warmup", default=5, type=int, help="number of initial batches excluded from speed stats")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.weight:
        cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=False)
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)

    rank1, rank5, mAP = do_inference_with_speed(cfg, model, val_loader, num_query, args.speed_warmup)
    logger.info("Rank-1: {:.1%}, Rank-5: {:.1%}, mAP: {:.1%}".format(rank1, rank5, mAP))

    if args.save_features:
        features, _, _, paths = extract_features(cfg, model, val_loader)
        save_rankings(features, num_query, paths, args.save_features, args.topk)
        logger.info("Saved features and top-{} rankings to {}".format(args.topk, args.save_features))


if __name__ == "__main__":
    main()
