import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from loss.make_loss import make_loss
from model.make_model_clipreid import apply_qat_to_clipreid_model, make_model
from model.clip.qat_layers import (
    disable_fake_quant,
    enable_fake_quant,
    enable_qat_observers,
)
from processor.processor_clipreid_stage2 import do_inference, do_train_stage2
from solver.lr_scheduler import WarmupMultiStepLR
from solver.make_optimizer_prompt import make_optimizer_2stage
from utils.logger import setup_logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def load_fp32_checkpoint(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_state = model.state_dict()
    loaded, skipped = 0, []
    for key, value in checkpoint.items():
        clean_key = key.replace("module.", "")
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            model_state[clean_key].copy_(value)
            loaded += 1
        else:
            skipped.append(key)

    if loaded == 0:
        raise RuntimeError(f"Khong load duoc tensor nao tu checkpoint: {weight_path}")

    print(f"Loaded {loaded} tensors from {weight_path}")
    if skipped:
        print(f"Skipped {len(skipped)} tensors because key/shape did not match.")


@torch.no_grad()
def calibrate_qat(model, train_loader, cfg, num_batches, device):
    if num_batches <= 0:
        return

    qat_model = unwrap_model(model)
    qat_model.eval()
    enable_qat_observers(qat_model)
    disable_fake_quant(qat_model)

    print(f"Calibrating QAT observers with {num_batches} batches ...")
    for step, (img, vid, target_cam, target_view) in enumerate(train_loader):
        if step >= num_batches:
            break

        img = img.to(device)
        target = vid.to(device)
        cam_label = target_cam.to(device) if cfg.MODEL.SIE_CAMERA else None
        view_label = target_view.to(device) if cfg.MODEL.SIE_VIEW else None

        with amp.autocast(enabled=torch.cuda.is_available()):
            qat_model(label=target, get_text=True)
            qat_model(x=img, label=target, cam_label=cam_label, view_label=view_label)

    enable_fake_quant(qat_model)


def parse_args():
    parser = argparse.ArgumentParser(description="QAT finetune CLIP-ReID from a trained checkpoint")
    parser.add_argument(
        "--config_file",
        default="configs/person/vit_clipreid.yml",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        required=True,
        type=str,
        help="FP32/ReID checkpoint to load before QAT patching",
    )
    parser.add_argument(
        "--calib_batches",
        default=50,
        type=int,
        help="number of train batches used to calibrate QAT observers before finetune",
    )
    parser.add_argument(
        "--qat_epochs",
        default=None,
        type=int,
        help="override SOLVER.STAGE2.MAX_EPOCHS for QAT finetune",
    )
    parser.add_argument(
        "--qat_lr",
        default=1e-6,
        type=float,
        help="learning rate used for QAT finetune",
    )
    parser.add_argument(
        "--disable_observer_epoch",
        default=None,
        type=int,
        help="epoch to freeze observer scales; defaults to MODEL.QAT.DISABLE_OBSERVER_EPOCH",
    )
    parser.add_argument(
        "--quantize_attention_internals",
        action="store_true",
        help="also replace MultiheadAttention internals with QAT attention",
    )
    parser.add_argument(
        "--eval_before",
        action="store_true",
        help="run validation after QAT patch/calibration and before finetune",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.MODEL.QAT.ENABLED = False
    cfg.TEST.WEIGHT = args.weight
    cfg.SOLVER.STAGE2.BASE_LR = args.qat_lr
    if args.qat_epochs is not None:
        cfg.SOLVER.STAGE2.MAX_EPOCHS = args.qat_epochs
    if args.disable_observer_epoch is not None:
        cfg.MODEL.QAT.DISABLE_OBSERVER_EPOCH = args.disable_observer_epoch
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("QAT finetune from checkpoint: {}".format(args.weight))
    logger.info(args)

    train_loader_stage2, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_fp32_checkpoint(model, args.weight)

    quantize_attention = args.quantize_attention_internals or cfg.MODEL.QAT.QUANTIZE_ATTENTION_INTERNALS
    model = apply_qat_to_clipreid_model(
        model,
        quantize_attention_internals=quantize_attention,
    )

    cfg.defrost()
    cfg.MODEL.QAT.ENABLED = True
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    calibrate_qat(model, train_loader_stage2, cfg, args.calib_batches, device)

    if args.eval_before:
        do_inference(cfg, model, val_loader, num_query)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query,
        0,
    )
