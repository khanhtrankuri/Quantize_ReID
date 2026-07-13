from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import apply_qat_to_clipreid_model, make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2
import copy
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _remove_module_prefix(key):
    return key[len("module."):] if key.startswith("module.") else key


def load_checkpoint(model, weight_path, logger=None):
    checkpoint = torch.load(weight_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for state_key in ("state_dict", "model", "model_state_dict"):
            if state_key in checkpoint and isinstance(checkpoint[state_key], dict):
                checkpoint = checkpoint[state_key]
                break

    if not hasattr(checkpoint, "items"):
        raise RuntimeError("Checkpoint {} does not contain a state_dict".format(weight_path))

    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in checkpoint.items():
        clean_key = _remove_module_prefix(key)
        if torch.is_tensor(value) and clean_key in model_state and model_state[clean_key].shape == value.shape:
            model_state[clean_key].copy_(value)
            loaded_keys.append(clean_key)
        elif torch.is_tensor(value) and clean_key in model_state and "fake_quant" in clean_key and clean_key.endswith((".scale", ".zero_point", ".min_val", ".max_val")):
            model_state[clean_key].resize_(value.shape)
            model_state[clean_key].copy_(value)
            loaded_keys.append(clean_key)
        else:
            skipped_keys.append(key)

    if not loaded_keys:
        raise RuntimeError("No tensors were loaded from checkpoint: {}".format(weight_path))

    message = "Loaded {} tensors from checkpoint: {}".format(len(loaded_keys), weight_path)
    if logger is not None:
        logger.info(message)
        if skipped_keys:
            logger.warning(
                "Skipped {} tensors because key/shape did not match. First skipped keys: {}".format(
                    len(skipped_keys), skipped_keys[:10]
                )
            )
    else:
        print(message)
        if skipped_keys:
            print("Skipped {} tensors. First skipped keys: {}".format(len(skipped_keys), skipped_keys[:10]))


def clone_for_fp32_train(source_cfg):
    train_cfg = source_cfg.clone()
    train_cfg.defrost()
    train_cfg.MODEL.QAT.ENABLED = False
    train_cfg.freeze()
    return train_cfg


def clone_for_qat_train(source_cfg, qat_epochs=None, quantize_attention_internals=False):
    qat_cfg = source_cfg.clone()
    qat_cfg.defrost()
    qat_cfg.MODEL.QAT.ENABLED = True
    if quantize_attention_internals:
        qat_cfg.MODEL.QAT.QUANTIZE_ATTENTION_INTERNALS = True
    if qat_epochs is not None:
        qat_cfg.SOLVER.STAGE2.MAX_EPOCHS = qat_epochs
    qat_cfg.freeze()
    return qat_cfg


def save_model_checkpoint(model, output_dir, checkpoint_name, logger=None):
    if not output_dir:
        output_dir = "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    torch.save(model_to_save.state_dict(), checkpoint_path)

    message = "Saved checkpoint: {}".format(checkpoint_path)
    if logger is not None:
        logger.info(message)
    else:
        print(message)
    return checkpoint_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("--resume", default="", type=str, help="path to checkpoint/state_dict to load before training")
    parser.add_argument(
        "--qat_after_fp32",
        action="store_true",
        help="train FP32 first, reload the FP32 checkpoint, then patch and finetune QAT",
    )
    parser.add_argument(
        "--qat_weight",
        default="",
        type=str,
        help="optional FP32 checkpoint to load for the QAT phase; defaults to the just-saved FP32 final checkpoint",
    )
    parser.add_argument("--qat_epochs", default=None, type=int, help="override SOLVER.STAGE2.MAX_EPOCHS for the QAT phase")
    parser.add_argument(
        "--quantize_attention_internals",
        action="store_true",
        help="replace MultiheadAttention with QAT attention during the QAT phase",
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    run_qat_after_fp32 = args.qat_after_fp32 or cfg.MODEL.QAT.ENABLED
    train_cfg = clone_for_fp32_train(cfg) if run_qat_after_fp32 else cfg

    set_seed(train_cfg.SOLVER.SEED)

    if train_cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = train_cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(train_cfg.OUTPUT_DIR))
    logger.info(args)
    if run_qat_after_fp32:
        logger.info("QAT-after-FP32 flow enabled: FP32 model will be trained first, then reloaded and patched for QAT.")

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running FP32 phase with config:\n{}".format(train_cfg))

    if train_cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(train_cfg)

    model = make_model(train_cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    if args.resume:
        load_checkpoint(model, args.resume, logger)

    loss_func, center_criterion = make_loss(train_cfg, num_classes=num_classes)

    if train_cfg.SOLVER.STAGE1.MAX_EPOCHS > 0:
        optimizer_1stage = make_optimizer_1stage(train_cfg, model)
        scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs = train_cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min = train_cfg.SOLVER.STAGE1.LR_MIN, \
                            warmup_lr_init = train_cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = train_cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)

        do_train_stage1(
            train_cfg,
            model,
            train_loader_stage1,
            optimizer_1stage,
            scheduler_1stage,
            args.local_rank
        )
    else:
        logger.info("Skipping FP32 stage1 because SOLVER.STAGE1.MAX_EPOCHS <= 0")

    if train_cfg.SOLVER.STAGE2.MAX_EPOCHS > 0:
        optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(train_cfg, model, center_criterion)
        scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, train_cfg.SOLVER.STAGE2.STEPS, train_cfg.SOLVER.STAGE2.GAMMA, train_cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                      train_cfg.SOLVER.STAGE2.WARMUP_ITERS, train_cfg.SOLVER.STAGE2.WARMUP_METHOD)

        do_train_stage2(
            train_cfg,
            model,
            center_criterion,
            train_loader_stage2,
            val_loader,
            optimizer_2stage,
            optimizer_center_2stage,
            scheduler_2stage,
            loss_func,
            num_query, args.local_rank
        )
    else:
        logger.info("Skipping FP32 stage2 because SOLVER.STAGE2.MAX_EPOCHS <= 0")

    if run_qat_after_fp32:
        fp32_checkpoint = args.qat_weight
        if not fp32_checkpoint:
            fp32_checkpoint = save_model_checkpoint(
                model,
                train_cfg.OUTPUT_DIR,
                "{}_fp32_final.pth".format(train_cfg.MODEL.NAME),
                logger,
            )

        qat_cfg = clone_for_qat_train(
            cfg,
            qat_epochs=args.qat_epochs,
            quantize_attention_internals=args.quantize_attention_internals,
        )
        logger.info("Running QAT phase with config:\n{}".format(qat_cfg))
        logger.info("Loading FP32 checkpoint before QAT patch: {}".format(fp32_checkpoint))
        load_checkpoint(model, fp32_checkpoint, logger)
        teacher_model = None
        if qat_cfg.MODEL.QAT.DISTILLATION.ENABLED:
            # Copy before patching: teacher has no fake quant and receives no gradients.
            try:
                teacher_model = copy.deepcopy(model).to(qat_cfg.MODEL.DEVICE).eval()
                for parameter in teacher_model.parameters():
                    parameter.requires_grad_(False)
                logger.info("Created frozen FP32 teacher for QAT distillation.")
            except RuntimeError as exc:
                logger.warning("Could not keep an online FP32 teacher (%s); continuing QAT without distillation.", exc)
                teacher_model = None

        model = apply_qat_to_clipreid_model(
            model,
            qat_options=qat_cfg.MODEL.QAT,
            quantize_attention_internals=qat_cfg.MODEL.QAT.QUANTIZE_ATTENTION_INTERNALS,
        )

        if qat_cfg.SOLVER.STAGE2.MAX_EPOCHS > 0:
            loss_func_qat, center_criterion_qat = make_loss(qat_cfg, num_classes=num_classes)
            optimizer_2stage_qat, optimizer_center_2stage_qat = make_optimizer_2stage(qat_cfg, model, center_criterion_qat)
            scheduler_2stage_qat = WarmupMultiStepLR(
                optimizer_2stage_qat,
                qat_cfg.SOLVER.STAGE2.STEPS,
                qat_cfg.SOLVER.STAGE2.GAMMA,
                qat_cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                qat_cfg.SOLVER.STAGE2.WARMUP_ITERS,
                qat_cfg.SOLVER.STAGE2.WARMUP_METHOD,
            )

            do_train_stage2(
                qat_cfg,
                model,
                center_criterion_qat,
                train_loader_stage2,
                val_loader,
                optimizer_2stage_qat,
                optimizer_center_2stage_qat,
                scheduler_2stage_qat,
                loss_func_qat,
                num_query,
                args.local_rank,
                teacher_model=teacher_model,
            )
        else:
            logger.info("Skipping QAT stage2 because SOLVER.STAGE2.MAX_EPOCHS <= 0")
        save_model_checkpoint(
            model,
            qat_cfg.OUTPUT_DIR,
            "{}_qat_final.pth".format(qat_cfg.MODEL.NAME),
            logger,
        )
