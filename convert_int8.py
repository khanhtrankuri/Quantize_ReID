import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.clip.int8_convert import bake_qat_fake_quant_weights, convert_linear_to_dynamic_int8
from model.make_model_clipreid import make_model
from utils.logger import setup_logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_checkpoint(model, weight_path):
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
        raise RuntimeError("No tensors were loaded from {}".format(weight_path))

    print("Loaded {} tensors from {}".format(loaded, weight_path))
    if skipped:
        print("Skipped {} tensors because key/shape did not match.".format(len(skipped)))


def count_modules(model):
    dynamic_linear_types = []
    try:
        dynamic_linear_types.append(torch.nn.quantized.dynamic.Linear)
    except AttributeError:
        pass

    counts = {
        "linear_fp32": 0,
        "linear_int8_dynamic": 0,
        "conv_fp32": 0,
    }
    dynamic_linear_types = tuple(dynamic_linear_types)

    for module in model.modules():
        if dynamic_linear_types and isinstance(module, dynamic_linear_types):
            counts["linear_int8_dynamic"] += 1
        elif isinstance(module, nn.Linear):
            counts["linear_fp32"] += 1
        elif isinstance(module, nn.Conv2d):
            counts["conv_fp32"] += 1

    return counts


def default_output_path(weight_path):
    base, _ = os.path.splitext(weight_path)
    if base.endswith("_qat"):
        return base[:-4] + "_int8.pt"
    return base + "_int8.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a QAT CLIP-ReID checkpoint to a CPU INT8 deploy model")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str)
    parser.add_argument("--weight", required=True, type=str, help="QAT checkpoint, for example ViT-B-16_qat_5.pth")
    parser.add_argument("--output", default="", type=str, help="output .pt path for the converted INT8 model")
    parser.add_argument("--engine", default="fbgemm", choices=["fbgemm", "qnnpack"], help="PyTorch quantized CPU backend")
    parser.add_argument(
        "--quantize_attention_internals",
        action="store_true",
        help="set this if the QAT checkpoint was trained with quantized attention internals",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.MODEL.QAT.ENABLED = True
    if args.quantize_attention_internals:
        cfg.MODEL.QAT.QUANTIZE_ATTENTION_INTERNALS = True
    cfg.TEST.WEIGHT = args.weight
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    output_path = args.output or default_output_path(args.weight)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=False)
    logger.info("Converting QAT checkpoint to INT8: {}".format(args.weight))
    logger.info(args)

    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_checkpoint(model, args.weight)

    model.cpu().float().eval()
    model = bake_qat_fake_quant_weights(model)
    baked_counts = count_modules(model)
    model_int8 = convert_linear_to_dynamic_int8(model, engine=args.engine)
    int8_counts = count_modules(model_int8)

    torch.save(
        {
            "model": model_int8,
            "config": cfg.dump(),
            "source_qat_weight": args.weight,
            "quantized_backend": args.engine,
            "baked_counts": baked_counts,
            "int8_counts": int8_counts,
        },
        output_path,
    )

    print("Saved INT8 converted model to {}".format(output_path))
    print("Before dynamic INT8: {}".format(baked_counts))
    print("After dynamic INT8: {}".format(int8_counts))
