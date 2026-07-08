import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from config import cfg
from model.clip.int8_convert import bake_qat_fake_quant_weights
from model.make_model_clipreid import make_model


def load_torch_package(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(package):
    if isinstance(package, OrderedDict):
        return package
    if isinstance(package, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in package and isinstance(package[key], dict):
                return package[key]
        if "model" in package and isinstance(package["model"], dict):
            return package["model"]
        if package and all(torch.is_tensor(value) for value in package.values()):
            return package
    return None


def clean_key(key):
    return key[len("module.") :] if key.startswith("module.") else key


def checkpoint_looks_qat(state_dict):
    qat_tokens = ("fake_quant", "activation_post_process", ".linear.", ".conv.")
    return any(any(token in key for token in qat_tokens) for key in state_dict.keys())


def infer_num_classes(state_dict):
    for key in ("classifier.weight", "classifier_proj.weight"):
        for raw_key, value in state_dict.items():
            if clean_key(raw_key) == key and torch.is_tensor(value):
                return int(value.shape[0])
    return None


def infer_camera_view(cfg, state_dict, camera_num, view_num):
    cv_embed_rows = None
    for raw_key, value in state_dict.items():
        if clean_key(raw_key) == "cv_embed" and torch.is_tensor(value):
            cv_embed_rows = int(value.shape[0])
            break

    if cfg.MODEL.SIE_CAMERA and not cfg.MODEL.SIE_VIEW and camera_num is None:
        camera_num = cv_embed_rows or 1
    if cfg.MODEL.SIE_VIEW and not cfg.MODEL.SIE_CAMERA and view_num is None:
        view_num = cv_embed_rows or 1

    if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
        if camera_num is None and view_num is None and cv_embed_rows is not None:
            camera_num, view_num = cv_embed_rows, 1
            print(
                "Warning: checkpoint has cv_embed rows={}, but both SIE_CAMERA and SIE_VIEW are enabled. "
                "Using camera_num={} and view_num={}; pass --camera_num/--view_num if this is wrong.".format(
                    cv_embed_rows, camera_num, view_num
                )
            )

    return camera_num or 1, view_num or 1


def load_state_dict_flexible(model, state_dict, weight_path):
    model_state = model.state_dict()
    loaded = 0
    skipped = []

    for key, value in state_dict.items():
        key = clean_key(key)
        if torch.is_tensor(value) and key in model_state and model_state[key].shape == value.shape:
            model_state[key].copy_(value)
            loaded += 1
        else:
            skipped.append(key)

    if loaded == 0:
        raise RuntimeError("No tensors were loaded from {}".format(weight_path))

    print("Loaded {} tensors from {}".format(loaded, weight_path))
    if skipped:
        print("Skipped {} unmatched tensors. First skipped keys: {}".format(len(skipped), skipped[:10]))


class ReIDOnnxWrapper(nn.Module):
    def __init__(self, model, use_camera_label=False, use_view_label=False):
        super().__init__()
        self.model = model
        self.use_camera_label = use_camera_label
        self.use_view_label = use_view_label

    def forward(self, image, cam_label=None, view_label=None):
        kwargs = {}
        if self.use_camera_label:
            kwargs["cam_label"] = cam_label
        if self.use_view_label:
            kwargs["view_label"] = view_label
        return self.model(image, **kwargs)


def default_output_path(weight_path):
    base, _ = os.path.splitext(weight_path)
    return base + ".onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CLIP-ReID .pt/.pth checkpoint to ONNX")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str)
    parser.add_argument("--weight", required=True, type=str, help="input .pt/.pth checkpoint")
    parser.add_argument("--output", default="", type=str, help="output .onnx path")
    parser.add_argument("--pretrain_path", default="", type=str, help="override MODEL.PRETRAIN_PATH while rebuilding model")
    parser.add_argument("--num_classes", default=None, type=int, help="override class count if it cannot be inferred")
    parser.add_argument("--camera_num", default=None, type=int, help="override camera count for SIE_CAMERA")
    parser.add_argument("--view_num", default=None, type=int, help="override view count for SIE_VIEW")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--height", default=None, type=int, help="input height; defaults to cfg.INPUT.SIZE_TEST[0]")
    parser.add_argument("--width", default=None, type=int, help="input width; defaults to cfg.INPUT.SIZE_TEST[1]")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", default=17, type=int)
    parser.add_argument("--qat", default="auto", choices=["auto", "true", "false"], help="whether to rebuild QAT modules")
    parser.add_argument("--no_bake_qat", action="store_true", help="keep QAT fake-quant modules instead of baking weights")
    parser.add_argument("--no_dynamic_batch", action="store_true", help="export fixed batch dimension")
    parser.add_argument("--check", action="store_true", help="run onnx.checker after export if onnx is installed")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    package = load_torch_package(args.weight)
    state_dict = extract_state_dict(package)
    packaged_model = package.get("model") if isinstance(package, dict) else package

    cfg.defrost()
    cfg.MODEL.DEVICE = args.device
    if args.pretrain_path:
        cfg.MODEL.PRETRAIN_PATH = args.pretrain_path

    if state_dict is not None:
        is_qat = checkpoint_looks_qat(state_dict) if args.qat == "auto" else args.qat == "true"
        cfg.MODEL.QAT.ENABLED = is_qat
        num_classes = args.num_classes or infer_num_classes(state_dict)
        if num_classes is None:
            raise ValueError("Cannot infer num_classes from checkpoint; pass --num_classes.")

        camera_num, view_num = infer_camera_view(cfg, state_dict, args.camera_num, args.view_num)
        cfg.freeze()

        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        load_state_dict_flexible(model, state_dict, args.weight)
        if is_qat and not args.no_bake_qat:
            model = bake_qat_fake_quant_weights(model)
    elif isinstance(packaged_model, nn.Module):
        cfg.freeze()
        model = packaged_model
    else:
        raise TypeError("Unsupported checkpoint format: {}".format(type(package).__name__))

    model.to(args.device).float().eval()

    height = args.height or int(cfg.INPUT.SIZE_TEST[0])
    width = args.width or int(cfg.INPUT.SIZE_TEST[1])
    image = torch.randn(args.batch_size, 3, height, width, device=args.device)

    input_names = ["image"]
    export_args = [image]
    use_camera_label = bool(cfg.MODEL.SIE_CAMERA)
    use_view_label = bool(cfg.MODEL.SIE_VIEW)

    if use_camera_label:
        input_names.append("cam_label")
        export_args.append(torch.zeros(args.batch_size, dtype=torch.long, device=args.device))
    if use_view_label:
        input_names.append("view_label")
        export_args.append(torch.zeros(args.batch_size, dtype=torch.long, device=args.device))

    wrapper = ReIDOnnxWrapper(model, use_camera_label, use_view_label).eval()
    output_path = args.output or default_output_path(args.weight)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dynamic_axes = None
    if not args.no_dynamic_batch:
        dynamic_axes = {"image": {0: "batch"}, "features": {0: "batch"}}
        if use_camera_label:
            dynamic_axes["cam_label"] = {0: "batch"}
        if use_view_label:
            dynamic_axes["view_label"] = {0: "batch"}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(export_args),
            output_path,
            input_names=input_names,
            output_names=["features"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True,
        )

    print("Saved ONNX model to {}".format(output_path))
    print("Input image shape: {}".format(tuple(image.shape)))
    print("Inputs: {}".format(input_names))

    if args.check:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX checker passed.")


if __name__ == "__main__":
    main()
