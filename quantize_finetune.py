"""QAT-only launcher: load an FP32 ReID checkpoint, patch, calibrate and adapt."""
import argparse
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description="Selective W8A8 CLIP-ReID QAT finetuning")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml")
    parser.add_argument("--weight", required=True, help="FP32 ReID checkpoint")
    parser.add_argument("--qat_epochs", type=int, default=5)
    parser.add_argument("--qat_lr", type=float, default=1e-6)
    parser.add_argument("--calib_batches", type=int, default=50)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Reuse the established stage-2 trainer, while explicitly skipping FP32
    # stages.  It loads `--qat_weight` before applying the selective patch.
    sys.argv = ["train_clipreid.py", "--config_file", args.config_file,
                "--qat_after_fp32", "--qat_weight", args.weight,
                "--qat_epochs", str(args.qat_epochs), "--local_rank", str(args.local_rank),
                "SOLVER.STAGE1.MAX_EPOCHS", "0", "SOLVER.STAGE2.MAX_EPOCHS", "0",
                # Guarantee the conventional ViT-B-16_qat_<epoch>.pth checkpoint
                # is emitted at the requested final QAT epoch.
                "SOLVER.STAGE2.CHECKPOINT_PERIOD", str(args.qat_epochs),
                "SOLVER.STAGE2.BASE_LR", str(args.qat_lr),
                "MODEL.QAT.CALIBRATION_BATCHES", str(args.calib_batches)] + args.opts
    runpy.run_path("train_clipreid.py", run_name="__main__")


if __name__ == "__main__":
    main()
