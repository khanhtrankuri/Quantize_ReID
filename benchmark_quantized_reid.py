"""Fair CPU benchmark for FP32, QAT and image-only INT8 CLIP-ReID models."""
import argparse, csv, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.metrics import R1_mAP_eval


def _is_resizable_qat_buffer(key):
    return ("fake_quant" in key and key.endswith((".scale", ".zero_point", ".min_val", ".max_val")))


def load_state(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict): state = state[key]; break
    target, loaded = model.state_dict(), []
    for key, value in state.items():
        key = key.removeprefix("module.")
        if torch.is_tensor(value) and key in target:
            if target[key].shape == value.shape:
                target[key].copy_(value); loaded.append(key)
            elif _is_resizable_qat_buffer(key):
                # Per-channel observers start with shape [1] but a trained
                # QAT checkpoint stores one qparam per output channel.
                target[key].resize_(value.shape)
                target[key].copy_(value)
                loaded.append(key)
    if not loaded: raise RuntimeError("No compatible tensor loaded from {}".format(path))


@torch.no_grad()
def evaluate(model, loader, num_query):
    model.eval(); evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm="yes", num_samples=len(loader.dataset)); evaluator.reset(); features=[]
    for image, pid, camid, camids, view, _ in loader:
        feature = model(image, cam_label=camids if hasattr(model, "cv_embed") else None,
                        view_label=view if hasattr(model, "cv_embed") else None)
        evaluator.update((feature, pid, camid)); features.append(feature.float().cpu())
    cmc, mAP, *_ = evaluator.compute()
    return {"mAP": float(mAP), "rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9])}, torch.cat(features)


@torch.no_grad()
def latency(model, images, warmup, repeat):
    model.eval()
    for _ in range(warmup): model(images)
    samples=[]
    for _ in range(repeat):
        start=time.perf_counter(); model(images); samples.append((time.perf_counter()-start)*1000)
    mean_latency_ms = float(np.mean(samples))
    return {"batch_size": int(images.shape[0]), "latency_ms_mean": mean_latency_ms,
            "latency_ms_per_image": float(mean_latency_ms / images.shape[0]),
            "latency_ms_median": float(np.median(samples)),
            "latency_ms_p90": float(np.percentile(samples,90)), "latency_ms_p95": float(np.percentile(samples,95)),
            "throughput": float(images.shape[0] * 1000 / mean_latency_ms)}


def compare_performance(baseline, candidate):
    """Return speed metrics for candidate relative to a FP32 baseline.

    A value above 1.0 for either speedup means that the candidate is faster.
    Both models are benchmarked on the same CPU batch and thread count.
    """
    baseline_latency = baseline["latency_ms_mean"]
    candidate_latency = candidate["latency_ms_mean"]
    baseline_throughput = baseline["throughput"]
    candidate_throughput = candidate["throughput"]
    return {
        "batch_size": candidate["batch_size"],
        "fp32_latency_ms_mean": baseline_latency,
        "int8_latency_ms_mean": candidate_latency,
        "latency_speedup_x": float(baseline_latency / candidate_latency),
        "latency_reduction_percent": float((baseline_latency - candidate_latency) * 100 / baseline_latency),
        "fp32_throughput": baseline_throughput,
        "int8_throughput": candidate_throughput,
        "throughput_speedup_x": float(candidate_throughput / baseline_throughput),
    }


def retrieval_metrics(reference, candidate):
    ref, cand = F.normalize(reference.float(), dim=1), F.normalize(candidate.float(), dim=1)
    cosine = F.cosine_similarity(ref, cand, dim=1); ref_sim, cand_sim = ref@ref.T, cand@cand.T
    ref_top = ref_sim.fill_diagonal_(-float("inf")).topk(min(10, ref.shape[0]-1), dim=1).indices
    cand_top = cand_sim.fill_diagonal_(-float("inf")).topk(min(10, cand.shape[0]-1), dim=1).indices
    return {"feature_cosine_mean":float(cosine.mean()), "feature_cosine_min":float(cosine.min()), "feature_cosine_std":float(cosine.std()),
            "pairwise_similarity_mse":float(F.mse_loss(ref_sim,cand_sim)), "top1_agreement":float((ref_top[:,0]==cand_top[:,0]).float().mean()),
            "top5_overlap":float(torch.stack([torch.isin(cand_top[i,:5],ref_top[i,:5]).float().mean() for i in range(ref.shape[0])]).mean()),
            "top10_overlap":float(torch.stack([torch.isin(cand_top[i],ref_top[i]).float().mean() for i in range(ref.shape[0])]).mean())}


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--config_file", default="configs/person/vit_clipreid.yml")
    p.add_argument("--fp32_weight", required=True); p.add_argument("--qat_weight", required=True); p.add_argument("--int8_model", required=True)
    p.add_argument("--batch_sizes", default="1,8,16,32"); p.add_argument("--num_threads", default="1,4,8")
    p.add_argument("--warmup_batches", type=int, default=20); p.add_argument("--repeat", type=int, default=5); p.add_argument("--output", default="benchmark_quantized_reid.json")
    return p.parse_args()


def main():
    args=parse_args(); cfg.merge_from_file(args.config_file); cfg.defrost(); cfg.MODEL.DEVICE="cpu"; cfg.MODEL.QAT.ENABLED=False; cfg.freeze()
    _, _, loader, num_query, classes, cameras, views=make_dataloader(cfg)
    fp32=make_model(cfg, classes,cameras,views).cpu(); load_state(fp32,args.fp32_weight)
    qat_cfg=cfg.clone(); qat_cfg.defrost(); qat_cfg.MODEL.QAT.ENABLED=True; qat_cfg.freeze()
    qat=make_model(qat_cfg,classes,cameras,views).cpu(); load_state(qat,args.qat_weight)
    # INT8 deploy files intentionally serialize the image-only module object.
    # This file is supplied locally by `convert_int8.py`; PyTorch >= 2.6 needs
    # an explicit opt-out from the safe weights-only default to restore it.
    try:
        saved = torch.load(args.int8_model, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        saved = torch.load(args.int8_model, map_location="cpu")
    int8=saved["model"] if isinstance(saved,dict) and "model" in saved else saved
    metrics={}; fp32_acc, reference=evaluate(fp32,loader,num_query); metrics["fp32"]=fp32_acc
    for name, model in (("qat",qat),("int8",int8)):
        accuracy, feature=evaluate(model,loader,num_query); accuracy.update(retrieval_metrics(reference,feature)); metrics[name]=accuracy
    first=next(iter(loader))[0]
    for threads in map(int,args.num_threads.split(",")):
        torch.set_num_threads(threads)
        for size in map(int,args.batch_sizes.split(",")):
            images=first[:size]
            for name,model in (("fp32",fp32),("qat",qat),("int8",int8)):
                metrics[name].setdefault("performance",{})["threads{}_batch{}".format(threads,size)] = latency(model,images,args.warmup_batches,args.repeat)
    # Compare the exact FP32 and INT8 measurements rather than separately
    # timing another batch. This makes the reported speedup reproducible.
    metrics["comparison"] = {"int8_vs_fp32": {}}
    for setting, fp32_performance in metrics["fp32"]["performance"].items():
        int8_performance = metrics["int8"]["performance"][setting]
        metrics["comparison"]["int8_vs_fp32"][setting] = compare_performance(
            fp32_performance, int8_performance
        )
    with open(args.output,"w") as f: json.dump(metrics,f,indent=2)
    with open(os.path.splitext(args.output)[0]+".csv","w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["record_type","setting","model","mAP","rank1","rank5","rank10","feature_cosine_mean","batch_size","latency_ms_mean","latency_ms_per_image","throughput","latency_speedup_x","latency_reduction_percent","throughput_speedup_x"])
        for name in ("fp32", "qat", "int8"):
            result = metrics[name]
            writer.writerow(["accuracy", "", name, result.get("mAP"), result.get("rank1"), result.get("rank5"), result.get("rank10"), result.get("feature_cosine_mean"), "", "", "", "", "", "", ""])
            for setting, performance in result.get("performance", {}).items():
                writer.writerow(["performance", setting, name, "", "", "", "", "", performance["batch_size"], performance["latency_ms_mean"], performance["latency_ms_per_image"], performance["throughput"], "", "", ""])
        for setting, comparison in metrics["comparison"]["int8_vs_fp32"].items():
            writer.writerow(["comparison", setting, "int8_vs_fp32", "", "", "", "", "", comparison["batch_size"], "", "", "", comparison["latency_speedup_x"], comparison["latency_reduction_percent"], comparison["throughput_speedup_x"]])
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
