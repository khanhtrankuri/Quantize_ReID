"""Per-module fake-quant sensitivity probe producing JSON, CSV and FP32 exclusions."""
import argparse, csv, json, os, sys
sys.path.insert(0, os.getcwd())
import torch
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model, apply_qat_to_clipreid_model
from model.clip.qat_layers import QATLinear, disable_fake_quant, enable_fake_quant
from benchmark_quantized_reid import load_state, evaluate, retrieval_metrics

def groups(model):
    return [name for name,module in model.named_modules() if isinstance(module,QATLinear)]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config_file",default="configs/person/vit_clipreid.yml"); p.add_argument("--fp32_weight",required=True)
    p.add_argument("--output",required=True); p.add_argument("--calib_batches",type=int,default=50); p.add_argument("--eval_batches",type=int,default=-1); args=p.parse_args()
    cfg.merge_from_file(args.config_file); cfg.defrost(); cfg.MODEL.DEVICE="cpu"; cfg.MODEL.QAT.ENABLED=False; cfg.freeze()
    _, train, loader, nq, nc, cams, views=make_dataloader(cfg)
    fp32=make_model(cfg,nc,cams,views).cpu(); load_state(fp32,args.fp32_weight); baseline, ref=evaluate(fp32,loader,nq)
    qat_cfg=cfg.clone(); qat_cfg.defrost(); qat_cfg.MODEL.QAT.ENABLED=True; qat_cfg.freeze()
    candidate=make_model(cfg,nc,cams,views).cpu(); load_state(candidate,args.fp32_weight); apply_qat_to_clipreid_model(candidate,qat_options=qat_cfg.MODEL.QAT)
    # Fill observers on a bounded calibration stream, then test one QAT module at a time.
    candidate.eval(); disable_fake_quant(candidate)
    with torch.no_grad():
        for idx,batch in enumerate(train):
            candidate(batch[0]);
            if idx+1>=args.calib_batches: break
    result={}; exclusions=[]
    for name in groups(candidate):
        disable_fake_quant(candidate); enable_fake_quant(dict(candidate.named_modules())[name])
        accuracy, features=evaluate(candidate,loader,nq); accuracy.update(retrieval_metrics(ref,features)); accuracy.update({"map_fp32":baseline["mAP"],"rank1_fp32":baseline["rank1"],"map_quant":accuracy["mAP"],"rank1_quant":accuracy["rank1"],"delta_map":accuracy["mAP"]-baseline["mAP"],"delta_rank1":accuracy["rank1"]-baseline["rank1"]}); result[name]=accuracy
        if accuracy["delta_map"] < -0.005 or accuracy["delta_rank1"] < -0.005: exclusions.append(name)
    with open(args.output,"w") as f: json.dump(result,f,indent=2)
    with open(os.path.splitext(args.output)[0]+".csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["module","delta_map","delta_rank1","feature_cosine_mean","top5_overlap"]); [w.writerow([n,r["delta_map"],r["delta_rank1"],r["feature_cosine_mean"],r["top5_overlap"]]) for n,r in result.items()]
    print("Suggested FP32_MODULE_PATTERNS:", json.dumps(exclusions))
if __name__=="__main__": main()
