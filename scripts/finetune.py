"""Fine-tunes Laya on the routing rows with Laya's own trainer (full fine-tune, RLCD loss).

python scripts/finetune.py [base] [out] [epochs]
Defaults: convaiinnovations/laya, ./laya-routing, 3. TRAIN, VAL and DEVICE override the data and device.
"""
import json, os, sys, time
import torch
import laya.train as T
from laya.train import TrainConfig, finetune, resolve_checkpoint_dir

DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available()
                                      else "mps" if torch.backends.mps.is_available() else "cpu")

# laya 0.4.0 leaves the loaded model on the CPU for the "before" evaluation, which fails on MPS.
_load = T.load_checkpoint
T.load_checkpoint = lambda d: (lambda m, tok, cfg: (m.to(DEVICE), tok, cfg))(*_load(d))

base = resolve_checkpoint_dir(sys.argv[1] if len(sys.argv) > 1 else "convaiinnovations/laya")
out = sys.argv[2] if len(sys.argv) > 2 else "laya-routing"
config = TrainConfig(epochs=int(sys.argv[3]) if len(sys.argv) > 3 else 3,
                     eval_data=os.environ.get("VAL", "data/val.jsonl"), log_every=25)
start = time.time()
summary = finetune(os.environ.get("TRAIN", "data/train.jsonl"), base, out, config, device=DEVICE)
print("SUMMARY", json.dumps(summary, default=str))
print(f"took {time.time() - start:.0f}s on {DEVICE}")
