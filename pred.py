from models import LitUNet
from utilities import predict
from phase_ordering import PhaseOrdering
import argparse
import numpy as np
import torch
from tqdm import tqdm
from time import time
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


parser = argparse.ArgumentParser()
parser.add_argument("--enf_mass", type=str2bool, default=True,
                     help="Enforce mass conservation at prediction time, and select the "
                          "checkpoint trained with this same setting (default: True)")
parser.add_argument("--log_dir", type=str, default="train_logs")
parser.add_argument("--ckpt_path", type=str, default=None,
                     help="Explicit checkpoint path; overrides the enf_mass-derived lookup")
parser.add_argument("--precision", type=str, default="double", choices=["single", "double"])
parser.add_argument("--out_dir", type=str, default="results")
args = parser.parse_args()

np.random.seed(42)

n_ahead = 100
enf_mass = args.enf_mass

if args.ckpt_path is not None:
    ckpt_path = args.ckpt_path
else:
    run_name = f"enf_mass_{str(enf_mass).lower()}"
    ckpt_dir = os.path.join(args.log_dir, "lightning_logs", run_name, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[-1])

precision = args.precision

nens = 10
seeds = np.random.randint(20000, 30000, nens)

l = 128
start = 100
end = 2000
dt = 0.01
off_values = [0.0, -0.3, 0.3]
end_ = end + n_ahead * dt
pred_len = int((end - start) / (n_ahead * dt))

np_dtype = np.float64
torch_dtype = torch.float64

if precision == "single":
    np_dtype = np.float32
    torch_dtype = torch.float32

print(f"\nLoading model from the checkpoint: {ckpt_path}\n")
model = LitUNet.load_from_checkpoint(ckpt_path)
model = model.double()

if torch_dtype == torch.float32:
    model = model.float()

t_org = 0
t_pred = 0
for off in off_values:
    system = PhaseOrdering(l=l, dt=dt, start=start, end=end_, off=off, dtype=np_dtype)
    org_trjs = np.empty((nens, pred_len+1, l, l))
    pred_trjs = []
    for i, seed in enumerate(tqdm(seeds, desc=f"Predicting for psi = {off}")):
        t1 = time()
        system.set_ic(seed)
        org_trjs[i, ...] = system.chc(system.ic)[::n_ahead, ...]
        t2 = time()
        t_org += (t2 - t1)

        X_t = org_trjs[i, 0, ...]
        
        t3 = time()
        pred_trjs.append(predict(model=model, X_t=X_t, n_rolls=pred_len, enf_mass=enf_mass))
        t4 = time()
        t_pred += (t4 - t3)

    pred_trjs = np.array(pred_trjs)

    metadata=dict(
        start=start,
        end=end,
        nens=nens,
        l=l,
        dt=dt,
        n_ahead=n_ahead,
        off=off,
        enf_mass=enf_mass,
        ckpt_path=ckpt_path,
        seeds=seeds.tolist(),
        t_org=t_org,
        t_pred=t_pred,
        pred_len=pred_len
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"result_{l}_{off}_enf_mass_{str(enf_mass).lower()}.npz")
    np.savez(out_path, pred_trjs=pred_trjs, org_trjs=org_trjs, metadata=metadata)
    print(f"[pred.py] saved: {out_path}")