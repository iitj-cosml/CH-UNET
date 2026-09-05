import argparse
import numpy as np
import torch
from utilities import load_dataset
from models import LitUNet
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


parser = argparse.ArgumentParser()
parser.add_argument("--enf_cons", type=str2bool, default=True,
                     help="Enforce order-parameter conservation during training (default: True)")
parser.add_argument("--seed", type=int, default=4276865)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--precision", type=str, default="double", choices=["single", "double"])
parser.add_argument("--log_dir", type=str, default="train_logs")
parser.add_argument("--fname", type=str, default="td_64_100_0_300_off.npz")
args = parser.parse_args()

seed_everything(args.seed, workers=True)

np_dtype = np.float64
torch_dtype = torch.float64

if args.precision == "single":
    np_dtype = np.float32
    torch_dtype = torch.float32

lr = args.lr
alpha = args.alpha
enf_cons = args.enf_cons
batch_size = args.batch_size
epochs = args.epochs
num_workers = args.num_workers
log_dir = args.log_dir
fname = args.fname

run_name = f"enf_cons_{str(enf_cons).lower()}"

# ................................ Training .............................................

train_loader, val_loader = load_dataset(fname=fname, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
pbar_update = int(len(train_loader)/10)

logger = TensorBoardLogger(save_dir=log_dir, name="lightning_logs", version=run_name)

checkpoint_callback = ModelCheckpoint(
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
            save_weights_only=False,
        )

trainer = Trainer(
            max_epochs=int(epochs),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            log_every_n_steps=50,
            default_root_dir=str(log_dir),
            logger=logger,
            callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=pbar_update)],
            precision=32 if args.precision == "single" else 64
        )

lit_model = LitUNet(lr=lr, alpha=alpha, enf_cons=enf_cons, use_attn=True, dtype=torch_dtype)
trainer.fit(lit_model, train_loader, val_loader)

print(f"\n[train.py] enf_cons={enf_cons} -> checkpoint dir: {checkpoint_callback.dirpath}\n")