import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

def enforce_conservation(y_pred, y_true):
    phi_bar_pred = y_pred.mean(dim=(1,2,3), keepdim=True)
    phi_bar_true  = y_true.mean(dim=(1,2,3), keepdim=True)
    return y_pred - phi_bar_pred + phi_bar_true

def predict(model, X_t, n_rolls, enf_cons=True):
    model.eval()
    device = model.device

    # prepare initial condition
    param_dtype = next(model.parameters()).dtype
    X_t = torch.tensor(X_t, dtype=param_dtype).unsqueeze(0).unsqueeze(0).to(device)

    pred_trj = [X_t.squeeze(0).squeeze(0).detach().cpu().numpy()]

    with torch.no_grad():
        for _ in tqdm(range(n_rolls)):
            y_hat = model(X_t)

            if enf_cons:
                y_hat = enforce_conservation(y_hat, X_t)
            
            X_t = y_hat
            pred_trj.append(
                y_hat.squeeze(0).squeeze(0).cpu().numpy()
            )
    pred_trj = np.array(pred_trj)
    return pred_trj

class SnapshotDataset(Dataset):
    def __init__(self, X, y):
     
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        X = X.unsqueeze(1)
        y = y.unsqueeze(1)

        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_dataset(fname, batch_size, num_workers=4, pin_memory=True):
    data = np.load(fname)

    train_X = data["train_X"]
    train_y = data["train_y"]
    val_X = data["val_X"]
    val_y = data["val_y"]

    train_X = train_X.reshape(-1, train_X.shape[-2], train_X.shape[-1])
    train_y = train_y.reshape(-1, train_y.shape[-2], train_y.shape[-1])

    val_X = val_X.reshape(-1, val_X.shape[-2], val_X.shape[-1])
    val_y = val_y.reshape(-1, val_y.shape[-2], val_y.shape[-1])


    train_ds = SnapshotDataset(train_X, train_y)
    val_ds = SnapshotDataset(val_X, val_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader    


def ls_from_correlation(snap, dx=1, zero_pad=True):
    f = snap - np.mean(snap)
    ny, nx = f.shape

    if zero_pad:
        # pad to avoid circular (wrap-around) correlation
        pad = ((0, ny), (0, nx))                
        g = np.pad(f, pad, mode='constant')
    else:
        g = f

    G = fft2(g)
    R = ifft2(np.abs(G)**2).real                 
    R /= R[0,0]                                 

    R = fftshift(R)

    # radial average
    Ny, Nx = R.shape
    y, x = np.indices((Ny, Nx))
    r_pix = np.hypot(x - Nx//2, y - Ny//2).astype(np.int32)
    sums = np.bincount(r_pix.ravel(), R.ravel())
    counts = np.bincount(r_pix.ravel())
    C_r = sums / np.maximum(counts, 1)

    r = np.arange(len(C_r)) * dx

    target = 0.0

    y = C_r
    # find first downward crossing of target
    idx = np.where((y[:-1] >= target) & (y[1:] < target))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    # linear interpolation between (r[i], y[i]) and (r[i+1], y[i+1])
    t = (target - y[i]) / (y[i+1] - y[i])
    ls = r[i] + t*(r[i+1] - r[i])
    
    return ls, r, C_r

def ls_trajectory(trj):
    ls = []
    r = []
    cr = []
    for i in range(trj.shape[0]):
        ls_, r_, cr_ = ls_from_correlation(trj[i, :, :])
        ls.append(ls_)
        r.append(r_)
        cr.append(cr_)

    return np.array(ls), np.array(r), np.array(cr)


