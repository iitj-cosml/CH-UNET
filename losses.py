import torch
import torch.nn as nn

def conservation_loss(y_pred, y_true, alpha=1.0):
    mse = nn.functional.mse_loss(y_pred, y_true)
    phi_bar_pred = y_pred.mean(dim=(1,2,3), keepdim=True)
    phi_bar_true = y_true.mean(dim=(1,2,3), keepdim=True)
    conservation_penalty = (phi_bar_pred - phi_bar_true).pow(2).mean()

    return mse + alpha * conservation_penalty