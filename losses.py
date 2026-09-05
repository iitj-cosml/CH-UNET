import torch
import torch.nn as nn

def pignn_loss(y_pred, y_true, alpha=1.0):
    mse = nn.functional.mse_loss(y_pred, y_true)
    m_pred = y_pred.mean(dim=(1,2,3), keepdim=True)
    m_true = y_true.mean(dim=(1,2,3), keepdim=True)
    mass_penalty = (m_pred - m_true).pow(2).mean()

    return mse + alpha * mass_penalty