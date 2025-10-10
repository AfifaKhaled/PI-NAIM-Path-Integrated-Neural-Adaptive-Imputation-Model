import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def rmse(pred, target, mask=None):
    if mask is None:
        mse = (pred - target).pow(2).mean().item()
    else:
        missing = 1.0 - mask
        den = missing.sum().item() if isinstance(missing, torch.Tensor) else float(missing.sum())
        den = max(den, 1.0)
        mse = ((missing * (pred - target)**2).sum().item()) / den
    return float(np.sqrt(mse + 1e-12))

def auroc(logits, targets):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        y = targets.cpu().numpy()
        try:
            return float(roc_auc_score(y, probs))
        except Exception:
            return float("nan")
