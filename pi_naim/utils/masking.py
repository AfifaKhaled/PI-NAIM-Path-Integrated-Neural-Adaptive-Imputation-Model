# pi_naim/utils/masking.py
import torch
import torch.nn as nn

def compute_missing_rate(mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        mask: (B, D) or (B, T, D) with 1 for observed, 0 for missing.
    Returns:
        per-sample missing rate tensor of shape (B,).
    """
    if mask.dim() == 3:
        B, T, D = mask.shape
        obs = mask.sum(dim=(1,2))
        total = T * D
    else:
        B, D = mask.shape
        obs = mask.sum(dim=1)
        total = D
    mr = 1.0 - (obs / total)
    return mr

def apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x * mask

def curriculum_mask(x: torch.Tensor, phase: str, p_miss_low=0.1, p_miss_high=0.6, device=None) -> torch.Tensor:
    """
    Create training masks for curriculum phases:
      - 'mcar': uniform random masking
      - 'mar' : mask probability correlated with feature magnitude
      - 'mnar': value-dependent masking (sigmoid of scaled values)
    Returns mask with 1 for observed, 0 for missing.
    """
    device = device or x.device
    if x.dim() == 3:
        B, T, D = x.shape
        shape = (B, T, D)
        X = x
    else:
        B, D = x.shape
        shape = (B, D)
        X = x

    if phase == "mcar":
        p = torch.empty(shape, device=device).uniform_(p_miss_low, p_miss_high)
        mask = (torch.rand(shape, device=device) > p).float()
    elif phase == "mar":
        norm = (X - X.mean(dim=-1, keepdim=True)) / (X.std(dim=-1, keepdim=True) + 1e-6)
        logits = -0.5 * norm
        prob_missing = torch.sigmoid(logits)
        mask = (torch.rand_like(prob_missing) > prob_missing).float()
    elif phase == "mnar":
        logits = 1.0 * X
        prob_missing = torch.sigmoid(logits)
        mask = (torch.rand_like(prob_missing) > prob_missing).float()
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return mask

class HomoscedasticUncertaintyWeights(nn.Module):
    def __init__(self, n_terms: int = 3):
        super().__init__()
        self.log_sigma2 = nn.Parameter(torch.zeros(n_terms))

    def forward(self) -> torch.Tensor:
        sigma2 = torch.exp(self.log_sigma2).clamp_min(1e-6)
        lambdas = 1.0 / (2.0 * sigma2)
        return lambdas