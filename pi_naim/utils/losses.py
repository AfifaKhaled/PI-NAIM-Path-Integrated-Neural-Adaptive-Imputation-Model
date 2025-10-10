# pi_naim/utils/losses.py
import torch
import torch.nn as nn
from torch import Tensor


class HomoscedasticUncertaintyWeights(nn.Module):
    """Learns σ_i; returns weights λ_i = 1/(2σ_i^2)."""

    def __init__(self, n_terms: int = 4, init: float = 0.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.full((n_terms,), init))

    def forward(self) -> Tensor:
        # Clip to prevent numerical issues
        log_sigma_clipped = torch.clamp(self.log_sigma, min=-6.0, max=6.0)
        sigma2 = torch.exp(2 * log_sigma_clipped)
        weights = 1.0 / (2.0 * sigma2)

        # Clip weights to reasonable range and ensure positivity
        weights = torch.clamp(weights, min=1e-6, max=1e6)

        return weights


def masked_mse(x_hat: Tensor, x: Tensor, m: Tensor) -> Tensor:
    diff = (1.0 - m) * (x_hat - x)
    num = torch.clamp((1.0 - m).sum(), min=1.0)
    mse = (diff ** 2).sum() / num

    # Prevent NaN/Inf and ensure positive loss
    if torch.isnan(mse) or torch.isinf(mse) or mse < 0:
        mse = torch.tensor(0.0, device=x.device, requires_grad=True)

    return torch.clamp(mse, min=0.0, max=100.0)  # Reasonable upper bound