import torch
import torch.nn as nn
from torch import Tensor

class TemporalSelfAttention(nn.Module):
    """Minimal temporal self-attention (B,T,D) → (B,T,D)."""
    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x: Tensor) -> Tensor:
        y, _ = self.attn(x, x, x)
        return self.norm(x + y)
