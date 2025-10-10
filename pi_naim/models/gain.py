# pi_naim/models/gain.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalSelfAttention(nn.Module):
    """Temporal attention with causal masking for MAR awareness"""

    def __init__(self, d_model: int, nhead: int = 8):
        super().__init__()
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = d_model + (nhead - d_model % nhead)

        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(d_model, d_model) if d_model != d_model else nn.Identity()

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Project to compatible dimension if needed
        x_proj = self.proj(x)

        # Causal attention mask for temporal dependencies
        seq_len = x_proj.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        y, _ = self.attn(x_proj, x_proj, x_proj, attn_mask=causal_mask)
        y = self.dropout(y)
        return self.norm(x + y)  # Residual connection with original x


class Critic(nn.Module):
    """WGAN-GP Critic"""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Generator(nn.Module):
    """GAIN Generator with residual connections"""

    def __init__(self, dim: int, hidden: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, dim)
        )
        self.residual = nn.Linear(dim, dim)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        input_cat = torch.cat([x, z], dim=-1)
        generated = self.net(input_cat)
        return generated + self.residual(x)


class TemporalGAIN(nn.Module):
    """Complete GAIN with WGAN-GP and temporal attention"""

    def __init__(self, dim: int, hidden: int = 256, temporal: bool = True,
                 p_drop: float = 0.1, n_heads: int = 4):  # Reduced n_heads to 4 for compatibility
        super().__init__()
        self.dim = dim
        self.temporal = temporal

        # Ensure dimension is compatible with attention heads
        attn_dim = dim
        if temporal and attn_dim % n_heads != 0:
            attn_dim = attn_dim + (n_heads - attn_dim % n_heads)

        self.tem = TemporalSelfAttention(attn_dim, n_heads) if temporal else None
        self.attn_proj = nn.Linear(dim, attn_dim) if temporal and dim != attn_dim else nn.Identity()

        self.gen = Generator(dim, hidden, p_drop)
        self.critic = Critic(dim, hidden)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        if self.temporal and x.dim() == 3:
            x_proj = self.attn_proj(x)
            x = self.tem(x_proj, m)

        # Create noise
        z = torch.randn_like(x)

        # Condition on observed values
        gen_input = torch.where(m > 0.5, x, z)

        # Generate imputations
        x_gen = self.gen(gen_input, z)

        # Apply mask
        return m * x + (1.0 - m) * x_gen

    def critic_loss(self, real: Tensor, fake: Tensor, gp_lambda: float = 10.0) -> Tensor:
        d_real = self.critic(real).mean()
        d_fake = self.critic(fake).mean()

        # Gradient penalty
        alpha = torch.rand(real.size(0), 1, device=real.device)
        if real.dim() > 2:
            alpha = alpha.unsqueeze(-1)

        x_hat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_hat = self.critic(x_hat)

        gradients = torch.autograd.grad(
            outputs=d_hat.sum(),
            inputs=x_hat,
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return -(d_real - d_fake) + gp_lambda * gradient_penalty

    def generator_loss(self, fake: Tensor, x: Tensor, m: Tensor, alpha: float = 1.0) -> Tensor:
        d_fake = self.critic(fake).mean()
        rec_loss = F.mse_loss(fake * (1 - m), x * (1 - m))
        return -d_fake + alpha * rec_loss