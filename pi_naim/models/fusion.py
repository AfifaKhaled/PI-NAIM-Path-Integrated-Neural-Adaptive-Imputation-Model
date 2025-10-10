# pi_naim/models/fusion.py
import torch
import torch.nn as nn


class MissingnessEmbedding(nn.Module):
    def __init__(self, dim_flat, embed_dim=64, hidden=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(dim_flat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim)
        )

    def forward(self, mask):
        if mask.dim() == 3:
            B, T, D = mask.shape
            mask = mask.reshape(B, T * D)
        return self.proj(mask)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, out_dim, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        # Ensure head_dim is valid
        assert self.head_dim * n_heads == out_dim, "out_dim must be divisible by n_heads"

        self.Wq = nn.Linear(q_dim, out_dim, bias=False)
        self.Wk = nn.Linear(k_dim, out_dim, bias=False)
        self.Wv = nn.Linear(v_dim, out_dim, bias=False)
        self.scale = self.head_dim ** 0.5
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # Project to same dimension
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn_scores, dim=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        out = self.out_proj(out)

        return out, attn


class AdaptiveFusion(nn.Module):
    def __init__(self, imp_dim, task_dim, num_classes=2):
        super().__init__()
        self.common_dim = max(imp_dim, task_dim)
        self.imp_proj = nn.Linear(imp_dim, self.common_dim)
        self.task_proj = nn.Linear(task_dim, self.common_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.ReLU(),
            nn.Linear(self.common_dim // 2, num_classes)
        )

        # Confidence-based gating mechanism
        self.lambda_gate = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, h_imp, h_task, c_imp, c_task):
        # Project both inputs to common dimension
        h_imp_proj = self.imp_proj(h_imp)
        h_task_proj = self.task_proj(h_task)

        # Confidence-based gating
        confidence_stack = torch.stack([c_imp, c_task], dim=-1)
        lambda_val = self.lambda_gate(confidence_stack)

        # Fuse features - ensure proper broadcasting
        h_fused = lambda_val * h_imp_proj + (1 - lambda_val) * h_task_proj

        # Classification - ensure output is [batch_size, num_classes]
        logits = self.classifier(h_fused)

        # If logits have extra dimensions, squeeze them
        if logits.dim() > 2:
            logits = logits.squeeze(1)  # Remove extra dimension

        return logits, lambda_val