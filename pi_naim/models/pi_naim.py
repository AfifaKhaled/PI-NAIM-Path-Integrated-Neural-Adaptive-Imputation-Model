# pi_naim/models/pi_naim.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from pi_naim.models.gain import TemporalGAIN
from pi_naim.models.fusion import MissingnessEmbedding, ScaledDotProductAttention, AdaptiveFusion
from pi_naim.models.mice import run_mice_impute
from pi_naim.utils.losses import HomoscedasticUncertaintyWeights, masked_mse
from pi_naim.utils.masking import compute_missing_rate


@dataclass
class RouteCfg:
    threshold: float = 0.15  # Lower default threshold
    temporal: bool = True
    n_bootstrap: int = 3
    gp_lambda: float = 10.0
    alpha_rec: float = 1.0
    gain_hidden: int = 64
    gain_dropout: float = 0.3  # Increased dropout
    mice_max_iter: int = 5
    learnable_threshold: bool = True
    n_critic: int = 3
    attention_heads: int = 2


class PINAIM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64,
                 embed_dim: int = 32, cfg: RouteCfg = RouteCfg(), task_dropout: float = 0.2):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temporal = cfg.temporal

        # Ensure embed_dim is divisible by attention_heads
        if embed_dim % cfg.attention_heads != 0:
            embed_dim = embed_dim + (cfg.attention_heads - embed_dim % cfg.attention_heads)

        # Learnable routing threshold
        if cfg.learnable_threshold:
            self.route_threshold = nn.Parameter(torch.tensor(cfg.threshold))
        else:
            self.register_buffer('route_threshold', torch.tensor(cfg.threshold))

        # Missingness embedding
        self.miss_emb = MissingnessEmbedding(input_dim, embed_dim, hidden=64)

        # GAIN path with temporal analysis
        self.gain = TemporalGAIN(
            dim=input_dim,
            hidden=cfg.gain_hidden,
            temporal=cfg.temporal,
            p_drop=cfg.gain_dropout,
            n_heads=min(2, cfg.attention_heads)
        )

        # Cross-path attention fusion - ensure compatible dimensions
        cross_attn_dim = max(embed_dim, input_dim)
        if cross_attn_dim % cfg.attention_heads != 0:
            cross_attn_dim = cross_attn_dim + (cfg.attention_heads - cross_attn_dim % cfg.attention_heads)

        self.cross_attn = ScaledDotProductAttention(
            q_dim=embed_dim,
            k_dim=input_dim,
            v_dim=input_dim,
            out_dim=cross_attn_dim,
            n_heads=cfg.attention_heads
        )

        # Task prediction head with enhanced regularization
        self.task_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(task_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(task_dropout)
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)

        # Dropout for uncertainty estimation
        self.dropout = nn.Dropout(task_dropout)

        # Fusion module
        self.fusion = AdaptiveFusion(
            imp_dim=cross_attn_dim,
            task_dim=hidden_dim // 2,
            num_classes=output_dim
        )

        # Uncertainty weights - increased terms for better stability
        self.uncertainty_weights = HomoscedasticUncertaintyWeights(n_terms=4)

        # Additional regularization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with better scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def route(self, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Dynamic path selection based on missingness rate (Page 6)"""
        mr = compute_missing_rate(mask)

        if self.cfg.learnable_threshold:
            # Learnable threshold with sigmoid constraint
            adaptive_threshold = torch.sigmoid(self.route_threshold)
        else:
            # Fixed threshold with batch adaptation
            batch_std_mr = mr.std()
            adaptive_threshold = self.cfg.threshold + 0.1 * batch_std_mr

        route_decisions = (mr >= adaptive_threshold).float()
        return route_decisions, adaptive_threshold

    def forward(self, x: Tensor, m: Tensor, return_uncertainty: bool = False) -> Dict[str, Tensor]:
        # Dynamic path selection
        route_gain, adaptive_threshold = self.route(m)

        # Missingness embeddings
        miss_emb = self.miss_emb(m)

        # GAIN path imputation
        x_imp_gain = self.gain(x, m)

        # MICE path imputation (only for low missingness samples)
        x_imp_mice = x.clone()
        mice_mask = (route_gain < 0.5)

        if mice_mask.any():
            mice_indices = mice_mask.nonzero(as_tuple=True)[0]
            if len(mice_indices) > 0:
                x_mice = x[mice_indices].detach().cpu().numpy()
                m_mice = m[mice_indices].detach().cpu().numpy()

                x_imp_mice_np, uncertainty_mice = run_mice_impute(
                    x_mice, m_mice,
                    max_iter=self.cfg.mice_max_iter,
                    bootstrap=self.cfg.n_bootstrap
                )

                x_imp_mice[mice_indices] = torch.tensor(x_imp_mice_np, device=x.device, dtype=x.dtype)

        # Combine imputations
        route_gain = route_gain.view(-1, 1, 1) if x.dim() == 3 else route_gain.view(-1, 1)
        x_imp = route_gain * x_imp_gain + (1.0 - route_gain) * x_imp_mice

        # Cross-path attention fusion
        h_imp, attn_weights = self.cross_attn(miss_emb, x_imp, x_imp)

        # If attention output has sequence dimension, take mean over it
        if h_imp.dim() == 3:
            h_imp = h_imp.mean(dim=1)  # Average over sequence dimension

        # Task prediction with uncertainty
        # If x_imp has sequence dimension, take mean over it
        if x_imp.dim() == 3:
            x_imp_flat = x_imp.mean(dim=1)
        else:
            x_imp_flat = x_imp

        h_task = self.task_head(x_imp_flat)
        if return_uncertainty:
            # MC dropout for uncertainty estimation
            h_task = self.dropout(h_task)

        task_logits = self.classifier(h_task)

        # Confidence scores
        with torch.no_grad():
            c_imp = torch.exp(-F.mse_loss(x_imp, x, reduction='none').mean(dim=-1))
            c_task = torch.softmax(task_logits, dim=-1).max(dim=-1)[0]

        # Adaptive fusion
        logits, lambda_gate = self.fusion(h_imp, h_task, c_imp, c_task)

        # Fix logits shape - ensure [batch_size, num_classes]
        if logits.dim() == 3:
            # If logits have shape [batch_size, seq_len, num_classes], take mean over seq_len
            logits = logits.mean(dim=1)
        elif logits.dim() == 2 and logits.size(0) != x.size(0):
            # If shape is wrong but 2D, reshape appropriately
            logits = logits.view(x.size(0), -1)

        # Ensure logits have the correct final shape [batch_size, num_classes]
        if logits.size(1) > self.output_dim:
            logits = logits[:, :self.output_dim]  # Take first num_classes dimensions
        elif logits.size(1) < self.output_dim:
            # Pad if needed (shouldn't happen normally)
            pad_size = self.output_dim - logits.size(1)
            logits = F.pad(logits, (0, pad_size))

        # Uncertainty quantification
        if return_uncertainty:
            # Multiple forward passes for epistemic uncertainty
            uncertainties = []
            for _ in range(3):  # Reduced from 5
                with torch.no_grad():
                    h_task_unc = self.task_head(self.dropout(x_imp_flat))
                    logits_unc = self.classifier(h_task_unc)
                    uncertainties.append(logits_unc)

            uncertainties = torch.stack(uncertainties)
            uncertainty_std = uncertainties.std(dim=0).mean(dim=-1)
        else:
            uncertainty_std = torch.zeros_like(c_imp)

        return {
            'logits': logits,
            'x_imp': x_imp,
            'x_imp_gain': x_imp_gain,
            'x_imp_mice': x_imp_mice,
            'route_gain': route_gain,
            'route_threshold': adaptive_threshold,
            'attn_weights': attn_weights,
            'lambda_gate': lambda_gate,
            'c_imp': c_imp,
            'c_task': c_task,
            'uncertainty': uncertainty_std,
            'miss_emb': miss_emb
        }

    def compute_losses(self, x: Tensor, m: Tensor, y: Tensor, outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Multi-task learning objective (Page 8)"""
        # Imputation loss
        loss_imp = masked_mse(outputs['x_imp'], x, m)

        # Task loss - ensure y has the correct shape for cross_entropy
        if outputs['logits'].dim() > 2:
            logits = outputs['logits'].mean(dim=1)
        else:
            logits = outputs['logits']

        # Ensure logits have correct shape [batch_size, num_classes]
        if logits.size(1) != self.output_dim:
            if logits.size(1) > self.output_dim:
                logits = logits[:, :self.output_dim]
            else:
                # Pad if needed
                pad_size = self.output_dim - logits.size(1)
                logits = F.pad(logits, (0, pad_size))

        loss_task = F.cross_entropy(logits, y)

        # Adversarial loss for GAIN with stability
        loss_adv_g = self.gain.generator_loss(outputs['x_imp_gain'], x, m, self.cfg.alpha_rec)
        loss_adv_g = torch.clamp(loss_adv_g, min=-3.0, max=3.0)  # Tighter clamping

        # Regularization
        loss_reg = self.regularization_loss()

        # Uncertainty-weighted total loss
        lambdas = self.uncertainty_weights()

        # DEBUG: Check for numerical issues
        if torch.isnan(loss_imp) or torch.isinf(loss_imp):
            loss_imp = torch.tensor(0.0, device=x.device, requires_grad=True)
        if torch.isnan(loss_task) or torch.isinf(loss_task):
            loss_task = torch.tensor(0.0, device=x.device, requires_grad=True)
        if torch.isnan(loss_adv_g) or torch.isinf(loss_adv_g):
            loss_adv_g = torch.tensor(0.0, device=x.device, requires_grad=True)
        if torch.isnan(loss_reg) or torch.isinf(loss_reg):
            loss_reg = torch.tensor(0.0, device=x.device, requires_grad=True)

        # Ensure lambdas are positive and reasonable
        lambdas = torch.clamp(lambdas, min=1e-6, max=50.0)  # Further reduced max

        loss_total = (lambdas[0] * loss_imp +
                      lambdas[1] * loss_task +
                      lambdas[2] * loss_adv_g +
                      lambdas[3] * loss_reg)

        # DEBUG: Print loss components for monitoring
        if torch.isnan(loss_total) or torch.isinf(loss_total) or loss_total < 0:
            print(f"WARNING: Invalid loss detected - Total: {loss_total.item():.6f}")
            print(f"Components - imp: {loss_imp.item():.6f}, task: {loss_task.item():.6f}, "
                  f"adv_g: {loss_adv_g.item():.6f}, reg: {loss_reg.item():.6f}")
            print(f"Lambdas: {lambdas.detach().cpu().numpy()}")

            # Fallback to simple sum if numerical issues
            loss_total = loss_imp + loss_task + 0.05 * loss_adv_g + 0.001 * loss_reg

        return {
            'total': loss_total,
            'imp': loss_imp,
            'task': loss_task,
            'adv_g': loss_adv_g,
            'reg': loss_reg,
            'lambdas': lambdas
        }

    def train_adversarial(self, x: Tensor, m: Tensor) -> Tensor:
        """Train critic with WGAN-GP (Page 7)"""
        if self.training:
            with torch.no_grad():
                x_imp_gain = self.gain(x, m)
            critic_loss = self.gain.critic_loss(x, x_imp_gain, self.cfg.gp_lambda)

            # Clip critic loss to prevent explosions
            if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                critic_loss = torch.tensor(0.0, device=x.device, requires_grad=False)

            return torch.clamp(critic_loss, min=-3.0, max=3.0)  # Tighter clamping
        return torch.tensor(0.0, device=x.device)

    def regularization_loss(self) -> Tensor:
        """L2 regularization with clipping"""
        reg_loss = sum(p.norm(2) for p in self.parameters() if p.requires_grad) * 0.001
        return torch.clamp(reg_loss, max=3.0)  # Tighter clamping

    def debug_loss_components(self, x: Tensor, m: Tensor, y: Tensor, outputs: Dict[str, Tensor]):
        """Debug method to print detailed loss information"""
        with torch.no_grad():
            # Calculate individual components
            loss_imp = F.mse_loss(outputs['x_imp'] * (1 - m), x * (1 - m))
            loss_task = F.cross_entropy(outputs['logits'], y)

            # Confidence metrics
            task_probs = torch.softmax(outputs['logits'], dim=1)
            task_confidence = task_probs.max(dim=1)[0].mean()
            imp_confidence = torch.exp(-F.mse_loss(outputs['x_imp'], x, reduction='none').mean())

            print(f"DEBUG - Loss imp: {loss_imp.item():.4f}, task: {loss_task.item():.4f}")
            print(f"DEBUG - Confidence: task={task_confidence.item():.3f}, imp={imp_confidence.item():.3f}")
            print(f"DEBUG - Routing: {outputs['route_gain'].mean().item():.3f} to GAIN")

            return loss_imp.item(), loss_task.item()

    def get_routing_stats(self) -> Dict[str, float]:
        """Get routing statistics for analysis"""
        if hasattr(self, 'route_threshold'):
            threshold = self.route_threshold.item() if self.cfg.learnable_threshold else self.route_threshold
            return {'threshold': float(threshold)}
        return {}