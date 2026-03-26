"""
models/router.py — 动态 K 专家路由器 (v2)
v2: 推理时向量化 topk 替代逐样本循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DynamicKRouter(nn.Module):

    def __init__(
        self,
        in_channels: int   = 64,
        spatial_size: int  = 4,
        hidden_dim: int    = 256,
        num_experts: int   = 12,
        dynamic_k: bool    = True,
        fixed_k: int       = 2,
        min_k: int         = 1,
        max_k: int         = 4,
        threshold_init: float = 0.08,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.dynamic_k   = dynamic_k
        self.fixed_k     = fixed_k
        self.min_k       = min_k
        self.max_k       = max_k
        self.temperature = temperature

        feat_dim = in_channels * spatial_size * spatial_size

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(spatial_size),
            nn.Flatten(),
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

        self.log_threshold = nn.Parameter(
            torch.tensor(threshold_init).log()
        )
        self.noise_scale = nn.Parameter(torch.ones(num_experts) * 0.1)

    def forward(
        self,
        feat: torch.Tensor,
        training: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if training is None:
            training = self.training

        logits = self.gate(feat)

        if training:
            noise        = torch.randn_like(logits) * F.softplus(self.noise_scale)
            logits_noisy = logits + noise
            gate_soft    = F.gumbel_softmax(
                logits_noisy, tau=self.temperature, hard=False
            )

            avg_usage    = gate_soft.mean(dim=0)
            balance_loss = (avg_usage * torch.log(avg_usage + 1e-8)).sum()
            balance_loss = balance_loss + torch.log(
                torch.tensor(float(self.num_experts))
            )

            if self.dynamic_k:
                threshold    = torch.sigmoid(self.log_threshold)
                soft_mask    = torch.sigmoid((gate_soft - threshold) * 10)
                gate_weights = gate_soft * soft_mask
                gate_weights = gate_weights / (
                    gate_weights.sum(dim=-1, keepdim=True) + 1e-8
                )
            else:
                gate_weights = gate_soft

            active_k = (gate_weights > 0.01).float().sum(dim=-1)

        else:
            # v2: 向量化推理，替代逐样本循环
            probs        = F.softmax(logits, dim=-1)
            balance_loss = torch.tensor(0.0, device=feat.device)

            if self.dynamic_k:
                threshold = torch.sigmoid(self.log_threshold).item()
                threshold = max(threshold, 1.0 / self.num_experts)

                # 每个样本取 max_k 个候选
                topk_vals, topk_idx = probs.topk(self.max_k, dim=-1)

                # 根据阈值计算每个样本的 K
                mask = probs > threshold
                active_k = mask.float().sum(dim=-1).clamp(self.min_k, self.max_k)

                # 生成位置 mask：对每个样本，只保留前 active_k 个
                k_range = torch.arange(
                    self.max_k, device=probs.device
                ).unsqueeze(0)
                k_mask = k_range < active_k.unsqueeze(1).long()

                # 散布有效的 topk 值
                topk_vals_masked = topk_vals * k_mask.float()
                gate_weights = torch.zeros_like(probs)
                gate_weights.scatter_(1, topk_idx, topk_vals_masked)
            else:
                topk_vals, topk_idx = probs.topk(self.fixed_k, dim=-1)
                gate_weights = torch.zeros_like(probs).scatter_(
                    1, topk_idx, topk_vals
                )
                active_k = torch.full(
                    (probs.size(0),), self.fixed_k, device=feat.device
                )

            gate_weights = gate_weights / (
                gate_weights.sum(dim=-1, keepdim=True) + 1e-8
            )

        return gate_weights, active_k, balance_loss

    def get_threshold(self) -> float:
        return torch.sigmoid(self.log_threshold).item()

    def get_avg_active_k(self, feat: torch.Tensor) -> float:
        with torch.no_grad():
            _, active_k, _ = self.forward(feat, training=False)
        return active_k.float().mean().item()
