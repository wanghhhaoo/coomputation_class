"""
models/experts.py — 同构自学习专家网络 (v5.1)

版本历史：
  v2: dropout_rate 可配置
  v5.1: 默认参数适配 MobileNetV3-Small 维度
        in_channels: 128 → 48（解压后通道）
        hidden_dim:  256 → 96（更轻量）
        num_layers:  3   → 2（输入特征图已够小）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class SelfLearnExpert(nn.Module):
    """
    同构自学习专家
    forward 返回 (logits, feat)
    """

    def __init__(
        self,
        in_channels: int  = 48,           # v5.1: 128 → 48
        hidden_dim: int   = 96,           # v5.1: 256 → 96
        num_classes: int  = 200,
        num_layers: int   = 2,            # v5.1: 3 → 2
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ExpertBlock(hidden_dim) for _ in range(num_layers)]
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.blocks(x)

        B, C, H, W = x.shape
        attn = self.se(x).view(B, C, 1, 1)
        x    = x * attn

        feat   = self.pool(x).flatten(1)
        logits = self.head(feat)

        return logits, feat


def build_experts(
    num_experts: int  = 4,
    in_channels: int  = 48,           # v5.1: 128 → 48
    hidden_dim: int   = 96,           # v5.1: 256 → 96
    num_classes: int  = 200,
    num_layers: int   = 2,            # v5.1: 3 → 2
    dropout_rate: float = 0.5,
) -> nn.ModuleList:
    experts = nn.ModuleList([
        SelfLearnExpert(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
        for _ in range(num_experts)
    ])
    total_params = sum(p.numel() for p in experts[0].parameters()) / 1e6
    print(f"[Experts] {num_experts} 个同构自学习专家  "
          f"单专家参数: {total_params:.2f}M  "
          f"合计: {total_params * num_experts:.1f}M  "
          f"Dropout: {dropout_rate}")
    return experts
