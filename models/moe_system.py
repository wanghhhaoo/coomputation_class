"""
models/moe_system.py — MoE 系统 (v5.2)

版本历史：
  v5.1: MobileNetV3-Small 主干，固定维度参数
  v5.2: 支持从 config dict 构建（供 search.py 使用）
        维度根据 cut_point 自动推导，无需手动传入
        保持旧的关键字参数接口向后兼容
  v5.2 Phase A: spatial_size=None → 无解压器
        主干截断特征直接喂给专家，验证增益上界
        expert_in = feat_ch（不经过解压器）

维度自动推导规则（config dict 模式）：
  spatial_size=None (Phase A): expert_in=feat_ch（无解压器）
  spatial_size=14   (Phase B): expert_in=feat_ch*2（有解压器，通道翻倍）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .backbone   import BackboneMobileNetV3
from .compressor import FeatureCompressor, FeatureDecompressor
from .experts    import build_experts
from .router     import DynamicKRouter


class MoESystemTiny(nn.Module):

    def __init__(
        self,
        # ── v5.2: config dict 接口（优先级高）──────────────
        config: dict = None,
        # ── 向后兼容的关键字参数（config=None 时生效）──────
        num_classes:     int   = 200,
        num_experts:     int   = 4,
        deploy_mode:     str   = "local",
        backbone_device: str   = "cuda:0",
        expert_devices:  Optional[List[str]] = None,
        dynamic_k:       bool  = True,
        quantize:        bool  = False,
        expert_dropout:  float = 0.3,
        initial_alpha:   float = 0.7,
        dropout_rate:    float = 0.5,
        # 旧接口维度参数（config=None 时才使用）
        compress_in:     int   = 24,
        compress_out:    int   = 24,
        decompress_out:  int   = 48,
        expert_hidden:   int   = 96,
        router_hidden:   int   = 64,
        # backbone 参数（config=None 时才使用）
        head_type:       str   = "full_network",
        cut_point:       str   = "stage1",
    ):
        super().__init__()

        # ── 解析参数：config dict 优先，否则用关键字参数 ──
        if config is not None:
            num_classes    = config.get("num_classes",        45)
            num_experts    = config.get("num_experts",        4)
            deploy_mode    = config.get("deploy_mode",        "local")
            expert_dropout = config.get("expert_dropout",     0.3)
            initial_alpha  = config.get("alpha_start",        0.7)
            dropout_rate   = config.get("dropout_rate",       0.5)
            head_type      = config.get("head_type",          "full_network")
            cut_point      = config.get("cut_point",          "stage1")
            expert_hidden  = config.get("expert_hidden_dim",  96)
            router_hidden  = config.get("router_hidden_dim",  64)
            expert_layers  = config.get("expert_num_layers",  2)
            # v5.2: spatial_size=None → Phase A（无解压器）
            spatial_size   = config.get("spatial_size",       None)
        else:
            expert_layers = 2   # 旧接口默认值
            spatial_size  = 4   # 旧接口默认值（向后兼容）

        self.deploy_mode    = deploy_mode
        self.num_experts    = num_experts
        self.num_classes    = num_classes
        self.expert_dropout = expert_dropout

        if expert_devices is None:
            expert_devices = [backbone_device] * num_experts
        self.backbone_device = backbone_device
        self.expert_devices  = expert_devices

        # ── 构建 backbone，自动读取截断维度 ────────────────
        self.backbone  = BackboneMobileNetV3(
            num_classes=num_classes,
            head_type=head_type,
            cut_point=cut_point,
        )
        feat_ch = self.backbone.feat_channels   # 24 or 40

        # ── 自动推导压缩器/解压器维度 ──────────────────────
        if config is not None:
            compress_in  = feat_ch
            compress_out = feat_ch          # 通道不压缩

            # v5.2 Phase A: spatial_size=None → 无解压器，expert 直接用压缩输出
            if spatial_size is None:
                self.use_decompressor = False
                expert_in = compress_out    # 24 (stage1) or 40 (stage2)
                decompress_out = expert_in  # 占位，不实际使用
            else:
                self.use_decompressor = True
                decompress_out = feat_ch * 2   # 解压时通道翻倍
                expert_in = decompress_out
        else:
            # 旧接口：始终使用解压器（向后兼容）
            self.use_decompressor = True
            expert_in = decompress_out

        # ── 构建各模块 ────────────────────────────────────
        self.compressor = FeatureCompressor(
            in_channels=compress_in,
            out_channels=compress_out,
            spatial_size=spatial_size,     # None → Identity pool（Phase A）
            quantize=quantize,
        )

        if self.use_decompressor:
            self.decompressor = FeatureDecompressor(
                in_channels=compress_out,
                out_channels=decompress_out,
            )
        else:
            self.decompressor = None       # Phase A: 无解压器

        self.router = DynamicKRouter(
            in_channels=compress_out,
            hidden_dim=router_hidden,
            num_experts=num_experts,
            dynamic_k=dynamic_k,
        )
        self.experts = build_experts(
            num_experts=num_experts,
            in_channels=expert_in,
            hidden_dim=expert_hidden,
            num_classes=num_classes,
            num_layers=expert_layers,
            dropout_rate=dropout_rate,
        )

        self.register_buffer('backbone_alpha', torch.tensor(initial_alpha))

        if deploy_mode == "distributed":
            self._distribute()

    # ── alpha 调度 ────────────────────────────────────────

    def set_alpha(self, value: float):
        self.backbone_alpha.fill_(value)

    def get_alpha(self) -> float:
        return self.backbone_alpha.item()

    # ── 分布式部署 ────────────────────────────────────────

    def _distribute(self):
        self.backbone   = self.backbone.to(self.backbone_device)
        self.compressor = self.compressor.to(self.backbone_device)
        self.router     = self.router.to(self.backbone_device)
        if self.use_decompressor:
            self.decompressors = nn.ModuleList([
                FeatureDecompressor(
                    in_channels=self.decompressor.conv1[0].in_channels,
                    out_channels=self.decompressor.conv2[0].out_channels,
                ).to(dev) for dev in self.expert_devices
            ])
        for i, dev in enumerate(self.expert_devices):
            self.experts[i] = self.experts[i].to(dev)

    # ── Expert Dropout ────────────────────────────────────

    def _get_active_indices(self) -> List[int]:
        if not self.training or self.expert_dropout <= 0:
            return list(range(self.num_experts))
        n_drop   = max(0, int(self.num_experts * self.expert_dropout))
        n_drop   = min(n_drop, self.num_experts - 2)
        drop_idx = torch.randperm(self.num_experts)[:n_drop].tolist()
        return [i for i in range(self.num_experts) if i not in drop_idx]

    # ── 前向传播 ─────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_extras: bool = False,
    ) -> Dict[str, torch.Tensor]:

        # 1. 主干特征提取
        backbone_feat   = self.backbone.forward_features(x)
        backbone_logits = self.backbone.forward_standalone(x)

        # 2. 特征压缩
        compressed = self.compressor(backbone_feat)

        # 3. 路由决策
        gate_weights, active_k, balance_loss = self.router(compressed)

        # 4. Expert Dropout
        active_indices = self._get_active_indices()

        # 5. 专家推理
        expert_logits = []
        expert_feats  = []

        if self.deploy_mode == "local":
            # v5.2 Phase A: 无解压器时直接用压缩输出
            decompressed = (self.decompressor(compressed)
                            if self.use_decompressor else compressed)
            for i in active_indices:
                logit, feat = self.experts[i](decompressed)
                expert_logits.append((i, logit))
                expert_feats.append(feat)
        else:
            for i in active_indices:
                dev   = self.expert_devices[i]
                comp  = compressed.to(dev)
                decomp = (self.decompressors[i](comp)
                          if self.use_decompressor else comp)
                logit, feat = self.experts[i](decomp)
                expert_logits.append((i, logit.to(self.backbone_device)))
                expert_feats.append(feat.to(self.backbone_device))

        # 6. 加权聚合
        if expert_logits:
            active_idx_tensor = torch.tensor(
                [idx for idx, _ in expert_logits], device=compressed.device
            )
            active_weights = gate_weights[:, active_idx_tensor]
            active_weights = active_weights / (
                active_weights.sum(dim=-1, keepdim=True) + 1e-8
            )
            stacked      = torch.stack([l for _, l in expert_logits], dim=1)
            expert_fused = (stacked * active_weights.unsqueeze(-1)).sum(dim=1)
        else:
            expert_fused = torch.zeros_like(backbone_logits)

        alpha  = self.backbone_alpha
        logits = alpha * backbone_logits + (1 - alpha) * expert_fused

        result = {
            "logits":         logits,
            "gate_weights":   gate_weights,
            "active_k":       active_k,
            "balance_loss":   balance_loss,
            "active_indices": active_indices,
            "expert_feats":   expert_feats,
        }

        if return_extras:
            result["backbone_logits"] = backbone_logits
            result["expert_logits"]   = [l for _, l in expert_logits]
            result["threshold"]       = self.router.get_threshold()
            result["alpha"]           = alpha.item()

        return result

    def forward_standalone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_standalone(x)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[MoE] 主干已冻结，仅训练专家和路由器")

    def param_groups(self, lr: float, lr_backbone: float):
        groups = [
            {"params": self.backbone.parameters(),   "lr": lr_backbone},
            {"params": self.compressor.parameters(), "lr": lr},
        ]
        if self.use_decompressor:
            groups.append({"params": self.decompressor.parameters(), "lr": lr})
        groups += [
            {"params": self.router.parameters(),   "lr": lr},
            {"params": self.experts.parameters(),  "lr": lr},
        ]
        return groups

    def print_params(self):
        def cnt(m): return sum(p.numel() for p in m.parameters()) / 1e6
        total     = cnt(self)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        phase = "A（无解压器）" if not self.use_decompressor else "B（有解压器）"
        print(f"[MoE-Tiny 参数量]  Phase {phase}")
        print(f"  Backbone (MobileNetV3-Small): {cnt(self.backbone):.2f}M")
        print(f"  Compressor:                  {cnt(self.compressor):.3f}M")
        if self.use_decompressor:
            print(f"  Decompressor:                {cnt(self.decompressor):.3f}M")
        else:
            print(f"  Decompressor:                (跳过，Phase A)")
        print(f"  Router:                      {cnt(self.router):.3f}M")
        print(f"  Experts × {self.num_experts}:               {cnt(self.experts):.2f}M")
        print(f"  ──────────────────────────────────")
        print(f"  Total:                       {total:.2f}M")
        print(f"  Trainable:                   {trainable:.2f}M")