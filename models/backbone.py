"""
models/backbone.py — 主干网络 (v5.2)

版本历史：
  v5.1.1: MobileNetV3-Small, stem stride=1（Tiny-ImageNet 64×64 适配）
  v5.2:   切换 NWPU-RESISC45 224×224
          Student: 恢复默认 stem stride=2（224×224 足够大，不需要改）
          截断点 stage1: (24, 28, 28)  stage2: (40, 14, 14)
          Teacher: 新增 image_size 参数，224×224 使用标准 ResNet-50 stem

Teacher（ResNet-50）新增 image_size 参数。
"""

import torch
import torch.nn as nn
from torchvision import models


# ── Teacher 辅助函数（保持不变）────────────────────────────

def load_pretrained_except_stem(base_model, pretrained_model):
    pretrained_sd = pretrained_model.state_dict()
    current_sd    = base_model.state_dict()
    skip_keys, loaded_keys = set(), set()
    for k, v in pretrained_sd.items():
        if k.startswith("conv1") or k.startswith("fc"):
            skip_keys.add(k); continue
        if k in current_sd and current_sd[k].shape == v.shape:
            current_sd[k] = v; loaded_keys.add(k)
        else:
            skip_keys.add(k)
    base_model.load_state_dict(current_sd)
    print(f"[Backbone] 预训练权重加载: {len(loaded_keys)} 层  "
          f"跳过: {len(skip_keys)} 层 (conv1/fc/shape不匹配)")
    return base_model


# ── v5.2 Student：MobileNetV3-Small（参数化）──────────────

class BackboneMobileNetV3(nn.Module):
    """
    Student 主干：MobileNetV3-Small (v5.2)

    cut_point:
      "stage1" → features[0:4] → (B, 24, 16, 16)  feat_channels=24
      "stage2" → features[0:7] → (B, 40,  8,  8)  feat_channels=40

    head_type（standalone head 强度）:
      "linear"       → Linear(C, 200)                           最弱
      "mlp_small"    → Linear(C,128) → ReLU → Linear(128,200)
      "mlp_medium"   → Linear(C,256) → ReLU → Dropout → Linear(256,200)
      "full_network" → features_late → GAP → Linear(576,200)   最强

    stem stride 固定为 1（v5.1.1 确认）。
    """

    # cut_point → (features_early slice, features_late slice, channels, spatial)
    # spatial 对应 224×224 输入、stem stride=2 时的特征图尺寸
    CUT_CONFIG = {
        "stage1": (slice(0, 4), slice(4, None), 24, 28),  # (B, 24, 28, 28)
        "stage2": (slice(0, 7), slice(7, None), 40, 14),  # (B, 40, 14, 14)
    }

    def __init__(
        self,
        num_classes: int = 200,
        pretrained:  bool = True,
        head_type:   str  = "full_network",
        cut_point:   str  = "stage1",
    ):
        super().__init__()
        assert cut_point in self.CUT_CONFIG, \
            f"cut_point 必须是 {list(self.CUT_CONFIG.keys())}，got {cut_point!r}"
        assert head_type in ("linear", "mlp_small", "mlp_medium", "full_network"), \
            f"未知 head_type: {head_type!r}"

        self.cut_point = cut_point
        self.head_type = head_type

        if pretrained:
            base = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
        else:
            base = models.mobilenet_v3_small(weights=None)

        # v5.2: 224×224 输入，保持默认 stem stride=2（不需要修改）
        # v5.1.1 的 stride=1 改动仅适用于 64×64 Tiny-ImageNet，224×224 不需要

        # 截断点
        early_sl, late_sl, feat_ch, feat_sp = self.CUT_CONFIG[cut_point]
        self.features_early = base.features[early_sl]
        self.features_late  = base.features[late_sl]
        self.feat_channels  = feat_ch   # 供 moe_system 读取
        self.feat_spatial   = feat_sp   # 供 moe_system 读取

        # standalone pool + head
        self.standalone_pool = nn.AdaptiveAvgPool2d(1)
        C = feat_ch

        if head_type == "linear":
            self.standalone_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(C, num_classes),
            )
        elif head_type == "mlp_small":
            self.standalone_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(C, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes),
            )
        elif head_type == "mlp_medium":
            self.standalone_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(C, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
        elif head_type == "full_network":
            # standalone 走完整网络，最终 576ch → 200
            self.standalone_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(576, num_classes),
            )

        print(f"[Backbone] MobileNetV3-Small  cut={cut_point}({feat_ch}ch,{feat_sp}×{feat_sp})  "
              f"head={head_type}  pretrained={pretrained}  "
              f"(224×224 输入，stem stride=2)")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """截断输出，喂给压缩器/专家"""
        return self.features_early(x)

    def forward_standalone(self, x: torch.Tensor) -> torch.Tensor:
        """独立推理，精度由 head_type 决定"""
        feat = self.features_early(x)
        if self.head_type == "full_network":
            feat = self.features_late(feat)
        out = self.standalone_pool(feat)
        return self.standalone_head(out)

    def forward(self, x: torch.Tensor, standalone: bool = False):
        if standalone:
            return self.forward_standalone(x)
        return self.forward_features(x)

    def freeze_layers(self, layer_names: list):
        layer_map = {
            "features_early": self.features_early,
            "features_late":  self.features_late,
        }
        for name in layer_names:
            if name in layer_map:
                for p in layer_map[name].parameters():
                    p.requires_grad = False
                print(f"[Backbone] 已冻结: {name}")


# ── Teacher：ResNet-50，完全不变 ────────────────────────────

class BackboneResNet50Teacher(nn.Module):
    """
    Teacher：ResNet-50

    image_size < 128（如 64×64 Tiny-ImageNet）：
      stem 改为 3×3 stride=1 + Identity maxpool（小图像适配）
    image_size >= 128（如 224×224 RESISC45）：
      保持标准 ResNet-50 stem（7×7 stride=2 + 3×3 maxpool）
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True,
                 image_size: int = 64):
        super().__init__()
        small_image = image_size < 128

        if pretrained:
            base_pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            base = models.resnet50(weights=None)
            if small_image:
                base.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                base.maxpool = nn.Identity()
            base.fc = nn.Linear(2048, num_classes)
            if small_image:
                base = load_pretrained_except_stem(base, base_pretrained)
            else:
                # 标准 stem 权重可以直接复用，只跳过 fc
                sd_pre = base_pretrained.state_dict()
                sd_cur = base.state_dict()
                loaded, skipped = 0, 0
                for k, v in sd_pre.items():
                    if k.startswith("fc"):
                        skipped += 1; continue
                    if k in sd_cur and sd_cur[k].shape == v.shape:
                        sd_cur[k] = v; loaded += 1
                    else:
                        skipped += 1
                base.load_state_dict(sd_cur)
                print(f"[Backbone] 预训练权重加载: {loaded} 层  跳过: {skipped} 层 (fc/shape不匹配)")
        else:
            base = models.resnet50(weights=None)
            if small_image:
                base.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                base.maxpool = nn.Identity()
            base.fc = nn.Linear(2048, num_classes)

        print(f"[Teacher] ResNet-50  num_classes={num_classes}  "
              f"image_size={image_size}  small_image={small_image}  pretrained={pretrained}")
        self.stem    = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool    = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = base.fc

    def forward_features(self, x):
        x = self.stem(x); x = self.pool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        feat = self.layer4(feat)
        out  = self.avgpool(feat).flatten(1)
        return self.fc(out)