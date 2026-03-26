"""
config.py — Tiny-ImageNet 版本配置 (v5.1.1)

版本历史：
  v5.1:   MobileNetV3-Small 主干，early_stopping_patience=15
  v5.1.1: early_stopping_patience 15→25（15 太激进，Stage 3 前 20 epoch 几乎不涨）
          backbone.py 中 stem stride 2→1（截断特征 8×8→16×16，config 无需改动）
"""

from dataclasses import dataclass, field
from typing import List, Optional

DEPLOY_MODE = "local"


@dataclass
class DataConfig:
    train_dir: str   = "data/tiny-imagenet-200/train"
    val_dir: str     = "data/tiny-imagenet-200/val"
    num_classes: int = 200
    image_size: int  = 64
    num_workers: int = 16
    pin_memory: bool = True
    prefetch_factor: int = 4


@dataclass
class BackboneConfig:
    arch: str         = "mobilenet_v3_small"   # v5.1: resnet18 → mobilenet_v3_small
    pretrained: bool  = True
    cut_layer: str    = "stage1"               # features[0:4]
    frozen_layers: List[str] = field(default_factory=list)


@dataclass
class CompressorConfig:
    in_channels: int  = 24       # v5.1: 128 → 24（MobileNetV3 stage1 输出）
    out_channels: int = 24       # v5.1: 32 → 24（不压缩通道，只做空间池化）
    spatial_size: int = 4
    quantize: bool    = False


@dataclass
class RouterConfig:
    in_channels: int      = 24       # v5.1: 32 → 24（压缩后通道）
    spatial_size: int     = 4
    hidden_dim: int       = 64       # v5.1: 128 → 64（更小的路由器）
    num_experts: int      = 4
    dynamic_k: bool       = True
    fixed_k: int          = 2
    min_k: int            = 1
    max_k: int            = 4
    threshold_init: float = 0.08
    temperature: float    = 1.0
    balance_loss_weight: float = 0.1


@dataclass
class ExpertDropoutConfig:
    expert_dropout: float = 0.3
    ortho_w: float        = 0.5           # v2: 0.1 → 0.5，防正交损失过快归零


@dataclass
class ExpertConfig:
    in_channels: int  = 48       # v5.1: 64 → 48（解压后通道）
    spatial_size: int = 8
    num_classes: int  = 200
    hidden_dim: int   = 96       # v5.1: 128 → 96
    num_layers: int   = 2        # v5.1: 3 → 2（输入已够小）
    dropout_rate: float = 0.5


@dataclass
class AlphaScheduleConfig:
    """Stage 3 backbone_alpha 退火调度"""
    alpha_start: float     = 0.7          # v4: standalone 更弱，起点降低（原 0.85）
    alpha_end: float       = 0.5          # v5: 0.4→0.5（v5 standalone 更强，最优alpha偏高）
    warmup_epochs: int     = 20           # v4: 20 epoch 足够退火（原 30）
    freeze_backbone: bool  = True         # Stage 3 冻结主干


@dataclass
class DistillConfig:
    teacher_arch: str        = "resnet50"
    teacher_pretrained: bool = True
    teacher_checkpoint: Optional[str] = None
    alpha_feat: float  = 1.0
    beta_kd: float     = 1.0
    gamma_ce: float    = 1.0
    temperature: float = 4.0
    stage: int         = 2


@dataclass
class TrainConfig:
    batch_size: int    = 512

    epochs_stage2: int = 15               # v5.1: 12 → 15（新主干可能需要更多蒸馏）
    epochs_stage3: int = 100              # v5: 60 → 100

    lr: float          = 8e-3
    lr_backbone: float = 8e-4
    weight_decay: float = 5e-4            # v2: 1e-4 → 5e-4
    warmup_epochs: int  = 3               # v2: 5 → 3
    grad_clip: float    = 1.0

    early_stopping_patience: int = 25    # v5.1.1: 15 → 25（15 太激进，前 20 epoch 几乎不涨）

    use_amp: bool       = True
    amp_dtype: str      = "bf16"

    use_data_parallel: bool = False
    gpu_ids: List[int] = field(default_factory=lambda: [1])
    primary_gpu: int   = 1

    cudnn_benchmark: bool = True

    save_dir: str     = "checkpoints_tiny"
    save_every: int   = 5
    log_interval: int = 20

    # v2: Stage 3 使用强数据增强
    stage3_use_autoaugment: bool = True
    stage3_use_mixup: bool       = True
    stage3_mixup_alpha: float    = 0.2


@dataclass
class DeployConfig:
    backbone_device: str = "cuda:1"
    expert_devices: List[str] = field(
        default_factory=lambda: ["cuda:1"] * 4     # v3: 4 个专家
    )
    aggregation_device: str = "cuda:1"


data_cfg        = DataConfig()
backbone_cfg    = BackboneConfig()
compressor_cfg  = CompressorConfig()
router_cfg      = RouterConfig()
expert_cfg      = ExpertConfig()
alpha_sched_cfg = AlphaScheduleConfig()
distill_cfg     = DistillConfig()
train_cfg       = TrainConfig()
expert_drop_cfg = ExpertDropoutConfig()
deploy_cfg      = DeployConfig()