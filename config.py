"""
config.py — NWPU-RESISC45 版本配置 (v5.2)

版本历史：
  v5.1.1: Tiny-ImageNet 64×64，stem stride=1，200 类
  v5.2:   切换 NWPU-RESISC45 224×224，45 类
          Phase A: 无压缩（spatial_size=None），验证专家增益上界
          Phase B: 逐步增大压缩率
"""

from dataclasses import dataclass, field
from typing import List, Optional

DEPLOY_MODE = "local"


@dataclass
class DataConfig:
    dataset: str       = "resisc45"                # v5.2: tiny-imagenet → resisc45
    data_dir: str      = "/home/wh/code14/datasets/nwpuresisc45/train/train"  # RESISC45 训练集
    num_classes: int   = 45                         # v5.2: 200 → 45
    image_size: int    = 224                        # v5.2: 64 → 224
    num_workers: int   = 16
    pin_memory: bool   = True
    prefetch_factor: int = 4
    train_ratio: float = 0.8                        # v5.2: 80/20 stratified split


@dataclass
class BackboneConfig:
    arch: str         = "mobilenet_v3_small"
    pretrained: bool  = True
    cut_point: str    = "stage1"                   # features[0:4] → (24, 28, 28) at 224×224
    head_type: str    = "full_network"
    frozen_layers: List[str] = field(default_factory=list)


@dataclass
class CompressorConfig:
    in_channels: int  = 24
    out_channels: int = 24
    spatial_size: int = None   # v5.2 Phase A: None = 不压缩（Identity pool）
    quantize: bool    = False


@dataclass
class RouterConfig:
    in_channels: int       = 24       # 压缩后通道（Phase A 等于截断特征通道）
    spatial_size: int      = 28       # v5.2: 28×28（Phase A 无空间压缩）
    hidden_dim: int        = 64
    num_experts: int       = 4
    dynamic_k: bool        = True
    fixed_k: int           = 2
    min_k: int             = 1
    max_k: int             = 4
    threshold_init: float  = 0.08
    temperature: float     = 1.0
    balance_loss_weight: float = 0.1


@dataclass
class ExpertDropoutConfig:
    expert_dropout: float = 0.3
    ortho_w: float        = 0.5


@dataclass
class ExpertConfig:
    in_channels: int    = 24       # v5.2 Phase A: 24（直接用截断特征，无解压）
    spatial_size: int   = 28       # v5.2: 28×28（无压缩）
    num_classes: int    = 45       # v5.2: 200 → 45
    hidden_dim: int     = 96
    num_layers: int     = 2
    dropout_rate: float = 0.5


@dataclass
class AlphaScheduleConfig:
    """Stage 3 backbone_alpha 退火调度"""
    alpha_start: float    = 0.7
    alpha_end: float      = 0.5
    warmup_epochs: int    = 20
    freeze_backbone: bool = True


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
    batch_size: int    = 128                      # v5.2: 512 → 128（224×224 显存更大）

    epochs_stage2: int = 20                       # v5.2: 15 → 20
    epochs_stage3: int = 100

    lr: float          = 8e-3
    lr_backbone: float = 8e-4
    weight_decay: float = 5e-4
    warmup_epochs: int  = 3
    grad_clip: float    = 1.0

    early_stopping_patience: int = 25

    use_amp: bool       = True
    amp_dtype: str      = "bf16"

    use_data_parallel: bool = False
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    primary_gpu: int   = 0

    cudnn_benchmark: bool = True

    save_dir: str     = "checkpoints_resisc"      # v5.2: 新 checkpoint 目录
    save_every: int   = 5
    log_interval: int = 20

    stage3_use_autoaugment: bool = True
    stage3_use_mixup: bool       = True
    stage3_mixup_alpha: float    = 0.2


@dataclass
class DeployConfig:
    backbone_device: str = "cuda:0"
    expert_devices: List[str] = field(
        default_factory=lambda: ["cuda:0"] * 4
    )
    aggregation_device: str = "cuda:0"


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
