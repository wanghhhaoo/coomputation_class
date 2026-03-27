# MoE-Tiny 项目文档

> 更新时间：2026-03-27

---

## 一、项目概述

**目标**：在无人机/边缘设备部署场景下，用"轻量主干 + 远端 MoE 专家"的分布式推理架构，突破单一轻量模型的精度上限。

**核心思路**：

```
图像输入 → MobileNetV3-Small（设备端，低功耗）
         ↓ 截断特征（轻量或无压缩）→ 传输到服务器
         服务器端：Router 动态选专家 → 多专家加权融合
         ↓ 最终预测 = α × 主干 + (1-α) × 专家集成
```

**版本沿革**：

| 版本 | 数据集 | 主干 | 核心改动 |
|------|--------|------|---------|
| v5.1.1 | Tiny-ImageNet 64×64 | MobileNetV3-Small (stride=1) | 主干够强，专家增益为零 |
| v5.2-search | Tiny-ImageNet 64×64 | 同上 | 架构搜索，Phase1 崩溃，验证问题在数据集 |
| **v5.2** | **NWPU-RESISC45 224×224** | **MobileNetV3-Small (stride=2)** | **换数据集 + 低压缩率验证** |

**换数据集的根因**：Tiny-ImageNet 64×64 信息量太少，主干一次提取即覆盖大部分有用特征，专家没有互补学习空间。NWPU-RESISC45 特征：256×256 遥感图像（裁剪到 224×224）、45 类、高类内多样性，更贴近无人机部署目标。

---

## 二、目录结构

```
cla_5_2/
├── config.py                  # 全局配置（数据集、模型、训练超参）
├── train.py                   # 主训练脚本（Stage 2 + Stage 3）
├── pretrain_teacher.py        # Teacher 预训练脚本（新建于 v5.2）
├── eval_experts.py            # 评估：逐步启用专家，观察增益曲线
├── eval_alpha_sweep.py        # 评估：扫描不同 alpha 值下的系统精度
├── cache_clear.py             # 清理 __pycache__ 工具
├── script.sh                  # 一键启动训练脚本
├── PROJECT.md                 # 项目文档（本文件）
│
├── data/
│   └── dataset.py             # 数据加载（RESISC45）
│
├── models/
│   ├── backbone.py            # Student (MobileNetV3) + Teacher (ResNet-50)
│   ├── compressor.py          # 特征压缩器（支持无压缩 Phase A）
│   ├── experts.py             # 同构自学习专家网络
│   ├── router.py              # 动态 K 专家路由器
│   └── moe_system.py          # MoE 系统总装（整合所有模块）
│
├── distill/
│   └── losses.py              # 蒸馏损失函数（CE + KD + balance + ortho）
│
├── search/
│   ├── search_space.py        # 搜索空间定义（Phase1/2 变量范围）
│   └── search.py              # 三阶段架构搜索主脚本
│
├── checkpoints_resisc/        # v5.2 RESISC45 实验（alpha_end=0.5）
│   ├── teacher_best.pth       # RESISC45 Teacher checkpoint
│   ├── stage2_best.pt         # Stage 2 最优 checkpoint
│   └── stage3_best.pt         # Stage 3 最优 checkpoint（训练完成后生成）
│
└── checkpoints_resisc_a03/    # v5.2 RESISC45 实验（alpha_end=0.3）
    ├── stage2_best.pt
    └── stage3_best.pt
```

---

## 三、各文件详细说明

### 3.1 config.py

全局配置中心，所有超参均从此读取。

| dataclass | 作用 |
|-----------|------|
| `DataConfig` | 数据集路径、类别数、图像尺寸、split 比例 |
| `BackboneConfig` | 主干架构、截断点（stage1/stage2）、head 类型 |
| `CompressorConfig` | 压缩率设置，`spatial_size=None` 为 Phase A 无压缩 |
| `RouterConfig` | 路由器输入维度、专家数、动态 K 参数 |
| `ExpertConfig` | 专家输入维度、类别数、隐层、层数 |
| `AlphaScheduleConfig` | alpha 退火：start→end，决定主干与专家的权重比 |
| `DistillConfig` | 蒸馏温度、各损失权重 |
| `TrainConfig` | batch size、epoch、lr、save_dir 等训练参数 |
| `ExpertDropoutConfig` | Expert Dropout 比例 + 正交损失权重 |
| `DeployConfig` | 分布式部署时各模块的 GPU 分配 |

**当前 v5.2 关键值**：

```python
dataset    = "resisc45"
data_dir   = "/home/wh/code14/datasets/nwpuresisc45/train/train"
num_classes = 45
image_size  = 224
spatial_size = None   # Phase A：无压缩
save_dir   = "checkpoints_resisc"
```

---

### 3.2 数据层

#### `data/dataset.py`

支持两个数据集：

**RESISC45（v5.2 当前）**

```python
build_resisc45_dataloaders(data_dir, image_size=224, batch_size=64,
                            num_workers=16, train_ratio=0.8, use_strong_aug=False)
```
- 目录结构：`root/class_name/class_name_NNN.jpg`（45类×600张）
- 按类别做 stratified 80/20 split，固定随机种子=42
- 训练增强：RandomResizedCrop + H/V Flip + ColorJitter [+ AutoAugment]
- 验证增强：Resize(256) + CenterCrop(224)

**Tiny-ImageNet（v5.1.x 历史，保留向后兼容）**

```python
build_dataloaders(train_dir, val_dir, image_size=64, ...)
```

---

### 3.3 模型层

#### `models/backbone.py`

**BackboneMobileNetV3（Student，设备端）**

| 参数 | 选项 | 说明 |
|------|------|------|
| `cut_point` | `"stage1"` / `"stage2"` | 截断位置 |
| `head_type` | `"linear"` / `"mlp_small"` / `"mlp_medium"` / `"full_network"` | standalone head 强度 |

v5.2 维度（224×224，stem stride=2）：

```
features[0:4] → (B, 24, 28, 28)   cut_point="stage1"（默认）
features[0:7] → (B, 40, 14, 14)   cut_point="stage2"
```

主要方法：
- `forward_features(x)` → 截断特征，喂给压缩器/专家
- `forward_standalone(x)` → 主干独立推理，精度由 head_type 决定

**BackboneResNet50Teacher（Teacher，服务器端）**

- `image_size >= 128`：标准 ResNet-50 stem（7×7 stride=2），适合 224×224
- `image_size < 128`：3×3 stride=1 + Identity maxpool，适合 64×64
- `forward_features(x)` → layer3 输出 `(B, 1024, 14, 14)`，用于特征对齐蒸馏

---

#### `models/compressor.py`

**FeatureCompressor**

| `spatial_size` | 模式 | 压缩率 | 适用阶段 |
|----------------|------|--------|---------|
| `None` | 无压缩（Identity pool）| 1× | **Phase A（当前）** |
| `14` | 28→14 | 4× | Phase B |
| `7` | 28→7 | 16× | Phase B |

通道不同时自动加 1×1 Conv；相同时直接透传（Phase A 完全无参数）。

**FeatureDecompressor**：通道扩展 + 双线性上采样（Phase A 不使用）。

---

#### `models/experts.py`

**SelfLearnExpert（同构自学习专家）**

```
输入 (B, C, H, W)
  → 1×1 proj (C→hidden)
  → N × ExpertBlock（3×3 Conv + BN + ReLU，残差）
  → SE 注意力（通道自校准）
  → GAP → 全连接分类头
输出：(logits, feat)
```

v5.2 Phase A 默认：`in_channels=24`（直接接收截断特征），`num_classes=45`

**build_experts()**：批量构建 ModuleList，打印参数量。

---

#### `models/router.py`

**DynamicKRouter**

- 输入：压缩后特征 → GAP → MLP → gate logits
- 动态 K：根据 softmax 分布与阈值，自动决定激活几个专家（1~4）
- 输出：`gate_weights`（权重分布）、`active_k`（激活数）、`balance_loss`（负载均衡损失）

---

#### `models/moe_system.py`

**MoESystemTiny（总装）**，整合所有模块。

两种构建接口：

```python
# v5.2 推荐：config dict（自动处理 Phase A/B 维度）
model = MoESystemTiny(config={
    "head_type": "full_network",
    "cut_point": "stage1",
    "num_classes": 45,
    "spatial_size": None,      # Phase A
    "num_experts": 4,
    ...
})

# 旧接口（向后兼容）
model = MoESystemTiny(num_classes=200, compress_in=24, ...)
```

**Phase A 逻辑**（`spatial_size=None`）：
- `use_decompressor = False`
- 专家直接接收压缩器输出（= 主干截断特征）
- 跳过 FeatureDecompressor，节省参数和计算

**前向流程**：

```
x → backbone.forward_features → compressor
                              ↘ router → gate_weights
  [Phase A: 直接] / [Phase B: decompressor] → experts
→ weighted sum → α × backbone_logits + (1-α) × expert_fused
```

**关键方法**：
- `set_alpha(v)` / `get_alpha()` — 控制 alpha 退火
- `freeze_backbone()` — Stage 3 冻结主干
- `param_groups(lr, lr_bb)` — 为优化器提供分组 lr
- `print_params()` — 打印各模块参数量

---

### 3.4 蒸馏损失

#### `distill/losses.py`

| 函数/类 | 作用 |
|---------|------|
| `BackboneDistillLoss` | Stage 2：CE(student, label) + MSE(student_feat, teacher_feat) |
| `MoEDistillLoss` | Stage 3：CE + KD(KL散度) + balance_loss + ortho_loss，支持 Mixup |
| `orthogonal_loss` | 专家特征正交性惩罚，防止专家同质化 |
| `build_teacher(checkpoint, num_classes, image_size)` | 加载 Teacher ResNet-50 checkpoint |

---

### 3.5 训练脚本

#### `pretrain_teacher.py`（v5.2 新建）

在 RESISC45 上 fine-tune ResNet-50 作为 Teacher。

两阶段训练：
1. **Phase 1（前 5 epoch）**：冻结 stem/layer1~3，只训 layer4 + FC，lr=1e-3
2. **Phase 2（剩余 epoch）**：解冻全部，全局 fine-tune，lr=5e-4

```bash
python pretrain_teacher.py --gpus 1 --epochs 50 --save_dir checkpoints_resisc
```

---

#### `train.py`

Stage 2（主干蒸馏）+ Stage 3（完整 MoE 蒸馏）。

**Stage 2**：只训主干和压缩器，冻结专家和路由器，Teacher 监督特征对齐。
**Stage 3**：冻结主干，训专家+路由器+解压器，alpha 从 0.7 退火到 alpha_end。

关键命令行参数（v5.2 新增）：

| 参数 | 说明 |
|------|------|
| `--save_dir` | 覆盖 checkpoint 输出目录 |
| `--teacher_ckpt` | 覆盖 teacher checkpoint 路径（多实验共享 teacher）|
| `--alpha_end` | 覆盖 alpha 退火终点（不同实验并行时使用）|
| `--gpus` | 指定使用的 GPU 编号 |

```bash
# 全流程（Stage 2 → Stage 3 → 自动评估）
python train.py --stage all --gpus 1 --save_dir checkpoints_resisc \
                --teacher_ckpt checkpoints_resisc/teacher_best.pth --alpha_end 0.5
```

---

#### `eval_experts.py`

逐步启用 1→2→3→4 个专家，观察精度随专家数的变化曲线，验证专家协作增益。

```bash
python eval_experts.py --ckpt checkpoints_resisc/stage3_best.pt --gpus 1
```

---

#### `eval_alpha_sweep.py`

扫描 alpha ∈ [0.0, 1.0]，画出"主干权重 vs 系统精度"曲线，找到最优推理 alpha。

```bash
python eval_alpha_sweep.py --ckpt checkpoints_resisc/stage3_best.pt --gpus 1
```

---

### 3.6 搜索脚本（v5.2-search 遗留）

#### `search/search_space.py`

定义搜索空间：

```python
PHASE1_SPACE = {
    "head_type":  ["linear", "mlp_small", "mlp_medium", "full_network"],
    "alpha_end":  [0.3, 0.4, 0.5, 0.6],
    "cut_point":  ["stage1", "stage2"],
}   # 32 个组合

PHASE2_SPACE = {
    "expert_hidden_dim":   [64, 128],
    "num_experts":         [2, 4],
    "expert_num_layers":   [1, 2],
    "balance_loss_weight": [0.05, 0.1],
}   # 16 个组合（在 Phase1 top-K 上扩展）
```

---

#### `search/search.py`

三阶段搜索（当前仅用于 Phase B 压缩率搜索，Phase A 验证优先）：

```
Stage2 共享训练（按 cut_point 各跑一次）
  ↓
Phase 1：32 配置 × 10 epoch → 选 top-3
  ↓
Phase 2：48 配置 × 10 epoch → 选 top-3
  ↓
Phase 3：3 配置 × 100 epoch → 完整训练 + 自动 eval
```

**v5.2 bug 修复**：加载 Stage 2 权重时过滤 `standalone_head` 键（不同 head_type 维度不兼容会导致加载失败）。

```bash
python search/search.py --phase all --gpus 1
```

---

#### `run_exps.py` / `run_grid.py`（历史遗留，Tiny-ImageNet 时代）

- `run_exps.py`：手选 5 个补充配置（alpha=0.7/0.8，stage2 截断），复用 search.py 的训练逻辑
- `run_grid.py`：24 个配置分 6 组并行跑满 GPU，用于 Tiny-ImageNet 的密集搜索

**⚠️ 这两个脚本目前仍依赖旧版 `build_dataloaders`（Tiny-ImageNet 接口），在 RESISC45 配置下不可直接使用。**

---

### 3.7 工具脚本

#### `script.sh`

一键启动训练流程，当前配置：

```bash
# Step 1: Teacher 预训练（GPU 1，50 epoch）
pretrain_teacher.py --gpus 1 --save_dir checkpoints_resisc

# Step 2: 两实验并行（teacher 完成后自动触发）
# Exp-A (GPU 1): alpha_end=0.5 → checkpoints_resisc/
# Exp-B (GPU 0): alpha_end=0.3 → checkpoints_resisc_a03/
```

```bash
bash script.sh           # 后台运行全流程
tail -f checkpoints_resisc/teacher.log
tail -f checkpoints_resisc/full_a05.log
tail -f checkpoints_resisc_a03/full_a03.log
```

#### `cache_clear.py`

递归删除所有 `__pycache__` 目录。

```bash
python cache_clear.py [目录]
```

---

## 四、本次 v5.2 改动详情

### 4.1 改动文件总览

| 文件 | 改动类型 | 核心内容 |
|------|---------|---------|
| `config.py` | 修改 | 切换 RESISC45（45类、224×224），spatial_size=None |
| `models/backbone.py` | 修改 | 去掉 stride=1 stem 改动；CUT_CONFIG 维度更新为 28/14；Teacher 新增 image_size 参数 |
| `models/compressor.py` | 修改 | spatial_size=None → Identity pool（无压缩透传） |
| `models/experts.py` | 修改 | 默认 in_channels=24、num_classes=45 |
| `models/moe_system.py` | 修改 | Phase A 无解压器支持；param_groups/print_params 适配 |
| `data/dataset.py` | 修改 | 新增 RESISC45Dataset + build_resisc45_dataloaders |
| `distill/losses.py` | 修改 | build_teacher 增加 image_size 参数 |
| `train.py` | 修改 | 切换 RESISC45；config dict 构建模型；新增 --save_dir/--alpha_end/--teacher_ckpt 参数 |
| `search/search.py` | 修改 | 修复 standalone_head 权重加载 bug；更新数据加载 |
| `pretrain_teacher.py` | **新建** | Teacher 两阶段 fine-tune 脚本 |
| `script.sh` | 修改 | 更新为 RESISC45 双 GPU 并行实验布局 |

---

### 4.2 核心改动详解

#### A. 换数据集（Tiny-ImageNet → NWPU-RESISC45）

**为什么换**：七轮实验证明，当 standalone ≥ 40% 时专家增益为零。64×64 图像信息量太少，主干一次提取即覆盖所有有用信息，专家没有互补空间。

NWPU-RESISC45 优势：
- 256×256 遥感图像（使用 224×224），信息量是 64×64 的 5.3 倍
- 45 类（standalone 不会太强，约 70~80%）
- 高类内多样性（需要多专家协同）
- 直接对应无人机部署目标

#### B. Phase A 无压缩验证

策略转变：

```
旧做法：高压缩率 → 专家从压缩特征学 → 失败
新做法：Phase A（无压缩）→ 先验证专家有增益 → Phase B 再逐步加压缩
```

实现：`spatial_size=None` → compressor.pool = Identity → moe_system 跳过 decompressor → 专家直接接收 `(B, 24, 28, 28)` 原始截断特征。

#### C. backbone 维度修正

| | v5.1.1（Tiny-ImageNet 64×64） | v5.2（RESISC45 224×224）|
|--|------|------|
| stem stride | 1（改过） | 2（标准，不需改）|
| stage1 截断 | (24, 16, 16) | (24, 28, 28) |
| stage2 截断 | (40,  8,  8) | (40, 14, 14) |

#### D. Teacher 适配

- 224×224 使用标准 ResNet-50 stem（7×7 stride=2），不再改 stem
- `BackboneResNet50Teacher` 新增 `image_size` 参数，自动选择 stem 模式
- `build_teacher()` 同步更新，向下兼容旧接口

#### E. search.py Bug 修复

**问题**：Stage 2 用 `full_network` head 训练保存 checkpoint，Stage 3 加载时换了 `mlp_small` head，`standalone_head` 权重维度不匹配，`strict=False` 会报 warning 但不会出错，实际加载了错误/部分权重。

**修复**：加载前显式过滤 `standalone_head` 相关 key：

```python
filtered = {k: v for k, v in state_dict.items()
            if "standalone_head" not in k}
model.load_state_dict(filtered, strict=False)
```

---

## 五、训练流程

### 完整流程

```bash
# 0. 激活环境
conda activate jkw
cd /home/wh/code14/cla_5_2

# 1. 一键启动（推荐）
bash script.sh

# 或分步运行：
# 1. Teacher 预训练
python pretrain_teacher.py --gpus 1 --epochs 50 --save_dir checkpoints_resisc

# 2. Stage 2（主干蒸馏，20 epoch）
python train.py --stage 2 --gpus 1 \
    --save_dir checkpoints_resisc \
    --teacher_ckpt checkpoints_resisc/teacher_best.pth

# 3. Stage 3（MoE 蒸馏，100 epoch）
python train.py --stage 3 --gpus 1 \
    --save_dir checkpoints_resisc \
    --teacher_ckpt checkpoints_resisc/teacher_best.pth \
    --alpha_end 0.5

# 4. 评估
python eval_experts.py --ckpt checkpoints_resisc/stage3_best.pt --gpus 1
python eval_alpha_sweep.py --ckpt checkpoints_resisc/stage3_best.pt --gpus 1
```

### 当前运行中的实验（2026-03-27）

| 实验 | GPU | alpha_end | checkpoint 目录 | 日志 |
|------|-----|-----------|----------------|------|
| Teacher | GPU 1 | — | `checkpoints_resisc/` | `teacher.log` |
| Exp-A（teacher 后自动启动）| GPU 1 | 0.5 | `checkpoints_resisc/` | `full_a05.log` |
| Exp-B（teacher 后自动启动）| GPU 0 | 0.3 | `checkpoints_resisc_a03/` | `full_a03.log` |

Teacher 约 24 秒/epoch，50 epoch ≈ 20 分钟。

---

## 六、实验计划

### Phase A：无压缩验证（当前）

**目标**：确认在 RESISC45 上专家能否超越 standalone。

- 主干截断：stage1 → (24, 28, 28)
- 压缩：无（Identity）
- 专家：4 个，in_channels=24，hidden_dim=96
- alpha：0.7 → 0.5（Exp-A）/ 0.7 → 0.3（Exp-B）

**判断标准**：
- 专家有增益（哪怕 +1%）→ 进入 Phase B，逐步加压缩
- 专家无增益 → 问题不在压缩，审视专家架构（→ v5.3 异构专家）

### Phase B：压缩率梯度实验（待）

| 实验 | 空间压缩 | 通道压缩 | 专家输入 |
|------|---------|---------|---------|
| B1 | 28→14 | 无 | (24, 14, 14) |
| B2 | 28→7 | 无 | (24, 7, 7) |
| B3 | 28→7 | 24→12 | (12, 7, 7) |
| B4 | 28→4 | 24→12 | (12, 4, 4) |

---

## 七、预期效果

| 指标 | Tiny-ImageNet (v5.1.1) | RESISC45 (v5.2 预期) |
|------|----------------------|---------------------|
| 图像尺寸 | 64×64 | 224×224 |
| 类别数 | 200 | 45 |
| standalone 精度 | 61.63% | 70~80% |
| 截断特征维度 | 6,144 | 18,816 |
| 专家提升（无压缩）| ~0% | 预期 +2~5% |
| 专家提升（4× 压缩）| — | 预期 +1~3% |
