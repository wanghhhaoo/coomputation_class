"""
search_space.py — 架构搜索空间定义 (v5.2)

使用方法：
  from search_space import generate_phase1_configs, generate_phase2_configs, SearchConfig
"""

import itertools
from dataclasses import dataclass, field


# ── Phase 1：核心变量 ──────────────────────────────────────

PHASE1_SPACE = {
    # standalone head 强度：决定 standalone 精度上限
    # linear      → 基于截断特征直接分类，最弱
    # mlp_small   → 加一层隐层，中等
    # mlp_medium  → 加 Dropout，稍强
    # full_network → 走完整网络，最强（48~52%）
    "head_type": ["linear", "mlp_small", "mlp_medium", "full_network"],

    # alpha_end：Stage 3 退火终点，决定专家承担多少权重
    "alpha_end": [0.3, 0.4, 0.5, 0.6],

    # 截断点：决定喂给专家的特征信息量
    # stage1 → features[0:4] → (24, 16, 16)  较丰富
    # stage2 → features[0:7] → (40,  8,  8)  语义更强但空间更小
    "cut_point": ["stage1", "stage2"],
}

# ── Phase 2：次要变量 ──────────────────────────────────────

PHASE2_SPACE = {
    "expert_hidden_dim":   [64, 128],
    "num_experts":         [2, 4],
    "expert_num_layers":   [1, 2],
    "balance_loss_weight": [0.05, 0.1],
}

# Phase 1 搜索时固定的 Phase 2 默认值
PHASE2_DEFAULTS = {
    "expert_hidden_dim":   96,
    "num_experts":         4,
    "expert_num_layers":   2,
    "balance_loss_weight": 0.1,
}

# ── 搜索流程配置 ───────────────────────────────────────────

@dataclass
class SearchConfig:
    phase1_epochs:  int = 10    # Phase 1/2 短训练轮数
    phase2_epochs:  int = 10
    phase3_epochs:  int = 100   # Phase 3 完整训练
    phase1_top_k:   int = 3     # Phase 1 选出几个进 Phase 2
    phase3_top_k:   int = 3     # 最终选几个做完整训练
    save_dir:       str = "./search_results"
    alpha_start:    float = 0.7  # Stage 3 退火起点（固定）


# ── 配置生成器 ─────────────────────────────────────────────

def generate_phase1_configs() -> list:
    """
    生成 Phase 1 的所有配置（4×4×2 = 32 个）
    每个配置包含 Phase 1 变量 + Phase 2 默认值
    """
    keys   = list(PHASE1_SPACE.keys())
    values = list(PHASE1_SPACE.values())
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg.update(PHASE2_DEFAULTS)    # 填入 Phase 2 默认值
        configs.append(cfg)
    return configs


def generate_phase2_configs(base_config: dict) -> list:
    """
    在给定的 Phase 1 配置基础上，生成 Phase 2 的所有组合（2×2×2×2 = 16 个）
    """
    keys   = list(PHASE2_SPACE.keys())
    values = list(PHASE2_SPACE.values())
    configs = []
    for combo in itertools.product(*values):
        cfg = base_config.copy()
        cfg.update(dict(zip(keys, combo)))
        configs.append(cfg)
    return configs


def config_to_str(config: dict) -> str:
    """把配置字典转为简短的可读字符串，用于日志"""
    p1_keys = list(PHASE1_SPACE.keys())
    p2_keys = list(PHASE2_SPACE.keys())
    parts = []
    for k in p1_keys:
        if k in config:
            parts.append(f"{k}={config[k]}")
    for k in p2_keys:
        if k in config:
            parts.append(f"{k}={config[k]}")
    return "  ".join(parts)
