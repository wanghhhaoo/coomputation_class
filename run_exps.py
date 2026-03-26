#!/usr/bin/env python3
"""
run_exps.py — 在指定 GPU 上运行手选配置的完整实验 (100 epoch)

实验目的：
  search.py 的搜索空间 alpha_end ∈ [0.3,0.6]，短训练 10 epoch 筛选。
  本脚本补充 search.py 覆盖不到的区域：
    1. alpha_end=0.7/0.8（扩展上界，基于 v5.1.1 分析最优推理 alpha≈0.8）
    2. stage2 截断点 + full_network head（search.py Phase3 不一定覆盖）
    3. 弱 standalone (mlp_small) + 大专家容量（给专家更多学习空间）

运行：
  python run_exps.py --gpus 1
  nohup python -u run_exps.py --gpus 1 > search_results/extra.log 2>&1 &
"""

import os, sys, time, json, argparse, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'search'))   # search_space.py, search.py
sys.path.insert(0, ROOT)                            # config, models, data

from config import data_cfg, train_cfg
from data.dataset import build_dataloaders
from distill.losses import build_teacher
import search as S   # 复用 search.py 中的 run_single_config 等函数


# ── 手选实验配置 ────────────────────────────────────────────
# search.py 的 Phase1 空间：alpha_end in [0.3,0.4,0.5,0.6]，10 epoch 筛选
# 本脚本补充：alpha_end=0.7/0.8，以及其他值得全程训练的配置

EXTRA_CONFIGS = [
    # ── 扩展 alpha_end 上界 ─────────────────────────────────
    {
        "name": "s1_full_a07",
        "desc": "stage1+full_network+alpha=0.7（延伸搜索空间上界）",
        "head_type": "full_network", "cut_point": "stage1",
        "alpha_end": 0.7,
        "num_experts": 4, "expert_hidden_dim": 96,
        "expert_num_layers": 2, "balance_loss_weight": 0.1,
    },
    {
        "name": "s1_full_a08",
        "desc": "stage1+full_network+alpha=0.8（极高主干权重，专家兜底）",
        "head_type": "full_network", "cut_point": "stage1",
        "alpha_end": 0.8,
        "num_experts": 4, "expert_hidden_dim": 96,
        "expert_num_layers": 2, "balance_loss_weight": 0.1,
    },
    # ── stage2 截断（更深特征）──────────────────────────────
    {
        "name": "s2_full_a05",
        "desc": "stage2+full_network+alpha=0.5（更深特征，与v5.1.1同alpha）",
        "head_type": "full_network", "cut_point": "stage2",
        "alpha_end": 0.5,
        "num_experts": 4, "expert_hidden_dim": 96,
        "expert_num_layers": 2, "balance_loss_weight": 0.1,
    },
    {
        "name": "s2_full_a07",
        "desc": "stage2+full_network+alpha=0.7（深特征+高alpha）",
        "head_type": "full_network", "cut_point": "stage2",
        "alpha_end": 0.7,
        "num_experts": 4, "expert_hidden_dim": 96,
        "expert_num_layers": 2, "balance_loss_weight": 0.1,
    },
    # ── 弱 standalone + 大专家容量 ──────────────────────────
    {
        "name": "s1_mlp_a04_h128",
        "desc": "stage1+mlp_small+alpha=0.4+hidden=128（弱standalone给专家空间）",
        "head_type": "mlp_small", "cut_point": "stage1",
        "alpha_end": 0.4,
        "num_experts": 4, "expert_hidden_dim": 128,
        "expert_num_layers": 2, "balance_loss_weight": 0.05,
    },
]

FULL_EPOCHS = 100


# ── 等待 checkpoint ────────────────────────────────────────

def wait_for_ckpt(path: str, poll: int = 30):
    if os.path.exists(path):
        return
    print(f"  [等待] {path}", flush=True)
    while not os.path.exists(path):
        time.sleep(poll)
    print(f"  [就绪] {path}", flush=True)


def get_stage2_ckpt(cut_point: str) -> str:
    """
    优先用 search.py 生成的精确 ckpt（stage2_{cut_point}.pt）。
    若是 stage1 且 search.py 还没跑完，用 train.py 生成的 stage2_best.pt 也可以。
    """
    primary = os.path.join(train_cfg.save_dir, f"stage2_{cut_point}.pt")
    fallback = os.path.join(train_cfg.save_dir, "stage2_best.pt")

    if os.path.exists(primary):
        return primary
    if cut_point == "stage1" and os.path.exists(fallback):
        print(f"  [Stage2] 使用 train.py 的 stage2_best.pt 替代", flush=True)
        return fallback
    # 都没有，等 primary
    wait_for_ckpt(primary)
    return primary


# ── 主流程 ────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=str, default="1")
    args = p.parse_args()

    gpu_id = int(args.gpus.split(",")[0])
    device = torch.device(f"cuda:{gpu_id}")

    print(f"\n{'═'*60}")
    print(f"  run_exps.py — 手选额外实验  GPU: cuda:{gpu_id}")
    print(f"  配置数: {len(EXTRA_CONFIGS)}  每个 {FULL_EPOCHS} epoch")
    print(f"{'═'*60}\n")

    os.makedirs("search_results/extra", exist_ok=True)
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    # 共享数据加载器和 teacher（节省显存和初始化时间）
    train_loader, val_loader, _ = build_dataloaders(
        data_cfg.train_dir, data_cfg.val_dir,
        image_size=data_cfg.image_size,
        batch_size=train_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        use_strong_aug=train_cfg.stage3_use_autoaugment,
    )
    teacher = build_teacher(
        checkpoint=f"{train_cfg.save_dir}/teacher_best.pth"
    ).to(device)

    all_results = []

    for i, raw_cfg in enumerate(EXTRA_CONFIGS):
        name = raw_cfg.pop("name")
        desc = raw_cfg.pop("desc")
        cfg  = raw_cfg   # 剩余键值即为 MoESystemTiny config dict

        stage2_ckpt = get_stage2_ckpt(cfg["cut_point"])
        save_dir    = f"search_results/extra/{name}"

        print(f"\n{'═'*60}")
        print(f"  [{i+1}/{len(EXTRA_CONFIGS)}] {name}")
        print(f"  说明: {desc}")
        print(f"  Stage2: {stage2_ckpt}")
        print(f"  保存至: {save_dir}")
        print(f"{'═'*60}\n")

        result = S.run_single_config(
            config=cfg,
            epochs=FULL_EPOCHS,
            stage2_ckpt=stage2_ckpt,
            device=device,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
        )

        result["name"] = name
        result["desc"] = desc
        result["config"] = cfg
        all_results.append(result)

        print(f"\n  [{name}] 完成  "
              f"Val:{result['best_val_acc']:.2f}%  "
              f"SA:{result['standalone_acc']:.2f}%  "
              f"增益:{result['best_val_acc']-result['standalone_acc']:+.2f}%")

        with open(f"{save_dir}/result.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # 写汇总
    all_results.sort(key=lambda x: x["best_val_acc"], reverse=True)
    summary_path = "search_results/extra/summary.txt"
    lines = ["═"*60, "  run_exps.py — 额外实验结果汇总", "═"*60, ""]
    for r in all_results:
        lines += [
            f"  {r['name']}",
            f"    {r['desc']}",
            f"    Val Acc@1:  {r['best_val_acc']:.2f}%",
            f"    Standalone: {r['standalone_acc']:.2f}%",
            f"    专家增益:   {r['best_val_acc']-r['standalone_acc']:+.2f}%",
            "",
        ]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"  汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
