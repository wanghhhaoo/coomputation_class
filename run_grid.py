#!/usr/bin/env python3
"""
run_grid.py — GPU 网格实验，多进程并行跑满 GPU 0

24 个配置分成 6 组，每组 4 个配置顺序执行。
6 组同时跑 → 最大化 GPU 利用率。

用法（分别在 6 个 tmux 窗口中启动）：
  python run_grid.py --group 0 --gpus 0
  python run_grid.py --group 1 --gpus 0
  ...
  python run_grid.py --group 5 --gpus 0
"""

import os, sys, time, json, argparse, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'search'))
sys.path.insert(0, ROOT)

from config import data_cfg, train_cfg
from data.dataset import build_dataloaders
from distill.losses import build_teacher
import search as S


# ── 完整网格（24 个配置） ──────────────────────────────────
# 覆盖 search.py 以外的区域 + 重要变量全扫描
# 每行：name, head_type, cut_point, alpha_end, num_experts, expert_hidden_dim, expert_num_layers, balance_loss_weight

ALL_CONFIGS = [
    # ── Group 0: stage1 + full_network，alpha 全扫 ─────────
    ("s1_fn_a03",   "full_network", "stage1", 0.3, 4, 96, 2, 0.1),
    ("s1_fn_a04",   "full_network", "stage1", 0.4, 4, 96, 2, 0.1),
    ("s1_fn_a06",   "full_network", "stage1", 0.6, 4, 96, 2, 0.1),
    ("s1_fn_a07",   "full_network", "stage1", 0.7, 4, 96, 2, 0.1),

    # ── Group 1: stage2 + full_network，alpha 全扫 ─────────
    ("s2_fn_a03",   "full_network", "stage2", 0.3, 4, 96, 2, 0.1),
    ("s2_fn_a04",   "full_network", "stage2", 0.4, 4, 96, 2, 0.1),
    ("s2_fn_a06",   "full_network", "stage2", 0.6, 4, 96, 2, 0.1),
    ("s2_fn_a08",   "full_network", "stage2", 0.8, 4, 96, 2, 0.1),

    # ── Group 2: stage1 + mlp_small，alpha 全扫 ───────────
    ("s1_ms_a03",   "mlp_small",    "stage1", 0.3, 4, 96, 2, 0.1),
    ("s1_ms_a04",   "mlp_small",    "stage1", 0.4, 4, 96, 2, 0.1),
    ("s1_ms_a05",   "mlp_small",    "stage1", 0.5, 4, 96, 2, 0.1),
    ("s1_ms_a06",   "mlp_small",    "stage1", 0.6, 4, 96, 2, 0.1),

    # ── Group 3: stage2 + mlp_small，alpha 全扫 ───────────
    ("s2_ms_a03",   "mlp_small",    "stage2", 0.3, 4, 96, 2, 0.1),
    ("s2_ms_a04",   "mlp_small",    "stage2", 0.4, 4, 96, 2, 0.1),
    ("s2_ms_a05",   "mlp_small",    "stage2", 0.5, 4, 96, 2, 0.1),
    ("s2_ms_a06",   "mlp_small",    "stage2", 0.6, 4, 96, 2, 0.1),

    # ── Group 4: 大专家容量（hidden=128） ─────────────────
    ("s1_fn_a05_h128_ne4",  "full_network", "stage1", 0.5, 4, 128, 2, 0.1),
    ("s1_fn_a04_h128_ne4",  "full_network", "stage1", 0.4, 4, 128, 2, 0.05),
    ("s2_fn_a05_h128_ne4",  "full_network", "stage2", 0.5, 4, 128, 2, 0.1),
    ("s1_ms_a04_h128_ne4",  "mlp_small",    "stage1", 0.4, 4, 128, 2, 0.05),

    # ── Group 5: 小专家容量 + 不同专家数 ──────────────────
    ("s1_fn_a05_h64_ne2",   "full_network", "stage1", 0.5, 2, 64,  2, 0.1),
    ("s1_fn_a05_ne2",       "full_network", "stage1", 0.5, 2, 96,  2, 0.1),
    ("s1_fn_a05_h64_ne4",   "full_network", "stage1", 0.5, 4, 64,  1, 0.1),
    ("s1_fn_a04_ne2_bl005", "full_network", "stage1", 0.4, 2, 96,  2, 0.05),
]

KEYS = ["head_type", "cut_point", "alpha_end", "num_experts",
        "expert_hidden_dim", "expert_num_layers", "balance_loss_weight"]

FULL_EPOCHS = 100
NUM_GROUPS  = 6


def make_cfg(row):
    name = row[0]
    cfg  = dict(zip(KEYS, row[1:]))
    return name, cfg


def wait_for_ckpt(path, poll=30):
    if os.path.exists(path):
        return
    print(f"  [等待] {path}", flush=True)
    while not os.path.exists(path):
        time.sleep(poll)
    print(f"  [就绪] {path}", flush=True)


def get_stage2_ckpt(cut_point):
    primary  = os.path.join(train_cfg.save_dir, f"stage2_{cut_point}.pt")
    fallback = os.path.join(train_cfg.save_dir, "stage2_best.pt")
    if os.path.exists(primary):
        return primary
    if cut_point == "stage1" and os.path.exists(fallback):
        return fallback
    wait_for_ckpt(primary)
    return primary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, required=True,
                   help=f"组编号 0~{NUM_GROUPS-1}")
    p.add_argument("--gpus",  type=str, default="0")
    args = p.parse_args()

    assert 0 <= args.group < NUM_GROUPS, f"group 需在 0~{NUM_GROUPS-1}"
    gpu_id = int(args.gpus.split(",")[0])
    device = torch.device(f"cuda:{gpu_id}")

    # 本组负责的配置（循环分配）
    my_rows = [r for i, r in enumerate(ALL_CONFIGS) if i % NUM_GROUPS == args.group]

    print(f"\n{'═'*60}")
    print(f"  run_grid.py  Group={args.group}/{NUM_GROUPS}  GPU=cuda:{gpu_id}")
    print(f"  本组 {len(my_rows)} 个配置，每个 {FULL_EPOCHS} epoch")
    for r in my_rows:
        print(f"    {r[0]}")
    print(f"{'═'*60}\n")

    os.makedirs("search_results/grid", exist_ok=True)
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(
        data_cfg.train_dir, data_cfg.val_dir,
        image_size=data_cfg.image_size,
        batch_size=train_cfg.batch_size,
        num_workers=max(4, data_cfg.num_workers // NUM_GROUPS),  # 共享 CPU workers
        use_strong_aug=train_cfg.stage3_use_autoaugment,
    )
    teacher = build_teacher(
        checkpoint=f"{train_cfg.save_dir}/teacher_best.pth"
    ).to(device)

    group_results = []

    for row in my_rows:
        name, cfg = make_cfg(row)
        stage2_ckpt = get_stage2_ckpt(cfg["cut_point"])
        save_dir    = f"search_results/grid/{name}"

        print(f"\n{'─'*60}")
        print(f"  [{args.group}] {name}  cut={cfg['cut_point']}  "
              f"alpha={cfg['alpha_end']}  ne={cfg['num_experts']}  "
              f"h={cfg['expert_hidden_dim']}")
        print(f"{'─'*60}\n")

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
        result.update({"name": name, "group": args.group, "config": cfg})
        group_results.append(result)

        print(f"\n  [{name}]  Val:{result['best_val_acc']:.2f}%  "
              f"SA:{result['standalone_acc']:.2f}%  "
              f"增益:{result['best_val_acc']-result['standalone_acc']:+.2f}%")

        with open(f"{save_dir}/result.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # 组内汇总
    group_results.sort(key=lambda x: x["best_val_acc"], reverse=True)
    summary = f"search_results/grid/group{args.group}_summary.txt"
    lines   = [f"Group {args.group} 结果汇总", "─"*40]
    for r in group_results:
        lines.append(f"  {r['name']:25s}  Val:{r['best_val_acc']:.2f}%  "
                     f"SA:{r['standalone_acc']:.2f}%  "
                     f"增益:{r['best_val_acc']-r['standalone_acc']:+.2f}%")
    with open(summary, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Group {args.group} 全部完成！汇总：{summary}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
