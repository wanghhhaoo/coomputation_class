#!/usr/bin/env bash
# =============================================================
# v5.2 NWPU-RESISC45 训练脚本
# =============================================================
# 数据集: /home/wh/code14/datasets/nwpuresisc45/train/train/
# GPU 0: ~27 GB 空闲   GPU 1: ~70 GB 空闲
#
# 实验布局:
#   Step 1 — Teacher 预训练（GPU 1，50 epoch，共享 ckpt）
#   Step 2 — 两个实验同时跑（Teacher 完成后）:
#     Exp-A (GPU 1): alpha_end=0.5  → checkpoints_resisc/
#     Exp-B (GPU 0): alpha_end=0.3  → checkpoints_resisc_a03/
#
# 运行方式:
#   bash script.sh          # 一键启动全流程（后台运行）
#   tail -f checkpoints_resisc/teacher.log
#   tail -f checkpoints_resisc/full_a05.log
#   tail -f checkpoints_resisc_a03/full_a03.log
# =============================================================

set -e
cd "$(dirname "$0")"

CONDA_ENV=jkw
PYTHON="conda run -n $CONDA_ENV python -u"

# ── 目录 ──────────────────────────────────────────────────
mkdir -p checkpoints_resisc
mkdir -p checkpoints_resisc_a03

# ── Step 1: Teacher 预训练 (GPU 1, 50 epoch) ─────────────
echo "[$(date '+%H:%M:%S')] 启动 Teacher 预训练 (GPU 1)..."
$PYTHON pretrain_teacher.py \
    --gpus 1 \
    --epochs 50 \
    --batch_size 128 \
    --save_dir checkpoints_resisc \
    > checkpoints_resisc/teacher.log 2>&1

echo "[$(date '+%H:%M:%S')] Teacher 预训练完成！"

# ── Step 2: 两个实验并行 (teacher 共享) ────────────────────

# Exp-A: alpha_end=0.5，GPU 1
echo "[$(date '+%H:%M:%S')] 启动 Exp-A (GPU 1, alpha_end=0.5)..."
$PYTHON train.py \
    --stage all \
    --auto_eval \
    --gpus 1 \
    --save_dir checkpoints_resisc \
    --teacher_ckpt checkpoints_resisc/teacher_best.pth \
    --alpha_end 0.5 \
    > checkpoints_resisc/full_a05.log 2>&1 &
PID_A=$!
echo "  Exp-A PID: $PID_A"

# Exp-B: alpha_end=0.3，GPU 0
echo "[$(date '+%H:%M:%S')] 启动 Exp-B (GPU 0, alpha_end=0.3)..."
$PYTHON train.py \
    --stage all \
    --auto_eval \
    --gpus 0 \
    --save_dir checkpoints_resisc_a03 \
    --teacher_ckpt checkpoints_resisc/teacher_best.pth \
    --alpha_end 0.3 \
    > checkpoints_resisc_a03/full_a03.log 2>&1 &
PID_B=$!
echo "  Exp-B PID: $PID_B"

echo ""
echo "  等待两个实验完成..."
wait $PID_A && echo "[$(date '+%H:%M:%S')] Exp-A 完成！"
wait $PID_B && echo "[$(date '+%H:%M:%S')] Exp-B 完成！"

echo ""
echo "========================================================"
echo "  全部完成！结果目录:"
echo "    checkpoints_resisc/        (alpha_end=0.5)"
echo "    checkpoints_resisc_a03/    (alpha_end=0.3)"
echo "========================================================"


# ── 单独命令（按需取消注释）──────────────────────────────

# Teacher 单独重跑
# nohup conda run -n jkw python -u pretrain_teacher.py --gpus 1 --epochs 50 \
#   > checkpoints_resisc/teacher.log 2>&1 &

# Stage 2 只跑
# nohup conda run -n jkw python -u train.py --stage 2 --gpus 1 \
#   --save_dir checkpoints_resisc --teacher_ckpt checkpoints_resisc/teacher_best.pth \
#   > checkpoints_resisc/stage2.log 2>&1 &

# Stage 3 只跑
# nohup conda run -n jkw python -u train.py --stage 3 --gpus 1 \
#   --save_dir checkpoints_resisc --teacher_ckpt checkpoints_resisc/teacher_best.pth \
#   > checkpoints_resisc/stage3.log 2>&1 &

# Eval
# conda run -n jkw python -u eval_experts.py \
#   --ckpt checkpoints_resisc/stage3_best.pt --gpus 1 \
#   > checkpoints_resisc/eval_experts.log 2>&1 &
# conda run -n jkw python -u eval_alpha_sweep.py \
#   --ckpt checkpoints_resisc/stage3_best.pt --gpus 1 \
#   > checkpoints_resisc/eval_alpha_sweep.log 2>&1 &
