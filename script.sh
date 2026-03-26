# nohup python -u pretrain_teacher.py --gpus 1 > checkpoints_tiny/teacher.log 2>&1 && \
# python -u train.py --stage 2 --gpus 1 > checkpoints_tiny/stage2.log 2>&1 && \
# python -u train.py --stage 3 --gpus 1 > checkpoints_tiny/stage3.log 2>&1 &

# nohup python -u train.py --stage 2 --gpus 1 > checkpoints_tiny/stage2.log 2>&1 && \
# python -u train.py --stage 3 --gpus 1 > checkpoints_tiny/stage3.log 2>&1 &

# nohup python -u train.py --stage 3 --gpus 1 > checkpoints_tiny/stage3.log 2>&1 &


# step：Loss / Acc@1 / Acc@5 / LR / grad_norm / imgs/s / GPU-Util / 显存 / ETA
# epoch：Train Loss / Train Acc@1 / Train Acc@5 / Val Acc@1 / Val Acc@5 / Best Acc@1 / 本轮耗时 / 预计剩余总时间



# Eval

# nohup python -u eval_experts.py --ckpt checkpoints_tiny/stage3_best.pth --gpus 1 > checkpoints_tiny/eval_experts.log 2>&1 &

# nohup python -u eval_alpha_sweep.py --ckpt checkpoints_tiny/stage3_best.pth --gpus 1 > checkpoints_tiny/eval_alpha_sweep.log 2>&1 &



mkdir -p checkpoints_tiny

# Teacher 不需要重新训练，沿用已有的 teacher_best.pth
# nohup python -u pretrain_teacher.py --gpus 1 > checkpoints_tiny/teacher.log 2>&1 &

# Stage 2（12 epochs）
# Stage 3（60 epochs，冻结主干 + alpha 退火 + Mixup）
# 评估
# nohup python -u train.py --stage 2 --gpus 1 > checkpoints_tiny/stage2.log 2>&1 && \
# python -u train.py --stage 3 --gpus 1 > checkpoints_tiny/stage3.log 2>&1 && \
# python -u eval_experts.py --ckpt checkpoints_tiny/stage3_best.pt --gpus 1 > checkpoints_tiny/eval_experts.log 2>&1 && \
# python -u eval_alpha_sweep.py --ckpt checkpoints_tiny/stage3_best.pt --gpus 1 > checkpoints_tiny/eval_alpha_sweep.log 2>&1 &

nohup python -u train.py --stage all --auto_eval --gpus 1 > checkpoints_tiny/full.log 2>&1 &
