"""
train.py — NWPU-RESISC45 MoE 训练 (v5.2)

v5.2 核心变更：
  1. 切换数据集 Tiny-ImageNet → NWPU-RESISC45（224×224，45 类）
  2. 使用 config dict 接口构建 MoESystemTiny（自动处理 Phase A 无解压器）
  3. 更新 build_teacher 调用（num_classes=45，image_size=224）

运行：
  python pretrain_teacher.py --gpus 0 > checkpoints_resisc/teacher.log 2>&1 &&
  python train.py --stage 2 --gpus 0 > checkpoints_resisc/stage2.log 2>&1 &&
  python train.py --stage 3 --gpus 0 > checkpoints_resisc/stage3.log 2>&1
"""

import os, sys, time, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    data_cfg, backbone_cfg, distill_cfg, train_cfg,
    deploy_cfg, DEPLOY_MODE, expert_drop_cfg, expert_cfg,
    alpha_sched_cfg, router_cfg, compressor_cfg,
)


def _build_dataloaders(use_strong_aug=False):
    """根据 data_cfg.dataset 选择正确的数据加载器"""
    if data_cfg.dataset == "resisc45":
        return build_resisc45_dataloaders(
            data_cfg.data_dir,
            image_size=data_cfg.image_size,
            batch_size=train_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            train_ratio=data_cfg.train_ratio,
            use_strong_aug=use_strong_aug,
        )
    else:
        return build_dataloaders(
            data_cfg.train_dir, data_cfg.val_dir,
            image_size=data_cfg.image_size,
            batch_size=train_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            use_strong_aug=use_strong_aug,
        )


def _build_moe_model(deploy_mode="local"):
    """用 config dict 接口构建 MoESystemTiny（自动处理 Phase A/B）"""
    return MoESystemTiny(config={
        "head_type":          backbone_cfg.head_type,
        "cut_point":          backbone_cfg.cut_point,
        "num_classes":        data_cfg.num_classes,
        "num_experts":        router_cfg.num_experts,
        "expert_dropout":     expert_drop_cfg.expert_dropout,
        "expert_hidden_dim":  expert_cfg.hidden_dim,
        "expert_num_layers":  expert_cfg.num_layers,
        "router_hidden_dim":  router_cfg.hidden_dim,
        "alpha_start":        alpha_sched_cfg.alpha_start,
        "balance_loss_weight": router_cfg.balance_loss_weight,
        "dropout_rate":       expert_cfg.dropout_rate,
        "spatial_size":       compressor_cfg.spatial_size,
        "deploy_mode":        deploy_mode,
    })
from data.dataset      import build_dataloaders, build_resisc45_dataloaders
from models.moe_system import MoESystemTiny
from distill.losses    import BackboneDistillLoss, MoEDistillLoss, build_teacher


# ── 工具 ──────────────────────────────────────────────────

def unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m
def sep(c="─", w=72): print(c * w)
def eta_str(s): return str(datetime.timedelta(seconds=int(s)))

def get_amp_dtype():
    s = train_cfg.amp_dtype.lower()
    if s == "bf16": return torch.bfloat16
    if s == "fp16": return torch.float16
    return torch.float32

def get_gpu_stats(dev) -> str:
    if not torch.cuda.is_available(): return "N/A"
    alloc = torch.cuda.memory_allocated(dev) / 1024**3
    total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    try:
        import subprocess
        idx  = dev.index if hasattr(dev, 'index') else int(str(dev).split(':')[-1])
        util = subprocess.check_output(
            ["nvidia-smi", f"--id={idx}",
             "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"], timeout=2
        ).decode().strip()
        return f"Mem:{alloc:.1f}/{total:.0f}GB  GPU-Util:{util}%"
    except Exception:
        return f"Mem:{alloc:.1f}/{total:.0f}GB"

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.count=0
    def update(self, val, n=1):
        self.sum+=val*n; self.count+=n; self.avg=self.sum/self.count

def accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk=max(topk); B=target.size(0)
        _,pred=output.topk(maxk,1,True,True); pred=pred.t()
        correct=pred.eq(target.view(1,-1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).mul_(100./B) for k in topk]

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    print(f"  [Checkpoint] {path}")


# ── v2: 合并 Teacher 前向 ──────────────────────────────────

def extract_teacher_feat_and_logits(teacher, imgs):
    """一次前向同时取 logits 和 layer3 特征（v2: 避免双次前向）"""
    feats = {}
    h = unwrap(teacher).layer3.register_forward_hook(
        lambda m, i, o: feats.update({"f": o})
    )
    with torch.no_grad():
        t_logits = teacher(imgs)
    h.remove()
    return t_logits, feats["f"]


# ── v2: Mixup 数据增强 ─────────────────────────────────────

def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    mixed_x  = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


# ── 验证 ──────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, mode="moe", amp_dtype=None):
    unwrap(model).eval()
    a1m=AverageMeter(); a5m=AverageMeter()
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = (unwrap(model).forward_standalone(imgs)
                      if mode == "standalone"
                      else unwrap(model)(imgs)["logits"])
        a1, a5 = accuracy(logits.float(), labels)
        a1m.update(a1.item(), imgs.size(0))
        a5m.update(a5.item(), imgs.size(0))
    return a1m.avg, a5m.avg


# ── 打印 epoch 汇总 ───────────────────────────────────────

def print_epoch_summary(stage, epoch, total_epochs,
                        train_loss, train_a1, train_a5,
                        val_a1, val_a5, best_acc1,
                        epoch_time, avg_epoch_time, remaining,
                        extra_lines=None):
    is_best = val_a1 >= best_acc1
    sep()
    print(
        f"  Stage{stage} Epoch [{epoch:03d}/{total_epochs}] 完成\n"
        f"  Train │ Loss:{train_loss:.4f}  "
        f"Acc@1:{train_a1:.2f}%  Acc@5:{train_a5:.2f}%\n"
        f"  Val   │ Acc@1:{val_a1:.2f}%  Acc@5:{val_a5:.2f}%  "
        f"Best:{max(best_acc1, val_a1):.2f}%"
        + ("  ✓ 新最佳！" if is_best else "") + "\n"
        f"  耗时: {epoch_time:.1f}s  "
        f"均速: {avg_epoch_time:.1f}s/epoch  "
        f"剩余总ETA: {eta_str(remaining)}"
    )
    if extra_lines:
        for line in extra_lines:
            print(line)
    sep(); print()


# ── Stage 2: Backbone 蒸馏 ────────────────────────────────

def train_stage2(args, device):
    amp_dtype  = get_amp_dtype() if train_cfg.use_amp else None
    use_scaler = train_cfg.use_amp and train_cfg.amp_dtype.lower() == "fp16"
    scaler     = GradScaler(enabled=use_scaler)

    sep("═")
    print("  Stage 2: Backbone 蒸馏训练")
    print(f"  设备: {device}  BS: {train_cfg.batch_size}  "
          f"AMP: {train_cfg.amp_dtype if train_cfg.use_amp else 'off'}  "
          f"Epochs: {train_cfg.epochs_stage2}")
    sep("═"); print()

    train_loader, val_loader, _ = _build_dataloaders(use_strong_aug=False)
    total_steps = len(train_loader)

    teacher_ckpt = (args.teacher_ckpt or f"{train_cfg.save_dir}/teacher_best.pth")
    teacher = build_teacher(
        checkpoint=teacher_ckpt,
        num_classes=data_cfg.num_classes,
        image_size=data_cfg.image_size,
    ).to(device)

    model = _build_moe_model().to(device)
    for p in model.experts.parameters():  p.requires_grad = False
    for p in model.router.parameters():   p.requires_grad = False

    criterion = BackboneDistillLoss(distill_cfg.alpha_feat, distill_cfg.gamma_ce)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.lr_backbone, weight_decay=train_cfg.weight_decay
    )

    # v2: 修复 T_max，确保 cosine 不回弹
    effective_epochs = train_cfg.epochs_stage2 - train_cfg.warmup_epochs
    warmup_sched = LinearLR(
        optimizer, start_factor=0.1, total_iters=train_cfg.warmup_epochs
    )
    cosine_sched = CosineAnnealingLR(optimizer, T_max=effective_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[train_cfg.warmup_epochs]
    )

    best_acc1   = 0.
    epoch_times = []

    for epoch in range(1, train_cfg.epochs_stage2 + 1):
        lr  = optimizer.param_groups[0]["lr"]
        now = datetime.datetime.now().strftime("%H:%M:%S")
        sep("═")
        print(f"  Stage2 │ Epoch [{epoch:02d}/{train_cfg.epochs_stage2}] │ "
              f"LR: {lr:.3e} │ {now}")
        sep("═")

        model.train(); teacher.eval()
        lm   = {k: AverageMeter() for k in ["total","ce","feat"]}
        a1m  = AverageMeter(); a5m = AverageMeter()
        t0   = time.time(); imgs_seen = 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype is not None)):
                # v2: 单次前向，同时取 logits 和 feat
                with torch.no_grad():
                    t_logits, t_feat = extract_teacher_feat_and_logits(
                        teacher, imgs
                    )
                s_logits = model.forward_standalone(imgs)
                s_feat   = model.backbone.forward_features(imgs)
                ld       = criterion(s_logits, t_logits, s_feat, t_feat, labels)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(ld["total"]).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip).item()
                scaler.step(optimizer); scaler.update()
            else:
                ld["total"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip).item()
                optimizer.step()

            a1, a5     = accuracy(s_logits.float(), labels)
            imgs_seen += imgs.size(0)
            for k,v in ld.items(): lm[k].update(v.item(), imgs.size(0))
            a1m.update(a1.item(), imgs.size(0))
            a5m.update(a5.item(), imgs.size(0))

            if i % train_cfg.log_interval == 0:
                elapsed    = time.time() - t0
                imgs_per_s = imgs_seen / max(elapsed, 1e-6)
                eta_epoch  = elapsed / max(i, 1) * (total_steps - i)
                if epoch_times:
                    remaining = sum(epoch_times)/len(epoch_times) * \
                                (train_cfg.epochs_stage2 - epoch) + eta_epoch
                    total_eta = f"  总ETA:{eta_str(remaining)}"
                else:
                    total_eta = ""

                print(
                    f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] "
                    f"[{i:04d}/{total_steps}] "
                    f"│ Loss:{lm['total'].avg:.4f} "
                    f"(CE:{lm['ce'].avg:.4f} Feat:{lm['feat'].avg:.4f}) "
                    f"│ Acc@1:{a1m.avg:.2f}%  Acc@5:{a5m.avg:.2f}% "
                    f"│ LR:{lr:.3e}  GradNorm:{grad_norm:.3f} "
                    f"│ {imgs_per_s:.0f}imgs/s "
                    f"│ {get_gpu_stats(device)} "
                    f"│ EpochETA:{eta_str(eta_epoch)}"
                    f"{total_eta}"
                )

        scheduler.step()
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        avg_t      = sum(epoch_times) / len(epoch_times)
        remaining  = avg_t * (train_cfg.epochs_stage2 - epoch)

        # v4: standalone 变弱后明确打印精度，避免误解
        val_a1_sa, val_a5_sa = validate(model, val_loader, device, "standalone", amp_dtype)
        val_a1, val_a5 = val_a1_sa, val_a5_sa   # best checkpoint 仍用 standalone 精度选
        print(f"  [Stage2] Standalone: {val_a1_sa:.2f}%  (用于保底，v4预期 45~50%)")

        print_epoch_summary(
            stage=2, epoch=epoch, total_epochs=train_cfg.epochs_stage2,
            train_loss=lm["total"].avg, train_a1=a1m.avg, train_a5=a5m.avg,
            val_a1=val_a1, val_a5=val_a5, best_acc1=best_acc1,
            epoch_time=epoch_time, avg_epoch_time=avg_t, remaining=remaining,
        )

        if val_a1 > best_acc1:
            best_acc1 = val_a1
            save_ckpt({"epoch":epoch,"model":model.state_dict(),"acc1":val_a1},
                      f"{train_cfg.save_dir}/stage2_best.pt")
        if epoch % train_cfg.save_every == 0:
            save_ckpt({"epoch":epoch,"model":model.state_dict()},
                      f"{train_cfg.save_dir}/stage2_epoch{epoch}.pt")

    print(f"\n  Stage2 完成！最佳 Val Acc@1: {best_acc1:.2f}%\n")


# ── v2: Alpha 退火调度 ────────────────────────────────────

def get_alpha_for_epoch(epoch: int, total_epochs: int) -> float:
    """线性退火：从 alpha_start 降到 alpha_end"""
    cfg = alpha_sched_cfg
    if epoch <= cfg.warmup_epochs:
        progress = epoch / cfg.warmup_epochs
        return cfg.alpha_start - (cfg.alpha_start - cfg.alpha_end) * progress
    return cfg.alpha_end


# ── Stage 3: 完整 MoE 蒸馏 ───────────────────────────────

def train_stage3(args, device):
    amp_dtype  = get_amp_dtype() if train_cfg.use_amp else None
    use_scaler = train_cfg.use_amp and train_cfg.amp_dtype.lower() == "fp16"
    scaler     = GradScaler(enabled=use_scaler)

    use_mixup = train_cfg.stage3_use_mixup

    sep("═")
    print("  Stage 3: 完整 MoE 蒸馏训练 (v2)")
    print(f"  设备: {device}  BS: {train_cfg.batch_size}  "
          f"AMP: {train_cfg.amp_dtype if train_cfg.use_amp else 'off'}  "
          f"Epochs: {train_cfg.epochs_stage3}  动态K: True")
    print(f"  冻结主干: {alpha_sched_cfg.freeze_backbone}  "
          f"Alpha退火: {alpha_sched_cfg.alpha_start}→{alpha_sched_cfg.alpha_end} "
          f"({alpha_sched_cfg.warmup_epochs}ep)")
    print(f"  Mixup: {use_mixup}  AutoAugment: {train_cfg.stage3_use_autoaugment}  "
          f"Balance_w: {router_cfg.balance_loss_weight}  "
          f"Ortho_w: {expert_drop_cfg.ortho_w}")
    sep("═"); print()

    # Stage 3 使用强数据增强
    train_loader, val_loader, _ = _build_dataloaders(
        use_strong_aug=train_cfg.stage3_use_autoaugment
    )
    total_steps = len(train_loader)

    teacher_ckpt = (args.teacher_ckpt or f"{train_cfg.save_dir}/teacher_best.pth")
    teacher = build_teacher(
        checkpoint=teacher_ckpt,
        num_classes=data_cfg.num_classes,
        image_size=data_cfg.image_size,
    ).to(device)

    model = _build_moe_model(deploy_mode=args.mode).to(device)

    ckpt2 = f"{train_cfg.save_dir}/stage2_best.pt"
    if os.path.exists(ckpt2):
        ck = torch.load(ckpt2, map_location="cpu", weights_only=False)
        state_dict = ck["model"]
        model_state = model.state_dict()
        # 只过滤形状不匹配的 key（不同 head_type 导致的形状差异），保留 standalone_head
        filtered = {k: v for k, v in state_dict.items()
                    if k not in model_state or v.shape == model_state[k].shape}
        skipped  = [k for k in state_dict if k not in filtered]
        miss, unex = model.load_state_dict(filtered, strict=False)
        print(f"  [Stage3] 加载Stage2权重  加载:{len(filtered)}  "
              f"跳过(shape不匹配):{len(skipped)}  "
              f"缺失:{len(miss)}  多余:{len(unex)}\n")
    else:
        print("  [Stage3] 未找到Stage2权重，随机初始化\n")

    # v2: 冻结主干
    if alpha_sched_cfg.freeze_backbone:
        model.freeze_backbone()

    criterion = MoEDistillLoss(
        distill_cfg.gamma_ce, distill_cfg.beta_kd,
        router_cfg.balance_loss_weight,     # v2: 从 config 读取
        expert_drop_cfg.ortho_w,            # v2: 从 config 读取
        distill_cfg.temperature,
    )

    # v2: param_groups 不再包含 backbone_alpha
    optimizer = optim.AdamW(
        model.param_groups(train_cfg.lr, train_cfg.lr_backbone),
        weight_decay=train_cfg.weight_decay
    )
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=train_cfg.warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=train_cfg.epochs_stage3 - train_cfg.warmup_epochs
    )
    scheduler = SequentialLR(optimizer, [warmup, cosine], [train_cfg.warmup_epochs])

    best_acc1        = 0.
    epoch_times      = []
    no_improve_count = 0   # v5.1: 早停计数器

    for epoch in range(1, train_cfg.epochs_stage3 + 1):
        router_temp = max(0.3, 1.0 - epoch / train_cfg.epochs_stage3 * 0.7)

        # v2: alpha 退火调度
        current_alpha = get_alpha_for_epoch(epoch, train_cfg.epochs_stage3)
        model.set_alpha(current_alpha)

        threshold   = model.router.get_threshold()
        lr          = optimizer.param_groups[0]["lr"]
        now         = datetime.datetime.now().strftime("%H:%M:%S")

        sep("═")
        print(f"  Stage3 │ Epoch [{epoch:03d}/{train_cfg.epochs_stage3}] │ "
              f"LR:{lr:.3e} │ Temp:{router_temp:.3f} │ "
              f"Thr:{threshold:.3f} │ α:{current_alpha:.3f} │ {now}")
        sep("═")

        model.train(); teacher.eval()
        lm    = {k: AverageMeter() for k in ["total","ce","kd","balance","ortho"]}
        a1m   = AverageMeter(); a5m = AverageMeter()
        km    = AverageMeter()
        gate_accum = []
        t0    = time.time(); imgs_seen = 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # v2: Mixup 数据增强
            if use_mixup:
                imgs, y_a, y_b, lam = mixup_data(
                    imgs, labels,
                    alpha=train_cfg.stage3_mixup_alpha,
                    device=device
                )
            else:
                y_a, y_b, lam = labels, None, 1.0

            with autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype is not None)):
                with torch.no_grad():
                    t_logits = teacher(imgs)
                raw_out = model(imgs, return_extras=True)
                ld      = criterion(raw_out, t_logits, y_a, y_b=y_b, lam=lam)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(ld["total"]).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip).item()
                scaler.step(optimizer); scaler.update()
            else:
                ld["total"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip).item()
                optimizer.step()

            model.router.temperature = router_temp

            # Acc 用 y_a 计算（Mixup 时仅用于监控趋势）
            a1, a5     = accuracy(raw_out["logits"].float(), y_a)
            imgs_seen += imgs.size(0)
            for k,v in ld.items(): lm[k].update(v.item(), imgs.size(0))
            a1m.update(a1.item(), imgs.size(0))
            a5m.update(a5.item(), imgs.size(0))
            km.update(raw_out["active_k"].float().mean().item(), imgs.size(0))

            if i % 50 == 0:
                gate_accum.append(raw_out["gate_weights"].detach().float().cpu())

            if i % train_cfg.log_interval == 0:
                elapsed    = time.time() - t0
                imgs_per_s = imgs_seen / max(elapsed, 1e-6)
                eta_epoch  = elapsed / max(i, 1) * (total_steps - i)
                if epoch_times:
                    remaining = sum(epoch_times)/len(epoch_times) * \
                                (train_cfg.epochs_stage3 - epoch) + eta_epoch
                    total_eta = f"  总ETA:{eta_str(remaining)}"
                else:
                    total_eta = ""

                print(
                    f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] "
                    f"[{i:04d}/{total_steps}] "
                    f"│ Loss:{lm['total'].avg:.4f} "
                    f"(CE:{lm['ce'].avg:.3f} KD:{lm['kd'].avg:.3f} "
                    f"Bal:{lm['balance'].avg:.4f} Ortho:{lm['ortho'].avg:.4f}) "
                    f"│ Acc@1:{a1m.avg:.2f}%  Acc@5:{a5m.avg:.2f}% "
                    f"│ AvgK:{km.avg:.2f} "
                    f"│ LR:{lr:.3e}  GradNorm:{grad_norm:.3f} "
                    f"│ {imgs_per_s:.0f}imgs/s "
                    f"│ {get_gpu_stats(device)} "
                    f"│ EpochETA:{eta_str(eta_epoch)}"
                    f"{total_eta}"
                )

        scheduler.step()
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        avg_t     = sum(epoch_times) / len(epoch_times)
        remaining = avg_t * (train_cfg.epochs_stage3 - epoch)

        val_a1, val_a5 = validate(model, val_loader, device, "moe", amp_dtype)

        # 专家激活分布
        extra = []
        if gate_accum:
            g = torch.cat(gate_accum, 0).mean(0).tolist()
            rows = "  [专家激活分布]"
            for idx, w in enumerate(g):
                bar = "█" * int(w * 40)
                rows += f"\n    expert_{idx:02d}: {w:.3f}  {bar}"
            extra.append(rows)

        print_epoch_summary(
            stage=3, epoch=epoch, total_epochs=train_cfg.epochs_stage3,
            train_loss=lm["total"].avg, train_a1=a1m.avg, train_a5=a5m.avg,
            val_a1=val_a1, val_a5=val_a5, best_acc1=best_acc1,
            epoch_time=epoch_time, avg_epoch_time=avg_t, remaining=remaining,
            extra_lines=extra,
        )

        if val_a1 > best_acc1:
            best_acc1        = val_a1
            no_improve_count = 0
            save_ckpt({"epoch":epoch,"model":model.state_dict(),"acc1":val_a1},
                      f"{train_cfg.save_dir}/stage3_best.pt")
        else:
            no_improve_count += 1

        if epoch % train_cfg.save_every == 0:
            save_ckpt({"epoch":epoch,"model":model.state_dict()},
                      f"{train_cfg.save_dir}/stage3_epoch{epoch}.pt")

        # v5.1: 早停
        if no_improve_count >= train_cfg.early_stopping_patience:
            print(f"\n  [Early Stop] 连续 {no_improve_count} epoch 无提升，停止训练")
            break

    print(f"\n  Stage3 完成！最佳 Val Acc@1: {best_acc1:.2f}%\n")

    # v5.1: 训练结束自动评估
    if args.auto_eval:
        print("\n  [Auto Eval] 自动运行专家评估...\n")
        from eval_experts import run_eval as run_expert_eval
        from eval_alpha_sweep import run_sweep as run_alpha_sweep
        eval_args = argparse.Namespace(
            ckpt=f"{train_cfg.save_dir}/stage3_best.pt",
            gpus=args.gpus
        )
        run_expert_eval(eval_args)
        run_alpha_sweep(eval_args)


# ── 主入口 ────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",       type=str,   default="2", choices=["2", "3", "all"])
    p.add_argument("--mode",        type=str,   default="local", choices=["local","distributed"])
    p.add_argument("--gpus",        type=str,   default="0")
    p.add_argument("--auto_eval",   action="store_true", default=True)
    # v5.2: 允许命令行覆盖关键配置，方便多实验并行
    p.add_argument("--save_dir",    type=str,   default=None,
                   help="覆盖 train_cfg.save_dir（不同实验用不同目录）")
    p.add_argument("--teacher_ckpt",type=str,   default=None,
                   help="覆盖 teacher checkpoint 路径")
    p.add_argument("--alpha_end",   type=float, default=None,
                   help="覆盖 alpha_sched_cfg.alpha_end")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()

    # 命令行参数覆盖 config
    if args.save_dir:
        train_cfg.save_dir = args.save_dir
    if args.alpha_end is not None:
        alpha_sched_cfg.alpha_end = args.alpha_end

    gpu_id = int(args.gpus.split(",")[0])
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    if train_cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    sep("═")
    print("  MoE-Tiny v5.2 — NWPU-RESISC45 224×224，Phase A 无压缩验证")
    sep("═")
    print(f"  设备:    {device}")
    print(f"  Stage:   {args.stage}")
    print(f"  Batch:   {train_cfg.batch_size}")
    print(f"  AMP:     {train_cfg.amp_dtype if train_cfg.use_amp else 'disabled'}")
    print(f"  Workers: {data_cfg.num_workers}")
    print(f"  LR:      {train_cfg.lr}  (backbone: {train_cfg.lr_backbone})")
    print(f"  Warmup:  {train_cfg.warmup_epochs} epochs")
    print(f"  早停:    patience={train_cfg.early_stopping_patience}")
    print(f"  开始:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sep("═"); print()

    tmp = _build_moe_model()
    tmp.print_params(); del tmp; print()

    if args.stage == "all":
        train_stage2(args, device)
        train_stage3(args, device)
    elif args.stage == "2":
        train_stage2(args, device)
    elif args.stage == "3":
        train_stage3(args, device)
