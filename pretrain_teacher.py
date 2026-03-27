"""
pretrain_teacher.py — NWPU-RESISC45 教师网络预训练 (v5.2)

使用 ResNet-50（ImageNet 预训练权重）在 RESISC45 上 fine-tune。
224×224 输入，保持标准 stem（7×7 stride=2），45 类输出。

运行：
  python pretrain_teacher.py --gpus 0 > checkpoints_resisc/teacher.log 2>&1
"""

import os, sys, time, argparse, datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(__file__))

from config            import data_cfg, train_cfg
from data.dataset      import build_resisc45_dataloaders
from models.backbone   import BackboneResNet50Teacher


# ── 工具 ──────────────────────────────────────────────────

def sep(c="─", w=72): print(c * w)
def eta_str(s): return str(datetime.timedelta(seconds=int(s)))

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk); B = target.size(0)
        _, pred = output.topk(maxk, 1, True, True); pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).mul_(100. / B) for k in topk]

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    print(f"  [Save] {path}")


# ── 验证 ──────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, amp_dtype=None):
    model.eval()
    a1m = AverageMeter(); a5m = AverageMeter()
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype is not None)):
            out  = model(imgs)
        a1, a5 = accuracy(out.float(), labels)
        a1m.update(a1.item(), imgs.size(0))
        a5m.update(a5.item(), imgs.size(0))
    return a1m.avg, a5m.avg


# ── 主训练函数 ────────────────────────────────────────────

def train_teacher(args):
    gpu_id = int(args.gpus.split(",")[0])
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    amp_dtype  = torch.bfloat16 if train_cfg.use_amp else None
    use_scaler = False  # bf16 不需要 GradScaler
    scaler     = GradScaler('cuda', enabled=use_scaler)

    os.makedirs(train_cfg.save_dir, exist_ok=True)

    # ── 数据 ──────────────────────────────────────────────
    sep("═")
    print(f"  Teacher 预训练  数据集: NWPU-RESISC45  类别: {data_cfg.num_classes}")
    print(f"  图像尺寸: {data_cfg.image_size}×{data_cfg.image_size}  "
          f"设备: {device}  Epochs: {args.epochs}")
    sep("═"); print()

    train_loader, val_loader, num_classes = build_resisc45_dataloaders(
        data_cfg.data_dir,
        image_size=data_cfg.image_size,
        batch_size=args.batch_size,
        num_workers=data_cfg.num_workers,
        train_ratio=data_cfg.train_ratio,
        use_strong_aug=True,
    )
    total_steps = len(train_loader)

    # ── 模型（标准 ResNet-50，224×224 stem）────────────────
    model = BackboneResNet50Teacher(
        num_classes=num_classes,
        pretrained=True,
        image_size=data_cfg.image_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── 分阶段学习率：先只训 FC，再全局 fine-tune ───────────
    # Phase 1（前 5 epoch）：冻结 stem/layer1~3，只训 layer4 + fc
    for name, p in model.named_parameters():
        if any(name.startswith(k) for k in ("stem.", "pool.", "layer1.", "layer2.", "layer3.")):
            p.requires_grad = False

    freeze_epochs = min(5, args.epochs // 4)
    full_epochs   = args.epochs - freeze_epochs

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=train_cfg.weight_decay,
    )
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=2)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(freeze_epochs - 2, 1))
    scheduler    = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[2])

    best_acc1  = 0.
    epoch_times = []

    for epoch in range(1, args.epochs + 1):

        # 解冻全部层（进入 full fine-tune 阶段）
        if epoch == freeze_epochs + 1:
            print(f"\n  [Unfreeze] Epoch {epoch}: 解冻所有层，全局 fine-tune\n")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr_full, weight_decay=train_cfg.weight_decay,
            )
            warmup2  = LinearLR(optimizer, start_factor=0.1, total_iters=2)
            cosine2  = CosineAnnealingLR(optimizer, T_max=max(full_epochs - 2, 1))
            scheduler = SequentialLR(optimizer, [warmup2, cosine2], milestones=[2])

        lr = optimizer.param_groups[0]["lr"]
        now = datetime.datetime.now().strftime("%H:%M:%S")
        sep("═")
        print(f"  Teacher Epoch [{epoch:03d}/{args.epochs}] │ LR:{lr:.3e} │ {now}  "
              f"{'[FC only]' if epoch <= freeze_epochs else '[全局]'}")
        sep("═")

        model.train()
        lm  = AverageMeter(); a1m = AverageMeter(); a5m = AverageMeter()
        t0  = time.time()

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype is not None)):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

            a1, a5 = accuracy(logits.float(), labels)
            lm.update(loss.item(), imgs.size(0))
            a1m.update(a1.item(), imgs.size(0))
            a5m.update(a5.item(), imgs.size(0))

            if i % train_cfg.log_interval == 0:
                elapsed   = time.time() - t0
                eta_epoch = elapsed / max(i, 1) * (total_steps - i)
                print(f"  [{i:04d}/{total_steps}] "
                      f"Loss:{lm.avg:.4f}  Acc@1:{a1m.avg:.2f}%  Acc@5:{a5m.avg:.2f}%  "
                      f"ETA:{eta_str(eta_epoch)}")

        scheduler.step()
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        avg_t     = sum(epoch_times) / len(epoch_times)
        remaining = avg_t * (args.epochs - epoch)

        val_a1, val_a5 = validate(model, val_loader, device, amp_dtype)
        is_best = val_a1 > best_acc1
        if is_best:
            best_acc1 = val_a1
            save_ckpt({"epoch": epoch, "model": model.state_dict(), "acc1": val_a1},
                      f"{train_cfg.save_dir}/teacher_best.pt")

        sep()
        print(f"  Epoch {epoch:03d}/{args.epochs}  "
              f"Train Loss:{lm.avg:.4f}  Acc@1:{a1m.avg:.2f}%  "
              f"Val Acc@1:{val_a1:.2f}%  Acc@5:{val_a5:.2f}%  "
              f"Best:{best_acc1:.2f}%{'  ✓' if is_best else ''}  "
              f"ETA:{eta_str(remaining)}")
        sep(); print()

        if epoch % train_cfg.save_every == 0:
            save_ckpt({"epoch": epoch, "model": model.state_dict()},
                      f"{train_cfg.save_dir}/teacher_epoch{epoch}.pt")

    print(f"\n  Teacher 预训练完成！最佳 Val Acc@1: {best_acc1:.2f}%")
    print(f"  checkpoint: {train_cfg.save_dir}/teacher_best.pt\n")


# ── 主入口 ────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Teacher ResNet-50 预训练 (RESISC45)")
    p.add_argument("--gpus",       type=str,   default="0")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr_head",    type=float, default=1e-3,
                   help="Phase 1 (FC only) 学习率")
    p.add_argument("--lr_full",    type=float, default=5e-4,
                   help="Phase 2 (全局 fine-tune) 学习率")
    p.add_argument("--save_dir",   type=str,   default=None,
                   help="覆盖 train_cfg.save_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.save_dir:
        train_cfg.save_dir = args.save_dir
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    train_teacher(args)
