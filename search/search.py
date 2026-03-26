"""
search.py — 轻量级架构搜索 (v5.2)

三阶段搜索：
  Phase 1: 搜核心变量（head_type × alpha_end × cut_point），每个配置 10 epoch
  Phase 2: 在 Phase 1 top-K 上搜次要变量，每个配置 10 epoch
  Phase 3: 对全局 top-K 做完整 100 epoch 训练 + 自动评估

用法：
  python search.py --phase all  --gpus 1   # 全自动一键过夜
  python search.py --phase 1    --gpus 1   # 只跑 Phase 1
  python search.py --phase 2    --gpus 1   # 只跑 Phase 2（需要 phase1_results.json）
  python search.py --phase 3    --gpus 1   # 只跑 Phase 3（需要 phase2_results.json）

输出目录：search_results/
  phase1_results.json
  phase2_results.json
  phase3_results/config_001/ ... config_003/
  search_summary.txt
"""

import os
import sys
import json
import time
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    data_cfg, distill_cfg, train_cfg, expert_drop_cfg,
)
from data.dataset      import build_dataloaders
from models.moe_system import MoESystemTiny
from distill.losses    import BackboneDistillLoss, MoEDistillLoss, build_teacher
from search_space      import (
    SearchConfig, generate_phase1_configs, generate_phase2_configs,
    config_to_str, PHASE1_SPACE,
)

# ── 全局搜索配置 ───────────────────────────────────────────
search_cfg = SearchConfig()


# ── 工具函数 ───────────────────────────────────────────────

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

def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def get_amp_dtype():
    s = train_cfg.amp_dtype.lower()
    if s == "bf16": return torch.bfloat16
    if s == "fp16": return torch.float16
    return torch.float32

def extract_teacher_feat_and_logits(teacher, imgs):
    feats = {}
    h = teacher.layer3.register_forward_hook(lambda m, i, o: feats.update({"f": o}))
    with torch.no_grad():
        t_logits = teacher(imgs)
    h.remove()
    return t_logits, feats["f"]

def compute_alpha(epoch, total_epochs, alpha_start, alpha_end, warmup=20):
    """线性退火：前 warmup epoch 从 alpha_start 降到 alpha_end"""
    if epoch <= warmup:
        return alpha_start - (alpha_start - alpha_end) * (epoch / warmup)
    return alpha_end


# ── 结果存取 ───────────────────────────────────────────────

def save_results(results: list, filename: str):
    os.makedirs(search_cfg.save_dir, exist_ok=True)
    path = os.path.join(search_cfg.save_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [Save] {path}")

def load_results(filename: str) -> list:
    path = os.path.join(search_cfg.save_dir, filename)
    assert os.path.exists(path), f"找不到结果文件: {path}"
    with open(path) as f:
        return json.load(f)

def print_top_k(results: list, k: int = 5):
    sep("═")
    print(f"  Top-{min(k, len(results))} 配置排名：")
    sep()
    for i, r in enumerate(results[:k]):
        print(f"  [{i+1}] Val={r['best_val_acc']:.2f}%  "
              f"SA={r['standalone_acc']:.2f}%  "
              f"{config_to_str(r['config'])}")
    sep("═")


# ── 验证 ───────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, mode="moe", amp_dtype=None):
    model.eval()
    a1m = AverageMeter(); a5m = AverageMeter()
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = (model.forward_standalone(imgs)
                      if mode == "standalone"
                      else model(imgs)["logits"])
        a1, a5 = accuracy(logits.float(), labels)
        a1m.update(a1.item(), imgs.size(0))
        a5m.update(a5.item(), imgs.size(0))
    return a1m.avg, a5m.avg


# ── Stage 2 共享训练（按 cut_point 各跑一次）──────────────

def run_stage2_once(cut_point: str, device, train_loader, val_loader, teacher):
    """
    为指定 cut_point 跑一次 Stage 2，保存到 stage2_{cut_point}.pt
    Stage 2 只训练主干特征提取能力，和 head_type 无关，可共享。
    """
    ckpt_path = os.path.join(train_cfg.save_dir, f"stage2_{cut_point}.pt")
    if os.path.exists(ckpt_path):
        print(f"  [Stage2] 已存在 {ckpt_path}，跳过训练")
        return ckpt_path

    sep("═")
    print(f"  Stage 2 共享训练  cut_point={cut_point}  epochs={train_cfg.epochs_stage2}")
    sep("═")

    amp_dtype  = get_amp_dtype() if train_cfg.use_amp else None
    use_scaler = train_cfg.use_amp and train_cfg.amp_dtype.lower() == "fp16"
    scaler     = GradScaler('cuda', enabled=use_scaler)

    # 用 full_network head 跑 Stage 2（head 类型不影响特征提取）
    model = MoESystemTiny(config={
        "head_type": "full_network",
        "cut_point": cut_point,
        "num_experts": 4,
    }).to(device)
    for p in model.experts.parameters():  p.requires_grad = False
    for p in model.router.parameters():   p.requires_grad = False

    criterion = BackboneDistillLoss(distill_cfg.alpha_feat, distill_cfg.gamma_ce)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.lr_backbone, weight_decay=train_cfg.weight_decay
    )
    effective = train_cfg.epochs_stage2 - train_cfg.warmup_epochs
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, total_iters=train_cfg.warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=max(effective, 1)),
    ], milestones=[train_cfg.warmup_epochs])

    best_acc, total_steps = 0., len(train_loader)

    for epoch in range(1, train_cfg.epochs_stage2 + 1):
        model.train(); teacher.eval()
        lm = {k: AverageMeter() for k in ["total", "ce", "feat"]}
        a1m = AverageMeter()

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype is not None)):
                with torch.no_grad():
                    t_logits, t_feat = extract_teacher_feat_and_logits(teacher, imgs)
                s_logits = model.forward_standalone(imgs)
                s_feat   = model.backbone.forward_features(imgs)
                ld       = criterion(s_logits, t_logits, s_feat, t_feat, labels)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(ld["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                ld["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

            a1, _ = accuracy(s_logits.float(), labels)
            for k, v in ld.items(): lm[k].update(v.item(), imgs.size(0))
            a1m.update(a1.item(), imgs.size(0))

            if i % train_cfg.log_interval == 0:
                print(f"  [{epoch:02d}/{train_cfg.epochs_stage2}]"
                      f"[{i:04d}/{total_steps}] "
                      f"Loss:{lm['total'].avg:.4f} "
                      f"(CE:{lm['ce'].avg:.4f} Feat:{lm['feat'].avg:.4f}) "
                      f"Acc@1:{a1m.avg:.2f}%")

        scheduler.step()
        val_a1, _ = validate(model, val_loader, device, "standalone", amp_dtype)
        print(f"  [Stage2-{cut_point}] Epoch {epoch:02d}  "
              f"Val(standalone):{val_a1:.2f}%")

        if val_a1 > best_acc:
            best_acc = val_a1
            os.makedirs(train_cfg.save_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "acc1": val_a1}, ckpt_path)

    print(f"\n  Stage2({cut_point}) 完成！最佳 Val:{best_acc:.2f}%  保存至 {ckpt_path}\n")
    return ckpt_path


# ── 单配置 Stage 3 短训练 ──────────────────────────────────

def run_single_config(
    config: dict,
    epochs: int,
    stage2_ckpt: str,
    device,
    teacher,
    train_loader,
    val_loader,
    save_dir: str = None,
) -> dict:
    """
    对单个架构配置跑 N epoch Stage 3，返回验证精度结果。

    关键设计：
    - 所有 head_type 共享同一个 cut_point 的 Stage 2 checkpoint
    - strict=False 加载（head 结构可能不同）
    - 主干冻结，只训专家和路由器
    """
    amp_dtype  = get_amp_dtype() if train_cfg.use_amp else None
    use_scaler = train_cfg.use_amp and train_cfg.amp_dtype.lower() == "fp16"
    scaler     = GradScaler('cuda', enabled=use_scaler)

    # 构建模型
    model = MoESystemTiny(config=config).to(device)

    # 加载 Stage 2 权重（strict=False，head 结构可能不同）
    if os.path.exists(stage2_ckpt):
        ck = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
        miss, unex = model.load_state_dict(ck["model"], strict=False)
        print(f"  [Load] Stage2 ckpt  缺失:{len(miss)}  多余:{len(unex)}")
    else:
        print(f"  [Warn] 未找到 Stage2 ckpt: {stage2_ckpt}，随机初始化")

    # 冻结主干
    model.freeze_backbone()

    # 损失函数
    balance_w = config.get("balance_loss_weight", 0.1)
    criterion = MoEDistillLoss(
        gamma_ce=distill_cfg.gamma_ce,
        beta_kd=distill_cfg.beta_kd,
        balance_w=balance_w,
        ortho_w=expert_drop_cfg.ortho_w,
        temperature=distill_cfg.temperature,
    )

    # 优化器
    optimizer = optim.AdamW(
        model.param_groups(train_cfg.lr, train_cfg.lr_backbone),
        weight_decay=train_cfg.weight_decay,
    )
    warmup_ep = min(train_cfg.warmup_epochs, epochs)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_ep),
        CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_ep, 1)),
    ], milestones=[warmup_ep])

    # Alpha 退火参数
    alpha_start = search_cfg.alpha_start
    alpha_end   = config.get("alpha_end", 0.5)
    use_mixup   = train_cfg.stage3_use_mixup
    total_steps = len(train_loader)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        router_temp   = max(0.3, 1.0 - epoch / epochs * 0.7)
        current_alpha = compute_alpha(epoch, epochs, alpha_start, alpha_end)
        model.set_alpha(current_alpha)
        model.train(); teacher.eval()

        lm  = {k: AverageMeter() for k in ["total", "ce", "kd", "balance", "ortho"]}
        a1m = AverageMeter()

        for i, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_mixup:
                imgs, y_a, y_b, lam = mixup_data(
                    imgs, labels, alpha=train_cfg.stage3_mixup_alpha, device=device
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                ld["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

            model.router.temperature = router_temp
            a1, _ = accuracy(raw_out["logits"].float(), y_a)
            for k, v in ld.items(): lm[k].update(v.item(), imgs.size(0))
            a1m.update(a1.item(), imgs.size(0))

            if i % train_cfg.log_interval == 0:
                print(f"    [{epoch:02d}/{epochs}][{i:04d}/{total_steps}] "
                      f"Loss:{lm['total'].avg:.4f} "
                      f"Acc@1:{a1m.avg:.2f}%  α:{current_alpha:.3f}")

        scheduler.step()
        val_a1, _ = validate(model, val_loader, device, "moe", amp_dtype)
        best_val_acc = max(best_val_acc, val_a1)
        print(f"  Epoch {epoch:02d}/{epochs}  Val:{val_a1:.2f}%  Best:{best_val_acc:.2f}%")

    # 评估 standalone 精度
    sa_acc, _ = validate(model, val_loader, device, "standalone", amp_dtype)

    # Phase 3 保存完整 checkpoint
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "config": config, "acc1": best_val_acc},
            os.path.join(save_dir, "stage3_best.pt")
        )

    return {
        "config":          config,
        "best_val_acc":    best_val_acc,
        "standalone_acc":  sa_acc,
    }


# ── Phase 1 ────────────────────────────────────────────────

def run_phase1(args, device):
    sep("═")
    print("  Phase 1：搜索核心变量")
    print(f"  配置数: 4×4×2 = 32  每个 {search_cfg.phase1_epochs} epoch")
    sep("═")

    amp_dtype = get_amp_dtype() if train_cfg.use_amp else None
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

    # 按 cut_point 各跑一次 Stage 2
    stage2_ckpts = {}
    for cp in PHASE1_SPACE["cut_point"]:
        # 单独加载无增强的 loader 用于 Stage 2
        s2_train, s2_val, _ = build_dataloaders(
            data_cfg.train_dir, data_cfg.val_dir,
            image_size=data_cfg.image_size,
            batch_size=train_cfg.batch_size,
            num_workers=data_cfg.num_workers,
        )
        stage2_ckpts[cp] = run_stage2_once(cp, device, s2_train, s2_val, teacher)

    # 生成所有 Phase 1 配置并逐个搜索
    configs = generate_phase1_configs()
    print(f"\n  开始 Phase 1 搜索，共 {len(configs)} 个配置\n")

    results = []
    t_start = time.time()
    for i, config in enumerate(configs):
        sep()
        print(f"  [{i+1:02d}/{len(configs)}] {config_to_str(config)}")
        sep()

        stage2_ckpt = stage2_ckpts[config["cut_point"]]
        result = run_single_config(
            config, search_cfg.phase1_epochs,
            stage2_ckpt, device, teacher,
            train_loader, val_loader,
        )
        results.append(result)

        elapsed = time.time() - t_start
        avg_t   = elapsed / (i + 1)
        eta     = avg_t * (len(configs) - i - 1)
        print(f"  → Val:{result['best_val_acc']:.2f}%  "
              f"SA:{result['standalone_acc']:.2f}%  "
              f"ETA:{eta_str(eta)}")

    results.sort(key=lambda x: x["best_val_acc"], reverse=True)
    save_results(results, "phase1_results.json")
    print_top_k(results, k=5)


# ── Phase 2 ────────────────────────────────────────────────

def run_phase2(args, device):
    sep("═")
    print("  Phase 2：在 Phase 1 top-K 上搜次要变量")
    sep("═")

    phase1_results = load_results("phase1_results.json")
    top_k_configs  = phase1_results[:search_cfg.phase1_top_k]
    print(f"  Phase 1 top-{search_cfg.phase1_top_k} 配置：")
    for i, r in enumerate(top_k_configs):
        print(f"    [{i+1}] Val={r['best_val_acc']:.2f}%  {config_to_str(r['config'])}")
    print()

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
    t_start = time.time()

    for base_result in top_k_configs:
        base_cfg = base_result["config"]
        p2_configs = generate_phase2_configs(base_cfg)
        stage2_ckpt = os.path.join(
            train_cfg.save_dir, f"stage2_{base_cfg['cut_point']}.pt"
        )
        print(f"\n  基配置: {config_to_str(base_cfg)}")
        print(f"  搜索 {len(p2_configs)} 个次要变量组合\n")

        for j, config in enumerate(p2_configs):
            sep()
            print(f"  [{j+1:02d}/{len(p2_configs)}] {config_to_str(config)}")
            result = run_single_config(
                config, search_cfg.phase2_epochs,
                stage2_ckpt, device, teacher,
                train_loader, val_loader,
            )
            all_results.append(result)
            elapsed = time.time() - t_start
            print(f"  → Val:{result['best_val_acc']:.2f}%  "
                  f"SA:{result['standalone_acc']:.2f}%")

    all_results.sort(key=lambda x: x["best_val_acc"], reverse=True)
    save_results(all_results, "phase2_results.json")
    print_top_k(all_results, k=5)


# ── Phase 3 ────────────────────────────────────────────────

def run_phase3(args, device):
    sep("═")
    print("  Phase 3：完整训练 top-K 配置")
    sep("═")

    phase2_results = load_results("phase2_results.json")
    top_k = phase2_results[:search_cfg.phase3_top_k]
    print(f"  选取 top-{search_cfg.phase3_top_k} 配置做完整训练：")
    for i, r in enumerate(top_k):
        print(f"    [{i+1}] Val={r['best_val_acc']:.2f}%  {config_to_str(r['config'])}")
    print()

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

    final_results = []

    for rank, base_result in enumerate(top_k):
        config = base_result["config"]
        run_dir = os.path.join(search_cfg.save_dir, "phase3_results", f"config_{rank+1:03d}")
        os.makedirs(run_dir, exist_ok=True)

        stage2_ckpt = os.path.join(
            train_cfg.save_dir, f"stage2_{config['cut_point']}.pt"
        )

        sep("═")
        print(f"  Phase3 [{rank+1}/{search_cfg.phase3_top_k}] "
              f"完整训练 {search_cfg.phase3_epochs} epoch")
        print(f"  配置: {config_to_str(config)}")
        sep("═")

        result = run_single_config(
            config, search_cfg.phase3_epochs,
            stage2_ckpt, device, teacher,
            train_loader, val_loader,
            save_dir=run_dir,
        )
        final_results.append(result)
        print(f"  → 完整训练结束  Val:{result['best_val_acc']:.2f}%  "
              f"SA:{result['standalone_acc']:.2f}%")

        # 自动跑 eval_experts + eval_alpha_sweep
        ckpt_path = os.path.join(run_dir, "stage3_best.pt")
        if os.path.exists(ckpt_path):
            print(f"\n  [Auto Eval] 运行专家评估...")
            try:
                from eval_experts     import run_eval     as run_expert_eval
                from eval_alpha_sweep import run_sweep    as run_alpha_sweep
                eval_args = argparse.Namespace(ckpt=ckpt_path, gpus=args.gpus)
                import contextlib
                with open(os.path.join(run_dir, "eval_experts.log"), "w") as f:
                    with contextlib.redirect_stdout(f):
                        run_expert_eval(eval_args)
                with open(os.path.join(run_dir, "eval_alpha_sweep.log"), "w") as f:
                    with contextlib.redirect_stdout(f):
                        run_alpha_sweep(eval_args)
                print(f"  [Auto Eval] 结果已写入 {run_dir}/")
            except Exception as e:
                print(f"  [Auto Eval] 评估失败: {e}")

    # 写搜索总结
    _write_summary(final_results)


def _write_summary(final_results: list):
    """写 search_summary.txt"""
    path = os.path.join(search_cfg.save_dir, "search_summary.txt")
    lines = ["═" * 50, "  v5.2-search 搜索结果", "═" * 50, ""]

    for i, r in enumerate(final_results):
        cfg = r["config"]
        sa  = r["standalone_acc"]
        moe = r["best_val_acc"]
        lines += [
            f"  Top-{i+1} 配置:",
            f"    head_type:           {cfg.get('head_type')}",
            f"    cut_point:           {cfg.get('cut_point')}",
            f"    alpha_end:           {cfg.get('alpha_end')}",
            f"    expert_hidden_dim:   {cfg.get('expert_hidden_dim')}",
            f"    num_experts:         {cfg.get('num_experts')}",
            f"    expert_num_layers:   {cfg.get('expert_num_layers')}",
            f"    balance_loss_weight: {cfg.get('balance_loss_weight')}",
            f"",
            f"    Standalone Acc@1:    {sa:.2f}%",
            f"    MoE System Acc@1:    {moe:.2f}%",
            f"    专家提升:            {moe - sa:+.2f}%",
            "",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  [Summary] 搜索结果已保存: {path}")
    print("\n".join(lines))


# ── 主入口 ────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, default="1",
                   choices=["1", "2", "3", "all"])
    p.add_argument("--gpus", type=str, default="1")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    gpu_id = int(args.gpus.split(",")[0])
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    os.makedirs(search_cfg.save_dir, exist_ok=True)
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    sep("═")
    print("  MoE v5.2-search — 轻量级架构搜索")
    print(f"  设备: {device}  Phase: {args.phase}")
    print(f"  Phase1: 32 配置 × {search_cfg.phase1_epochs} epoch")
    print(f"  Phase2: 48 配置 × {search_cfg.phase2_epochs} epoch")
    print(f"  Phase3: {search_cfg.phase3_top_k} 配置 × {search_cfg.phase3_epochs} epoch")
    print(f"  输出目录: {search_cfg.save_dir}/")
    print(f"  开始: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sep("═"); print()

    if args.phase in ("1", "all"):
        run_phase1(args, device)
    if args.phase in ("2", "all"):
        run_phase2(args, device)
    if args.phase in ("3", "all"):
        run_phase3(args, device)

    print(f"\n  搜索完成  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
