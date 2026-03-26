"""
eval_alpha_sweep.py — backbone_alpha 扫描评估 (v2)
v2: 直接传入 alpha_override 而非修改模型参数
"""

import os, sys, argparse
import torch
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))

from config       import data_cfg, expert_cfg, router_cfg, compressor_cfg
from data.dataset import build_dataloaders
from models.moe_system import MoESystemTiny


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

def sep(c="─", w=68): print(c*w)


@torch.no_grad()
def eval_with_k_experts(model, val_loader, device,
                        expert_indices, alpha_override,
                        amp_dtype=torch.bfloat16):
    model.eval()
    a1m = AverageMeter()
    a5m = AverageMeter()

    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast('cuda', dtype=amp_dtype):
            backbone_feat   = model.backbone.forward_features(imgs)
            backbone_logits = model.backbone.forward_standalone(imgs)

            if len(expert_indices) == 0:
                logits = backbone_logits
            else:
                compressed   = model.compressor(backbone_feat)
                decompressed = model.decompressor(compressed)
                expert_logit_list = []
                for idx in expert_indices:
                    logit, _ = model.experts[idx](decompressed)
                    expert_logit_list.append(logit)
                expert_fused = torch.stack(expert_logit_list, dim=1).mean(dim=1)
                logits = alpha_override * backbone_logits + \
                         (1 - alpha_override) * expert_fused

        a1, a5 = accuracy(logits.float(), labels)
        a1m.update(a1.item(), imgs.size(0))
        a5m.update(a5.item(), imgs.size(0))

    return a1m.avg, a5m.avg


@torch.no_grad()
def get_expert_order(model, val_loader, device, amp_dtype=torch.bfloat16):
    weight_sum = torch.zeros(model.num_experts)
    count = 0
    for imgs, _ in val_loader:
        imgs = imgs.to(device)
        with autocast('cuda', dtype=amp_dtype):
            feat = model.backbone.forward_features(imgs)
            comp = model.compressor(feat)
            gate_w, _, _ = model.router(comp, training=False)
        weight_sum += gate_w.mean(dim=0).cpu()
        count += 1
        if count >= 20:
            break
    weights = (weight_sum / count).tolist()
    return sorted(range(model.num_experts),
                  key=lambda i: weights[i], reverse=True), weights


def run_sweep(args):
    device    = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    _, val_loader, _ = build_dataloaders(
        data_cfg.train_dir, data_cfg.val_dir,
        image_size=data_cfg.image_size, batch_size=256,
        num_workers=data_cfg.num_workers,
    )

    model = MoESystemTiny(
        num_classes=200, expert_dropout=0.0,
        num_experts=router_cfg.num_experts,
        dropout_rate=expert_cfg.dropout_rate,
        compress_in=compressor_cfg.in_channels,
        compress_out=compressor_cfg.out_channels,
        decompress_out=expert_cfg.in_channels,
        expert_hidden=expert_cfg.hidden_dim,
        router_hidden=router_cfg.hidden_dim,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    trained_alpha = model.get_alpha()
    print(f"[Eval] 模型训练时的 backbone_alpha = {trained_alpha:.3f}")

    sorted_indices, weights = get_expert_order(model, val_loader, device)
    print(f"[Eval] 专家排序: {sorted_indices}")

    alpha_list = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    all_results = {}

    sep("═")
    print("  backbone_alpha 扫描 × 专家渐进评估")
    print(f"  扫描范围: alpha ∈ {alpha_list}")
    sep("═")

    for alpha in alpha_list:
        print(f"\n  ── alpha = {alpha:.1f} "
              f"(主干占{alpha*100:.0f}% / 专家占{(1-alpha)*100:.0f}%) ──")
        sep()

        results = []
        acc1_backbone, _ = eval_with_k_experts(
            model, val_loader, device, [], alpha, amp_dtype
        )
        results.append((0, acc1_backbone))
        print(f"    k=00  Acc@1={acc1_backbone:.2f}%  [主干]")

        active = []
        for expert_idx in sorted_indices:
            active.append(expert_idx)
            acc1, _ = eval_with_k_experts(
                model, val_loader, device, active, alpha, amp_dtype
            )
            delta = acc1 - results[-1][1]
            arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
            results.append((len(active), acc1))
            print(f"    k={len(active):02d}  Acc@1={acc1:.2f}%  "
                  f"Δ={delta:+.2f}% {arrow}  [+expert_{expert_idx:02d}]")

        all_results[alpha] = results
        best = max(results, key=lambda x: x[1])
        print(f"    → 峰值: {best[1]:.2f}% (k={best[0]})  "
              f"主干: {acc1_backbone:.2f}%  "
              f"提升空间: {best[1]-acc1_backbone:+.2f}%")

    sep("═")
    print("\n  汇总对比\n")
    sep()
    print(f"  {'alpha':>6}  {'主干Acc@1':>10}  {'峰值Acc@1':>10}  "
          f"{'最优K':>6}  {'专家提升':>8}")
    sep()

    for alpha, results in all_results.items():
        backbone_acc = results[0][1]
        best         = max(results, key=lambda x: x[1])
        gain         = best[1] - backbone_acc
        print(f"  {alpha:>6.1f}  {backbone_acc:>9.2f}%  "
              f"{best[1]:>9.2f}%  {best[0]:>6}个  {gain:>+7.2f}%")

    sep()
    print(f"\n  参考：训练时 alpha={trained_alpha:.3f}")
    sep("═")

    save_path = os.path.join(os.path.dirname(args.ckpt), "alpha_sweep_results.txt")
    with open(save_path, "w") as f:
        f.write("alpha,专家数,Acc@1\n")
        for alpha, results in all_results.items():
            for k, acc1 in results:
                f.write(f"{alpha},{k},{acc1:.4f}\n")
    print(f"\n  结果已保存: {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints_tiny/stage3_best.pth")
    p.add_argument("--gpus", type=str, default="1")
    args = p.parse_args()
    run_sweep(args)
