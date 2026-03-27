"""
eval_experts.py — 专家渐进评估脚本 (v2)
v2: 适配 backbone_alpha 为 buffer 的新 API
"""

import os, sys, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from config       import data_cfg, expert_cfg, router_cfg, compressor_cfg
from data.dataset import build_resisc45_dataloaders
from models.moe_system import MoESystemTiny
from torch.amp import autocast


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

def sep(c="─", w=60): print(c*w)


@torch.no_grad()
def eval_with_experts(model, val_loader, device, expert_indices,
                      alpha_override=None, amp_dtype=torch.bfloat16):
    model.eval()
    a1m = AverageMeter()
    a5m = AverageMeter()

    # v2: 使用 buffer 值或 override
    if alpha_override is not None:
        alpha = alpha_override
    else:
        alpha = model.get_alpha()

    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with autocast('cuda', dtype=amp_dtype):
            backbone_feat   = model.backbone.forward_features(imgs)
            backbone_logits = model.backbone.forward_standalone(imgs)

            if len(expert_indices) == 0:
                logits = backbone_logits
            else:
                compressed   = model.compressor(backbone_feat)
                decompressed = model.decompressor(compressed) if model.use_decompressor else compressed

                expert_logit_list = []
                for idx in expert_indices:
                    logit, _ = model.experts[idx](decompressed)
                    expert_logit_list.append(logit)

                expert_fused = torch.stack(expert_logit_list, dim=1).mean(dim=1)
                logits = alpha * backbone_logits + (1 - alpha) * expert_fused

        a1, a5 = accuracy(logits.float(), labels)
        a1m.update(a1.item(), imgs.size(0))
        a5m.update(a5.item(), imgs.size(0))

    return a1m.avg, a5m.avg


@torch.no_grad()
def get_expert_weights(model, val_loader, device, amp_dtype):
    weight_sum = torch.zeros(model.num_experts)
    count = 0
    model.eval()
    for imgs, _ in val_loader:
        imgs = imgs.to(device)
        with autocast('cuda', dtype=amp_dtype):
            backbone_feat = model.backbone.forward_features(imgs)
            compressed    = model.compressor(backbone_feat)
            gate_w, _, _  = model.router(compressed, training=False)
        weight_sum += gate_w.mean(dim=0).cpu()
        count += 1
        if count >= 20:
            break
    return (weight_sum / count).tolist()


def run_eval(args):
    device    = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    _, val_loader, _ = build_resisc45_dataloaders(
        data_cfg.data_dir,
        image_size=data_cfg.image_size, batch_size=256,
        num_workers=data_cfg.num_workers,
    )

    model = MoESystemTiny(config={
        "num_classes":   data_cfg.num_classes,
        "spatial_size":  compressor_cfg.spatial_size,
        "feat_ch":       compressor_cfg.in_channels,
        "compress_ch":   compressor_cfg.out_channels,
        "expert_hidden": expert_cfg.hidden_dim,
        "num_experts":   router_cfg.num_experts,
        "expert_dropout": 0.0,
    }).to(device)
    assert os.path.exists(args.ckpt), f"找不到 checkpoint: {args.ckpt}"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[Eval] 加载权重: {args.ckpt}")
    print(f"[Eval] 专家总数: {model.num_experts}")
    print(f"[Eval] backbone_alpha: {model.get_alpha():.3f}")

    num_experts = model.num_experts
    results     = []

    sep("═")
    print(f"  专家渐进评估：激活 0 ~ {num_experts} 个专家")
    print(f"  融合方式：平均融合（排除门控权重干扰）")
    sep("═")

    acc1, acc5 = eval_with_experts(model, val_loader, device, [], amp_dtype=amp_dtype)
    results.append((0, acc1, acc5, []))
    print(f"  专家数=00  Acc@1={acc1:.2f}%  Acc@5={acc5:.2f}%  [只用主干]")

    expert_weights = get_expert_weights(model, val_loader, device, amp_dtype)
    sorted_indices = sorted(range(num_experts),
                           key=lambda i: expert_weights[i], reverse=True)

    print(f"\n  专家按门控权重排序（从高到低）:")
    for rank, idx in enumerate(sorted_indices):
        print(f"    第{rank+1}位: expert_{idx:02d}  平均权重={expert_weights[idx]:.4f}")
    sep()

    active = []
    for rank, expert_idx in enumerate(sorted_indices):
        active.append(expert_idx)
        acc1, acc5 = eval_with_experts(
            model, val_loader, device, active, amp_dtype=amp_dtype
        )
        results.append((len(active), acc1, acc5, active.copy()))
        delta = acc1 - results[-2][1]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  专家数={len(active):02d}  Acc@1={acc1:.2f}%  "
              f"Acc@5={acc5:.2f}%  "
              f"Δ={delta:+.2f}% {arrow}  "
              f"[+expert_{expert_idx:02d}]")

    sep("═")
    print("\n  完整结果汇总：")
    sep()
    print(f"  {'专家数':>6}  {'Acc@1':>8}  {'Acc@5':>8}  {'Δ Acc@1':>8}")
    sep()
    for i, (n, a1, a5, _) in enumerate(results):
        delta_str = ""
        if i > 0:
            delta = a1 - results[i-1][1]
            delta_str = f"{delta:+.2f}%"
        bar = "█" * int(a1 / 2)
        print(f"  {n:>6}个  {a1:>7.2f}%  {a5:>7.2f}%  {delta_str:>8}  {bar}")
    sep()

    best     = max(results, key=lambda x: x[1])
    baseline = results[0][1]
    print(f"\n  主干基线:    {baseline:.2f}%")
    print(f"  精度峰值:    {best[1]:.2f}%  (使用 {best[0]} 个专家)")
    print(f"  总提升:      {best[1]-baseline:+.2f}%")

    acc_series = [r[1] for r in results]
    peak_idx   = acc_series.index(max(acc_series))
    if peak_idx < len(acc_series) - 1:
        print(f"\n  ⚠ 精度在第 {peak_idx} 个专家时达到峰值后开始下降")
    else:
        print(f"\n  ✓ 精度随专家数单调递增（或趋于平稳）")
    sep("═")

    save_path = os.path.join(os.path.dirname(args.ckpt), "expert_eval_results.txt")
    with open(save_path, "w") as f:
        f.write("专家数,Acc@1,Acc@5,激活专家列表\n")
        for n, a1, a5, idxs in results:
            f.write(f"{n},{a1:.4f},{a5:.4f},{idxs}\n")
    print(f"\n  结果已保存: {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints_resisc/stage3_best.pt")
    p.add_argument("--gpus", type=str, default="1")
    args = p.parse_args()
    run_eval(args)
