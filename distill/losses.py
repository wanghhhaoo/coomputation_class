"""
distill/losses.py — 蒸馏损失 (v2)

v2 变更：
  1. orthogonal_loss 增加下界，防止过快归零
  2. 新增 FeatureProjector，解决 Student/Teacher 维度不匹配
  3. MoEDistillLoss 支持 Mixup 标签
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


def kd_loss(student_logits, teacher_logits, temperature=4.0):
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits    / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


class FeatureProjector(nn.Module):
    """v2: Student→Teacher 特征维度投影"""
    def __init__(self, in_channels=256, out_channels=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.proj(x)


def feature_alignment_loss(student_feat, teacher_feat):
    """维度一致后直接 MSE"""
    if student_feat.shape != teacher_feat.shape:
        # 降级处理：GAP 后按通道统计对齐
        if student_feat.dim() == 4:
            s = F.adaptive_avg_pool2d(student_feat, 1).flatten(1)
            t = F.adaptive_avg_pool2d(teacher_feat, 1).flatten(1)
        else:
            s, t = student_feat, teacher_feat
        s = F.normalize(s, dim=-1)
        t = F.normalize(t, dim=-1)
        return F.mse_loss(s.mean(dim=-1), t.mean(dim=-1)) + \
               F.mse_loss(s.std(dim=-1), t.std(dim=-1))
    return F.mse_loss(student_feat, teacher_feat)


def orthogonal_loss(expert_feats: List[torch.Tensor]) -> torch.Tensor:
    """
    正交损失 (v2): 增加持续推动多样性的下界

    即使平均相似度降到 target 以下，仍保留微小惩罚，
    防止专家特征虽然"正交"但没有意义（高维随机向量天然正交）
    """
    if len(expert_feats) < 2:
        device = expert_feats[0].device if expert_feats else torch.device("cpu")
        return torch.tensor(0.0, device=device)

    vecs = torch.stack([
        F.normalize(f.mean(dim=0), dim=-1)
        for f in expert_feats
    ], dim=0)

    sim_matrix = torch.mm(vecs, vecs.t())

    n    = vecs.size(0)
    mask = torch.triu(torch.ones(n, n, device=vecs.device), diagonal=1).bool()
    off_diag = sim_matrix[mask]

    raw = off_diag.abs().mean()

    # v2: 主损失 + 持续微小惩罚
    # 高于 target 时正常惩罚，低于 target 时仍有 0.05 * raw 的推力
    target = 0.1
    loss = F.relu(raw - target) + 0.05 * raw

    return loss


class BackboneDistillLoss(nn.Module):
    def __init__(self, alpha_feat=1.0, gamma_ce=1.0):
        super().__init__()
        self.alpha_feat = alpha_feat
        self.gamma_ce   = gamma_ce
        self.ce         = nn.CrossEntropyLoss()

    def forward(self, s_logits, t_logits, s_feat, t_feat, labels):
        l_ce   = self.ce(s_logits, labels)
        l_feat = feature_alignment_loss(s_feat, t_feat.detach())
        total  = self.gamma_ce * l_ce + self.alpha_feat * l_feat
        return {"total": total, "ce": l_ce, "feat": l_feat}


class MoEDistillLoss(nn.Module):
    """
    Stage 3 完整 MoE 蒸馏损失 (v2)
    支持 Mixup：当 y_a, y_b, lam 传入时，CE 改为 Mixup 损失
    """

    def __init__(
        self,
        gamma_ce=1.0, beta_kd=1.0,
        balance_w=0.1,              # v2: 0.01 → 0.1
        ortho_w=0.5,                # v2: 0.1 → 0.5
        temperature=4.0,
    ):
        super().__init__()
        self.gamma_ce    = gamma_ce
        self.beta_kd     = beta_kd
        self.balance_w   = balance_w
        self.ortho_w     = ortho_w
        self.temperature = temperature
        self.ce          = nn.CrossEntropyLoss()

    def forward(
        self,
        moe_output: Dict,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        y_b: torch.Tensor = None,        # v2: Mixup 第二标签
        lam: float = 1.0,                # v2: Mixup 系数
    ) -> Dict[str, torch.Tensor]:

        s_logits     = moe_output["logits"]
        balance_loss = moe_output["balance_loss"]
        expert_feats = moe_output.get("expert_feats", [])

        # v2: 支持 Mixup
        if y_b is not None and lam < 1.0:
            l_ce = lam * self.ce(s_logits, labels) + \
                   (1 - lam) * self.ce(s_logits, y_b)
        else:
            l_ce = self.ce(s_logits, labels)

        l_kd    = kd_loss(s_logits, teacher_logits.detach(), self.temperature)
        l_bal   = balance_loss
        l_ortho = orthogonal_loss(expert_feats)

        total = (
            self.gamma_ce  * l_ce  +
            self.beta_kd   * l_kd  +
            self.balance_w * l_bal +
            self.ortho_w   * l_ortho
        )

        return {
            "total":   total,
            "ce":      l_ce,
            "kd":      l_kd,
            "balance": l_bal,
            "ortho":   l_ortho,
        }


def build_teacher(arch="resnet50", num_classes=200,
                  pretrained=False, checkpoint=None):
    from models.backbone import BackboneResNet50Teacher
    teacher = BackboneResNet50Teacher(num_classes=num_classes)
    if checkpoint and __import__("os").path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu",
                           weights_only=False)
        teacher.load_state_dict(state.get("model", state))
        print(f"[Teacher] 加载 checkpoint: {checkpoint}")
    else:
        print(f"[Teacher] 未找到 checkpoint，使用随机初始化")
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher
