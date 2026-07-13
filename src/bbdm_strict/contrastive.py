"""
建议 2：监督对比损失（Supervised Contrastive, Khosla et al. 2020）。

用途：在 BBDM 训练时，对"翻译得到的源域风格潜表示"按类别做对比——
同类拉近、异类推远，强制翻译保留类别判别结构，缓解"翻译把病灶抹平"的问题。

接入方式见文件底部 INTEGRATION 注释。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = float(temperature)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, D] 特征向量（会做 L2 归一化）
        labels: [B] 整数类别
        """
        if feats.dim() != 2:
            feats = feats.flatten(1)
        feats = F.normalize(feats, dim=1)
        B = feats.shape[0]
        device = feats.device

        sim = (feats @ feats.t()) / self.t
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()  # 数值稳定

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float().to(device)
        self_mask = torch.eye(B, device=device)
        pos_mask = pos_mask - self_mask                       # 正样本：同类且非自身

        exp_sim = torch.exp(sim) * (1.0 - self_mask)          # 分母排除自身
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        denom = pos_mask.sum(dim=1).clamp_min(1.0)
        loss = -(pos_mask * log_prob).sum(dim=1) / denom
        # 只对有同类正样本的样本计损失
        valid = (pos_mask.sum(dim=1) > 0).float()
        return (loss * valid).sum() / valid.sum().clamp_min(1.0)


# ============================ INTEGRATION ============================
# 在 train_strict_bbdm.py 里这样接（latent 模式）：
#
# 1) 顶部 import：
#    from contrastive import SupConLoss
#    supcon = SupConLoss(temperature=0.07)
#
# 2) 数据集需要给出类别标签。StrictBBDMPairedDataset 的 batch 里
#    用 x_A/x_B 配对时是 class-consistent，取该对的标签即可：
#    y = batch["label"].to(device)      # 若没有该字段，加一个返回 label 的字段
#
# 3) 已经算出 x_a_hat_bridge（预测的源域风格潜表示）后：
#    feat = x_a_hat_bridge.mean(dim=(2, 3))          # [B, C] 全局池化
#    loss_supcon = supcon(feat, y) if y is not None else x_a_img.new_zeros(())
#
# 4) 加进总损失：
#    loss_total = loss_total + float(cfg.get("lambda_supcon", 0.0)) * loss_supcon
#
# 5) config 里加： "lambda_supcon": 0.1   （建议 0.05~0.2 起步）
#
# 说明：也可改成对“解码图过分类器的特征”做 SupCon（更贴合判别空间，但需载入
# 冻结分类器、每步多一次前向）。先用 latent 池化版最省事。
