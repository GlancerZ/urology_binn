# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:37:26 2026

@author: 10857
"""

"""
PerPathwayGatedFusion
逐通路门控融合模块

核心思想：
    每个 H6 生物通路节点都拥有独立的门控权重 α，
    实现单一 PRS 标量对多维生物特征的精细调节。

核心公式：
    alpha_vec = Sigmoid(MLP([H6, PRS_encoded]))   # (B, h6_dim)，每个通路独立一个 α
    P_mapped  = Linear(PRS_encoded)               # (B, h6_dim)
    Fused     = (1 - alpha_vec) · H6 + alpha_vec · P_mapped
    out       = LayerNorm(H6 + Fused)             # 残差连接

物理意义：
    alpha_vec[i] → 1：第 i 个通路节点主要受遗传风险（PRS）驱动
    alpha_vec[i] → 0：第 i 个通路节点主要受组学特征（H6）驱动
    相比标量 α，向量 α 能捕捉 PRS 对不同通路的差异化影响

与 GatedResidualCrossModalAttention（凸组合版）的区别：
    凸组合版：α 由 [H6, PRS] → MLP → h6_dim 生成，结构相似
    本模块：中间隐藏层更宽（128维），PRS 编码维度更高（64维），
            表达能力更强，适合通路节点数较多的场景

调用方式：
    from binn.model.per_pathway_gated_fusion import PerPathwayGatedFusion
    fusion = PerPathwayGatedFusion(h6_dim=29)
    out, alpha_vec = fusion(h6, prs)
"""
import torch
import torch.nn as nn


class PerPathwayGatedFusion(nn.Module):
    """
    逐通路门控融合模块

    参数：
        h6_dim   : int，BINN 最后隐层维度（如 29）
        prs_embed: int，PRS 编码隐藏维度，默认 64
    """

    def __init__(self, h6_dim: int, prs_embed: int = None):
        super().__init__()
        self.h6_dim = h6_dim
        # prs_embed 默认与 h6_dim 保持一致
        if prs_embed is None:
            prs_embed = h6_dim

        # ── 1. PRS 升维编码：1 → prs_embed
        #    捕获遗传风险的复杂表征
        self.prs_enc = nn.Sequential(
            nn.Linear(1, prs_embed),
            nn.Tanh(),
        )

        # ── 2. PRS 空间对齐：prs_embed → h6_dim
        #    将 PRS 编码映射到与 H6 相同的维度
        self.prs_projector = nn.Linear(prs_embed, h6_dim)

        # ── 3. 逐通路 Alpha 向量生成网络
        #    输入：[H6, PRS_encoded] 拼接
        #    输出：长度为 h6_dim 的向量 α，每个通路节点独立一个权重
        #    Sigmoid 保证 α ∈ [0, 1]
        self.alpha_net = nn.Sequential(
            nn.Linear(h6_dim + prs_embed, 128),
            nn.ReLU(),
            nn.Linear(128, h6_dim),   # 输出维度必须等于 h6_dim
            nn.Sigmoid(),
        )

        # ── 残差连接后的 LayerNorm
        self.norm = nn.LayerNorm(h6_dim)

    def forward(self, h6: torch.Tensor, prs: torch.Tensor):
        """
        参数：
            h6  : (B, h6_dim)  BINN 最后隐层特征（生物通路激活信号）
            prs : (B, 1)       PRS 值（遗传背景风险）

        返回：
            out      : (B, h6_dim)  融合后特征（经过残差连接 + LayerNorm）
            alpha_vec: (B, h6_dim)  逐通路门控权重，可用于可解释性分析
        """
        # Step 1: PRS 升维编码
        p_latent = self.prs_enc(prs)                                    # (B, prs_embed)

        # Step 2: PRS 投影到 H6 维度
        p_mapped = self.prs_projector(p_latent)                         # (B, h6_dim)

        # Step 3: 计算逐通路 Alpha 向量
        # 每个通路节点都有独立的门控权重
        alpha_vec = self.alpha_net(torch.cat([h6, p_latent], dim=1))    # (B, h6_dim)

        # Step 4: 元素级凸组合融合
        # 每个通路节点独立决定 H6 和 PRS 的混合比例
        fused = (1 - alpha_vec) * h6 + alpha_vec * p_mapped             # (B, h6_dim)

        # Step 5: 残差连接 + LayerNorm
        out = self.norm(h6 + fused)                                      # (B, h6_dim)

        return out, alpha_vec