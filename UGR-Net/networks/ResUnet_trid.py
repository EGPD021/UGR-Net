from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils import prune
from networks.unet import UnetBlock
import torch.nn.functional as F
import torch
from torch import nn
from networks.fetureweight import extract_features, ACBBlock, DepthwiseSeparableConv
from networks.MBpool import MBPOOL
from networks.SCSA import SCSA
from networks.CSHA import CSHA
from networks.CHWS import CHWS
from networks.confidence import ConfidenceGuidedChannelSpatialEvidential
from networks.resnest import resnest50
import torch.nn.functional as F
from networks.vit import VisionTransformer
import random

import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from networks.Rmcsam import RMCSAM, RMCSAM_CBAM
from networks.mixstyle_kernel import TriD,MixStyle,EFDMix,DomainLearner,DomainClassMixAugmentation
from networks.ChannelAttention import ChannelAttention, SpatialAttention, AFF,iAFF,FAFF,MultiModalFusionModule

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def vec_from_any(x):
    if x.dim() == 4:  # [B,C,H,W]
        return F.adaptive_avg_pool2d(x, 1).flatten(1)
    if x.dim() == 3:  # [B,C,L]
        return F.adaptive_avg_pool1d(x, 1).flatten(1)
    if x.dim() == 2:  # [B,C]
        return x
    raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

class ResidualEvidFuser(nn.Module):
    """
    主干：z_base = cat([x1, x2, ...], dim=1)
    残差：z_res  = Σ_i w_i * proj_i(x_i)      （w 由 evid 计算）
    输出：z = z_base + gamma * z_res

    in_ch_list: 每个分支的通道数列表，如 [64,64,64,64]
    out_ch:     残差分支的通道数，默认等于 sum(in_ch_list) 以便和 z_base 相加
    tau:        softmax 温度（越大越均匀）
    beta:       不确定度映射幂指数（(1 - û)^beta）
    w_floor:    权重下限，避免某个分支被压成 0
    gamma_init: 残差强度初值（可在训练中动态调整）
    """
    def __init__(self, in_ch_list, out_ch=None, tau=2.0, beta=0.2, w_floor=0.2, gamma_init=0.2):
        super().__init__()
        self.in_ch_list = list(in_ch_list)
        self.out_ch = int(out_ch if out_ch is not None else sum(self.in_ch_list))
        self.tau = float(tau)
        self.beta = float(beta)
        self.w_floor = float(w_floor)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)), requires_grad=False)  # 需要可学习就改 True

        self.proj = nn.ModuleList([nn.Conv2d(c, self.out_ch, kernel_size=1, bias=False)
                                   for c in self.in_ch_list])
        for p in self.proj:
            nn.init.kaiming_normal_(p.weight, nonlinearity="linear")

    @torch.no_grad()
    def set_gamma(self, g: float):
        self.gamma.data.fill_(float(g))

    def _evid_to_weight(self, evid_list):
        """
        evid: dict，至少包含 'uncertainty' 或 'epistemic_mi'，形状 [B,1]
        返回 w: [B, n_levels]，行归一化
        """
        scores = []
        for evid in evid_list:
            # MI 优先作为“置信度”参考，退化到 u
            u_gate = evid.get("epistemic_mi", evid["uncertainty"])  # [B,1]
            # 批内 min-max 归一
            u_min = u_gate.min()
            u_max = u_gate.max()
            u_hat = (u_gate - u_min) / (u_max - u_min + 1e-6)
            # 置信度分数（越大越可信）
            s = torch.pow((1.0 - u_hat).clamp(0, 1), self.beta)     # [B,1]
            scores.append(s)
        S = torch.cat(scores, dim=1)            # [B, n_levels]
        w = torch.softmax(S / self.tau, dim=1)  # 温度软化
        w = torch.clamp(w, min=self.w_floor)    # 加地板
        w = w / w.sum(dim=1, keepdim=True)      # 归一
        return w                                 # [B, n_levels]

    def forward(self, xs, evid_list):
        """
        xs:        list of feature maps，每个 [B,C_i,H,W]，H/W 必须相同（你当前就是这样）
        evid_list: 与 xs 对齐的 evid 字典列表
        """
        assert len(xs) == len(self.in_ch_list) == len(evid_list)
        B, _, H, W = xs[0].shape
        # 主干：直接拼接（信息完整保留）
        z_base = torch.cat(xs, dim=1)  # [B, sum(C_i), H, W]

        # 权重（按 batch 逐样本）
        w = self._evid_to_weight(evid_list)     # [B, n_levels]

        # 残差：每个分支 1x1 投影到 out_ch，再按 w 加权求和
        z_res = 0.0
        for i, (x_i, proj_i) in enumerate(zip(xs, self.proj)):
            z_i = proj_i(x_i)                   # [B, out_ch, H, W]
            wi  = w[:, i].view(B, 1, 1, 1)      # [B,1,1,1]
            z_res = z_res + wi * z_i

        # 温和残差注入
        z = z_base + self.gamma * z_res         # [B, sum(C_i), H, W]（因为 out_ch = sum(C_i)）
        return z, w                              # w 可用于监控

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, num_states):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(num_states, num_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        return self.softmax(self.fc(state))

# 创建策略网络和优化器
num_actions = 2  # 假设有两个数据增强方式
num_states = 10  # 假设有10个状态特征
policy_network = PolicyNetwork(num_actions, num_states)
optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

#创建置信度反馈的特征模块
class ConfidenceGuidedChannelModulation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ConfidenceGuidedChannelModulation, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conf_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        feat = self.gap(x).view(b, c)
        conf = self.conf_fc(feat).view(b, c, 1, 1)
        x_mod = x * conf  # channel-wise scaling
        return x_mod, conf




import numpy as np
import torch
import torch.nn as nn


class GateWithUncertainty(nn.Module):
    """
    用 Dirichlet 不确定性 u 对注意力特征做残差混合门控。
    x_out = κ * x_att + (1-κ) * x_raw
    其中 κ = (1-u)^γ
    """

    def __init__(self, gamma):
        """
        Args:
            gamma: 指数缩放因子，可以是单个值或列表
        """
        super(GateWithUncertainty, self).__init__()
        # 存储所有可能的 gamma 值
        self.gamma_values = gamma if isinstance(gamma, list) else [gamma]
        # 当前使用的 gamma 索引（默认为0）
        self.current_gamma_idx = 0
        # 当前使用的 gamma 值
        self.current_gamma = self.gamma_values[self.current_gamma_idx]

        # 打印初始化的 gamma 值
        print(f"GateWithUncertainty 初始化: 可用 gamma 值 = {self.gamma_values}")
        # print(f"当前使用的 gamma = {self.current_gamma}")

    def next_gamma(self):
        """切换到下一个 gamma 值"""
        # 增加索引，如果超过列表长度则循环回到开头
        self.current_gamma_idx = (self.current_gamma_idx + 1) % len(self.gamma_values)
        self.current_gamma = self.gamma_values[self.current_gamma_idx]
        print(f"切换到下一个 gamma: {self.current_gamma} (索引: {self.current_gamma_idx})")
        return self.current_gamma

    def set_gamma(self, gamma_idx):
        """
        设置当前使用的 gamma 值

        Args:
            gamma_idx: gamma 值在列表中的索引
        """
        if 0 <= gamma_idx < len(self.gamma_values):
            self.current_gamma_idx = gamma_idx
            self.current_gamma = self.gamma_values[gamma_idx]
            print(f"设置 gamma 为: {self.current_gamma} (索引: {gamma_idx})")
        else:
            print(f"错误: 索引 {gamma_idx} 超出范围 [0, {len(self.gamma_values) - 1}]")

    def forward(self, x_att: torch.Tensor, x_raw: torch.Tensor, u: torch.Tensor = None):
        """
        Args:
            x_att: 注意力后的特征 [B,C,H,W]
            x_raw: 注意力前的特征 [B,C,H,W]
            u:     不确定性张量 [B,1]，来自 Dirichlet 分布 M/S
        Returns:
            x_out: 门控后的特征 [B,C,H,W]
        """
        if u is None:
            return x_att

        # 使用当前 gamma 值
        # print(f"当前使用的 gamma = {self.current_gamma}")
        kappa = torch.clamp((1.0 - u) ** self.current_gamma, 0.0, 1.0)  # [B,1]

        return kappa.view(-1, 1, 1, 1) * x_att + (1.0 - kappa).view(-1, 1, 1, 1) * x_raw

# class GateWithUncertainty(nn.Module):
#     """
#     用 Dirichlet 不确定性 u 对注意力特征做残差混合门控。
#     x_out = κ * x_att + (1-κ) * x_raw
#     其中 κ = (1-u)^γ
#     """
#     def __init__(self, gamma ):
#         """
#         Args:
#             gamma: 指数缩放因子，γ > 0；γ 越大，高不确定样本抑制越强
#         """
#         super(GateWithUncertainty, self).__init__()
#         self.gamma = gamma if gamma else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
#
#
#     def forward(self, x_att: torch.Tensor, x_raw: torch.Tensor, u: torch.Tensor = None):
#         """
#         Args:
#             x_att: 注意力后的特征 [B,C,H,W]
#             x_raw: 注意力前的特征 [B,C,H,W]
#             u:     不确定性张量 [B,1]，来自 Dirichlet 分布 M/S
#         Returns:
#             x_out: 门控后的特征 [B,C,H,W]
#         """
#         if u is None:
#             # print("没有u")
#             return x_att
#         # κ = (1-u)^γ
#         # print("u:",u)
#         kappa = torch.clamp((1.0 - u) ** self.gamma, 0.0, 1.0)  # [B,1]
#         return kappa.view(-1, 1, 1, 1) * x_att + (1.0 - kappa).view(-1, 1, 1, 1) * x_raw


#通道空间置信度反馈注意力模块
# class ConfidenceGuidedChannelSpatialModulation(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         # 通道注意力 MLP
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels),
#             nn.Sigmoid()
#         )
#
#         # 空间注意力部分
#         self.spatial_pool = nn.Conv2d(2, 1, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#         # 输出置信度（可选用于可视化/判别）
#         self.confidence_score = None
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # ---------------- 通道注意力 ---------------- #
#         y = self.avg_pool(x).view(b, c)          # [B, C, 1, 1] -> [B, C]
#         channel_weight = self.mlp(y).view(b, c, 1, 1)
#         x_channel_att = x * channel_weight       # 通道调制
#
#         # ---------------- 空间注意力 ---------------- #
#         avg_out = torch.mean(x_channel_att, dim=1, keepdim=True)   # [B, 1, H, W]
#         max_out, _ = torch.max(x_channel_att, dim=1, keepdim=True)
#         spatial_input = torch.cat([avg_out, max_out], dim=1)       # [B, 2, H, W]
#         spatial_weight = self.sigmoid(self.spatial_pool(spatial_input))  # [B, 1, H, W]
#         x_att = x_channel_att * spatial_weight
#
#         # ---------------- 置信度估计（可选） ---------------- #
#         channel_conf = channel_weight.view(b, c)
#         spatial_conf = spatial_weight.view(b, -1).mean(dim=1, keepdim=True)  # 每图均值
#         self.confidence_score = (channel_conf.mean(dim=1, keepdim=True) + spatial_conf) / 2  # 归一化置信度评分
#         # x_att, confidence_score, channel_weight, spatial_weight
#         return x_att, self.confidence_score, channel_weight.detach(), spatial_weight.detach()
import torch
import torch.nn as nn
import math

# ---------- 通用工具：Top-B 内随机 Dropout + 总和保持重标定 ----------
def dropout_topk_attention_1d(attn_1d: torch.Tensor, top_B: int, drop_ratio: float,
                              keep_sum: bool = True, eps: float = 1e-12):
    """
    attn_1d: [B, N]，每样本 N 个注意力分数(建议∈[0,1])
    top_B:   最高的 B 个中进行随机丢弃
    drop_ratio: 在 top_B 内随机丢弃的比例 (0,1]
    keep_sum:   丢弃后是否把剩余权重按“保持旧总和”重标定
    返回: attn_new [B,N]、每样本被丢弃的 indices 列表
    """
    B, N = attn_1d.shape
    top_B = min(top_B, N)
    attn_new = attn_1d.clone()

    # Top-B indices (降序，不必排序输出)
    _, topk_idx = torch.topk(attn_new, k=top_B, dim=1, largest=True, sorted=False)

    k_drop = max(1, math.ceil(top_B * drop_ratio))
    dropped = []
    for b in range(B):
        perm = torch.randperm(top_B, device=attn_new.device)
        drop_idx = topk_idx[b, perm[:k_drop]]
        dropped.append(drop_idx)
        attn_new[b, drop_idx] = 0.0  # 硬置零

    if keep_sum:
        old_sum = attn_1d.sum(dim=1, keepdim=True).clamp_min(eps)
        new_sum = attn_new.sum(dim=1, keepdim=True).clamp_min(eps)
        attn_new = attn_new * (old_sum / new_sum)

    return attn_new.clamp(0.0, 1.0), dropped


# class ConfidenceGuidedCSM_FromEvid(nn.Module):
#     """
#     forward(x, evid): evid 来自 DirichletHead(feats)
#         evid["uncertainty"] = u ∈ (0,1], [B,1]
#         evid["alpha"]       = alpha,    [B,M]   (可选参与缩放)
#     """
#     def __init__(self, in_channels, reduction=16,
#                  # 门控阈值
#                  t_ch=0.3, t_sp=0.3,
#                  # Top-B 超参
#                  topB_ch=8,  drop_ratio_ch=0.5,
#                  topB_sp=256, drop_ratio_sp=0.3,
#                  # 置信幅度缩放：w <- w * (1 - u)^beta
#                  use_alpha_scale=True, beta=1.0):
#         super().__init__()
#         self.t_ch, self.t_sp = float(t_ch), float(t_sp)
#         self.topB_ch, self.drop_ratio_ch = int(topB_ch), float(drop_ratio_ch)
#         self.topB_sp, self.drop_ratio_sp = int(topB_sp), float(drop_ratio_sp)
#         self.use_alpha_scale, self.beta = bool(use_alpha_scale), float(beta)
#
#         # 通道注意力
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         hid = max(1, in_channels // reduction)
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, hid), nn.ReLU(inplace=True),
#             nn.Linear(hid, in_channels), nn.Sigmoid()
#         )
#         # 空间注意力
#         self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     @staticmethod
#     def _gate_mask(u: torch.Tensor, t: float):
#         # u: [B,1] or [B] → [B]；返回 True 表示触发TopB-Dropout
#         return u.view(-1) >= t
#
#     def _maybe_topb_channel(self, w_ch: torch.Tensor, mask_do: torch.Tensor):
#         B, C = w_ch.shape[:2]
#         w = w_ch.view(B, C)
#         dropped = [torch.empty(0, dtype=torch.long, device=w.device) for _ in range(B)]
#         if mask_do.any():
#             idx = mask_do.nonzero(as_tuple=False).view(-1)
#             w_do = w[idx]
#             w_do, dropped_do = dropout_topk_attention_1d(
#                 w_do, self.topB_ch, self.drop_ratio_ch, keep_sum=True)
#             w[idx] = w_do
#             for j, b in enumerate(idx.tolist()): dropped[b] = dropped_do[j]
#         return w.view(B, C, 1, 1), dropped
#
#     def _maybe_topb_spatial(self, w_sp: torch.Tensor, mask_do: torch.Tensor):
#         B, _, H, W = w_sp.shape
#         w = w_sp.view(B, -1)
#         dropped = [torch.empty(0, dtype=torch.long, device=w.device) for _ in range(B)]
#         if mask_do.any():
#             idx = mask_do.nonzero(as_tuple=False).view(-1)
#             w_do = w[idx]
#             w_do, dropped_do = dropout_topk_attention_1d(
#                 w_do, min(self.topB_sp, H * W), self.drop_ratio_sp, keep_sum=True)
#             w[idx] = w_do
#             for j, b in enumerate(idx.tolist()): dropped[b] = dropped_do[j]
#         return w.view(B, 1, H, W), dropped
#
#     def forward(self, x: torch.Tensor, evid: dict):
#         """
#         x: [B,C,H,W]
#         evid: 来自 DirichletHead 的输出字典（至少包含 'uncertainty'，可包含 'alpha'）
#         """
#         u = evid["uncertainty"]           # [B,1]
#         # 可选：利用 alpha 的强度/置信度来调整幅度；这里用 (1 - u)^beta
#         scale = (1.0 - u).clamp(0, 1) ** self.beta  # [B,1]
#         B, C, H, W = x.shape
#
#         # ===== Channel attention =====
#         ch_vec = self.gap(x).view(B, C)
#         w_ch = self.mlp(ch_vec).view(B, C, 1, 1)           # [B,C,1,1] ∈ [0,1]
#         if self.use_alpha_scale:
#             w_ch = w_ch * scale.view(B, 1, 1, 1)           # 置信放缩
#
#         mask_ch = self._gate_mask(u, self.t_ch)            # 是否触发TopB
#         w_ch, dropped_ch = self._maybe_topb_channel(w_ch, mask_ch)
#         x_ch = x * w_ch
#
#         # ===== Spatial attention =====
#         avg = x_ch.mean(dim=1, keepdim=True)
#         mx, _ = x_ch.max(dim=1, keepdim=True)
#         sp_in = torch.cat([avg, mx], dim=1)
#         w_sp = self.sigmoid(self.spatial_conv(sp_in))      # [B,1,H,W]
#         if self.use_alpha_scale:
#             w_sp = w_sp * scale.view(B, 1, 1, 1)           # 置信放缩
#
#         mask_sp = self._gate_mask(u, self.t_sp)
#         w_sp, dropped_sp = self._maybe_topb_spatial(w_sp, mask_sp)
#
#         x_out = x_ch * w_sp
#
#         # 置信分数（方便记录）
#         conf_ch = w_ch.view(B, C).mean(dim=1, keepdim=True)
#         conf_sp = w_sp.view(B, -1).mean(dim=1, keepdim=True)
#         confidence = (conf_ch + conf_sp) / 2
#
#         return (x_out, confidence, w_ch.detach(), w_sp.detach(),
#                 dropped_ch, dropped_sp)

class ConfidenceGuidedCSM_FromEvid(nn.Module):
    """
    forward(x, evid): evid 由 DirichletHead(feats) 给出
      必含:
        evid["uncertainty"] = u_total ∈ (0,∞), [B,1]
        evid["epistemic_mi"] = mi ∈ [0, +),   [B,1]
      可含:
        evid["alpha"], evid["strength"] 等
    """

    def __init__(self, in_channels, reduction=16,
                 # 门控“基准阈值”（将被分位/标准化后再进入sigmoid）
                 t_ch=0.8, t_sp=0.8,
                 # Top-B 超参
                 topB_ch=8, drop_ratio_ch=0.0,
                 topB_sp=256, drop_ratio_sp=0.0,#禁用/放松 Top-B 丢弃：drop_ratio_ch=0.0、drop_ratio_sp=0.0（或显著调小），先确认不是被硬阈值“掐死”。
                 # 置信幅度缩放：w <- w * (1 - û)^beta
                 use_alpha_scale=True, beta=0.3,#设置beta=1.0时，将特征全部关在门外，因此设置在0.1-0.3之间
                 # 软门控温度
                 init_tau=1.5, #降到 1.0~1.5（越小越“软”），避免 m_ch/m_sp 一上来接近 0 或 1
                 # 是否用动态分位阈值
                 use_dynamic_thresh=True, q_ch=0.7, q_sp=0.7):
        super().__init__()
        self.t_ch, self.t_sp = float(t_ch), float(t_sp)
        self.topB_ch, self.drop_ratio_ch = int(topB_ch), float(drop_ratio_ch)
        self.topB_sp, self.drop_ratio_sp = int(topB_sp), float(drop_ratio_sp)
        self.use_alpha_scale, self.beta = bool(use_alpha_scale), float(beta)
        self.use_dynamic_thresh = use_dynamic_thresh
        self.q_ch, self.q_sp = float(q_ch), float(q_sp)

        # 可学习温度(越大→门更“硬”，建议从2~4起步)
        self.tau_ch = nn.Parameter(torch.tensor(init_tau))
        self.tau_sp = nn.Parameter(torch.tensor(init_tau))

        # 通道注意力 (SE)
        self.gap = nn.AdaptiveAvgPool2d(1)
        hid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, in_channels), nn.Sigmoid()
        )
        # 空间注意力 (CBAM风格)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _topb_dropout_1d(w_flat, topB, drop_ratio, keep_sum=True):
        """
        w_flat: [B, L]  ∈ (0,1).  对每个样本丢弃其 Top-B 分量的一部分（比例 drop_ratio）
        返回形状不变的 w'，可选 keep_sum 让权重和保持近似一致。
        """
        B, L = w_flat.shape
        k = min(topB, L)
        if k <= 0 or drop_ratio <= 1e-6:
            return w_flat, [torch.empty(0, dtype=torch.long, device=w_flat.device) for _ in range(B)]
        w = w_flat.clone()
        dropped_idx_list = []
        for b in range(B):
            topk = torch.topk(w[b], k=k, largest=True).indices
            # 在这 k 个里再随机/按幅度丢 drop_ratio 部分
            m = max(1, int(k * drop_ratio))
            drop_idx = topk[torch.randperm(k, device=w.device)[:m]]
            dropped_idx_list.append(drop_idx)
            w[b, drop_idx] = 0.0
            if keep_sum:
                # 重新归一化到原始和（避免全幅降低）
                s_old = w_flat[b].sum().clamp_min(1e-6)
                s_new = w[b].sum().clamp_min(1e-6)
                w[b] = w[b] * (s_old / s_new)
        return w, dropped_idx_list

    @staticmethod
    def _percentile(x, q):
        # 返回按 q 分位的阈值（逐 batch 计算）
        x_sorted, _ = torch.sort(x.view(-1))
        idx = max(0, min(int((len(x_sorted) - 1) * q), len(x_sorted) - 1))
        return x_sorted[idx].detach()

    def forward(self, x, evid):
        B, C, H, W = x.shape
        mi = evid.get("epistemic_mi", None)
        u = evid["uncertainty"]

        # ---- 门控信号（阻断梯度 + 批内归一）----
        # ---- 门控信号（阻断梯度 + 批内归一，DG 请用 MI 优先）----
        with torch.no_grad():
            u_gate = evid.get("epistemic_mi", evid["uncertainty"]).detach()  # MI优先
            u_hat = (u_gate - u_gate.min()) / (u_gate.max() - u_gate.min() + 1e-6)
            # scale = (1.0 - u_hat).clamp(0, 1) ** 0.5  # beta=0.5 起步
            scale = (1.0 - u_hat).clamp(0.0, 1.0) ** 0.3 #避免scal=0
            scale = scale.clamp_min(0.5)
        # # ---- Channel attention ----
        ch_vec = self.gap(x).view(B, C)
        # # w_ch = self.mlp(ch_vec).view(B, C)
        # w_ch = self.mlp(ch_vec).view(B, C) #把 SE/CBAM 的权重映射到 [0.5, 1.0]（下限 0.5）这样即使权重小也不会把通路杀死：
        # w_ch = 0.5 + 0.5 * w_ch
        #
        # with torch.no_grad():
        #     t_ch = self._percentile(u_hat, self.q_ch) if self.use_dynamic_thresh else torch.tensor(self.t_ch,
        #                                                                                            device=x.device)
        #     w_ch_drop, _ = self._topb_dropout_1d(w_ch.detach(), self.topB_ch, self.drop_ratio_ch, keep_sum=True)
        #     w_ch_drop = w_ch_drop.to(w_ch.dtype)
        #
        # m_ch = torch.sigmoid(self.tau_ch * (u_hat.view(-1) - t_ch)).view(B, 1)  # tau=2.0
        # w_ch = (1 - m_ch) * w_ch + m_ch * w_ch_drop
        # w_ch = w_ch * scale if self.use_alpha_scale else w_ch
        # x_ch = x * w_ch.view(B, C, 1, 1)
        #
        # # ---- Spatial attention（可先关或降强度）----
        # avg = x_ch.mean(dim=1, keepdim=True)
        # mx, _ = x_ch.max(dim=1, keepdim=True)
        # sp_in = torch.cat([avg, mx], dim=1)
        # # w_sp_2d = self.sigmoid(self.spatial_conv(sp_in))  # [B,1,H,W]
        # w_sp_2d = self.sigmoid(self.spatial_conv(sp_in))
        # w_sp_2d = 0.5 + 0.5 * w_sp_2d
        # w_sp = w_sp_2d.view(B, -1)
        #
        # with torch.no_grad():
        #     t_sp = self._percentile(u_hat, self.q_sp) if self.use_dynamic_thresh else torch.tensor(self.t_sp,
        #                                                                                            device=x.device)
        #     # 简易省显存：直接 keep_sum=False；或先 F.avg_pool2d 后再 topk
        #     w_sp_drop, _ = self._topb_dropout_1d(w_sp.detach(), min(self.topB_sp, H * W), self.drop_ratio_sp, keep_sum=False)
        #     w_sp_drop = w_sp_drop.to(w_sp.dtype)
        #
        # m_sp = torch.sigmoid(self.tau_sp * (u_hat.view(-1) - t_sp)).view(B, 1)
        # w_sp = (1 - m_sp) * w_sp + m_sp * w_sp_drop
        # w_sp = w_sp * scale if self.use_alpha_scale else w_sp
        # w_sp = w_sp.view(B, 1, H, W)
        #
        # x_out = x_ch * w_sp
        # 通道分支
        g_ch = self.mlp(ch_vec).view(B, C)  # 预激活
        scale_ch = 1.0 + 0.3 * torch.tanh(g_ch / self.tau_ch)  # ∈ [0.7, 1.3]
        if self.use_alpha_scale:
            scale_ch = scale_ch * scale.view(B, 1).clamp_min(0.5)
        x_ch = x * scale_ch.view(B, C, 1, 1)

        # 空间分支
        avg = x_ch.mean(dim=1, keepdim=True)
        mx, _ = x_ch.max(dim=1, keepdim=True)
        sp_in = torch.cat([avg, mx], dim=1)
        g_sp = self.spatial_conv(sp_in)
        scale_sp = 1.0 + 0.3 * torch.tanh(g_sp / self.tau_sp)  # ∈ [0.7, 1.3]
        if self.use_alpha_scale:
            scale_sp = scale_sp * scale.view(B, 1, 1, 1).clamp_min(0.5)

        x_out = x_ch * scale_sp
        vis_data = {
            'uncertainty': u.detach(),
            'epistemic_mi': mi.detach() if mi is not None else u.detach(),
            'u_hat': u_hat.detach(),
            'scale': scale.detach(),
            'ch_vec': ch_vec.detach(),
            'g_ch': g_ch.detach(),
            'scale_ch': scale_ch.detach(),
            'g_sp': g_sp.detach(),
            'scale_sp': scale_sp.detach(),
            'tau_ch': self.tau_ch.detach(),
            'tau_sp': self.tau_sp.detach(),
        }
        # 统计项可只在 eval 计算，训练时直接返回 None 进一步省显存
        # return x_out, None, w_ch.detach().view(B, C, 1, 1), w_sp.detach(), None, None
        return x_out,vis_data,None,None,None,None
        # return x_att, self.confidence_score  # 可选输出 confidence_score

# class ConfidenceGuidedChannelSpatialModulation(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         # 通道注意力
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels),
#             nn.Sigmoid()
#         )
#         # 空间注意力
#         self.spatial_pool = nn.Conv2d(2, 1, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, u: torch.Tensor = None):
#         b, c, h, w = x.size()
#
#         # ---- 通道注意力 ----
#         y = self.avg_pool(x).view(b, c)
#         channel_weight = self.mlp(y).view(b, c, 1, 1)
#         if u is not None:
#             channel_weight = channel_weight * (1 - u).view(b, 1, 1, 1)  # 不确定性越高→抑制注意力
#         x_channel_att = x * channel_weight
#
#         # ---- 空间注意力 ----
#         avg_out = torch.mean(x_channel_att, dim=1, keepdim=True)
#         max_out, _ = torch.max(x_channel_att, dim=1, keepdim=True)
#         spatial_input = torch.cat([avg_out, max_out], dim=1)
#         spatial_weight = self.sigmoid(self.spatial_pool(spatial_input))
#         if u is not None:
#             spatial_weight = spatial_weight * (1 - u).view(b, 1, 1, 1)
#         x_att = x_channel_att * spatial_weight
#
#         return x_att
def _u_from_evid(evid):
    if evid is None or not isinstance(evid, dict): return None
    u = evid.get("epistemic_mi", evid.get("uncertainty", None))
    if u is None: return None
    u = torch.as_tensor(u).float().view(-1,1)
    return (u - u.min()) / (u.max() - u.min() + 1e-6)

class EvidFuseFixed(nn.Module):
    """ 输入每层 64 维，[B,64]×L → [B,dim_out]；权重=softmax(-û/τ) """
    def __init__(self, n_levels=4, in_dim=1, dim_out=512, tau=0.5, w_floor=0.1):
        super().__init__()
        self.tau = tau; self.w_floor = w_floor
        self.proj = nn.ModuleList([nn.Linear(in_dim, dim_out) for _ in range(n_levels)])
        for p in self.proj:
            nn.init.kaiming_normal_(p.weight, nonlinearity='linear')
            nn.init.zeros_(p.bias)

    def forward(self, vec_list, evid_list):
        B = vec_list[0].size(0); dev = vec_list[0].device
        projed = [p(v) for p, v in zip(self.proj, vec_list)]  # -> [B,dim_out]*L
        with torch.no_grad():
            us = []
            for e in evid_list:
                u = _u_from_evid(e)
                if u is None: u = torch.zeros(B,1, device=dev)
                us.append(u)
            u_stack = torch.cat(us, dim=1)                 # [B,L]
            w = F.softmax(-u_stack / self.tau, dim=1)      # [B,L]
            if self.w_floor > 0:
                w = self.w_floor + (1 - self.w_floor * w.size(1)) * w
                w = w / w.sum(dim=1, keepdim=True)
        fused = 0
        for i, v in enumerate(projed):
            fused = fused + w[:, i:i+1] * v
        return fused, w.detach()

# class DirichletHead(nn.Module):
#     def __init__(self, in_dim, num_classes):
#         super().__init__()
#         self.fc = nn.Linear(in_dim, num_classes)
#
#     def forward(self, feats):
#         logits = self.fc(feats)
#         evidence = F.softplus(logits)
#         alpha = evidence + 1.0
#         S = alpha.sum(dim=-1, keepdim=True)
#         p = alpha / S
#         M = alpha.size(-1)
#         u = M / S   # 不确定性
#         return {"logits": logits, "alpha": alpha, "p": p, "uncertainty": u, "strength": S}
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DirichletHead(nn.Module):
    """
    返回:
      logits: [B,M]
      alpha:  [B,M]
      p:      [B,M]
      uncertainty (u): [B,1]
      epistemic_mi (mi): [B,1]
      strength (S): [B,1]
    """
    def __init__(self, in_dim: int = None, num_classes: int = 2,
                 init_T: float = 1.0, learnable_T: bool = True, lazy: bool = False):
        super().__init__()
        self.num_classes = int(num_classes)
        self.lazy = bool(lazy or in_dim is None)
        if self.lazy:
            self.fc = None  # 首次 forward 根据输入维度自动创建
        else:
            self.fc = nn.Linear(int(in_dim), self.num_classes)
        self.log_T = nn.Parameter(torch.log(torch.tensor(init_T))) if learnable_T else None

    def _ensure_2d(self, feats: torch.Tensor) -> torch.Tensor:
        # 接受 [B,C] 或 [B,C,1,1] 或 [B,C,H,W]；统一转成 [B,C]
        if feats.dim() == 2:
            return feats
        if feats.dim() == 4:
            if feats.size(-1) == 1 and feats.size(-2) == 1:
                return feats.flatten(1)                   # [B,C,1,1] -> [B,C]
            # 若是一般 [B,C,H,W]，做 GAP 再 flatten
            return F.adaptive_avg_pool2d(feats, 1).flatten(1)
        raise ValueError(f"DirichletHead expects 2D/4D, got {feats.shape}")

    def _build_if_needed(self, in_dim: int, device):
        if self.fc is None:
            self.fc = nn.Linear(in_dim, self.num_classes).to(device)
            nn.init.kaiming_normal_(self.fc.weight, nonlinearity='linear')
            nn.init.zeros_(self.fc.bias)

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        if self.log_T is None:
            return logits
        T = torch.exp(self.log_T).clamp(0.5, 10.0)
        return logits / T

    @staticmethod
    def _evidential_metrics(alpha: torch.Tensor, eps: float = 1e-8):
        S = alpha.sum(dim=-1, keepdim=True)                    # [B,1]
        p = alpha / S                                          # [B,M]
        pred_entropy = -(p * (p.clamp_min(eps)).log()).sum(dim=-1, keepdim=True)
        e_log_p = torch.digamma(alpha) - torch.digamma(S)      # [B,M]
        exp_entropy = -(p * e_log_p).sum(dim=-1, keepdim=True)
        mi = (pred_entropy - exp_entropy).clamp_min(0.0)       # [B,1]
        M = alpha.size(-1)
        u = (M / S).clamp_min(0.0)                             # [B,1]
        return S, p, u, mi

    def forward(self, feats: torch.Tensor, detach_input: bool = False):
        x = self._ensure_2d(feats)                # -> [B,C]
        if detach_input:
            x = x.detach()

        if self.lazy:
            self._build_if_needed(x.size(1), x.device)
        else:
            # 严格检查，第一时间暴露维度错误
            assert x.size(1) == self.fc.in_features, \
                f"DirichletHead in_features={self.fc.in_features}, got {x.size(1)}"

        logits = self.fc(x)                        # [B,M]
        logits = self._apply_temperature(logits)   # 温度缩放
        evidence = F.softplus(logits)              # e>=0
        alpha = evidence + 1.0                     # [B,M]
        S, p, u, mi = self._evidential_metrics(alpha)
        return {
            "logits": logits, "alpha": alpha, "p": p,
            "uncertainty": u, "epistemic_mi": mi, "strength": S
        }

import torch
import torch.nn as nn
import torch.nn.functional as F

# 通用：把任意张量变成 [B, C]（通道向量）
def vec_from_any(x):
    if x.dim() == 4:   # [B,C,H,W]
        return F.adaptive_avg_pool2d(x, 1).flatten(1)
    if x.dim() == 3:   # [B,C,L]
        return F.adaptive_avg_pool1d(x, 1).flatten(1)
    if x.dim() == 2:   # [B,C]
        return x
    raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

# 懒初始化的线性适配器：第一次看到输入维度时再建 Linear(C_in -> out_dim)
class LinearAdaptor(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj = None

    def _build(self, in_dim, device):
        self.proj = nn.Linear(in_dim, self.out_dim).to(device)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):                 # x 可是 [B,C] 或 [B,C,H,W] 等
        x = vec_from_any(x)              # 统一成 [B,C_in]
        if self.proj is None:
            self._build(x.size(1), x.device)
        return self.proj(x)              # -> [B,out_dim]


class ResnetModel(nn.Module):
    """ This is a subclass from nn.Module that creates a pre-trained resnet50 model.

    Attributes:
        cnn: The convolutional network, a pretrained resnet50.
        fc_meta_data: A fully connected network.
        classifier: The last layer in the network.
    """

    def __init__(self, resnet='resnest50',num_classes=2, pretrained=False, mixstyle_layers=[],random_type=None, p=0.5):
        """ The __init__ function.
        Args:
            n_columns: the number of columns in data (the meta data).
        """
        super(ResnetModel, self).__init__()
        # 新增部分
        # self.gamma_u = 1.0  # 门控指数，可调 0.5~2.0
        self.dir_head_u = nn.Linear(2048, 2)  # x5: [B,2048,H,W] -> GAP -> [B,2048]
        # 新增部分end

        self.mixstyle_layers = mixstyle_layers

        self.num_classes = num_classes

        self.p = p

        if mixstyle_layers:
            if random_type == 'TriD':#ACC:0.8525 AUC:0.8773
                self.random = TriD(p=p)
            elif random_type == 'MixStyle':
                self.random = MixStyle(p=p, mix='random')
            elif random_type == 'EFDMixStyle':
                self.random = EFDMix(p=p, mix='random')
            elif random_type == 'DomainClassMixAugmentation':
                # self.domain_classifier = DomainLearner(self.feature_dim, 4)
                # self.domain_classifier = DomainLearner(self.feature_dim, 4)

                # 定义超参数
                hparams = {
                    "threshold": 0.7,
                    "threshold_lower_bound": 0.3,
                    "value_to_change": 0.01,
                    "step_to_change": 1000,
                    "probability_to_discard": 0.1
                }

                # 初始化域增强模块 - 需要传入必要的参数
                self.random = DomainClassMixAugmentation(
                    batch_size=12,  # 例如 32
                    num_classes=2,  # 例如 2（青光眼分类）
                    num_domains=4,  # 您的4个域
                    hparams=hparams
                )
            else:
                raise ValueError('The random method type is wrong!')
            print(random_type)
            print('Insert Random Style after the following layers: {}'.format(mixstyle_layers))
        # res2net预训练好的模型
        # self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # resnext预训练好的模型
        image = 3
        self.weight = extract_features(image)
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv_feature1 = nn.Conv2d(18, 1, kernel_size=1, stride=1, padding=1,
                                       bias=False)
        self.conv_feature2 = nn.Conv2d(18, 1, kernel_size=3, stride=1, padding=2,
                                       bias=False)
        self.conv_feature3 = nn.Conv2d(18, 1, kernel_size=5, stride=1, padding=3,
                                       bias=False)
        self.resnet = resnest50(pretrained=True)

        # cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
        # self.resize = transforms.Resize([224,224])
        self.conv4_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=1,
                               bias=False)
        self.conv4_2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=2,
                                 bias=False)
        self.conv4_3 = nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=3,
                                 bias=False)
        self.acb1 = ACBBlock(128, 64)
        self.bn4 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.conv5_1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=1,
                               bias=False)
        self.conv5_2 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=2,
                               bias=False)
        self.conv5_3 = nn.Conv2d(512, 64, kernel_size=5, stride=1, padding=3,
                               bias=False)
        self.acb2 = ACBBlock(64, 64)
        self.bn5 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv6_1 = nn.Conv2d(1024, 64, kernel_size=1, stride=2, padding=1,
                               bias=False)
        self.conv6_2 = nn.Conv2d(1024, 64, kernel_size=3, stride=2, padding=2,
                               bias=False)
        self.conv6_3 = nn.Conv2d(1024, 64, kernel_size=5, stride=2, padding=3,
                               bias=False)

        self.bn6 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        # self.avg6 = nn.AvgPool2d(3, stride=2)
        self.avg6 = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = norm_layer(1024)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = norm_layer(512)
        self.conv3_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=1,
                                 bias=False)
        self.bn3_1 = norm_layer(64)
        self.conv3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.bn3_2 = norm_layer(128)
        self.conv3_3 = nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.bn3_3 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)
        self.trans1 = VisionTransformer()
        self.conv1_1= nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=1,  # kernel_size=7, stride=2, padding=3,

                             bias=False)
        self.conv1_2= nn.Conv2d(2048, 64, kernel_size=3, stride=1, padding=2,  # kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_3 = nn.Conv2d(2048, 64, kernel_size=5, stride=1, padding=3,  # kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.acb4 = ACBBlock(1024, 64)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(2048, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn7 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn8 = norm_layer(3)
        # self.relu = nn.ReLU(inplace=True)
        self.inplanes = 64
        self.k_size = 3
        #加入CBAM模块
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
        #加入置信度反馈模块,使用多层嵌套方法对于主干网络每层使用置信度打分机制，要清楚每层通道数ConfidenceGuidedChannelSpatialModulation

        self.cgcfm1 = ConfidenceGuidedCSM_FromEvid(64,t_ch=0.8, t_sp=0.8, topB_ch=64,  drop_ratio_ch=0.5,
        topB_sp=256, drop_ratio_sp=0.3,
        use_alpha_scale=True, beta=1.0)#use_alpha_scale=False,软硬门控用这个调整
        self.cgcfm2 = ConfidenceGuidedChannelSpatialEvidential(256)
        self.cgcfm3 = ConfidenceGuidedChannelSpatialEvidential(512)
        self.cgcfm4 = ConfidenceGuidedCSM_FromEvid(64,t_ch=0.8, t_sp=0.8, topB_ch=64,  drop_ratio_ch=0.5,
        topB_sp=256, drop_ratio_sp=0.3,
        use_alpha_scale=True, beta=1.0)
        self.cgcfm5 = ConfidenceGuidedChannelSpatialEvidential(2048)
        self.dir_head = DirichletHead(in_dim=None, num_classes=2, init_T=1.0, learnable_T=True)
        self.scsa = SCSA(dim=64, head_num=4)
        self.csha = CSHA(64)
        self.chws = CHWS(64)
        # self.dir_head = DirichletHead(in_dim=64, num_classes=2)
        # self.inplanes2 = 256
        # self.k_size = 3
        #
        # self.ca2 = ChannelAttention(self.inplanes2)
        # self.sa2 = SpatialAttention()
        # self.inplanes3 = 512
        # self.ca3 = ChannelAttention(self.inplanes3)
        # self.sa3 = SpatialAttention()
        # self.inplanes4 = 1024
        # self.ca4 = ChannelAttention(self.inplanes4)
        # self.sa4 = SpatialAttention()
        # self.inplanes1 = 2048
        # self.ca1 = ChannelAttention(self.inplanes1)
        # self.sa1 = SpatialAttention()
        self.aff = AFF()
        self.iaff = iAFF()
        self.faff = FAFF()

        self.feature_dim_1 = 64
        self.RMCSAM_1 = RMCSAM(self.feature_dim_1)
        self.feature_dim_2 = 128
        self.RMCSAM_2 = RMCSAM(self.feature_dim_2)
        self.feature_dim_3 = 256
        self.RMCSAM_3 = RMCSAM(self.feature_dim_3)
        self.feature_dim_4 = 512
        self.RMCSAM_4 = RMCSAM(self.feature_dim_4)
        self.feature_dim = 64
        self.RMCSAM = RMCSAM(self.feature_dim)  # (先点这一行最前面输入#)
        # self.RMCSAM = RMCSAM_CBAM(self.feature_dim)  #(调整一下这一行，删除这一行最前面的#)
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.4)  # dropout训练
        self.fc2 = nn.Linear(4288, 4096)
        num_channels = 64
        self.MB1 = MBPOOL(num_channels)
        self.MC = MultiModalFusionModule(num_channels)

        self.depth = DepthwiseSeparableConv(192, 64, 3, 1, 1)
        self.depth1 = DepthwiseSeparableConv(192,64,3,1,1)
        self.depth2 = DepthwiseSeparableConv(256, 64, 3, 1, 1)
        self.depth3 = DepthwiseSeparableConv(512, 64, 3, 1, 1)
        self.depth4 = DepthwiseSeparableConv(1024, 64, 3, 1, 1)
        # self.MB2 = MBPOOL(num_channels)
        # self.MB3 = MBPOOL(num_channels)
        # self.MB4 = MBPOOL(num_channels)
        # self.fc3 = nn.Linear(1000, 512)

        # self.classifier = nn.Sequential(nn.Linear(1000 + 250, 1))
        # self.fc = nn.Linear(16384, 256)
        # self.reduce_dim = nn.Linear(335872, 1024)
        self.fuser =EvidFuseFixed(n_levels=4, in_dim=64, dim_out=512, tau=1.5, w_floor=0.1)
        self.target_dim = 64
        self.dir_head = DirichletHead(in_dim=self.target_dim, num_classes=2, init_T=1.0, learnable_T=True)

        # 四个分支都用懒适配到 64（不需要知道 C_img/C2/C3/C4）
        self.adapt_img = LinearAdaptor(self.target_dim)
        self.adapt_y2 = LinearAdaptor(self.target_dim)
        self.adapt_y3 = LinearAdaptor(self.target_dim)
        self.adapt_y4 = LinearAdaptor(self.target_dim)
        self.fused_dim = 512
        self.res_evid_fuser = ResidualEvidFuser(
            in_ch_list=[64, 64, 64, 64],
            out_ch=64 * 4,  # 和 cat([..., ..., ..., ...], 1) 的通道数一致，才能相加
            tau=2.0,  # 前期更“软”
            beta=0.2,  # 证据映射强度适中
            w_floor=0.2,  # 保底权重，避免早期某一路为 0
            gamma_init=0.2  # 残差强度起步小，后面可慢慢调大
        )
        self.out_ch = self.num_classes
        self.cls_head = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # <- 这里改成 1
        )
        self.classifier = nn.Sequential(nn.Linear(256, 1), nn.Dropout(0.5))#221952,3504384,avg:861184
        # self.z = nn.BatchNorm1d(num_features=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        gamma_list = [0.3, 0.5, 0.7,0.9,1.1]
        # gate_module = GateWithUncertainty(gamma_list)
        self.gate_u = GateWithUncertainty(gamma_list,)
        # self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        #
        # self.trans_conv = nn.ConvTranspose2d(192, 64, kernel_size=4, stride=4, padding=0)

        self.intermediate_outputs = {}

    def DS_Combin_two(self, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))

            u[v] = self.num_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.num_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a
    # def forward(self, image, data):
    @torch.no_grad()
    def set_text_proto(self, proto: torch.Tensor):
        proto = proto.to(self.text_proto.device)
        assert proto.dim() == 2 and proto.size(0) == 2
        assert proto.size(1) == self.prompt_dim
        self.text_proto.copy_(proto)
        self.text_ready = True
    def forward(self, x, is_train):
        # 把torch.size(4,4,3,240,240)变成(4,3,240,240)
        # x = x[:, 0, 0, :, :, :]


        # print(x.size())
        # x = torch.tensor(x)
        # x = self.weight(x)
        # x = torch.tensor(x)
        # x = x.to(device)
        # xf1 = self.conv_feature1(x)
        # xf2 = self.conv_feature2(x)
        # xf3 = self.conv_feature3(x)  # x.size[4, 64, 320, 320]
        # print(xf1.size())
        # print(xf2.size())
        # print(xf3.size())
        # x= torch.cat((xf1,xf2,xf3), dim=1)
        # x = torch.add(x, xf3)
        # print('x')
        # print('x.size()75',x.size())
        sfs = []
        if random.random() > 0.5:
            data_aug = True
        else:
            data_aug = False
        # data_aug = True
        # # max_x1 = torch.max(x)
        # # print(max_x1)
        # mix1 = None

        if is_train:
            if 'layer0' in self.mixstyle_layers and data_aug ==True:
                mix1 = self.random(x)
                # mix1 = self.random(x, y, domain, class_gradient, domain_gradient)
                # print("mix1", mix1)
                sfs.append(mix1)
                # z=self.conv(x)
                x1 = self.resnet.conv1(mix1)
                # x1, conf1 = self.cgcfm1(x1)
            else:
                x1 = self.resnet.conv1(x)
                # x1, conf1 = self.cgcfm1(x1)
        #
        #
        else:
            sfs=[]
            mix1=None
            x1 = self.resnet.conv1(x)
            # x1, conf1 = self.cgcfm1(x1)

        # x1 = self.resnet.conv1(x)
        x = self.resnet.bn1(x1)  # x.size()78 torch.Size([4, 64, 320, 320])
        x = self.resnet.relu(x)

        x1 = self.resnet.maxpool(x)  # x1.size()83 torch.Size([4, 64, 160, 160])

        if is_train:
            if 'layer0' in self.mixstyle_layers and data_aug == False:
                mix2 = self.random(x1)
                # mix2 = self.random(x1, y, domain, class_gradient, domain_gradient)
                # print("mix2",mix2)
                sfs.append(mix2)
                x2 = self.resnet.layer1(mix2)
                # x2, conf2 = self.cgcfm2(x2)
            else:
                x2 = self.resnet.layer1(x1)
                # x2, conf2= self.cgcfm2(x2)
        else:
            sfs=[]
            mix2=None
            x2 = self.resnet.layer1(x1)

            # x2, conf2 = self.cgcfm2(x2)
        x3 = self.resnet.layer2(x2)  # x3.size()88 torch.Size([4, 512, 80, 80])
        # x3, conf3 = self.cgcfm3(x3)
        x4 = self.resnet.layer3(x3)  # x4.size()90 torch.Size([4, 1024, 40, 40])
        # x4, conf4 = self.cgcfm4(x4)
        x5 = self.resnet.layer4(x4)  # x5.size()92 torch.Size([4, 2048, 20, 20])
        # x5, conf5 = self.cgcfm5(x5)
        # 新增内容8.30
        gp5 = F.adaptive_avg_pool2d(x5, 1).view(x5.size(0), -1)  # [B,2048]
        dir_logits_u = self.dir_head_u(gp5)  # [B,num_classes]
        evidence_u = F.softplus(dir_logits_u)  # e >= 0
        alpha_u = evidence_u + 1.0  # α = e + 1
        S_u = alpha_u.sum(dim=-1, keepdim=True)  # [B,1]
        p_u = alpha_u / S_u  # [B,M]
        M = alpha_u.size(-1)
        u = M / S_u  # [B,1] 不确定性
        u_det_y2 = u.detach()  # 门控时先阻断梯度，稳定训练
        u_det_y3 = u.detach()
        u_det_y4 = u.detach()
        u_det_img = u.detach()
        # # 打包（作为整体模型输出的一部分返回）
        # dir_out = {"logits": dir_logits_u, "alpha": alpha_u, "p": p_u, "uncertainty": u, "strength": S_u}
        # 新增内容end
        # 多尺度拼接特征
        # y2_1 = self.conv4_1(x2)
        y2_2 = self.conv4_2(x2)
        y2_raw = y2_2
        self.intermediate_outputs['y2_raw'] = y2_raw.detach().clone()
        # y2_3 = self.conv4_3(x2)
        #
        y2 = self.relu(y2_2)  # size(4,64,60,60)
        y2 = self.bn4(y2)
        # y2, conf1 = self.cgcfm1(y2)

        # networks/ResUnet_trid.py 里
        # 假设 y2_raw: [B, C, H, W]
        feats = F.adaptive_avg_pool2d(y2_raw, output_size=1).flatten(1)  # [B, C]
        evid_y2 = self.dir_head(feats)
        y2_att, conf_y2, cw_y2, sw_y2, dropped_ch, dropped_sp = self.cgcfm1(y2, evid_y2)
        y2 = self.gate_u(y2_att, y2_raw, u_det_y2)
        # self.intermediate_outputs['y2'] = y2.detach().clone()
        # y2, conf4,(a_c, b_c, a_s, b_s) = self.cgcfm4(evid_y2)

        # y2 = self.ca(y2_2) * y2
        # y2 = self.sa(y2) * y2


        # y2 = self.scsa(y2)
        # y2 = self.csha(y2)#FRECSA
        # y2 = self.chws(y2)
        output = y2
        # output1 = y2
        # self.intermediate_outputs['output1'] = output1.detach().clone()
        y2 = self.avg6(y2)
        # print('y2conv shape:', y2.size())
        y2 = torch.flatten(y2, 1)

        y2 = self.dropout(y2)

        y3_2 = self.conv5_2(x3)

        y3 = self.relu(y3_2)  # size(4,64,30,30)
        y3 = self.bn5(y3)
        # y3, conf4 = self.cgcfm4(y3)
        y3_raw=y3
        feats = F.adaptive_avg_pool2d(y3_raw, output_size=1).flatten(1)  # [B, C]
        evid_y3 = self.dir_head(feats)
        y3_att, conf_y3, cw_y3, sw_y3, dropped_ch, dropped_sp = self.cgcfm1(y3, evid_y3)
        # y3_att, conf_y3, cw_y3, sw_y3  = self.cgcfm4(y3)
        # # print("u_det_y3:", u_det_y3)
        y3 = self.gate_u(y3_att, y3_raw, u_det_y3)

        # y3 = self.ca(y3) * y3
        # y3 = self.sa(y3) * y3
        # y3, conf2 = self.cgcfm2(y3)
        # y3 = self.scsa(y3)
        # y3 = self.csha(y3)
        # y3 = self.chws(y3)
        y3 = self.avg6(y3)
        # print('y3conv shape:', y3.size())
        y3 = torch.flatten(y3, 1)

        y3 = self.dropout(y3)

        y4_2 = self.conv6_2(x4)


        y4 = self.relu(y4_2)  # size(4,64,8,8)
        y4 = self.bn6(y4)
        # y4, conf4 = self.cgcfm4(y4)
        y4_raw = y4
        feats = F.adaptive_avg_pool2d(y4_raw, output_size=1).flatten(1)
        evid_y4 = self.dir_head(feats)
        y4_att, conf_y4, cw_y4, sw_y4, dropped_ch, dropped_sp = self.cgcfm1(y4, evid_y4)
        # # y4_att, conf_y4, cw_y4, sw_y4  = self.cgcfm4(y4)
        # # print("u_det_y4:", u_det_y4)
        y4 = self.gate_u(y4_att, y4_raw, u_det_y4)

        # y4 = self.ca(y4) * y4
        # y4 = self.sa(y4) * y4
        # y4, conf3 = self.cgcfm3(y4)
        # y4 = self.scsa(y4)
        # y4 = self.csha(y4)
        # y4 = self.chws(y4)
        y4 = self.avg6(y4)
        # print('y4conv shape:', y4.size())
        y4 = torch.flatten(y4, 1)

        y4 = self.dropout(y4)



        image_2 = self.conv1_2(x5)
        # image_3 = self.conv1_3(x5)
        #
        #
        image = self.relu(image_2)
        image = self.bn1(image)
        # image, conf4 = self.cgcfm4(image)
        image_raw = image
        feats = F.adaptive_avg_pool2d(image_raw, output_size=1).flatten(1)
        evid_image = self.dir_head(feats)
        # # self.intermediate_outputs['image_raw'] = image_raw.detach().clone()
        image_att, conf_image, cw_image, sw_image, dropped_ch, dropped_sp = self.cgcfm1(image, evid_image)

        # print("u_det_y2:",u_det_y2)
        # y2 = self.gate_u(y2_att, y2_raw, u_det_y2)
        # self.intermediate_outputs['image_att'] = image_att.detach().clone()
        # image_att, conf_img, cw_img, sw_img  = self.cgcfm4(image)
        # print("u_det_img:", u_det_img)
        #     image = self.gate_u(image_att, image_raw, u_det_img )

        # image = self.ca(image) * image
        # image = self.sa(image) * image
        # image, conf4 = self.cgcfm4(image_att)
        # image = self.scsa(image)
        # image = self.csha(image)
        # image = self.chws(image)
        output1 = image
        self.intermediate_outputs['output1'] = output1.detach().clone()


        image = self.avg6(image)
        # print('imageconv shape:', image.size())
        image = torch.flatten(image, 1)
        image = self.dropout(image)

#按照插值操作来拼接
        # y2 = F.interpolate(y2, size=(8, 8), mode='bilinear', align_corners=False)
        # #
        # image = torch.nn.functional.interpolate(image, size=(28, 28), mode='bilinear',align_corners=False)
        # y3 = torch.nn.functional.interpolate(y3, size=(28, 28), mode='bilinear', align_corners=False)
        # y4 = torch.nn.functional.interpolate(y4, size=(28, 28), mode='bilinear',align_corners=False)
        #
        # y2 = y2[:, :4096]
        # y3 = y3[:, :4096]
        # y4 = y4[:, :4096]
        # 直接把“特征图或向量”喂给 adaptor，它内部会 GAP 成 [B,C] 再投到 64


        # img64 = self.adapt_img(image_evid)  # [B,64]
        # y2_64 = self.adapt_y2(y2_evid)  # [B,64]
        # y3_64 = self.adapt_y3(y3_evid)  # [B,64]
        # y4_64 = self.adapt_y4(y4_evid)  # [B,64]
        #
        # # evid 与 fuser 都用“同一套 64 向量”
        # evid_image = self.dir_head(img64, detach_input=True)
        # evid_y2 = self.dir_head(y2_64, detach_input=True)
        # evid_y3 = self.dir_head(y3_64, detach_input=True)
        # evid_y4 = self.dir_head(y4_64, detach_input=True)
        # evid_f1 = self.DS_Combin_two(evid_y2["alpha"],evid_y3["alpha"])
        # evid_f2 = self.DS_Combin_two(evid_f1, evid_y4["alpha"])
        # evid_f3 = self.DS_Combin_two(evid_f2, evid_image["alpha"])
        # alpha = evid_f3["alpha"]  # [B, num_classes]
        # S = alpha.sum(dim=1, keepdim=True)  # 证据强度
        # M = alpha.size(1)  # 类别数
        # u = M / S  # [B,1] 不确定度

        # fused_map, w_levels = self.res_evid_fuser(
        #     [image, y2, y3, y4],
        #     [evid_image, evid_y2, evid_y3, evid_y4]
        # )  # fused_map: [B, 256, H, W]
        #
        # # 分类向量
        # fused_vec = F.adaptive_avg_pool2d(fused_map, 1).flatten(1)  # [B,256]
        # x = self.cls_head(fused_vec)  # [B,1]（BCE）

        # # …前面省略：img64, y2_64, y3_64, y4_64 的计算，以及 evid 的计算…
        #
        # fused_vec, w_levels = self.fuser(
        #     [img64, y2_64, y3_64, y4_64],
        #     [evid_image, evid_y2, evid_y3, evid_y4]
        # )  # fused_vec: [B, self.fused_dim]
        #
        # assert fused_vec.dim() == 2 and fused_vec.size(1) == self.fused_dim
        # x = self.cls_head(fused_vec)
        # output=w_levels


        classtoken1 = torch.cat([y2, image], dim=1)
        classtoken2 = torch.cat([classtoken1, y3], 1)
        classtoken3 = torch.cat([classtoken2, y4], 1)
        # z = torch.cat([image, y2, y3, y4], dim=1)  # 主通路
        # h = proj(z)  # 1x1 conv or Linear 到 target_dim
        # g = gate_from_evidence(evid)  # 映射到 [0,1]，tau 高一点
        # fused = z + gamma * (g * h)  # 残差加性注入，gamma 从 0.0→1.0 逐步升
        # logits = head(GAP(fused))

        # x = self.classifier(F.adaptive_avg_pool2d(classtoken3, (1, 1)).squeeze())
        # output = classtoken3

        classtoken3 = torch.flatten(classtoken3, 1)

        logits_txt = None
        if self.text_ready:
            z = F.normalize(self.img_prompt_proj(classtoken3), dim=-1)  # [B, D]
            t = F.normalize(self.text_proto, dim=-1)  # [2, D]
            scale = self.logit_scale.exp().clamp(1.0, 100.0)
            logits_txt = scale * (z @ t.t())  # [B, 2]

        x = self.classifier(classtoken3)
        # image=torch.flatten(image, 1)
        # feat = F.adaptive_avg_pool2d(image, 1).flatten(1)
        x = self.classifier(classtoken3)
        # prune.l1_unstructured(linear, name='weight', amount=0.5)
        # if data_aug == True:
        #     output= mix1
        # else:

        # print(f"conf1 ty+pe: {type(conf2)}")
        conf_dict = {
            "logits_txt": logits_txt,
            # "alpha_img": evid_image["alpha"],  # [B,2]
            # "alpha_y2": evid_y2["alpha"],  # [B,2]
            # "alpha_y3": evid_y3["alpha"],  # [B,2]
            # "alpha_y4": evid_y4["alpha"],  # [B,2]
            # "evid_f3":evid_f3,
            "y2_raw":y2_raw,
            # "image_raw":image_raw,
            # "image_att":image_att,
            # "classtoken3":classtoken3,
            "output1":output1
        }

        if is_train:
            return x, output, None, conf_dict  # 训练时多输出不确定性
        else:
            return x, output, None, conf_dict



