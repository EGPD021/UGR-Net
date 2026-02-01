import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceGuidedChannelSpatialEvidential(nn.Module):
    """
    通道-空间 Evidential 注意力：
      - 通道：p_c (Sigmoid) + s_c (Softplus) -> Beta(alpha_c, beta_c)
      - 空间：p_s (Sigmoid) + s_s (Softplus) -> Beta(alpha_s, beta_s)
    输出：
      x_att: 加权后的特征
      stats: 置信相关统计（期望/证据/不确定性）方便日志与可视化
    """
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()

        # 通道注意力：全局池化 -> 两个头（p与s）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(4, in_channels // reduction)
        self.mlp_p = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels), nn.Sigmoid()  # p_c \in (0,1)
        )
        self.mlp_s = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels), nn.Softplus() # s_c \in (0,+inf)
        )

        # 空间注意力：avg/max 池化拼接 -> 两个头（p与s）
        pad = spatial_kernel // 2
        self.spatial_p = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=pad),
            nn.Sigmoid()  # p_s \in (0,1)
        )
        self.spatial_s = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=pad),
            nn.Softplus() # s_s \in (0,+inf)
        )

    @staticmethod
    def beta_params(p, s):
        # p \in (0,1), s \in (0, +inf)
        alpha = p * s + 1.0
        beta  = (1.0 - p) * s + 1.0
        return alpha, beta

    @staticmethod
    def beta_total_evidence(alpha, beta):
        return alpha + beta

    @staticmethod
    def beta_uncertainty(alpha, beta):
        # 一个简单、单调的“不确定性”指标（证据越大越确定）
        return 2.0 / (alpha + beta)

    @staticmethod
    def beta_entropy(alpha, beta, eps=1e-8):
        # Beta 熵（可选）：H = log B(a,b) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b)
        from torch.special import digamma, gammaln
        a, b = alpha, beta
        ab = a + b
        return (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(ab)
                - (a-1)*digamma(a+eps) - (b-1)*digamma(b+eps)
                + (ab-2)*digamma(ab+eps))

    def forward(self, x):
        b, c, h, w = x.size()

        # ------- 通道 evidential ------- #
        y = self.avg_pool(x).view(b, c)
        p_c = self.mlp_p(y).view(b, c, 1, 1)
        s_c = self.mlp_s(y).view(b, c, 1, 1)
        a_c, b_c = self.beta_params(p_c, s_c)          # Beta 参数
        u_c = self.beta_uncertainty(a_c, b_c)          # 不确定性
        w_c = p_c                                      # 用期望作为门控（也可用 alpha/(alpha+beta) 等价）
        x_c = x * w_c

        # ------- 空间 evidential ------- #
        avg_out = torch.mean(x_c, dim=1, keepdim=True)
        max_out, _ = torch.max(x_c, dim=1, keepdim=True)
        s_in = torch.cat([avg_out, max_out], dim=1)

        p_s = self.spatial_p(s_in)                     # [B,1,H,W]
        s_s = self.spatial_s(s_in)
        a_s, b_s = self.beta_params(p_s, s_s)
        u_s = self.beta_uncertainty(a_s, b_s)
        w_s = p_s
        x_att = x_c * w_s

        # ------- 统计量（日志/可视化用） ------- #
        stats = {
            "p_channel_mean": p_c.mean(dim=[1,2,3]),   # [B]
            "p_spatial_mean": p_s.mean(dim=[1,2,3]),   # [B]
            "evi_channel":    (a_c+b_c).mean(dim=[1,2,3]),
            "evi_spatial":    (a_s+b_s).mean(dim=[1,2,3]),
            "unc_channel":    u_c.mean(dim=[1,2,3]),
            "unc_spatial":    u_s.mean(dim=[1,2,3]),
            # 也可返回熵，但一般统计量够用了
        }
        return x_att, stats, (a_c, b_c, a_s, b_s)
