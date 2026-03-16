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
 
    def __init__(self, in_ch_list, out_ch=None, tau=2.0, beta=0.2, w_floor=0.2, gamma_init=0.2):
        super().__init__()
        self.in_ch_list = list(in_ch_list)
        self.out_ch = int(out_ch if out_ch is not None else sum(self.in_ch_list))
        self.tau = float(tau)
        self.beta = float(beta)
        self.w_floor = float(w_floor)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)), requires_grad=False)  

        self.proj = nn.ModuleList([nn.Conv2d(c, self.out_ch, kernel_size=1, bias=False)
                                   for c in self.in_ch_list])
        for p in self.proj:
            nn.init.kaiming_normal_(p.weight, nonlinearity="linear")

    @torch.no_grad()
    def set_gamma(self, g: float):
        self.gamma.data.fill_(float(g))

    def _evid_to_weight(self, evid_list):

        scores = []
        for evid in evid_list:

            u_gate = evid.get("epistemic_mi", evid["uncertainty"])  # [B,1]

            u_min = u_gate.min()
            u_max = u_gate.max()
            u_hat = (u_gate - u_min) / (u_max - u_min + 1e-6)

            s = torch.pow((1.0 - u_hat).clamp(0, 1), self.beta)     # [B,1]
            scores.append(s)
        S = torch.cat(scores, dim=1)           
        w = torch.softmax(S / self.tau, dim=1)  
        w = torch.clamp(w, min=self.w_floor)    
        w = w / w.sum(dim=1, keepdim=True)     
        return w                                 

    def forward(self, xs, evid_list):

        assert len(xs) == len(self.in_ch_list) == len(evid_list)
        B, _, H, W = xs[0].shape

        z_base = torch.cat(xs, dim=1)  # [B, sum(C_i), H, W]

        w = self._evid_to_weight(evid_list)     # [B, n_levels]

        z_res = 0.0
        for i, (x_i, proj_i) in enumerate(zip(xs, self.proj)):
            z_i = proj_i(x_i)                   # [B, out_ch, H, W]
            wi  = w[:, i].view(B, 1, 1, 1)      # [B,1,1,1]
            z_res = z_res + wi * z_i
        z = z_base + self.gamma * z_res         
        return z, w                           

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, num_states):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(num_states, num_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        return self.softmax(self.fc(state))


num_actions = 2  
num_states = 10  
policy_network = PolicyNetwork(num_actions, num_states)
optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

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
    def __init__(self, gamma):
        super(GateWithUncertainty, self).__init__()
        self.gamma_values = gamma if isinstance(gamma, list) else [gamma]   
        self.current_gamma_idx = 0
        self.current_gamma = self.gamma_values[self.current_gamma_idx]
      
    def next_gamma(self):
        self.current_gamma_idx = (self.current_gamma_idx + 1) % len(self.gamma_values)
        self.current_gamma = self.gamma_values[self.current_gamma_idx]
        return self.current_gamma

    def set_gamma(self, gamma_idx):

        if 0 <= gamma_idx < len(self.gamma_values):
            self.current_gamma_idx = gamma_idx
            self.current_gamma = self.gamma_values[gamma_idx]
            print(f"设置 gamma 为: {self.current_gamma} (索引: {gamma_idx})")
        else:
            print(f"错误: 索引 {gamma_idx} 超出范围 [0, {len(self.gamma_values) - 1}]")

    def forward(self, x_att: torch.Tensor, x_raw: torch.Tensor, u: torch.Tensor = None):
       
        if u is None:
            return x_att

        kappa = torch.clamp((1.0 - u) ** self.current_gamma, 0.0, 1.0)  # [B,1]

        return kappa.view(-1, 1, 1, 1) * x_att + (1.0 - kappa).view(-1, 1, 1, 1) * x_raw
spatial_weight.detach()
import torch
import torch.nn as nn
import math


def dropout_topk_attention_1d(attn_1d: torch.Tensor, top_B: int, drop_ratio: float,
                              keep_sum: bool = True, eps: float = 1e-12):
   
    B, N = attn_1d.shape
    top_B = min(top_B, N)
    attn_new = attn_1d.clone()

    _, topk_idx = torch.topk(attn_new, k=top_B, dim=1, largest=True, sorted=False)

    k_drop = max(1, math.ceil(top_B * drop_ratio))
    dropped = []
    for b in range(B):
        perm = torch.randperm(top_B, device=attn_new.device)
        drop_idx = topk_idx[b, perm[:k_drop]]
        dropped.append(drop_idx)
        attn_new[b, drop_idx] = 0.0  

    if keep_sum:
        old_sum = attn_1d.sum(dim=1, keepdim=True).clamp_min(eps)
        new_sum = attn_new.sum(dim=1, keepdim=True).clamp_min(eps)
        attn_new = attn_new * (old_sum / new_sum)

    return attn_new.clamp(0.0, 1.0), dropped


class ConfidenceGuidedCSM_FromEvid(nn.Module):
    
    def __init__(self, in_channels, reduction=16,   
                 t_ch=0.8, t_sp=0.8,
                 topB_ch=8, drop_ratio_ch=0.0,
                 topB_sp=256, drop_ratio_sp=0.0,                 
                 use_alpha_scale=True, beta=0.3,
                 init_tau=1.5, 
                 use_dynamic_thresh=True, q_ch=0.7, q_sp=0.7):
        super().__init__()
        self.t_ch, self.t_sp = float(t_ch), float(t_sp)
        self.topB_ch, self.drop_ratio_ch = int(topB_ch), float(drop_ratio_ch)
        self.topB_sp, self.drop_ratio_sp = int(topB_sp), float(drop_ratio_sp)
        self.use_alpha_scale, self.beta = bool(use_alpha_scale), float(beta)
        self.use_dynamic_thresh = use_dynamic_thresh
        self.q_ch, self.q_sp = float(q_ch), float(q_sp)
        self.tau_ch = nn.Parameter(torch.tensor(init_tau))
        self.tau_sp = nn.Parameter(torch.tensor(init_tau))
        self.gap = nn.AdaptiveAvgPool2d(1)
        hid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, in_channels), nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _topb_dropout_1d(w_flat, topB, drop_ratio, keep_sum=True):
        
        B, L = w_flat.shape
        k = min(topB, L)
        if k <= 0 or drop_ratio <= 1e-6:
            return w_flat, [torch.empty(0, dtype=torch.long, device=w_flat.device) for _ in range(B)]
        w = w_flat.clone()
        dropped_idx_list = []
        for b in range(B):
            topk = torch.topk(w[b], k=k, largest=True).indices
            m = max(1, int(k * drop_ratio))
            drop_idx = topk[torch.randperm(k, device=w.device)[:m]]
            dropped_idx_list.append(drop_idx)
            w[b, drop_idx] = 0.0
            if keep_sum:                
                s_old = w_flat[b].sum().clamp_min(1e-6)
                s_new = w[b].sum().clamp_min(1e-6)
                w[b] = w[b] * (s_old / s_new)
        return w, dropped_idx_list

    @staticmethod
    def _percentile(x, q):      
        x_sorted, _ = torch.sort(x.view(-1))
        idx = max(0, min(int((len(x_sorted) - 1) * q), len(x_sorted) - 1))
        return x_sorted[idx].detach()

    def forward(self, x, evid):
        B, C, H, W = x.shape
        mi = evid.get("epistemic_mi", None)
        u = evid["uncertainty"]
        with torch.no_grad():
            u_gate = evid.get("epistemic_mi", evid["uncertainty"]).detach()  
            u_hat = (u_gate - u_gate.min()) / (u_gate.max() - u_gate.min() + 1e-6)
            scale = (1.0 - u_hat).clamp(0.0, 1.0) ** 0.3 
            scale = scale.clamp_min(0.5)      
        ch_vec = self.gap(x).view(B, C)        
        g_ch = self.mlp(ch_vec).view(B, C)  
        scale_ch = 1.0 + 0.3 * torch.tanh(g_ch / self.tau_ch)  # ∈ [0.7, 1.3]
        if self.use_alpha_scale:
            scale_ch = scale_ch * scale.view(B, 1).clamp_min(0.5)
        x_ch = x * scale_ch.view(B, C, 1, 1)
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
        return x_out,vis_data,None,None,None,None
       
def _u_from_evid(evid):
    if evid is None or not isinstance(evid, dict): return None
    u = evid.get("epistemic_mi", evid.get("uncertainty", None))
    if u is None: return None
    u = torch.as_tensor(u).float().view(-1,1)
    return (u - u.min()) / (u.max() - u.min() + 1e-6)

class EvidFuseFixed(nn.Module):    
    def __init__(self, n_levels=4, in_dim=1, dim_out=512, tau=0.5, w_floor=0.1):
        super().__init__()
        self.tau = tau; self.w_floor = w_floor
        self.proj = nn.ModuleList([nn.Linear(in_dim, dim_out) for _ in range(n_levels)])
        for p in self.proj:
            nn.init.kaiming_normal_(p.weight, nonlinearity='linear')
            nn.init.zeros_(p.bias)

    def forward(self, vec_list, evid_list):
        B = vec_list[0].size(0); dev = vec_list[0].device
        projed = [p(v) for p, v in zip(self.proj, vec_list)]  
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

class DirichletHead(nn.Module):    
    def __init__(self, in_dim: int = None, num_classes: int = 2,
                 init_T: float = 1.0, learnable_T: bool = True, lazy: bool = False):
        super().__init__()
        self.num_classes = int(num_classes)
        self.lazy = bool(lazy or in_dim is None)
        if self.lazy:
            self.fc = None  
        else:
            self.fc = nn.Linear(int(in_dim), self.num_classes)
        self.log_T = nn.Parameter(torch.log(torch.tensor(init_T))) if learnable_T else None

    def _ensure_2d(self, feats: torch.Tensor) -> torch.Tensor:        
        if feats.dim() == 2:
            return feats
        if feats.dim() == 4:
            if feats.size(-1) == 1 and feats.size(-2) == 1:
                return feats.flatten(1)                   
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
            assert x.size(1) == self.fc.in_features, \
                f"DirichletHead in_features={self.fc.in_features}, got {x.size(1)}"

        logits = self.fc(x)                        # [B,M]
        logits = self._apply_temperature(logits)   
        evidence = F.softplus(logits)              # e>=0
        alpha = evidence + 1.0                     # [B,M]
        S, p, u, mi = self._evidential_metrics(alpha)
        return {
            "logits": logits, "alpha": alpha, "p": p,
            "uncertainty": u, "epistemic_mi": mi, "strength": S
        }


def vec_from_any(x):
    if x.dim() == 4:   # [B,C,H,W]
        return F.adaptive_avg_pool2d(x, 1).flatten(1)
    if x.dim() == 3:   # [B,C,L]
        return F.adaptive_avg_pool1d(x, 1).flatten(1)
    if x.dim() == 2:   # [B,C]
        return x
    raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

class LinearAdaptor(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj = None

    def _build(self, in_dim, device):
        self.proj = nn.Linear(in_dim, self.out_dim).to(device)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):               
        x = vec_from_any(x)            
        if self.proj is None:
            self._build(x.size(1), x.device)
        return self.proj(x)            


class ResnetModel(nn.Module):
    def __init__(self, resnet='resnest50',num_classes=2, pretrained=False, mixstyle_layers=[],random_type=None, p=0.5):
        super(ResnetModel, self).__init__()
        
        self.dir_head_u = nn.Linear(2048, 2)  # x5: [B,2048,H,W] -> GAP -> [B,2048]
  

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
               
                hparams = {
                    "threshold": 0.7,
                    "threshold_lower_bound": 0.3,
                    "value_to_change": 0.01,
                    "step_to_change": 1000,
                    "probability_to_discard": 0.1
                }

        
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
        
        self.inplanes = 64
        self.k_size = 3
       
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
       

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
       
        self.fuser =EvidFuseFixed(n_levels=4, in_dim=64, dim_out=512, tau=1.5, w_floor=0.1)
        self.target_dim = 64
        self.dir_head = DirichletHead(in_dim=self.target_dim, num_classes=2, init_T=1.0, learnable_T=True)

        self.adapt_img = LinearAdaptor(self.target_dim)
        self.adapt_y2 = LinearAdaptor(self.target_dim)
        self.adapt_y3 = LinearAdaptor(self.target_dim)
        self.adapt_y4 = LinearAdaptor(self.target_dim)
        self.fused_dim = 512
        self.res_evid_fuser = ResidualEvidFuser(
            in_ch_list=[64, 64, 64, 64],
            out_ch=64 * 4,  
            tau=2.0,  
            beta=0.2,  
            w_floor=0.2,  
            gamma_init=0.2  
        )
        self.out_ch = self.num_classes
        self.cls_head = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  
        )
        self.classifier = nn.Sequential(nn.Linear(256, 1), nn.Dropout(0.5))#221952,3504384,avg:861184
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        gamma_list = [0.3, 0.5, 0.7,0.9,1.1]
       
        self.gate_u = GateWithUncertainty(gamma_list,)        
        self.intermediate_outputs = {}

    def DS_Combin_two(self, alpha1, alpha2):       
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))

            u[v] = self.num_classes / S[v]

        bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        K = bb_sum - bb_diag
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))

        S_a = self.num_classes / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

        sfs = []
        if random.random() > 0.5:
            data_aug = True
        else:
            data_aug = False
 
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
        else:
            sfs=[]
            mix1=None
            x1 = self.resnet.conv1(x)           
        x = self.resnet.bn1(x1)  # x.size()78 torch.Size([4, 64, 320, 320])
        x = self.resnet.relu(x)

        x1 = self.resnet.maxpool(x)  # x1.size()83 torch.Size([4, 64, 160, 160])

        if is_train:
            if 'layer0' in self.mixstyle_layers and data_aug == False:
                mix2 = self.random(x1)                
                sfs.append(mix2)
                x2 = self.resnet.layer1(mix2)               
            else:
                x2 = self.resnet.layer1(x1)             
        else:
            sfs=[]
            mix2=None
            x2 = self.resnet.layer1(x1)
           
        x3 = self.resnet.layer2(x2)  # x3.size()88 torch.Size([4, 512, 80, 80])
        
        x4 = self.resnet.layer3(x3)  # x4.size()90 torch.Size([4, 1024, 40, 40])
       
        x5 = self.resnet.layer4(x4)  # x5.size()92 torch.Size([4, 2048, 20, 20])
        
        gp5 = F.adaptive_avg_pool2d(x5, 1).view(x5.size(0), -1)  # [B,2048]
        dir_logits_u = self.dir_head_u(gp5)  # [B,num_classes]
        evidence_u = F.softplus(dir_logits_u)  # e >= 0
        alpha_u = evidence_u + 1.0  # α = e + 1
        S_u = alpha_u.sum(dim=-1, keepdim=True)  # [B,1]
        p_u = alpha_u / S_u  # [B,M]
        M = alpha_u.size(-1)
        u = M / S_u  
        u_det_y2 = u.detach()  
        u_det_y3 = u.detach()
        u_det_y4 = u.detach()
        u_det_img = u.detach()      
        y2_2 = self.conv4_2(x2)
        y2_raw = y2_2
        self.intermediate_outputs['y2_raw'] = y2_raw.detach().clone()        
        y2 = self.relu(y2_2)  
        y2 = self.bn4(y2)

        feats = F.adaptive_avg_pool2d(y2_raw, output_size=1).flatten(1)  # [B, C]
        evid_y2 = self.dir_head(feats)
        y2_att, conf_y2, cw_y2, sw_y2, dropped_ch, dropped_sp = self.cgcfm1(y2, evid_y2)
        y2 = self.gate_u(y2_att, y2_raw, u_det_y2) 
        output = y2        
        y2 = self.avg6(y2)
      
        y2 = torch.flatten(y2, 1)

        y2 = self.dropout(y2)

        y3_2 = self.conv5_2(x3)

        y3 = self.relu(y3_2)  # size(4,64,30,30)
        y3 = self.bn5(y3)       
        y3_raw=y3
        feats = F.adaptive_avg_pool2d(y3_raw, output_size=1).flatten(1)  # [B, C]
        evid_y3 = self.dir_head(feats)
        y3_att, conf_y3, cw_y3, sw_y3, dropped_ch, dropped_sp = self.cgcfm1(y3, evid_y3)        
        y3 = self.gate_u(y3_att, y3_raw, u_det_y3)
        y3 = self.avg6(y3)

        y3 = torch.flatten(y3, 1)

        y3 = self.dropout(y3)

        y4_2 = self.conv6_2(x4)


        y4 = self.relu(y4_2)  
        y4 = self.bn6(y4)
       
        y4_raw = y4
        feats = F.adaptive_avg_pool2d(y4_raw, output_size=1).flatten(1)
        evid_y4 = self.dir_head(feats)
        y4_att, conf_y4, cw_y4, sw_y4, dropped_ch, dropped_sp = self.cgcfm1(y4, evid_y4)
        
        y4 = self.gate_u(y4_att, y4_raw, u_det_y4)

        y4 = self.avg6(y4)
        y4 = torch.flatten(y4, 1)
        y4 = self.dropout(y4)
        image_2 = self.conv1_2(x5)        
        image = self.relu(image_2)
        image = self.bn1(image)
        image_raw = image
        feats = F.adaptive_avg_pool2d(image_raw, output_size=1).flatten(1)
        evid_image = self.dir_head(feats)        
        image_att, conf_image, cw_image, sw_image, dropped_ch, dropped_sp = self.cgcfm1(image, evid_image)
        output1 = image
        self.intermediate_outputs['output1'] = output1.detach().clone()
        image = self.avg6(image)       
        image = torch.flatten(image, 1)
        image = self.dropout(image)

        classtoken1 = torch.cat([y2, image], dim=1)
        classtoken2 = torch.cat([classtoken1, y3], 1)
        classtoken3 = torch.cat([classtoken2, y4], 1)
        classtoken3 = torch.flatten(classtoken3, 1)        
        x = self.classifier(classtoken3)       

        conf_dict = {
            "y2_raw":y2_raw,          
            "output1":output1
        }

        if is_train:
            return x, output, None, conf_dict  
        else:
            return x, output, None, conf_dict



