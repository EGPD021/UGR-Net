import torch.nn as nn
import torch



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1#padding = 3修改为2

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class FAFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=2):
        super(FAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(channels, inter_channels),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class MultiModalFusionModule1(nn.Module):
    def __init__(self, num_modalities, num_channels):
        super(MultiModalFusionModule1, self).__init__()

        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_modalities, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input_modalities):
        # 输入模态的形状：(batch_size, num_modalities, channels, height, width)
        batch_size, num_modalities, channels, height, width = input_modalities.size()

        # 通道注意力融合
        channel_weights = self.channel_attention(input_modalities.view(-1, channels, height, width))
        channel_weights = channel_weights.view(batch_size, num_modalities, 1, 1, 1)
        fused_modalities = input_modalities * channel_weights

        # 空间注意力融合
        spatial_weights = self.spatial_attention(input_modalities.view(-1, num_modalities, height, width))
        spatial_weights = spatial_weights.view(batch_size, 1, num_modalities, height, width)
        fused_modalities = fused_modalities * spatial_weights

        # 将多模态数据沿通道维度进行求和
        fused_modalities = fused_modalities.sum(dim=2)

        return fused_modalities


import torch
import torch.nn as nn

class MultiModalFusionModule(nn.Module):
    def __init__(self, num_channels):
        super(MultiModalFusionModule, self).__init__()

        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input_gray, input_green,input_rgb):
        # 输入灰度图像的形状：(batch_size, 1, height, width)
        # 输入绿色通道的形状：(batch_size, 1, height, width)
        # 输入原始RGB图像的形状：(batch_size, 3, height, width)
        batch_size, _, height, width = input_gray.size()

        # 通道注意力融合
        fused_gray = input_gray * self.channel_attention(input_gray)
        fused_green = input_green * self.channel_attention(input_green)
        # 将数据类型转换为浮点类型
        input_rgb = input_rgb.float()
        fused_rgb = input_rgb * self.channel_attention(input_rgb)

        # 将多个模态数据融合
        fused_modalities = torch.add(fused_gray, fused_green, alpha=10 )
        fused_modalities = torch.add(fused_modalities, fused_rgb, alpha=10 )

        return fused_modalities
