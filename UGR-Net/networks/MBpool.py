import torch
from torch import nn
import torch.nn.functional as F
# 定义ACB卷积模块
class ACBConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ACBConvolution, self).__init__()

        # 定义卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

        # 定义注意力模块
        self.attention = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.Sigmoid()
        )

    def forward(self, x):
        # 卷积操作
        conv_output = self.conv(x)

        # 计算注意力权重
        attention_weight = self.attention(conv_output)

         # 加权特征
        acb_output = conv_output * attention_weight

        return acb_output
class MBPOOL(nn.Module):
    def __init__(self, num_channels):
        super(MBPOOL, self).__init__()
        # 获取特征图尺寸
        # height, width, num_channels = feature_map.shape
        norm_layer = nn.BatchNorm2d
        # 卷积核1模块
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.avg1 = nn.AvgPool2d(3,stride=2)
        self.avg2 = nn.AvgPool2d(4,stride=2)
        # 卷积核2模块
        self.conv2 = nn.Conv2d(num_channels, 64, kernel_size=1, stride=2, padding=1,
                               bias=False)
        # self.ACB1 = ACBConvolution(48,32,3)
        self.bn2 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.avg3 = nn.AvgPool2d(6, stride=2)
        # 卷积核2模块
        self.conv3 = nn.Conv2d(num_channels, 64, kernel_size=1, stride=2, padding=1,
                               bias=False)
        self.avg4 = nn.AvgPool2d(6, stride=2)


    def forward(self, x):
        output1 = self.conv1(x)
        output1 = self.bn1(output1)
        output1 = self.relu(output1)
        output1 = self.avg1(output1)
        output1 = self.avg2(output1)
        output2 = self.conv2(x)
        # output2 = self.ACB1(output2)
        output2 = self.bn2(output2)
        output2 = self.relu(output2)
        output2 = self.avg3(output2)
        output3 = self.conv3(x)
        output3 = self.avg4(output3)
        output1 = torch.tensor(output1)
        output2 = torch.tensor(output2)
        output3 = torch.tensor(output3)
        output1 = output1[:,:,:27,:27]
        # output2 = F.interpolate(output2, size=(28, 28), mode='bilinear',
        #                                  align_corners=False)
        # output3 = F.interpolate(output3, size=(28, 28), mode='bilinear',
        #                         align_corners=False)
        # print('output1:', output1.size())
        # print('output2:', output2.size())
        # print('output3:', output3.size())
        output = torch.cat((output1,output2,output3),dim=0)
        # output = torch.cat((output1,output3),dim=3)

        return output
