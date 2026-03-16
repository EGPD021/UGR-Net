import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.transforms import (Compose, ToTensor, Normalize, CenterCrop,
                                    RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective,ColorJitter)

# from sklearn.linear_model import LogisticRegression
# from sklearn import svm
# from dataset import MelanomaDataset
from networks.resnest import resnest50

# train_img_path = 'C:/Users/abc/Desktop/tangyilin/erfenlei/data/train_re/'
# data_train = pd.read_csv('C:/Users/abc/Desktop/tangyilin/erfenlei/data/train_youwu.csv',dtype='object'
#                              )
# transform_train = Compose([
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# dataset_train = MelanomaDataset(data_train, train_img_path, transform=transform_train)
# training_loader = DataLoader(dataset_train, batch_size=2 , shuffle=True)
# X_train = []
# y_train = []

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#ACB卷积块
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ACBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=3):
        super(ACBBlock, self).__init__()

        # 瓶颈结构
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 瓶颈结构
        out = self.bottleneck(x)

        # 注意力机制
        attention_weights = self.attention(out)
        out = out * attention_weights

        return out

# 3. 提取特征
class extract_features(nn.Module):
    def __init__(self,image):
        self.image = image
        super(extract_features, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.resnet = resnest50(pretrained=True)
        self.conv1 = nn.Conv2d(image, 12, kernel_size=1, stride=1, padding=1,  # kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.acb1 = ACBBlock(12,192)
        self.bn1 = norm_layer(192)
        self.relu = nn.PReLU(num_parameters=1,init=0.25)
        self.conv2 = nn.Conv2d(image, 16, kernel_size=1, stride=1, padding=1,  # kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.acb2 = ACBBlock(16, 3)
        self.acb3 = ACBBlock(3, 192)
        self.bn2 = norm_layer(192)
        self.relu = nn.PReLU(num_parameters=1,init=0.25)
        self.conv3 = nn.Conv2d(image, 192, kernel_size=1, stride=1, padding=1,  # kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.avg1 = nn.AvgPool2d(kernel_size=1)
        self.conv4 = nn.Conv2d(576, 8, kernel_size=1, stride=1, padding=1,  # kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.PReLU(num_parameters=1,init=0.25)
        self.conv5 = nn.Conv2d(8, 32, kernel_size=1, stride=1, padding=1,  # kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.avg2 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.max = nn.MaxPool2d(kernel_size=3, stride=2)
        self.acb4 = ACBBlock(384, 192)
        self.acb5 = ACBBlock(192, 32)
        self.acb6 = ACBBlock(32, 16)
        self.acb7 = ACBBlock(16, 3)

    def forward(self, x):
            # x = x.to(device)
            x1 = self.conv1(x)  # x.size[4, 64, 320, 320]
            # print('x')
            # print('x.size()75',x.size())
            x1 = self.acb1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x2 = self.conv2(x)
            x2 = self.acb2(x2)
            x2 = self.acb3(x2)
            x2 = self.bn2(x2)
            x2 = self.relu(x2)
            x3 = self.conv3(x)
            x3 = self.avg1(x3)
            # print(x1.size())
            # print(x2.size())
            # print(x3.size())
            cha_f1 = torch.cat((x1, x2, x3),dim=0)
            # cha_f1 = cha_f1[:,:,:240,:240]
            cha_f1 = cha_f1[:2,:,:,:]
            x3 = x3[:2, :, :, :]
            # print(cha_f1.size())
            # print(x3.size())
            cha_f2 = cha_f1*x3
            x4 = self.avg2(cha_f2)
            x5 = self.max(cha_f2)
            # print(x4.size())
            # print(x5.size())
            x6 = torch.cat((x4,x5),dim=1)
            x6 = self.acb4(x6)
            x6 = self.acb5(x6)
            x6 = self.acb6(x6)
            x6 = self.acb7(x6)
            weight = torch.sigmoid(x6)
            return weight

# extract_features.train()

# for i, (x, y) in enumerate(training_loader):
#     if torch.cuda.is_available():
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
#
#     x = torch.tensor(x)
#     y = torch.tensor(y)
#     y = y.to(device)
#     x = x.to(device)
#     # 3. 提取特征
#
#     model = extract_features(x)
#     model = model.to(device)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1)
#     # optimizer.zero_grad()
#     features = model(x)
#     print("Feature Weights:",features.size())
#     # features = features.to('cpu')
#     # features = features.cpu()
#     # y_pred = y_pred.to('cpu')
#     # features = features.to(device)
#     # 4. 标记数据
# #     label = y
# #     torch.cuda.empty_cache()
# #     # print(features.size())
# #     # print(label.size())
# #     # 6. 创建样本数据集
# #     X_train.append(features)
# #     y_train.append(label)
# #
# # X_train = np.array(X_train)
# # print(X_train)
# # y_train = np.array(y_train)
# #
# # model1 = svm.SVC()
# #
# # model1.fit(X_train, y_train)
# #
# # #     num_classes = 2
# # #     epsilon = 0.2  #
# # #     smy = (1 - epsilon) * y + epsilon / num_classes  # 标签平滑
# # #     y_all_train.extend(y.detach().numpy())
# # #     y_pred_all_train.extend(y_pred.detach().numpy())
# #
# # # 获取特征权重
# # feature_weights = model1.coef_
# #
# # print("Feature Weights:", feature_weights)