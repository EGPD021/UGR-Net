# coding:utf-8
import os
import cv2
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from networks.ResUnet_trid import ResnetModel
from utils.metrics import calculate_metrics
import sys

sys.path.append('./torch-cam-main')
from config import *
from torchnet import meter
import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchnet import meter

# 你的评估函数保持不变
# from xxx import calculate_classification_metrics1, calculate_classification_metrics2
# from networks.ResUnet_trid import ResnetModel   # 确保能导入

class Test:
    def __init__(self,
                 config=None,
                 test_loader=None,
                 model: nn.Module = None,
                 device=None,
                 model_path: str = None,
                 model_type: str = None,
                 seg_cost: nn.Module = None,
                 load_from_disk: bool = False):
        """
        两种用法：
        1) 训练期验证：传入 model=当前模型实例，load_from_disk=False（默认）
        2) 离线评估 best：model=None, load_from_disk=True + 提供 config 或手动提供 backbone/out_ch 等
        """
        # 数据与设备
        self.test_loader = test_loader
        self.device = device if device is not None else (config.device if config else None)

        # 基本配置
        self.model_path = model_path if model_path is not None else (config.model_path if config else None)
        self.model_type = model_type if model_type is not None else (config.model_type if config else None)
        self.backbone   = getattr(config, 'backbone', None)
        self.out_ch     = getattr(config, 'out_ch', 2)
        self.image_size = getattr(config, 'image_size', None)
        self.mode       = getattr(config, 'mode', None)
        self.target     = getattr(config, 'Target_Dataset', None)

        # 损失（默认 BCEWithLogits）
        self.seg_cost = seg_cost if seg_cost is not None else nn.BCEWithLogitsLoss()

        # 是否加载磁盘 best
        self.load_from_disk = load_from_disk

        # 模型
        if model is not None:
            # ✅ 训练期验证：直接用外部传入的当前模型，绝不加载磁盘
            self.model = model.to(self.device)
        else:
            # ✅ 离线评估：需要自行构建并（可选）加载 best
            self.model = self._build_model_from_config()
            if self.load_from_disk:
                self._load_best_weights()

        self.print_network(self.model)

    def _build_model_from_config(self) -> nn.Module:
        if self.model_type == 'Res_Unet':
            assert self.backbone is not None, "backbone 未提供"
            m = ResnetModel(resnet=self.backbone, num_classes=self.out_ch, pretrained=True, mixstyle_layers=[])
        else:
            raise ValueError('The model type is wrong!')
        return m.to(self.device).eval()

    def _load_best_weights(self):
        assert self.model_path is not None and self.model_type is not None, "缺少 model_path 或 model_type"
        best_fp = f"{self.model_path}/best-{self.model_type}.pth"
        try:
            ckpt = torch.load(best_fp, map_location=self.device, weights_only=True)
        except TypeError:
            # 兼容旧版 PyTorch 无 weights_only 参数
            ckpt = torch.load(best_fp, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()

    def print_network(self, model):
        num_params = sum(p.numel() for p in model.parameters())
        print("The number of parameters: {}".format(num_params))

    @torch.no_grad()
    def test(self):
        print("Testing and Saving the results... Domain Generalization Phase")
        print("--" * 15)

        was_training = self.model.training
        self.model.eval()

        loss_meter = meter.AverageValueMeter()
        loss_meter.reset()
        y_all, y_pred_all = [], []

        for batch, data in enumerate(self.test_loader):
            x = torch.from_numpy(data['data']).to(dtype=torch.float32, device=self.device)
            y = torch.from_numpy(data['mask']).to(dtype=torch.float32, device=self.device)
            out, feat, dir_out, conf_dict = self.model(x, is_train=True)
            # out = self.model(x, is_train=False)
            seg_logit = out[0] if isinstance(out, (tuple, list)) else out

            y_bce = y.view(-1, 1).float()
            if self.seg_cost is not None:
                loss = self.seg_cost(seg_logit, y_bce)
                loss_meter.add(loss.detach().item())

            seg_prob = torch.sigmoid(seg_logit).detach()
            y_all.append(y_bce.cpu().numpy())
            y_pred_all.append(seg_prob.cpu().numpy())

        if was_training:
            self.model.train()

        y_all_np  = np.concatenate(y_all, axis=0)
        y_pred_np = np.concatenate(y_pred_all, axis=0)

        if self.seg_cost is not None:
            print("Test ———— Total_Loss:{:.8f}".format(loss_meter.value()[0]))
        else:
            print("Test ———— Total_Loss:N/A (seg_cost is None)")

        # ===== 原有 ACC / AUC，保持 0.4 阈值逻辑不变 =====
        acc = calculate_classification_metrics1(y_pred_np, y_all_np)
        auc = calculate_classification_metrics2(y_pred_np, y_all_np)

        # ===== 新增 F1 / SPE / SEN / PREC，同样基于 p=0.4 =====
        f1, spe, sen, prec = calculate_extra_metrics(y_pred_np, y_all_np, p=0.4)

        result = {
            "ACC": acc,
            "AUC": auc,
            "F1": f1,
            "SPE": spe,
            "SEN": sen,
            "PREC": prec,
        }

        # 这里建议让“每个 epoch 写 txt”的逻辑放到 TrainDG 里，
        # test.py 就只负责返回结果，避免多进程 / 多次覆盖的问题
        print("Test Metrics: ", result)
        return result


def calculate_classification_metrics1(output, target):
    outputs = [pred[0] for pred in output]  # 提取每个输出的第一个元素
    outputs = np.array(outputs)  # 转换为 NumPy 数组

    # 应用阈值并将输出转换为二元预测
    p = 0.4
    pred = (outputs > p).astype(int)  # 使用内建的 int 类型

    # 计算准确率
    acc = accuracy_score(target, pred)
    return acc
def calculate_extra_metrics(output, target, p=0.4):
    """
    output: 模型的概率输出 (N,1) 或 (N,)，和上面的函数一样，用阈值 p 做二分类
    target: 真实标签 (N,1) 或 (N,)
    返回: F1, SPE, SEN, PREC
    """
    output = np.array(output).reshape(-1)  # [N]
    target = np.array(target).reshape(-1).astype(int)

    # 阈值 0.4，与 calculate_classification_metrics1 保持一致
    pred = (output > p).astype(int)

    # 防止全一类时 sklearn 报错
    try:
        f1 = f1_score(target, pred)
    except ValueError:
        f1 = float("nan")

    try:
        prec = precision_score(target, pred)
    except ValueError:
        prec = float("nan")

    try:
        sen = recall_score(target, pred)  # sensitivity = recall(正类)
    except ValueError:
        sen = float("nan")

    try:
        cm = confusion_matrix(target, pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spe = tn / (tn + fp + 1e-8)  # specificity
        else:
            spe = float("nan")
    except ValueError:
        spe = float("nan")

    return f1, spe, sen, prec


def calculate_classification_metrics2(output, target):
    try:
        output = [pred[0] for pred in output]
        auc = roc_auc_score(target, output)
        return auc
    except ValueError as e:
        print("ValueError occurred:", e)
        return -1


def smooth_grad_campp(model, x):
    # Forward pass through the model
    feature, logits = model(x, is_train=False)

    # Get the class prediction
    pred_class = logits.argmax(dim=1, keepdim=True)

    # Calculate gradients w.r.t. logits
    logits.backward(gradient=torch.ones_like(logits), retain_graph=True)

    # 获取最后卷积层的输出和梯度
    conv_outputs = model.feature_map  # 假设 feature_map 已经是通过 hook 提取的特征图
    grads = model.gradients  # 假设 gradients 已经通过 hook 提取的梯度

    # 全局平均池化（GAP）梯度
    weights = F.adaptive_avg_pool2d(grads, 1)

    # 创建 CAM
    cam = torch.mul(conv_outputs, weights).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # 上采样 CAM 到输入图像大小
    cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / cam.max()

    return cam
