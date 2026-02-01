import os

import torch
from matplotlib import pyplot as plt

from dataloaders.normalize import normalize_image
from torchnet import meter
from networks.EGPD_Net import ResnetModel, GateWithUncertainty
from config import *
import numpy as np
from tensorboardX import SummaryWriter
from test_attention import Test
import datetime
import torch.nn as nn
import torch.nn.functional as F
def kl_beta_uniform(alpha, beta, eps=1e-8):
    """
    KL( Beta(a,b) || Beta(1,1) ) = log B(1,1) - log B(a,b)
                                  + (a-1)[ψ(a)-ψ(a+b)]
                                  + (b-1)[ψ(b)-ψ(a+b)]
    其中 log B(1,1)=0
    """
    from torch.special import digamma
    lgB_ab = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    term = (alpha - 1) * (digamma(alpha + eps) - digamma(alpha + beta + eps)) \
         + (beta - 1)  * (digamma(beta  + eps) - digamma(alpha + beta + eps))
    return -lgB_ab + term

def evidential_regularizer(a_c, b_c, a_s, b_s, lambda_evi=1e-3):
    kl_c = kl_beta_uniform(a_c, b_c).mean()
    kl_s = kl_beta_uniform(a_s, b_s).mean()
    return lambda_evi * (kl_c + kl_s)


class TrainDG:
    def __init__(self, config, train_loader, valid_loader=None):
        # 配置相关
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # self.save_path = config.get('save_path', './saved_models')  # 使用配置中的路径或默认路径

        # 确保保存目录存在
        # os.makedirs(self.save_path, exist_ok=True)
        # 模型相关
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.model_type = config.model_type
        self.mixstyle_layers = config.mixstyle_layers
        self.random_type = config.random_type
        self.random_prob = config.random_prob

        # 损失函数
        self.seg_cost = nn.BCEWithLogitsLoss()

        # 优化器
        self.optim = config.optimizer
        self.lr_scheduler = config.lr_scheduler
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.betas = (config.beta1, config.beta2)

        # 训练设置
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # 路径设置
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.log_path = config.log_path

        # 设置验证频率，默认为 1
        self.valid_frequency = 1  # 从配置中获取 valid_frequency

        self.device = config.device
        # self.gamma = config.gamma
        self.gamma = [float(g) for g in config.gamma.split(',')]
        self.build_model()
        # self.print_network()


    def build_model(self):
        self.model = ResnetModel(resnet=self.config.backbone, num_classes=self.config.out_ch, pretrained=True,
                                 mixstyle_layers=self.config.mixstyle_layers, random_type=self.config.random_type,
                                 p=self.config.random_prob).to(self.device)

        # 优化器和学习率调度器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          betas=(self.config.beta1, self.config.beta2))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

        self.model = self.model.to(self.device)

    def train_and_evaluate(self, gamma):
        """
        训练并评估每个 gamma 值。

        Args:
            gamma: 当前实验的 gamma 值。

        Returns:
            result_dict: 包含 AUC 和 ACC 的字典。
        """
        # 用每个 gamma 创建 GateWithUncertainty
        # 实例化门控模块
        gamma_list = [2.0, 2.5, 3.0, 3.5, 4.0]
        gate = GateWithUncertainty(gamma_list)
        self.model.add_module('gate_with_uncertainty', gate)
        # self.model.add_module('gate_with_uncertainty', gate)
        self.model = self.model.to(self.device)

        # 训练阶段
        writer = SummaryWriter(self.config.log_path.replace('.log', f'_gamma_{gamma}.writer'))
        best_loss, best_epoch = np.inf, 0
        loss_meter = meter.AverageValueMeter()

        for epoch in range(self.num_epochs):
            self.model.train()
            print("Epoch:{}/{}".format(epoch + 1, self.num_epochs))
            print("Training...")
            print("Learning rate: " + str(self.optimizer.param_groups[0]["lr"]))
            loss_meter.reset()

            for batch, data in enumerate(self.train_loader):
                x, y = data['data'], data['mask']
                x = torch.from_numpy(normalize_image(x)).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                x, y = x.to(self.device), y.to(self.device)
                pred, feat, dir_out, conf_dict = self.model(x, is_train=True)

                loss = self.seg_cost(pred, y.view(-1, 1).float())
                loss_meter.add(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 记录每个 epoch 的 loss
            # writer.add_scalar('Total_Loss_Epoch', loss_meter.value()[0], epoch + 1)
            writer.add_scalar('Total_Loss_Epoch', loss_meter.value()[0], epoch + 1)

            # 保存最佳模型
            if loss_meter.value()[0] < best_loss:
                best_loss = loss_meter.value()[0]
                best_epoch = epoch + 1
                # torch.save(self.model.state_dict(), f"{self.model_path}/best_{gamma}.pth")
                torch.save(self.model.state_dict(), self.model_path + '/' + 'best' + '-' + self.model_type + '.pth')

            # 每轮验证

            # 每轮验证（val 上计算分类指标）
            if (epoch + 1) % self.valid_frequency == 0 and self.valid_loader is not None:
                tester = Test(
                    model=self.model,
                    test_loader=self.valid_loader,
                    device=self.device,
                    model_path=self.model_path,
                    model_type=self.model_type,
                    seg_cost=self.seg_cost,
                )
                result_dict = tester.test()  # 在 val 上计算所有指标

                step = (epoch + 1) // self.valid_frequency

                # ========= 1) 写入 TensorBoard =========
                writer.add_scalar('Val/ACC', result_dict['ACC'], step)
                writer.add_scalar('Val/AUC', result_dict['AUC'], step)
                writer.add_scalar('Val/F1', result_dict['F1'], step)
                writer.add_scalar('Val/SPE', result_dict['SPE'], step)
                writer.add_scalar('Val/SEN', result_dict['SEN'], step)
                writer.add_scalar('Val/PREC', result_dict['PREC'], step)

                # 顺便在终端打印一行，方便看
                print(f"[Val] Epoch {epoch + 1}: "
                      f"ACC={result_dict['ACC']:.4f} "
                      f"AUC={result_dict['AUC']:.4f} "
                      f"F1={result_dict['F1']:.4f} "
                      f"SPE={result_dict['SPE']:.4f} "
                      f"SEN={result_dict['SEN']:.4f} "
                      f"PREC={result_dict['PREC']:.4f}")

                # ========= 2) 追加写入 txt 文件 =========
                metrics_file = os.path.join(self.model_path, 'val_metrics_per_epoch.txt')
                file_exists = os.path.exists(metrics_file)

                with open(metrics_file, 'a', encoding='utf-8') as f:
                    # 如果是新文件，先写表头
                    if not file_exists:
                        f.write("epoch\tACC\tAUC\tF1\tSPE\tSEN\tPREC\n")
                    f.write(
                        f"{epoch + 1}\t"
                        f"{result_dict['ACC']:.6f}\t"
                        f"{result_dict['AUC']:.6f}\t"
                        f"{result_dict['F1']:.6f}\t"
                        f"{result_dict['SPE']:.6f}\t"
                        f"{result_dict['SEN']:.6f}\t"
                        f"{result_dict['PREC']:.6f}\n"
                    )

        writer.close()
        return best_loss, best_epoch

    def plot_results(self, results):
        """
        根据 gamma 的调参结果绘制 AUC 和 ACC 曲线

        Args:
            results: 每个 gamma 对应的结果字典，包含 AUC 和 ACC。
        """
        plt.figure(figsize=(12, 6))

        # Plot AUC vs Gamma
        plt.subplot(1, 2, 1)
        plt.plot(results['gamma'], results['AUC'], marker='o', label='AUC')
        plt.title('AUC vs Gamma')
        plt.xlabel('Gamma')
        plt.ylabel('AUC')
        plt.grid(True)

        # Plot ACC vs Gamma
        plt.subplot(1, 2, 2)
        plt.plot(results['gamma'], results['ACC'], marker='o', label='ACC')
        plt.title('ACC vs Gamma')
        plt.xlabel('Gamma')
        plt.ylabel('ACC')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, epoch, gamma, is_best=False):
        """
        保存模型检查点

        Args:
            epoch: 当前训练周期
            gamma: 当前使用的 gamma 值
            is_best: 是否是最佳模型
        """
        # 创建检查点目录（如果不存在）
        checkpoint_dir = os.path.join(self.save_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 准备检查点数据
        state = {
            'epoch': epoch,
            'gamma': gamma,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'best_loss': self.best_loss if hasattr(self, 'best_loss') else float('inf'),
        }

        # 保存检查点
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_gamma_{gamma}.pth')
        torch.save(state, checkpoint_path)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_model_path = os.path.join(checkpoint_dir, f'best_model_gamma_{gamma}.pth')
            torch.save(state, best_model_path)
            print(f"保存最佳模型: {best_model_path}")

        print(f"保存检查点: {checkpoint_path}")
    # def run(self):
    #     """
    #     运行整个调参流程，并记录每个 gamma 的表现。
    #     """
    #     results = {'gamma': [], 'AUC': [], 'ACC': []}
    #     # gamma_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    #     gate = GateWithUncertainty(gamma=self.gamma)
    #     for gamma in self.gamma:
    #         print(f"Running experiment for gamma={gamma}")
    #         gate.set_gamma_value(gamma)
    #         best_loss, best_epoch = self.train_and_evaluate(gamma)
    #         print(f"Best Loss: {best_loss} at Epoch {best_epoch}")
    #
    #         # Assume the result_dict contains 'ACC' and 'AUC' metrics
    #         tester = Test(model=self.model, test_loader=self.valid_loader, device=self.device, load_from_disk=False)
    #         result_dict = tester.test()
    #         results['gamma'].append(gamma)
    #         results['AUC'].append(result_dict['AUC'])
    #         results['ACC'].append(result_dict['ACC'])
    #
    #     # After all gamma experiments are done, plot the results
    #     self.plot_results(results)
    def run(self):
        """
        运行整个调参流程，并记录每个 gamma 的表现。
        """
        results = {'gamma': [], 'AUC': [], 'ACC': []}

        # 确保模型中有 gate_u 模块
        if not hasattr(self.model, 'gate_u'):
            print("警告: 模型中没有 gate_u 模块，将创建并添加")
            gamma_list = [2.0, 2.5, 3.0,3.5,4.0]
            self.model.gate_u = GateWithUncertainty(gamma_list)
            # 如果模型在多个设备上，需要确保 gate_u 也在正确的设备上
            self.model.gate_u = self.model.gate_u.to(self.device)

        # 遍历所有 gamma 值
        for i, gamma_value in enumerate(self.gamma):
            print(f"Running experiment for gamma={gamma_value}")

            # 设置当前 gamma 值
            self.model.gate_u.set_gamma(i)  # 使用索引设置 gamma

            # 训练和评估
            best_loss, best_epoch = self.train_and_evaluate(gamma_value)
            print(f"Best Loss: {best_loss} at Epoch {best_epoch}")

            # 测试模型性能
            tester = Test(model=self.model, test_loader=self.valid_loader, device=self.device, load_from_disk=False)
            result_dict = tester.test()

            # 记录结果
            results['gamma'].append(gamma_value)
            results['AUC'].append(result_dict['AUC'])
            results['ACC'].append(result_dict['ACC'])

            # 保存当前 gamma 的最佳模型
            # self.save_checkpoint(epoch=best_epoch, gamma=gamma_value, is_best=True)

        # 所有 gamma 实验完成后，绘制结果
        self.plot_results(results)

        # 找到最佳 gamma 值
        best_idx = np.argmax(results['AUC'])  # 假设 AUC 越高越好
        best_gamma = self.gamma[best_idx]
        print(f"最佳 gamma 值: {best_gamma}, AUC: {results['AUC'][best_idx]}")

        return best_gamma, results
