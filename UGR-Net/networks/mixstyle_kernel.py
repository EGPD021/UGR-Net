import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.svm import SVC
from torch.distributions import kl_divergence

class SaveMuVar():
    mu, var = None, None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.mu = output.detach().cpu().mean(dim=[2, 3], keepdim=True).squeeze(-1).squeeze(-1).numpy()
        self.var = output.detach().cpu().var(dim=[2, 3], keepdim=True).squeeze(-1).squeeze(-1).numpy()

    def remove(self):
        self.hook.remove()


class EFDMix(nn.Module):
    """EFDMix.

    Reference:
      Zhang et al. Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization. CVPR 2022.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
        x_view = x.view(B, C, -1)
        value_x, index_x = torch.sort(x_view)  # sort inputs
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        inverse_index = index_x.argsort(-1)
        x_view_copy = value_x[perm].gather(-1, inverse_index)
        new_x = x_view + (x_view_copy - x_view.detach()) * (1-lmda)
        return new_x.view(B, C, W, H)


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True  # Train: True, Test: False

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        return x_normed*sig_mix + mu_mix


class TriD(nn.Module):
    """TriD.
    Reference:
      Chen et al. Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation. MICCAI 2023.
    """
    def __init__(self, p=0.5, eps=1e-6, alpha=0.1):
        """
        Args:
          p (float): probability of using TriD.
          eps (float): scaling parameter to avoid numerical issues.
          alpha (float): parameter of the Beta distribution.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self._activated = True  # Train: True, Test: False
        self.beta = torch.distributions.Beta(alpha, alpha)

    def set_activation_status(self, status=True):
        self._activated = status




    def forward(self, x):
        # if not self._activated:
        #     return x
        #
        # if random.random() > self.p:
        #     return x

        N, C, H, W = x.shape

        mu = x.mean(dim=[2, 3], keepdim=True)

        var = x.var(dim=[2, 3], keepdim=True)


        if random.random() > 0.5:# 参数0.8代表随机选择的概率，可以调整，经过试验下来，发现为0.5效果较好

        # #     # 假设 input 是一个张量对象
            swap_index_cpu = torch.randperm(x.size(0))

            # 将生成的随机排列索引转移到 CUDA 设备上
            swap_index = swap_index_cpu.to(x.device)


            swap_mean = mu[swap_index]
            swap_std = var[swap_index]

            scale = swap_std / var
            shift = swap_mean - mu * scale
            output = x * scale + shift
        else:
            # print('aug randomly choice 选择TRID')
            # print('trid randomly choice:',random.random())
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            lmda = self.beta.sample((N, C, 1, 1))
            bernoulli = torch.bernoulli(lmda).to(x.device)
            # 获取当前数据分布，并根据目标数据分布更新混合比例
            # mixing_ratio = self.update_mixing_ratio()

            # 使用混合比例进行操作 高斯分布
            # 在 CPU 上生成随机数
            mu_random_cpu = torch.randn((N, C, 1, 1), dtype=torch.float32, device='cpu')

            # 将生成的随机数转移到 CUDA 设备上
            mu_random= mu_random_cpu.to(x.device)
            # mu_random = torch.randn((N, C, 1, 1), dtype=torch.float32).to(x.device)
            var_random_cpu = torch.randn((N, C, 1, 1), dtype=torch.float32,device='cpu')
            var_random = var_random_cpu.to(x.device)
            mu_mix = mu_random * bernoulli + mu * (1. - bernoulli)
            sig_mix = var_random * bernoulli + sig * (1. - bernoulli)
            output=x_normed * sig_mix + mu_mix

        return output


class DomainLearner(nn.Module):
    def __init__(self, feature_dim, num_domain):
        super(DomainLearner, self).__init__()
        self.network = nn.Linear(feature_dim, num_domain)
        print('initialise ResNet Domain learner')

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.network(x)
        return x


class DomainClassMixAugmentation(nn.Module):
    '''mixup'''

    def __init__(self, batch_size, num_classes, num_domains, hparams):
        super(DomainClassMixAugmentation, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_domains = num_domains

        self.hparams = hparams
        self.threshold = hparams["threshold"]
        self.threshold_lower_bound = hparams["threshold_lower_bound"]
        self.threshold_change = hparams["value_to_change"]
        self.step_to_change = hparams["step_to_change"]

        self.uniform = torch.distributions.Uniform(0, 1)
        self.p = hparams['probability_to_discard']

        self.call_num = 0

    def update_threshold(self):
        next_threshold = self.threshold - self.threshold_change
        if self.threshold == self.threshold_lower_bound:
            self.threshold = self.hparams["threshold"]
        elif next_threshold < self.threshold_lower_bound:
            self.threshold = self.threshold_lower_bound
        else:
            self.threshold = next_threshold

    def get_threshold(result, quantile):
        return torch.quantile(result, quantile)

    def sample_different_class_different_domain(idx, y, domain, y_target, domain_target):
        y_size = y.size(0)
        for trial in range(y_size * 4):
            current_idx = random.randrange(y_size)
            if trial <= y_size * 2:
                if (y[current_idx] != y_target) and (domain[current_idx] != domain_target):
                    return current_idx
            else:
                if y[current_idx] != y_target:
                    return current_idx
        return idx

    def sample_same_class_different_domain(idx, y, domain, y_target, domain_target):
        y_size = y.size(0)
        for trial in range(y_size * 4):
            current_idx = random.randrange(y_size)
            if trial <= y_size * 2:
                if (y[current_idx] == y_target) and (domain[current_idx] != domain_target):
                    return current_idx
            else:
                if (y[current_idx] == y_target) and (current_idx != idx):
                    return current_idx
        return idx

    def get_feature_decomposition(self, class_im, domain_im, feature):
        class_im = torch.mean(class_im, dim=(1, 2), keepdim=True)
        domain_im = torch.mean(domain_im, dim=(1, 2), keepdim=True)
        class_thr = DomainClassMixAugmentation.get_threshold(class_im, 0.5)
        domain_thr = DomainClassMixAugmentation.get_threshold(domain_im, self.threshold)

        cs_idx = class_im > class_thr
        cg_idx = class_im <= class_thr
        ds_idx = domain_im > domain_thr
        di_idx = domain_im <= domain_thr
        csds_mask = cs_idx * ds_idx
        csdi_mask = cs_idx * di_idx
        cgds_mask = cg_idx * ds_idx
        cgdi_mask = cg_idx * di_idx

        return feature * csds_mask, feature * csdi_mask, feature * cgds_mask, feature * cgdi_mask

    def forward(self, x, y, domain, class_gradient, domain_gradient):
        if (self.call_num % self.step_to_change == 0) and self.call_num != 0:
            self.update_threshold()

        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg * current_feature,
                                                                            current_dg * current_feature,
                                                                            current_feature)

            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f

        mixup_strength = self.uniform.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            diff_y = DomainClassMixAugmentation.sample_different_class_different_domain(b, y, domain, y_label,
                                                                                        domain_label)
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1 - mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1 - mixup_strength[b][1]) * cgds[diff_y]
            if prob > self.p:
                result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]
            else:
                result[b] = new_cgds + csdi[b] + cgdi[b]

        self.call_num += 1

        return result

    def clip_get_feature_decomposition(self, class_im, domain_im, feature):
        class_thr = DomainClassMixAugmentation.get_threshold(class_im, 0.5)
        domain_thr = DomainClassMixAugmentation.get_threshold(domain_im, self.threshold)

        cs_idx = class_im > class_thr
        cg_idx = class_im <= class_thr
        ds_idx = domain_im > domain_thr
        di_idx = domain_im <= domain_thr
        csds_mask = cs_idx * ds_idx
        csdi_mask = cs_idx * di_idx
        cgds_mask = cg_idx * ds_idx
        cgdi_mask = cg_idx * di_idx

        return feature * csds_mask, feature * csdi_mask, feature * cgds_mask, feature * cgdi_mask

    def clip_forward(self, x, y, domain, class_gradient, domain_gradient):
        if (self.call_num % self.step_to_change == 0) and self.call_num != 0:
            self.update_threshold()

        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b]
            csds_f, csdi_f, cgds_f, cgdi_f = self.clip_get_feature_decomposition(current_cg * current_feature,
                                                                                 current_dg * current_feature,
                                                                                 current_feature)

            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f

        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            diff_y = DomainClassMixAugmentation.sample_different_class_different_domain(b, y, domain, y_label,
                                                                                        domain_label)
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1 - mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1 - mixup_strength[b][1]) * cgds[diff_y]
            if prob > self.p:
                result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]
            else:
                result[b] = new_cgds + csdi[b] + cgdi[b]

        self.call_num += 1

        return result

    def no_discard(self, x, y, domain, class_gradient, domain_gradient):
        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg * current_feature,
                                                                            current_dg * current_feature,
                                                                            current_feature)

            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f

        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            diff_y = DomainClassMixAugmentation.sample_different_class_different_domain(b, y, domain, y_label,
                                                                                        domain_label)
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1 - mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1 - mixup_strength[b][1]) * cgds[diff_y]

            result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]

        return result

    def same_x(self, x, y, domain, class_gradient, domain_gradient):
        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg * current_feature,
                                                                            current_dg * current_feature,
                                                                            current_feature)

            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f

        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1 - mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1 - mixup_strength[b][1]) * cgds[same_y]

            result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]

        return result

    def same_class_x(self, x, y, domain, class_gradient, domain_gradient):
        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg * current_feature,
                                                                            current_dg * current_feature,
                                                                            current_feature)

            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f

        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]

            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            while True:
                same_y2 = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label,
                                                                                        domain_label)
                if same_y2 != same_y:
                    break

            new_csds = mixup_strength[b][0] * csds[b] + (1 - mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1 - mixup_strength[b][1]) * cgds[same_y2]

            result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]

        return result

    def same_domain_x(self, x, y, domain, class_gradient, domain_gradient):
        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg * current_feature,
                                                                            current_dg * current_feature,
                                                                            current_feature)

            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f

        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]

            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)
            diff_y = DomainClassMixAugmentation.sample_different_class_different_domain(b, y, domain, y_label,
                                                                                        domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1 - mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1 - mixup_strength[b][1]) * cgds[diff_y]

            result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]

        return result
