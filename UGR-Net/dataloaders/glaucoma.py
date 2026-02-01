import os
import os.path as osp
import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class GlaucomaDataset(DatasetBase):
    """青光眼数据集适配器"""

    domains = ["Refuge2", "Harvard", "ORIGA", "RIMONE"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        # 读取训练数据（源域）
        train_data = self._read_domain_data(root, source_domains, 'train')

        # 读取测试数据（目标域）
        test_data = self._read_domain_data(root, target_domains, 'test')

        # 可选：读取验证数据
        val_data = self._read_domain_data(root, source_domains, 'train')  # 可以用部分训练数据作为验证

        super().__init__(train_x=train_data, val=val_data, test=test_data)

    def _read_domain_data(self, root, domains, split):
        """读取指定域和划分的数据"""
        items = []

        for domain in domains:
            # 构建CSV文件路径
            if split == 'train':
                csv_file = osp.join(root, f"{domain}_train.csv")
            elif split == 'test':
                csv_file = osp.join(root, f"{domain}_test.csv")
            else:
                continue

            if not osp.exists(csv_file):
                print(f"警告: CSV文件不存在: {csv_file}")
                continue

            # 读取CSV文件
            df = pd.read_csv(csv_file)

            # 假设CSV文件有'image_path'和'label'列
            # 根据您的实际CSV结构调整
            for _, row in df.iterrows():
                # 构建图像路径
                if 'image_path' in df.columns:
                    img_path = row['image_path']
                else:
                    # 如果CSV没有完整路径，需要构建
                    img_name = row['image_name']  # 根据您的列名调整
                    img_path = osp.join(root, domain, split, img_name)

                # 获取标签
                label = int(row['label'])  # 根据您的列名调整

                # 检查图像文件是否存在
                if not osp.exists(img_path):
                    print(f"警告: 图像文件不存在: {img_path}")
                    continue

                # 创建Datum对象
                item = Datum(impath=img_path, label=label, domain=domain)
                items.append(item)

        print(f"从 {len(domains)} 个域读取了 {len(items)} 个{split}样本")
        return items
