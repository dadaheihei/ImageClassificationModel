import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_url

class CIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        """
        自定义 CIFAR10 数据集类
        
        参数:
        - root (str): 数据集存放路径
        - train (bool): 是否加载训练集
        - transform (callable): 预处理函数
        - download (bool): 是否下载数据集
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        # 如果指定下载数据集，则进行下载
        if self.download:
            self.download_cifar10()

        # 使用 torchvision 加载 CIFAR10 数据集
        self.dataset = datasets.CIFAR10(root=self.root, train=self.train, transform=self.transform, download=False)
    
    def download_cifar10(self):
        """
        如果 CIFAR10 数据集不存在，则下载它
        """
        # CIFAR-10 官方网址
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        data_dir = os.path.join(self.root, 'cifar-10-batches-py')  # CIFAR-10 数据集文件夹
    
        # 检查数据集文件夹是否存在
        if not os.path.exists(data_dir):
            os.makedirs(self.root, exist_ok=True)  # 如果根目录没有文件夹，创建它
            download_url(url, self.root, filename)
            print(f"数据集下载完成，并保存在：{data_dir}")
        else:
            print(f"数据集已经存在，跳过下载：{data_dir}")


    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签
        
        参数:
        - idx (int): 索引
        
        返回:
        - 处理后的图像和对应标签
        """
        image, label = self.dataset[idx]
        return image, label
