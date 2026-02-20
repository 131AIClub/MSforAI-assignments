import os
import gzip
import numpy as np
import urllib.request
from typing import Tuple

# 数据集存储路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class MNISTLoader:
    """
    MNIST 数据加载器
    负责下载、读取和预处理 MNIST 数据集。
    """
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.files = {
            'train_img': 'train-images-idx3-ubyte.gz',
            'train_lbl': 'train-labels-idx1-ubyte.gz',
            'test_img': 't10k-images-idx3-ubyte.gz',
            'test_lbl': 't10k-labels-idx1-ubyte.gz'
        }
        # 使用可靠的镜像源
        self.base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

    def download(self):
        """下载数据集如果不存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"创建数据目录: {self.data_dir}")

        for name, filename in self.files.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                url = self.base_url + filename
                print(f"正在下载 {filename} ...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    print(f"下载完成: {filepath}")
                except Exception as e:
                    print(f"下载失败 {url}: {e}")
                    raise

    def load_images(self, filename: str) -> np.ndarray:
        """读取图像数据并归一化"""
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            # 跳过前16个字节的头部信息 (Magic number, number of images, rows, cols)
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # Reshape: (N, 28*28) 并归一化到 [0, 1]
        return data.reshape(-1, 784).astype(np.float32) / 255.0

    def load_labels(self, filename: str) -> np.ndarray:
        """读取标签数据"""
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            # 跳过前8个字节的头部信息 (Magic number, number of items)
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载所有数据"""
        self.download()
        
        x_train = self.load_images(self.files['train_img'])
        y_train = self.load_labels(self.files['train_lbl'])
        x_test = self.load_images(self.files['test_img'])
        y_test = self.load_labels(self.files['test_lbl'])
        
        return x_train, y_train, x_test, y_test

def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    将整数标签转换为 One-hot 编码
    Example: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """
    return np.eye(num_classes)[labels]

class DataLoader:
    """
    自定义数据加载器
    支持批量加载 (Batching) 和 随机打乱 (Shuffling)
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = x.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / batch_size))
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        """返回迭代器"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取下一个 batch"""
        if self.current_idx >= self.num_samples:
            raise StopIteration
            
        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        
        self.current_idx += self.batch_size
        return batch_x, batch_y

    def __len__(self):
        return self.num_batches
