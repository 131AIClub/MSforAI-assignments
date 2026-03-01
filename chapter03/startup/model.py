import pickle
import numpy as np
from typing import List, Tuple, Optional

# 设置随机种子以保证结果可复现
np.random.seed(42)

class Linear:
    """
    全连接层 (Linear Layer)
    
    数学公式:
        Output = Input @ W + b
    
    参数:
        input_dim (int): 输入特征维度
        output_dim (int): 输出特征维度
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        # 初始化权重 (Weights) 和偏置 (Bias)
        # Xavier 初始化 (Glorot Initialization)
        # 保持方差一致，有助于 Sigmoid 网络收敛
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.w = np.random.randn(input_dim, output_dim).astype(np.float32) * scale
        # 偏置形状: (output_dim,)
        self.b = np.zeros(output_dim).astype(np.float32)
        
        # 用于存储前向传播的输入，供反向传播使用
        self.input_cache: Optional[np.ndarray] = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播 (Forward Pass)
        
        参数:
            x (np.ndarray): 输入数据，形状为 (batch_size, input_dim)
            
        返回:
            np.ndarray: 输出数据，形状为 (batch_size, output_dim)
        
        你需要补全该方法，实现前向传播
        公式: Output = Input @ W + b
        """
        return np.array(0)
    
    def backpropagation(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        """
        反向传播 (Backward Pass)
        
        参数:
            grad_output (np.ndarray): 上一层传回的梯度 dL/dY，形状为 (batch_size, output_dim)
            lr (float): 学习率
            
        返回:
            np.ndarray: 传递给下一层（即前一层）的梯度 dL/dX，形状为 (batch_size, input_dim)
        
        你需要补全该方法，实现反向传播，公式在README中。
        """
        return np.array(0)
    

class Sigmoid:
    """
    Sigmoid 激活函数
    
    数学公式:
        f(x) = 1 / (1 + e^(-x))
        f'(x) = f(x) * (1 - f(x))
    """
    def __init__(self) -> None:
        self.output_cache: Optional[np.ndarray] = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        参数:
            x (np.ndarray): 输入数据，形状为 (batch_size, input_dim)
            
        返回:
            np.ndarray: 输出数据，形状为 (batch_size, input_dim)
        你需要补全该方法，实现前向传播。
        公式: f(x) = 1 / (1 + e^(-x))
        """
        return np.array(0)
    
    def backpropagation(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        """
        反向传播
        注意: 激活函数没有可学习参数，所以不需要更新参数，只需传递梯度。
        你需要补全该方法，实现反向传播。
        公式: f'(x) = f(x) * (1 - f(x))
        """
        
        return np.array(0)

class Softmax:
    """
    Softmax 激活函数 (用于多分类输出层)
    
    数学公式:
        f(x)_i = e^(x_i) / sum(e^(x_j))
    """
    def __init__(self) -> None:
        self.output_cache: Optional[np.ndarray] = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        参数:
            x (np.ndarray): 输入数据，形状为 (batch_size, input_dim)
            
        返回:
            np.ndarray: 输出数据，形状为 (batch_size, input_dim)
        你需要补全该方法，实现前向传播。
        公式: f(x)_i = e^(x_i) / sum(e^(x_j))
        """
        return np.array(0)
    
    def backpropagation(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        """
        反向传播
        注意: 通常 Softmax 与 CrossEntropyLoss 结合使用，
        合并后的梯度计算更简单: pred - target。
        这里仅提供单独的 Softmax 梯度计算供参考。
        这里你需要补全该方法，实现反向传播。
        公式: dL/dx_i = f(x)_i - y_true_i
        你只需要考虑单独Softmax一个层即可，不需要考虑与CrossEntropyLoss的结合。
        """
        return np.array(0)

class CrossEntropyLoss:
    """
    交叉熵损失函数 (Cross Entropy Loss)
    通常用于多分类问题，结合 Softmax 使用
    
    数学公式:
        L = -sum(y_true * log(y_pred))
    """
    def __init__(self) -> None:
        self.pred_cache: Optional[np.ndarray] = None
        self.labels_cache: Optional[np.ndarray] = None
        self.epsilon = 1e-12 # 防止 log(0)
    
    def __call__(self, pred: np.ndarray, labels: np.ndarray) -> float:
        """
        计算损失值
        
        参数:
            pred (np.ndarray): 预测概率 (Softmax 输出)，形状 (batch_size, num_classes)
            labels (np.ndarray): 真实标签 (One-hot 编码)，形状 (batch_size, num_classes)
            
        返回:
            float: 交叉熵损失值
        你需要补全该方法，实现前向传播。
        公式: L = -sum(y_true * log(y_pred))
        """
        
        return 0.
    
    def backpropagation(self, lr: float) -> np.ndarray:
        """
        反向传播
        注意: 如果前一层是 Softmax，这里的梯度通常简化为 (pred - labels) / N
        你需要补全该方法，实现反向传播。
        公式: dL/dx_i = (f(x)_i - y_true_i) / N
        你只需要考虑单独CrossEntropyLoss一个层即可，不需要考虑与Softmax的结合。
        """
        
        return np.array(0)


class MLP:
    """
    多层感知机 (Multi-Layer Perceptron)
    支持自定义层结构，用于 MNIST 分类
    这个类不需要补全。
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        self.layers = []
        
        # 构建隐藏层
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i+1]))
            self.layers.append(Sigmoid()) # 隐藏层使用 Sigmoid 激活
            
        # 构建输出层
        self.layers.append(Linear(dims[-1], output_dim))
        self.layers.append(Softmax()) # 输出层使用 Softmax
        
        # 损失函数
        self.loss_fn = CrossEntropyLoss()
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad: np.ndarray, lr: float):
        """反向传播"""
        for layer in reversed(self.layers):
            grad = layer.backpropagation(grad, lr)

    def save(self, filepath: str, verbose: bool = True):
        """保存模型参数"""
        # 只保存有参数的层 (Linear)
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append({'w': layer.w, 'b': layer.b})
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        if verbose:
            print(f"模型已保存至: {filepath}")

    def load(self, filepath: str):
        """加载模型参数"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            
        param_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                if param_idx < len(params):
                    layer.w = params[param_idx]['w']
                    layer.b = params[param_idx]['b']
                    param_idx += 1
        print(f"模型已加载: {filepath}")