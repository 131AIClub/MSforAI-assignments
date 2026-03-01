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
        """
        self.input_cache = x
        # 线性变换: y = xW + b
        return x @ self.w + self.b
    
    def backpropagation(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        """
        反向传播 (Backward Pass)
        
        参数:
            grad_output (np.ndarray): 上一层传回的梯度 dL/dY，形状为 (batch_size, output_dim)
            lr (float): 学习率
            
        返回:
            np.ndarray: 传递给下一层（即前一层）的梯度 dL/dX，形状为 (batch_size, input_dim)
        """
        assert self.input_cache is not None, "必须先调用前向传播 (__call__) 才能调用反向传播"
        
        # 1. 计算梯度
        # dL/dW = X^T @ dL/dY
        # 形状: (input_dim, batch_size) @ (batch_size, output_dim) -> (input_dim, output_dim)
        grad_w = self.input_cache.T @ grad_output
        
        # dL/db = sum(dL/dY) 沿 batch 维度求和
        # 形状: (output_dim,)
        grad_b = np.sum(grad_output, axis=0)
        
        # dL/dX = dL/dY @ W^T
        # 形状: (batch_size, output_dim) @ (output_dim, input_dim) -> (batch_size, input_dim)
        grad_input = grad_output @ self.w.T
        
        # 2. 更新参数 (梯度下降)
        # W = W - lr * dL/dW
        self.w -= lr * grad_w
        # b = b - lr * dL/db
        self.b -= lr * grad_b
        
        return grad_input
        

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
        """
        # 计算 Sigmoid 输出
        # clip 防止溢出，虽然 float32 范围较大，但 exp(-x) 可能很大
        self.output_cache = 1 / (1 + np.exp(-x))
        return self.output_cache
    
    def backpropagation(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        """
        反向传播
        注意: 激活函数没有可学习参数，所以不需要更新参数，只需传递梯度。
        """
        assert self.output_cache is not None, "必须先调用前向传播"
        
        # 链式法则: dL/dX = dL/dY * dY/dX
        # dY/dX = y * (1 - y)
        # 这里的乘法是逐元素乘法 (Element-wise multiplication)
        grad_input = grad_output * self.output_cache * (1 - self.output_cache)
        return grad_input

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
        """
        # 减去最大值以防止指数爆炸 (数值稳定性技巧)
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.output_cache = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        assert self.output_cache is not None, "必须先调用前向传播"
        return self.output_cache
    
    def backpropagation(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        """
        反向传播
        注意: 通常 Softmax 与 CrossEntropyLoss 结合使用，
        合并后的梯度计算更简单: pred - target。
        这里仅提供单独的 Softmax 梯度计算供参考。
        """
        assert self.output_cache is not None, "必须先调用前向传播"
        
        batch_size, num_classes = self.output_cache.shape
        # 向量化计算梯度，避免使用 for 循环
        # dL/dX = S * (dL/dY) - S * (S . dL/dY)
        s = self.output_cache
        # 逐元素相乘: S * dL/dY
        term1 = s * grad_output
        # 计算点积部分 (S . dL/dY)，并保持维度以便广播
        dot_product = np.sum(term1, axis=1, keepdims=True)
        # S * (S . dL/dY)
        term2 = s * dot_product
        
        grad_input = term1 - term2
            
        return grad_input

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
        """
        self.pred_cache = pred
        self.labels_cache = labels
        
        # 裁剪预测值以避免 log(0)
        pred = np.clip(pred, self.epsilon, 1. - self.epsilon)
        
        # 计算交叉熵: -sum(y * log(p)) / N
        loss = -float(np.sum(labels * np.log(pred))) / pred.shape[0]
        return float(loss)
    
    def backpropagation(self, lr: float) -> np.ndarray:
        """
        反向传播
        dL/d_pred = -y_true / y_pred
        """
        assert self.pred_cache is not None and self.labels_cache is not None, "必须先计算损失"
        
        batch_size = self.pred_cache.shape[0]
        
        # 裁剪预测值，避免除以零
        # 注意：这里需要与前向传播使用相同的 epsilon 策略保持一致
        safe_pred = np.clip(self.pred_cache, self.epsilon, 1. - self.epsilon)
        
        # dL/d_pred = - (y_true / y_pred) / N
        # 这里的除以 N 是因为损失函数中对 batch 进行了平均
        grad_input = - (self.labels_cache / safe_pred) / batch_size
        
        return grad_input


class MLP:
    """
    多层感知机 (Multi-Layer Perceptron)
    支持自定义层结构，用于 MNIST 分类
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