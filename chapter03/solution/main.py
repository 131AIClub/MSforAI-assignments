"""
模块名称: main.py
功能描述: 手写数字识别模型训练的主程序。
学习目标:
    1. 掌握深度学习模型的完整训练流程（数据加载 -> 模型构建 -> 训练 -> 验证 -> 保存）。
    2. 学习如何使用 matplotlib 可视化训练过程中的损失和准确率变化。
    3. 理解超参数（学习率、Batch Size、Epoch）对模型性能的影响。

作者: AI Assistant
日期: 2026-02-20
"""

import os
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# 导入自定义模块
# 假设 model.py 和 data.py 在同一目录下
try:
    from model import Linear, Sigmoid, Softmax, CrossEntropyLoss
    from data import MNISTLoader, DataLoader, one_hot_encode
except ImportError:
    # 如果作为脚本直接运行，可能需要调整路径
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import Linear, Sigmoid, Softmax, CrossEntropyLoss
    from data import MNISTLoader, DataLoader, one_hot_encode

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

    def save(self, filepath: str):
        """保存模型参数"""
        # 只保存有参数的层 (Linear)
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append({'w': layer.w, 'b': layer.b})
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
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

def evaluate(model: MLP, x: np.ndarray, y: np.ndarray, batch_size: int = 128) -> Tuple[float, float]:
    """
    在验证集/测试集上评估模型
    返回: (平均损失, 准确率)
    """
    loader = DataLoader(x, one_hot_encode(y, 10), batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    for x_batch, y_batch in loader:
        # 前向传播
        pred = model.forward(x_batch)
        
        # 计算损失
        loss = model.loss_fn(pred, y_batch)
        total_loss += loss * x_batch.shape[0]
        
        # 计算准确率
        pred_labels = np.argmax(pred, axis=1)
        true_labels = np.argmax(y_batch, axis=1)
        correct += np.sum(pred_labels == true_labels)
        total_samples += x_batch.shape[0]
        
    return total_loss / total_samples, correct / total_samples

def train(args):
    """训练流程"""
    # 1. 准备数据
    print("正在加载 MNIST 数据集...")
    loader = MNISTLoader()
    x_train, y_train, x_test, y_test = loader.load_data()
    
    # 划分验证集 (从训练集中分出 5000 张)
    val_size = 5000
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    
    print(f"训练集: {x_train.shape[0]}, 验证集: {x_val.shape[0]}, 测试集: {x_test.shape[0]}")
    
    # 2. 构建模型
    # 784 -> 256 -> 128 -> 10
    model = MLP(input_dim=784, hidden_dims=args.hidden_dims, output_dim=10)
    
    # 3. 训练循环
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"开始训练: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练阶段
        train_loader = DataLoader(x_train, one_hot_encode(y_train, 10), batch_size=args.batch_size, shuffle=True)
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Forward
            pred = model.forward(x_batch)
            
            # Loss
            loss = model.loss_fn(pred, y_batch)
            epoch_loss += loss * x_batch.shape[0]
            
            # Accuracy
            pred_labels = np.argmax(pred, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            correct += np.sum(pred_labels == true_labels)
            total += x_batch.shape[0]
            
            # Backward
            # CrossEntropyLoss.backward 返回 (pred - y) / N
            grad = model.loss_fn.backpropagation(args.lr)
            model.backward(grad, args.lr)
            
        train_loss = epoch_loss / total
        train_acc = correct / total
        
        # 验证阶段
        val_loss, val_acc = evaluate(model, x_val, y_val, args.batch_size)
        
        # 记录日志
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            model.save(os.path.join(args.save_dir, 'best_model.pkl'))
            
    total_time = time.time() - start_time
    print(f"训练结束! 总耗时: {total_time:.2f}s")
    
    # 4. 测试集最终评估
    # 加载最佳模型
    model.load(os.path.join(args.save_dir, 'best_model.pkl'))
    test_loss, test_acc = evaluate(model, x_test, y_test)
    print(f"最终测试集准确率: {test_acc*100:.2f}%")
    
    # 5. 可视化
    plot_history(history, args.save_dir)

def plot_history(history, save_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path)
    print(f"训练曲线已保存至: {save_path}")
    # plt.show() # 在无头环境中可能无法显示

def main():
    parser = argparse.ArgumentParser(description='MNIST Training with NumPy')
    
    # 获取当前文件所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 默认模型保存路径为当前目录下的 model 文件夹
    default_model_dir = os.path.join(base_dir, 'model')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train') # 默认20个epoch以确保达到95%准确率
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128], help='Hidden layer dimensions')
    parser.add_argument('--save_dir', type=str, default=default_model_dir, help='Directory to save model and logs')
    
    args = parser.parse_args()
    
    # 确保 save_dir 是绝对路径
    if not os.path.isabs(args.save_dir):
        # 如果是相对路径，则相对于当前工作目录
        args.save_dir = os.path.join(os.getcwd(), args.save_dir)
        
    train(args)

if __name__ == "__main__":
    main()
