from __future__ import annotations
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# 导入自定义模块
try:
    from model import MLP, Linear, Sigmoid, Softmax, CrossEntropyLoss
    from data import MNISTLoader, DataLoader, one_hot_encode
except ImportError:
    # 如果作为脚本直接运行，可能需要调整路径
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import MLP, Linear, Sigmoid, Softmax, CrossEntropyLoss
    from data import MNISTLoader, DataLoader, one_hot_encode

def bench():
    print("开始验证实现正确性...")
    print("(1) Linear Layer")
    try:
        # Initialize
        linear = Linear(2, 3)
        # Manually set weights for deterministic testing
        linear.w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        linear.b = np.array([0.1, 0.2, 0.3])
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Forward check
        # Expected: x @ w + b
        # [[1*1+2*4, 1*2+2*5, 1*3+2*6], ...] + b
        # [[9, 12, 15], [19, 26, 33]] + [0.1, 0.2, 0.3]
        expected_out = x @ linear.w + linear.b
        out = linear(x)
        
        if not np.allclose(out, expected_out):
            print("  [FAIL] Forward pass incorrect")
            print(f"  Expected:\n{expected_out}")
            print(f"  Got:\n{out}")
        else:
            print("  [PASS] Forward pass")
            
        # Backward check
        # Use expected shape for grad_output, not out.shape (which might be wrong)
        grad_output = np.ones((2, 3)) 
        lr = 0.1
        
        # Store old params to check update
        w_old = linear.w.copy()
        b_old = linear.b.copy()
        
        # Call backward
        grad_input = linear.backpropagation(grad_output, lr)
        
        # Check grad_input: dL/dx = dL/dy @ w.T
        expected_grad_input = grad_output @ w_old.T
        if grad_input is None:
             print("  [FAIL] Backward pass returned None")
        elif not np.allclose(grad_input, expected_grad_input):
            print("  [FAIL] Backward pass (grad_input) incorrect")
            print(f"  Expected:\n{expected_grad_input}")
            print(f"  Got:\n{grad_input}")
        else:
            print("  [PASS] Backward pass (grad_input)")
            
        # Check Parameter Update
        # w_new = w - lr * dL/dw = w - lr * (x.T @ grad_output)
        # b_new = b - lr * dL/db = b - lr * sum(grad_output)
        
        expected_w = w_old - lr * (x.T @ grad_output)
        expected_b = b_old - lr * np.sum(grad_output, axis=0)
        
        if not np.allclose(linear.w, expected_w):
            print("  [FAIL] Parameter update (w) incorrect")
            print(f"  Expected:\n{expected_w}")
            print(f"  Got:\n{linear.w}")
        else:
            print("  [PASS] Parameter update (w)")
            
        if not np.allclose(linear.b, expected_b):
            print("  [FAIL] Parameter update (b) incorrect")
            print(f"  Expected:\n{expected_b}")
            print(f"  Got:\n{linear.b}")
        else:
            print("  [PASS] Parameter update (b)")
            
    except Exception as e:
        print(f"  [ERROR] Linear test failed with exception: {e}")

    print("\n(2) Sigmoid Activation")
    try:
        sigmoid = Sigmoid()
        x = np.array([[0.0, 2.0], [-2.0, 0.0]])
        # Forward
        out = sigmoid(x)
        expected_out = 1 / (1 + np.exp(-x))
        
        if not np.allclose(out, expected_out):
            print("  [FAIL] Forward pass incorrect")
            print(f"  Expected:\n{expected_out}")
            print(f"  Got:\n{out}")
        else:
            print("  [PASS] Forward pass")
            
        # Backward
        grad_output = np.ones((2, 2))
        grad_input = sigmoid.backpropagation(grad_output, 0.1)
        # dL/dx = dL/dy * y * (1-y)
        expected_grad = grad_output * expected_out * (1 - expected_out)
        
        if grad_input is None:
             print("  [FAIL] Backward pass returned None")
        elif not np.allclose(grad_input, expected_grad):
            print("  [FAIL] Backward pass incorrect")
            print(f"  Expected:\n{expected_grad}")
            print(f"  Got:\n{grad_input}")
        else:
            print("  [PASS] Backward pass")
            
    except Exception as e:
        print(f"  [ERROR] Sigmoid test failed: {e}")

    print("\n(3) Softmax Activation")
    try:
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        # Forward
        out = softmax(x)
        # Softmax: exp(x) / sum(exp(x))
        exp_x = np.exp(x)
        expected_out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        if not np.allclose(out, expected_out):
            print("  [FAIL] Forward pass incorrect")
            print(f"  Expected:\n{expected_out}")
            print(f"  Got:\n{out}")
        else:
            print("  [PASS] Forward pass")
            
        if not np.allclose(np.sum(out, axis=1), 1.0):
             print("  [FAIL] Output does not sum to 1")
        else:
             print("  [PASS] Output sums to 1")
        
        # Softmax backward is usually handled with Loss, so we skip rigorous numerical check 
        # unless user implemented full Jacobian. 
        # But we can check if it runs without error.
        try:
            grad_output = np.ones((2, 3))
            grad_input = softmax.backpropagation(grad_output, 0.1)
            # Just check shape matches
            if grad_input.shape != x.shape:
                print(f"  [WARN] Backward pass output shape mismatch. Expected {x.shape}, got {grad_input.shape}")
            else:
                print("  [PASS] Backward pass runs (shape check only)")
        except:
            print("  [WARN] Backward pass not implemented or failed")
             
    except Exception as e:
        print(f"  [ERROR] Softmax test failed: {e}")

    print("\n(4) CrossEntropyLoss")
    try:
        loss_fn = CrossEntropyLoss()
        # Batch size 2, 3 classes
        pred = np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]])
        y = np.array([[0, 1, 0], [0, 1, 0]]) # One-hot
        
        # Forward
        loss = loss_fn(pred, y)
        # Loss = -1/N * sum(sum(y * log(pred)))
        # For sample 1: -1 * log(0.5) = 0.693
        # For sample 2: -1 * log(0.8) = 0.223
        # Mean = (0.693 + 0.223) / 2 = 0.458
        expected_loss = -float(np.sum(y * np.log(pred + 1e-15))) / pred.shape[0]
        
        if not np.isclose(loss, expected_loss, atol=1e-5):
            print("  [FAIL] Forward pass incorrect")
            print(f"  Expected: {expected_loss}")
            print(f"  Got: {loss}")
        else:
            print("  [PASS] Forward pass")
            
        # Backward
        grad = loss_fn.backpropagation(0.1)
        # Expected: (pred - y) / N
        expected_grad = (pred - y) / pred.shape[0]
        
        if grad is None:
             print("  [FAIL] Backward pass returned None")
        elif not np.allclose(grad, expected_grad):
             print("  [FAIL] Backward pass incorrect")
             print(f"  Expected:\n{expected_grad}")
             print(f"  Got:\n{grad}")
        else:
             print("  [PASS] Backward pass")
             
    except Exception as e:
        print(f"  [ERROR] CrossEntropyLoss test failed: {e}")
    
    print("\n验证结束。注意：如果测试未通过，请检查 model.py 中的实现。")
    print("--------------------------------------------------")
    

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
    bench()
    main()
