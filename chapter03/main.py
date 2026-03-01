from __future__ import annotations
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
import sys
import importlib
import datetime

# 尝试导入 TUI 相关库
try:
    import pytest
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
    from rich.progress import (
        Progress, 
        SpinnerColumn, 
        TextColumn, 
        BarColumn, 
        TaskProgressColumn, 
        TimeRemainingColumn
    )
    from rich.layout import Layout
    from rich.live import Live
    from rich import print as rprint
    from rich.traceback import install
    install(show_locals=True) # Better traceback
except ImportError:
    print("Please install required packages: pip install pytest rich")
    sys.exit(1)

# Global placeholders for dynamic imports
MLP = None
Linear = None
Sigmoid = None
Softmax = None
CrossEntropyLoss = None
MNISTLoader = None
DataLoader = None
one_hot_encode = None

class Hyperparameters:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.hidden_dims = args.hidden_dims
        self.save_dir = args.save_dir
        self.last_update_time = None

    def update(self, key, value):
        setattr(self, key, value)
        self.last_update_time = datetime.datetime.now()

def load_modules(mode):
    """Dynamically load model and data modules based on mode"""
    global MLP, Linear, Sigmoid, Softmax, CrossEntropyLoss
    global MNISTLoader, DataLoader, one_hot_encode
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, mode)
    
    if not os.path.exists(target_dir):
        print(f"Error: Mode directory '{target_dir}' does not exist.")
        sys.exit(1)
        
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
    
    try:
        import model
        import data
        importlib.reload(model)
        importlib.reload(data)
        
        MLP = model.MLP
        Linear = model.Linear
        Sigmoid = model.Sigmoid
        Softmax = model.Softmax
        CrossEntropyLoss = model.CrossEntropyLoss
        
        MNISTLoader = data.MNISTLoader
        DataLoader = data.DataLoader
        one_hot_encode = data.one_hot_encode
        
        print(f"Successfully loaded modules from {mode}")
    except ImportError as e:
        print(f"Failed to import modules from {mode}: {e}")
        sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description='MNIST Training with NumPy')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用相对路径作为默认值，以便在 main 中更容易判断用户是否修改了它
    default_model_dir = 'model'
    
    parser.add_argument('--mode', type=str, choices=['startup', 'solution'], help='Run mode: startup or solution')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128], help='Hidden layer dimensions')
    parser.add_argument('--save_dir', type=str, default=default_model_dir, help='Directory to save model and logs')
    
    args, unknown = parser.parse_known_args()
    
    # 注意：这里我们不再自动将 save_dir 转换为绝对路径，
    # 而是留给 main 函数根据是否为 default_model_dir 来处理
    # 如果用户输入了相对路径（非默认值），main 函数也会处理
        
    return args

def run_pytest(args_list, mode):
    """Wrapper to run pytest and capture output"""
    console = Console()
    console.print(Panel(f"Running Tests in {mode} mode...", style="bold blue"))
    
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_bench.py")
    
    if not os.path.exists(test_file):
        console.print(f"[red]Error: Test file not found at {test_file}[/red]")
        return
        
    pytest_args = ["-v", test_file, "--mode", mode] + args_list
    
    result = pytest.main(pytest_args)
    
    if result == 0:
        console.print(Panel("All tests passed!", style="bold green"))
    else:
        console.print(Panel("Some tests failed. Check output above.", style="bold red"))
        
    console.input("\nPress Enter to continue...")

def configure_hyperparameters(config: Hyperparameters):
    """Interactive screen to configure hyperparameters"""
    console = Console()
    
    while True:
        console.clear()
        
        table = Table(title="Hyperparameters Configuration", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="dim", width=4)
        table.add_column("Parameter", style="green")
        table.add_column("Current Value", style="yellow")
        table.add_column("Description")
        
        table.add_row("1", "Learning Rate (lr)", str(config.lr), "Float (0.0001 - 1.0)")
        table.add_row("2", "Batch Size", str(config.batch_size), "Integer (1 - 1024)")
        table.add_row("3", "Epochs", str(config.epochs), "Integer (1 - 100)")
        table.add_row("4", "Hidden Dims", str(config.hidden_dims), "List of Ints (e.g. 256 128)")
        table.add_row("0", "Back", "", "Return to Main Menu")
        
        console.print(Panel(table, border_style="blue"))
        
        choice = IntPrompt.ask("Select parameter to edit", choices=["0", "1", "2", "3", "4"], default=0)
        
        if choice == 0:
            break
        elif choice == 1:
            new_lr = FloatPrompt.ask("Enter new Learning Rate", default=config.lr)
            if 0.0001 <= new_lr <= 1.0:
                config.update('lr', new_lr)
            else:
                console.print("[red]Invalid range! Must be between 0.0001 and 1.0[/red]")
                time.sleep(1.5)
        elif choice == 2:
            new_bs = IntPrompt.ask("Enter new Batch Size", default=config.batch_size)
            if 1 <= new_bs <= 1024:
                config.update('batch_size', new_bs)
            else:
                console.print("[red]Invalid range! Must be between 1 and 1024[/red]")
                time.sleep(1.5)
        elif choice == 3:
            new_epochs = IntPrompt.ask("Enter new Epochs", default=config.epochs)
            if 1 <= new_epochs <= 100:
                config.update('epochs', new_epochs)
            else:
                console.print("[red]Invalid range! Must be between 1 and 100[/red]")
                time.sleep(1.5)
        elif choice == 4:
            dims_str = Prompt.ask("Enter Hidden Dims (space separated)", default=" ".join(map(str, config.hidden_dims)))
            try:
                new_dims = [int(x) for x in dims_str.split()]
                if all(d > 0 for d in new_dims):
                    config.update('hidden_dims', new_dims)
                else:
                    console.print("[red]Dimensions must be positive integers![/red]")
                    time.sleep(1.5)
            except ValueError:
                console.print("[red]Invalid input! Use space separated integers.[/red]")
                time.sleep(1.5)

def generate_loss_sparkline(loss_history: List[float], width: int = 40) -> str:
    """Generate a simple text-based sparkline for loss"""
    if not loss_history:
        return ""
    
    # Use last N points
    data = loss_history[-width:]
    if not data:
        return ""
        
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val > min_val else 1.0
    
    # Unicode blocks for sparkline levels
    blocks = "  ▂▃▄▅▆▇█"
    
    sparkline = ""
    for val in data:
        normalized = (val - min_val) / range_val
        index = int(normalized * (len(blocks) - 1))
        sparkline += blocks[index]
        
    return sparkline

def bench(mode, initial_args=None):
    console = Console()
    # 如果提供了初始参数，则使用它，否则从命令行获取
    if initial_args:
        args = initial_args
    else:
        args = get_args()
        
    # Ensure args has mode
    args.mode = mode 
    
    # Create mutable config
    config = Hyperparameters(args)
    
    while True:
        console.clear()
        
        # Header
        title = f"""[bold cyan]
███╗   ███╗███╗   ██╗██╗███████╗████████╗
████╗ ████║████╗  ██║██║██╔════╝╚══██╔══╝
██╔████╔██║██╔██╗ ██║██║███████╗   ██║   
██║╚██╔╝██║██║╚██╗██║██║╚════██║   ██║   
██║ ╚═╝ ██║██║ ╚████║██║███████║   ██║   
╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝   ╚═╝   
        Model Verification & Benchmark
              [Mode: {mode.upper()}]
[/bold cyan]"""
        console.print(title, justify="center")
        
        # Menu Table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=4)
        table.add_column("Option", min_width=20)
        table.add_column("Description", justify="right")
        
        table.add_row("1", "Run All Tests", "Run complete test suite via Pytest")
        table.add_row("2", "Test Linear Layer", "Forward, Backward, Gradient, Stability")
        table.add_row("3", "Test Sigmoid", "Activation function checks")
        table.add_row("4", "Test Softmax", "Probability distribution checks")
        table.add_row("5", "Test CrossEntropyLoss", "Loss calculation & Gradient")
        table.add_row("6", "Start Training", "Train model on MNIST (main)")
        table.add_row("7", "Hyperparameters", "Configure training parameters")
        table.add_row("0", "Exit", "Exit application")
        
        console.print(Panel(table, title="Main Menu", border_style="blue"))
        
        choice = IntPrompt.ask("Select an option", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default=1)
        
        if choice == 0:
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)
        elif choice == 1:
            run_pytest([], mode)
        elif choice == 2:
            run_pytest(["-k", "Linear"], mode)
        elif choice == 3:
            run_pytest(["-k", "Sigmoid"], mode)
        elif choice == 4:
            run_pytest(["-k", "Softmax"], mode)
        elif choice == 5:
            run_pytest(["-k", "CrossEntropyLoss"], mode)
        elif choice == 6:
            try:
                train(config)
                console.input("\nPress Enter to return to menu...")
            except KeyboardInterrupt:
                console.print("\n[yellow]Training interrupted by user.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Training Error: {e}[/bold red]")
                console.print_exception()
                console.input("\nPress Enter to continue...")
        elif choice == 7:
            configure_hyperparameters(config)

def evaluate(model, x: np.ndarray, y: np.ndarray, batch_size: int = 128) -> Tuple[float, float]:
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

def train(config: Hyperparameters):
    """训练流程"""
    console = Console()
    
    # 1. 准备数据
    console.print(Panel("Loading MNIST Dataset...", style="bold green"))
    try:
        loader = MNISTLoader()
        x_train, y_train, x_test, y_test = loader.load_data()
    except Exception as e:
        console.print(f"[bold red]Failed to load data: {e}[/bold red]")
        raise e
    
    # 划分验证集
    val_size = 5000
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    
    console.print(f"Training Set: {x_train.shape[0]}, Validation Set: {x_val.shape[0]}, Test Set: {x_test.shape[0]}")
    
    # 2. 构建模型
    console.print(f"Building Model: 784 -> {config.hidden_dims} -> 10")
    model = MLP(input_dim=784, hidden_dims=config.hidden_dims, output_dim=10)
    
    # 3. 训练循环
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    last_save_time = None
    
    console.print(f"Start Training: Epochs={config.epochs}, Batch Size={config.batch_size}, LR={config.lr}")
    start_time = time.time()
    
    # 使用 Rich Live Display for advanced dashboard
    layout = Layout()
    layout.split_column(
        Layout(name="progress", size=3),
        Layout(name="stats", size=10)
    )
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )
    
    epoch_task = progress.add_task("[green]Training...", total=config.epochs)
    
    with Live(layout, refresh_per_second=4, console=console) as live:
        layout["progress"].update(progress)
        
        for epoch in range(config.epochs):
            # Update Status Panel
            last_update_str = config.last_update_time.strftime("%H:%M:%S") if config.last_update_time else "N/A"
            last_save_str = last_save_time.strftime("%H:%M:%S") if last_save_time else "N/A"
            sparkline = generate_loss_sparkline(history['train_loss'])
            
            stats_table = Table(show_header=False, box=None)
            stats_table.add_row("Last Param Update:", last_update_str)
            stats_table.add_row("Last Model Save:", last_save_str)
            stats_table.add_row("Current LR:", str(config.lr))
            stats_table.add_row("Loss Trend:", sparkline)
            
            if history['val_acc']:
                stats_table.add_row("Best Val Acc:", f"{best_val_acc:.4f}")
                
            layout["stats"].update(Panel(stats_table, title="Training Status", border_style="cyan"))
            
            # 训练阶段
            train_loader = DataLoader(x_train, one_hot_encode(y_train, 10), batch_size=config.batch_size, shuffle=True)
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
                grad = model.loss_fn.backpropagation(config.lr)
                model.backward(grad, config.lr)
                
            train_loss = epoch_loss / total
            train_acc = correct / total
            
            # 验证阶段
            val_loss, val_acc = evaluate(model, x_val, y_val, config.batch_size)
            
            # 记录日志
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            progress.update(epoch_task, advance=1, description=f"[green]Epoch {epoch+1}/{config.epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if not os.path.exists(config.save_dir):
                    try:
                        os.makedirs(config.save_dir, exist_ok=True)
                    except OSError as e:
                        console.print(f"[red]Error creating save directory {config.save_dir}: {e}[/red]")
                        
                try:
                    save_path = os.path.join(config.save_dir, 'best_model.pkl')
                    model.save(save_path, verbose=False)
                    last_save_time = datetime.datetime.now()
                    # 可以在这里添加保存成功的逻辑，例如记录日志
                except Exception as e:
                     console.print(f"[red]Failed to save model to {save_path}: {e}[/red]")
                     # 尝试备用路径
                     backup_dir = os.path.join(os.getcwd(), 'backup_models')
                     try:
                         os.makedirs(backup_dir, exist_ok=True)
                         backup_path = os.path.join(backup_dir, f'best_model_{mode}_{int(time.time())}.pkl')
                         model.save(backup_path, verbose=False)
                         console.print(f"[yellow]Model saved to backup path: {backup_path}[/yellow]")
                     except Exception as e2:
                         console.print(f"[bold red]Critical: Failed to save to backup path: {e2}[/bold red]")
            
    total_time = time.time() - start_time
    console.print(f"Training Finished! Total Time: {total_time:.2f}s", style="bold green")
    
    # 4. 测试集最终评估
    try:
        model.load(os.path.join(config.save_dir, 'best_model.pkl'))
        test_loss, test_acc = evaluate(model, x_test, y_test)
        console.print(f"Final Test Accuracy: {test_acc*100:.2f}%", style="bold cyan")
    except Exception as e:
        console.print(f"[red]Error loading best model for evaluation: {e}[/red]")
    
    # 5. 可视化
    try:
        plot_history(history, config.save_dir)
    except Exception as e:
        console.print(f"[red]Error plotting history: {e}[/red]")

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
    rprint(f"Training history saved to: [underline]{save_path}[/underline]")

def select_mode():
    """TUI for selecting mode"""
    console = Console()
    console.clear()
    
    title = """[bold cyan]
    Select Running Mode
    [/bold cyan]"""
    console.print(title, justify="center")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Mode", min_width=20)
    table.add_column("Description", justify="right")
    
    table.add_row("1", "Startup", "Run incomplete student version (for development)")
    table.add_row("2", "Solution", "Run reference solution (for verification)")
    
    console.print(Panel(table, title="Mode Selection", border_style="blue"))
    
    choice = IntPrompt.ask("Select a mode", choices=["1", "2"], default=1)
    
    if choice == 1:
        return "startup"
    else:
        return "solution"

def main():
    args = get_args()
    mode = args.mode
    
    if not mode:
        mode = select_mode()
        
    # 动态调整保存路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 检查 args.save_dir 是否等于默认值 'model'
    # 或者是否等于旧的默认绝对路径（为了兼容性）
    default_model_rel = 'model'
    default_model_abs = os.path.join(base_dir, 'model')
    
    if args.save_dir == default_model_rel or args.save_dir == default_model_abs:
        # 如果用户未指定特定路径（即使用默认路径），则根据 mode 自动调整
        # startup -> chapter03/startup/model
        # solution -> chapter03/solution/model
        
        # 修正：base_dir 已经是 chapter03 的绝对路径
        # 我们需要确保 target_dir 指向 chapter03/mode/model
        target_dir = os.path.join(base_dir, mode, 'model')
        args.save_dir = target_dir
        print(f"Auto-configured save directory for {mode} mode: {args.save_dir}")
    else:
        # 如果用户指定了路径，且是相对路径，转换为绝对路径
        if not os.path.isabs(args.save_dir):
            args.save_dir = os.path.join(os.getcwd(), args.save_dir)
            
    load_modules(mode)
    bench(mode, initial_args=args)

if __name__ == "__main__":
    main()
