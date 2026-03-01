import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import os
import sys

# 兼容不同版本的 Pillow 重采样滤镜
def get_lanczos_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    elif hasattr(Image, "LANCZOS"):
        return Image.LANCZOS
    else:
        return Image.ANTIALIAS

LANCZOS = get_lanczos_filter()

class HandwritingApp:
    def __init__(self, root, mlp_class, model_dir):
        self.root = root
        self.root.title("手写数字识别 (Handwriting Digit Recognition)")
        self.mlp_class = mlp_class
        self.model_dir = model_dir
        
        # 1. 创建界面元素
        # 画布设置
        self.canvas_width = 400
        self.canvas_height = 400
        self.bg_color = "white"
        self.draw_color = "black"
        self.line_width = 15  # 增加线宽以确保缩小后仍清晰
        
        # 主框架
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)
        
        # 标题
        tk.Label(main_frame, text="请在下方区域手写数字 (0-9)", font=("Arial", 12)).pack(pady=5)
        
        # 绘图区域
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color, cursor="cross", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(side=tk.TOP, pady=5)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # 用于保存绘图内容的PIL图像 (与Canvas同步)
        # 模式 'L' (8-bit pixels, black and white)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_obj = ImageDraw.Draw(self.image)
        
        # 按钮区域
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(side=tk.TOP, pady=10)
        
        self.btn_clear = tk.Button(btn_frame, text="清除 (Clear)", command=self.clear_canvas, width=15, height=2)
        self.btn_clear.pack(side=tk.LEFT, padx=10)
        
        self.btn_recognize = tk.Button(btn_frame, text="识别 (Recognize)", command=self.recognize, width=15, height=2, bg="#ddd")
        self.btn_recognize.pack(side=tk.LEFT, padx=10)
        
        # 结果显示标签
        self.lbl_result = tk.Label(main_frame, text="预测结果：等待绘制...", font=("Arial", 16, "bold"), fg="#333")
        self.lbl_result.pack(side=tk.TOP, pady=15)
        
        # 绘图状态
        self.last_x = None
        self.last_y = None
        
        # 加载模型
        self.model = None
        self.model_type = None
        self.load_model()
        
    def load_model(self):
        """加载模型文件"""
        # 优先查找 model.h5 (用户要求)，其次查找 best_model.pkl (本项目默认输出)
        # 搜索路径包括当前目录和 model 子目录
        possible_paths = [
            os.path.join(self.model_dir, "model.h5"),
            os.path.join(self.model_dir, "best_model.pkl"),
            "model.h5", 
            "best_model.pkl"
        ]
        
        loaded = False
        for full_path in possible_paths:
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(full_path):
                full_path = os.path.abspath(full_path)
                
            if os.path.exists(full_path):
                try:
                    if full_path.endswith(".h5"):
                        # 如果是 h5 文件，尝试使用 keras 加载
                        import tensorflow as tf
                        self.model = tf.keras.models.load_model(full_path)
                        self.model_type = "keras"
                        print(f"成功加载 Keras 模型: {full_path}")
                    else:
                        # 默认加载本项目自定义 MLP 模型
                        if self.mlp_class is None:
                            raise ImportError("无法导入 MLP 类")
                        # 注意：需要知道模型结构参数。默认使用 [256, 128]
                        self.model = self.mlp_class(784, [256, 128], 10)
                        self.model.load(full_path)
                        self.model_type = "custom"
                        print(f"成功加载自定义 MLP 模型: {full_path}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"尝试加载 {full_path} 失败: {e}")
        
        if not loaded:
            err_msg = f"未找到模型文件！\n请确保目录下存在 model.h5 或 best_model.pkl\n搜索路径: {self.model_dir}"
            self.lbl_result.config(text="错误：模型未加载", fg="red")
            messagebox.showerror("模型加载失败", err_msg)
    
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
        
    def draw(self, event):
        if self.last_x and self.last_y:
            x, y = event.x, event.y
            # 在 Canvas 上绘制 (用于显示)
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                  width=self.line_width, fill=self.draw_color, 
                                  capstyle=tk.ROUND, smooth=True)
            # 在 PIL Image 上绘制 (用于识别)
            # 注意：PIL 的 line 宽度处理可能与 tkinter 不同，稍作调整
            self.draw_obj.line([self.last_x, self.last_y, x, y], fill=0, width=self.line_width)
            self.last_x = x
            self.last_y = y
            
    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        # 重置 PIL 图像为全白
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="预测结果：等待绘制...", fg="black")
        
    def recognize(self):
        if self.model is None:
            messagebox.showerror("错误", "模型未加载，无法识别！")
            return
            
        try:
            # 1. 图像预处理
            # 调整大小为 28x28
            # 使用 LANCZOS 进行高质量下采样
            img_resized = self.image.resize((28, 28), LANCZOS)
            
            # 转换为 numpy 数组
            img_array = np.array(img_resized)
            
            # 反色处理 & 归一化
            # Canvas: 白底(255)黑字(0) -> MNIST: 黑底(0)白字(1.0)
            # 公式: (255 - pixel) / 255.0
            img_data = (255.0 - img_array) / 255.0
            
            # 展平并添加 batch 维度: (1, 784)
            img_data = img_data.reshape(1, 784).astype(np.float32)
            
            # 2. 模型推理
            
            pred = self.model.forward(img_data)
                
            # 3. 解析结果
            # pred 是概率分布 (Softmax 输出)
            # 找到最大概率对应的索引和概率值
            max_prob = np.max(pred)
            pred_class = np.argmax(pred)
            
            # 4. 结果展示
            confidence = max_prob * 100
            
            print(f"预测类别: {pred_class}, 置信度: {confidence:.2f}%")
            
            if confidence < 60:
                result_text = "无法识别，请重新绘制"
                fg_color = "red"
            else:
                result_text = f"预测结果：{pred_class} (置信度：{confidence:.1f}%)"
                fg_color = "green"
                
            self.lbl_result.config(text=result_text, fg=fg_color)
            
        except Exception as e:
            print(f"识别过程出错: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("识别错误", f"识别过程出错: {e}")

def run_app(mlp_class, model_dir):
    root = tk.Tk()
    # 设置窗口居中
    window_width = 450
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_cordinate = int((screen_width/2) - (window_width/2))
    y_cordinate = int((screen_height/2) - (window_height/2))
    root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
    
    app = HandwritingApp(root, mlp_class, model_dir)
    root.mainloop()

if __name__ == "__main__":
    # Standalone testing only
    print("This module is intended to be run from main.py")
    # Mock MLP for testing UI if run directly
    class MockMLP:
        def __init__(self, *args): pass
        def load(self, path): pass
        def forward(self, x): return np.array([[0.1]*10])
    
    run_app(MockMLP, ".")
