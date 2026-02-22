# MS for AI 第三章实践
本章的内容是NumPy入门

## 任务
### 任务说明
使用NumPy实现神经网络训练与推理. MNIST是非常有名的手写数字数据集. 其包含60000张训练样本和10000张测试样本. 

已经实现了MLP的代码框架, 你需要在startup补全以下内容:
`model.py`中的`Linear`, `Sigmoid`, `Softmax`, `CrossEntropyLoss`类.

### 设计指导
#### Linear层
Linear层是神经网络中最基本的层, 其本质是将输入从一个空间线性变换到另外一个空间(例如 $n$ 到 $m$ 维空间), 它的前向传播公式为:

$$
Y = XW + b 
$$

其中, $X\in \mathbb{R}^{n\times d}$是输入, $W\in \mathbb{R}^{d\times m}$是权重矩阵, $b\in \mathbb{R}^{1\times m}$是偏置向量, $Y\in \mathbb{R}^{n\times m}$是输出.

实际上, 在计算过程中, $b$ 会进行广播, 在数学上表现为前面乘以一个列向量$\bold{1}_N \in \mathbb{R}^{n\times 1}$, 所以, 实际上的表达式为:

$$
Y = XW + \bold{1}_N b 
$$

由于NumPy自带广播机制, 因此实现上的形式依然是第一个式子.

下面推导反向传播. 我们在进行Linear层的反向传播时, 得到的梯度张量是$\frac{\partial L}{\partial Y}$. 我们需要计算的是$\frac{\partial L}{\partial X}$, $\frac{\partial L}{\partial W}$和$\frac{\partial L}{\partial b}$. 下面我们从$(2)$式出发, 推导这三个量的计算过程:

$$
\begin{aligned}
\bold{d}Y &= \bold{d}(XW + \bold{1}_N b) \\
&= \bold{d}(XW) + \bold{d}(\bold{1}_N b) \\
&= (\bold{d}X)W + X\bold{d}W + (\bold{d1_N})b + \bold{1}_N\bold{d}b\\
&= (\bold{d}X)W + X\bold{d}W + \bold{1}_N\bold{d}b \\
\end{aligned}
$$

$\frac{\partial L}{\partial Y}\in\mathbb{R}^{n\times m}$, $\frac{\partial L}{\partial X}\in\mathbb{R}^{n\times d}$, $\frac{\partial L}{\partial W}\in\mathbb{R}^{d\times m}$, $\frac{\partial L}{\partial b}\in\mathbb{R}^{1\times m}$. 这是我们期望的形状. 我们知道:

$$
\bold{d}L = \text{tr}((\frac{\partial L}{\partial Y})^\bold{T}\bold{d}Y)
$$

将$\bold{d}Y$替换, 得到:

$$
\begin{aligned}
\bold{d}L &= \text{tr}\{(\frac{\partial L}{\partial Y})^\bold{T}[(\bold{d}X)W + X\bold{d}W + \bold{1}_N\bold{d}b]\}  \\
&= \text{tr}((\frac{\partial L}{\partial Y})^\bold{T}(\bold{d}X)W) + \text{tr}((\frac{\partial L}{\partial Y})^\bold{T}X\bold{d}W) + 
\text{tr}((\frac{\partial L}{\partial Y})^\bold{T}\bold{1}_N\bold{d}b) 
\end{aligned}
$$

因此, 我们可以得到:

$$
\begin{aligned}
\frac{\partial L}{\partial X} &= (W(\frac{\partial L}{\partial W})^\bold{T})^\bold{T} = \frac{\partial L}{\partial Y}W^\bold{T} \\
\frac{\partial L}{\partial W} &= ((\frac{\partial L}{\partial Y})^\bold{T}X)^\bold{T} = X^T\frac{\partial L}{\partial Y} \\
\frac{\partial L}{\partial b} &= ((\frac{\partial L}{\partial Y})^\bold{T}\bold{1}_N)^\bold{T} = \bold{1}_N^\bold{T}\frac{\partial L}{\partial Y} \\
\end{aligned}
$$

有关$\bold{1}_N^\bold{T}\frac{\partial L}{\partial Y}$, 在实现上就是对$\frac{\partial L}{\partial Y}$的每一行求和.

#### Sigmoid层
Sigmoid层的前向传播公式为:

$$
Y = \sigma(X) = \frac{1}{1 + e^{-X}}
$$

其中, $\sigma$是Sigmoid函数. 反向传播的公式为:

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}\odot Y\odot(1 - Y)
$$

在实现上, 可以将前向传播的运算结果$Y$暂时存储起来, 用于反向传播时的计算.
#### Softmax层
Softmax层的前向传播公式为:

$$
Y = \text{softmax}(X) = \frac{e^X}{\sum_{i=1}^N e^{X_i}}
$$

其中, $X\in \mathbb{R}^{n\times D}$是输入, $Y\in \mathbb{R}^{n\times D}$是输出.
反向传播的公式为:

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot Y \odot (1 - Y)
$$

#### CrossEntropyLoss层
CrossEntropyLoss层的前向传播公式为:

$$
L = -\frac{1}{N}\sum_{i=1}^N{\bold{y}_i}\log{\hat{\bold{y}}_i}
$$

其中, $\bold{y}$为标签向量, $\hat{\bold{y}}$为模型预测向量. $N$为向量长度.
反向传播的公式为:

$$
\frac{\partial L}{\partial \hat{\bold{y}}} = -\frac{1}{N}\frac{\bold{y}}{\bold{\hat{y}}}
$$

### 如何运行
在MSforAI-assignments目录下, 运行以下命令:
```bash
python chapter03/startup/main.py
```
会以默认配置运行正确性测试, 并且开始训练一个手写数字识别的模型.