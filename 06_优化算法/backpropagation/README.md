# 反向传播 (Backpropagation)

## 概述

反向传播（Backpropagation，简称BP）是训练神经网络的核心算法，由Rumelhart、Hinton和Williams在1986年提出。它通过链式法则高效地计算损失函数关于网络中每个参数的梯度，使得深度神经网络的训练成为可能。

## 核心思想

反向传播的本质是**利用链式法则进行梯度计算**：

1. **前向传播**：输入数据从输入层经过隐藏层到达输出层，逐层计算
2. **计算损失**：比较网络输出与真实标签，计算损失值
3. **反向传播**：从输出层开始，向输入层逐层传播梯度
4. **参数更新**：使用计算得到的梯度更新网络参数

## 数学原理

### 链式法则

反向传播的数学基础是微积分中的链式法则。对于复合函数 $f(g(x))$：

$$
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

在神经网络中，如果有 $L = f(g(h(x)))$，则：

$$
\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}
$$

### 前向传播

对于一个简单的三层网络：

$$
\begin{aligned}
z^{(1)} &= W^{(1)}x + b^{(1)} \\
a^{(1)} &= \sigma(z^{(1)}) \\
z^{(2)} &= W^{(2)}a^{(1)} + b^{(2)} \\
\hat{y} &= \sigma(z^{(2)}) \\
L &= \frac{1}{2}(\hat{y} - y)^2
\end{aligned}
$$

其中：
- $W^{(i)}$, $b^{(i)}$ 是第 $i$ 层的权重和偏置
- $\sigma$ 是激活函数（如Sigmoid）
- $L$ 是损失函数

### 反向传播

从输出层开始，逐层计算梯度：

**输出层：**
$$
\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = (\hat{y} - y) \cdot \sigma'(z^{(2)})
$$

**隐藏层：**
$$
\delta^{(1)} = \frac{\partial L}{\partial z^{(1)}} = (W^{(2)})^T \delta^{(2)} \cdot \sigma'(z^{(1)})
$$

**权重和偏置的梯度：**
$$
\begin{aligned}
\frac{\partial L}{\partial W^{(2)}} &= \delta^{(2)} (a^{(1)})^T \\
\frac{\partial L}{\partial b^{(2)}} &= \delta^{(2)} \\
\frac{\partial L}{\partial W^{(1)}} &= \delta^{(1)} x^T \\
\frac{\partial L}{\partial b^{(1)}} &= \delta^{(1)}
\end{aligned}
$$

## 计算图

计算图是表示计算过程的有向图，每个节点代表一个操作或变量。

### 计算图示例

```
输入 x ──→ [Linear] ──→ [Sigmoid] ──→ [MSE] ──→ 损失
         ↑                            ↑
      权重 W, b                    真实值 y
```

**前向传播**：沿着图的方向计算
**反向传播**：沿着图的反方向传播梯度

## 算法步骤

### 完整的训练流程

```
for epoch in range(num_epochs):
    # 1. 前向传播
    y_pred = forward(X)

    # 2. 计算损失
    loss = compute_loss(y_pred, y_true)

    # 3. 反向传播计算梯度
    grads = backward(loss)

    # 4. 更新参数
    update_parameters(grads)
```

### 详细步骤

1. **初始化**：随机初始化网络权重
2. **前向传播**：
   - 对每一层 $l$：计算 $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$
   - 应用激活函数：$a^{(l)} = \sigma(z^{(l)})$
3. **计算损失**：$L = \text{loss}(\hat{y}, y)$
4. **反向传播**：
   - 计算输出层梯度：$\delta^{(L)}$
   - 逐层向前计算：$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \cdot \sigma'(z^{(l)})$
   - 计算参数梯度：$\frac{\partial L}{\partial W^{(l)}}$, $\frac{\partial L}{\partial b^{(l)}}$
5. **参数更新**：$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$

## 代码实现

### 计算图节点

```python
class Node:
    """计算图中的节点"""

    def __init__(self, inputs=[]):
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

    def forward(self):
        """前向传播"""
        raise NotImplementedError

    def backward(self):
        """反向传播"""
        raise NotImplementedError
```

### 线性层

```python
class Linear(Node):
    """线性变换：y = Wx + b"""

    def forward(self):
        X = self.inputs[0].value
        W = self.inputs[1].value
        b = self.inputs[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        # 计算输入的梯度
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] += np.dot(grad_cost, self.inputs[1].value.T)
            self.gradients[self.inputs[1]] += np.dot(self.inputs[0].value.T, grad_cost)
            self.gradients[self.inputs[2]] += np.sum(grad_cost, axis=0)
```

### 简单神经网络

```python
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        # 初始化权重和偏置
        self.weights = []
        self.biases = []

    def forward(self, X):
        """前向传播"""
        self.activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, X, y):
        """反向传播"""
        delta = self.activations[-1] - y
        weight_grads = []
        bias_grads = []

        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / X.shape[0]
            db = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
            weight_grads.insert(0, dW)
            bias_grads.insert(0, db)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * \
                        self.activations[i] * (1 - self.activations[i])

        return weight_grads, bias_grads

    def train_step(self, X, y):
        """训练一步"""
        y_pred = self.forward(X)
        loss = np.mean((y_pred - y) ** 2)
        weight_grads, bias_grads = self.backward(X, y)

        # 更新参数
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_grads[i]
            self.biases[i] -= self.learning_rate * bias_grads[i]

        return loss
```

## 使用示例

### XOR问题

```python
# 创建XOR数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 创建网络
nn = NeuralNetwork([2, 4, 1], learning_rate=0.5)

# 训练
losses = nn.train(X, y, epochs=2000)

# 预测
predictions = nn.predict(X)
print(predictions)
```

### 函数拟合

```python
# 生成数据
X = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y = np.sin(X)

# 创建深层网络
nn = NeuralNetwork([1, 10, 10, 1], learning_rate=0.1)

# 训练
losses = nn.train(X, y, epochs=1000)

# 预测
y_pred = nn.predict(X)
```

## 常见激活函数及其导数

| 激活函数 | 表达式 | 导数 |
|---------|--------|------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $\tanh'(x) = 1-\tanh^2(x)$ |
| ReLU | $\text{ReLU}(x) = \max(0, x)$ | $\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ |
| Leaky ReLU | $f(x) = \max(0.01x, x)$ | $f'(x) = \begin{cases} 1 & x > 0 \\ 0.01 & x \leq 0 \end{cases}$ |

## 梯度消失和梯度爆炸

### 梯度消失

当网络很深时，梯度在反向传播过程中可能会变得非常小，导致前面层的权重几乎不更新。

**原因**：
- Sigmoid/Tanh激活函数的导数 < 1
- 链式法则导致梯度相乘，逐层衰减

**解决方法**：
- 使用ReLU等激活函数
- 使用残差连接（ResNet）
- 批量归一化（Batch Normalization）
- 使用LSTM/GRU（对于RNN）

### 梯度爆炸

梯度变得非常大，导致权重更新过大，训练不稳定。

**解决方法**：
- 梯度裁剪（Gradient Clipping）
- 权重正则化
- 使用更小的学习率
- 批量归一化

## 实践技巧

1. **权重初始化**
   - Xavier初始化：适用于Sigmoid/Tanh
   - He初始化：适用于ReLU
   - 避免全零初始化

2. **学习率选择**
   - 从小学习率开始（如0.001）
   - 使用学习率衰减
   - 尝试学习率调度策略

3. **批量大小**
   - 小批量（32-256）通常效果最好
   - 太大：收敛慢，泛化性差
   - 太小：训练不稳定

4. **正则化**
   - L2正则化（权重衰减）
   - Dropout
   - 早停（Early Stopping）

5. **梯度检查**
   - 使用数值梯度验证反向传播实现
   - 仅在调试时使用（计算成本高）

## 数值梯度检查

验证反向传播实现是否正确：

```python
def numerical_gradient(f, x, h=1e-5):
    """数值方法计算梯度"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        x[idx] = old_value + h
        fxh1 = f(x)

        x[idx] = old_value - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = old_value
        it.iternext()

    return grad
```

## 优缺点

### 优点
- ✅ **高效**：相比于数值梯度，计算效率高
- ✅ **精确**：可以准确计算梯度
- ✅ **通用**：适用于各种网络结构
- ✅ **自动化**：现代框架（PyTorch、TensorFlow）自动实现

### 缺点
- ❌ **实现复杂**：手动实现容易出错
- ❌ **梯度问题**：可能出现梯度消失或爆炸
- ❌ **局部最优**：可能陷入局部最优解

## 现代深度学习框架

现代框架提供了自动微分功能，无需手动实现反向传播：

**PyTorch示例**：
```python
import torch

# 定义网络
model = torch.nn.Sequential(
    torch.nn.Linear(2, 4),
    torch.nn.Sigmoid(),
    torch.nn.Linear(4, 1)
)

# 前向传播
output = model(x)
loss = ((output - y) ** 2).mean()

# 反向传播（自动）
loss.backward()

# 更新参数
optimizer.step()
```

## 历史意义

反向传播算法的提出是深度学习发展的里程碑：
- **1986年**：Rumelhart等人发表反向传播算法
- **2006年**：Hinton提出深度信念网络，深度学习复兴
- **2012年**：AlexNet在ImageNet上取得突破性成果
- **现在**：反向传播是所有深度学习模型的基础

## 参考文献

1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.
2. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (1998). Efficient backprop. Neural networks: Tricks of the trade, 9-50.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

## 文件说明

- `backpropagation.py`: Python实现（包含计算图和神经网络）
- `backpropagation.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档
