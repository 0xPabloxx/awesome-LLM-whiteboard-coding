# 梯度下降 (Gradient Descent)

## 概述

梯度下降是深度学习中最基础、最重要的优化算法。它通过迭代地沿着损失函数梯度的反方向更新参数，使损失函数逐步收敛到最小值。本模块实现了几种常见的梯度下降变体：SGD、Momentum、RMSprop和Adam。

## 核心思想

梯度下降的基本思想是：
1. **计算梯度**：计算损失函数关于参数的梯度
2. **反向更新**：沿着梯度的反方向更新参数
3. **迭代收敛**：重复上述过程直到损失收敛

梯度指向函数值增长最快的方向，因此沿着梯度的反方向可以最快地减小函数值。

## 数学原理

### 基本梯度下降

给定损失函数 $L(\theta)$ 和参数 $\theta$，基本的更新规则为：

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

其中：
- $\theta_t$ 是第 $t$ 步的参数
- $\eta$ 是学习率（步长）
- $\nabla L(\theta_t)$ 是损失函数在 $\theta_t$ 处的梯度

## 优化算法变体

### 1. SGD (随机梯度下降)

**更新规则**：
$$
\theta = \theta - \eta \nabla L(\theta)
$$

**特点**：
- ✅ 简单直接，易于实现
- ✅ 内存占用小
- ❌ 收敛速度慢
- ❌ 容易陷入局部最优
- ❌ 对学习率敏感

**适用场景**：简单的凸优化问题

### 2. Momentum (动量法)

**更新规则**：
$$
\begin{aligned}
v_t &= \beta v_{t-1} + \nabla L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta v_t
\end{aligned}
$$

**特点**：
- ✅ 加速收敛，特别是在相关方向上
- ✅ 减少震荡
- ✅ 能够跳出浅层局部最优
- ❌ 引入了额外的超参数 $\beta$

**物理类比**：就像一个球在斜坡上滚动，会累积动量加速前进

**适用场景**：损失函数存在狭长峡谷的情况

### 3. RMSprop

**更新规则**：
$$
\begin{aligned}
v_t &= \beta v_{t-1} + (1 - \beta) (\nabla L(\theta_t))^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \nabla L(\theta_t)
\end{aligned}
$$

**特点**：
- ✅ 自适应学习率
- ✅ 对每个参数使用不同的学习率
- ✅ 适合处理非平稳目标
- ✅ 对RNN效果好

**核心思想**：用梯度平方的移动平均来调整学习率

**适用场景**：非平稳、在线学习场景

### 4. Adam (Adaptive Moment Estimation)

**更新规则**：
$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t) \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{aligned}
$$

**特点**：
- ✅ 结合了Momentum和RMSprop的优点
- ✅ 计算高效，内存需求少
- ✅ 对超参数不敏感
- ✅ 适合大多数深度学习问题
- ✅ 默认参数通常表现良好

**核心组件**：
- $m_t$: 梯度的一阶矩估计（均值）
- $v_t$: 梯度的二阶矩估计（未中心化方差）
- 偏差修正：补偿初始化为0导致的偏差

**推荐参数**：
- $\eta = 0.001$
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

**适用场景**：大多数深度学习任务的首选优化器

## 优化器对比

| 优化器 | 学习率 | 收敛速度 | 内存占用 | 适用场景 |
|--------|--------|----------|----------|----------|
| SGD | 固定 | 慢 | 低 | 简单问题 |
| Momentum | 固定 | 中等 | 中等 | 峡谷形损失 |
| RMSprop | 自适应 | 快 | 中等 | 非平稳问题 |
| Adam | 自适应 | 快 | 中等 | 通用场景 |

## 学习率选择

学习率是梯度下降中最重要的超参数：

- **过大**：导致发散，损失震荡甚至增大
- **过小**：收敛太慢，可能陷入局部最优
- **合适**：快速且稳定地收敛到最优解

**常用策略**：
1. **固定学习率**：简单但不够灵活
2. **学习率衰减**：随着训练逐渐减小学习率
3. **学习率预热**：开始时使用小学习率，逐渐增大
4. **周期性学习率**：周期性地调整学习率

## 代码实现

### SGD实现

```python
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        return params - self.learning_rate * grads
```

### Momentum实现

```python
class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity + grads
        return params - self.learning_rate * self.velocity
```

### Adam实现

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # 更新一阶和二阶矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 更新参数
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

## 使用示例

```python
import numpy as np

# 定义损失函数和梯度
def loss_fn(params):
    return params[0]**2 + params[1]**2

def grad_fn(params):
    return 2 * params

# 初始化参数
params = np.array([5.0, 5.0])

# 使用不同的优化器
optimizers = {
    'SGD': SGD(learning_rate=0.1),
    'Momentum': Momentum(learning_rate=0.1, momentum=0.9),
    'Adam': Adam(learning_rate=0.5)
}

# 优化
for name, optimizer in optimizers.items():
    p = params.copy()
    for i in range(20):
        grads = grad_fn(p)
        p = optimizer.update(p, grads)
    print(f"{name}: 最终参数 = {p}, 损失 = {loss_fn(p):.6f}")
```

## 可视化

本模块提供了可视化功能来比较不同优化器：

1. **优化路径可视化**：在损失函数等高线上显示优化轨迹
2. **收敛速度对比**：绘制损失随迭代次数的变化曲线

## 实践建议

1. **首选Adam**：对于大多数深度学习任务，Adam是最好的起点
2. **调试用SGD**：SGD虽然慢，但更稳定，便于调试
3. **微调用SGD+Momentum**：在预训练模型的基础上微调时，SGD with Momentum通常表现更好
4. **学习率搜索**：使用学习率范围测试（LR Range Test）找到合适的学习率
5. **监控梯度**：注意梯度爆炸和梯度消失问题

## 常见问题

### 1. 为什么损失不下降？
- 学习率过大或过小
- 梯度计算错误
- 数据预处理问题
- 初始化不当

### 2. 为什么损失震荡？
- 学习率过大
- 批量大小过小
- 使用学习率衰减或Momentum

### 3. 如何选择优化器？
- 快速原型：Adam
- 追求最佳性能：SGD + Momentum + 学习率调度
- 大批量训练：LAMB或LARS

## 参考文献

1. Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv:1609.04747.
2. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
3. Sutskever, I., et al. (2013). On the importance of initialization and momentum in deep learning. ICML.

## 文件说明

- `gradient_descent.py`: Python实现（包含SGD、Momentum、RMSprop、Adam）
- `gradient_descent.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档
