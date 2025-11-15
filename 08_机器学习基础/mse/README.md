# 均方误差 (Mean Squared Error, MSE)

## 概述

均方误差（MSE）是机器学习中最常用的损失函数之一，特别是在回归问题中。它通过计算预测值与真实值差值的平方的平均值，来衡量模型的预测性能。MSE具有良好的数学性质，便于优化，是许多经典算法的基础。

## 核心思想

MSE的核心是通过**平方惩罚误差**：
1. 计算每个样本的预测值与真实值的差值（残差）
2. 对差值进行平方（放大大误差，减小小误差）
3. 求所有样本的平均值

平方操作使得：
- 大误差受到更严重的惩罚
- 损失函数处处可导
- 优化问题具有凸性（对于线性模型）

## 数学定义

### 基本形式

对于n个样本，MSE定义为：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

其中：
- $y_i$ 是第 $i$ 个样本的真实值
- $\hat{y}_i$ 是第 $i$ 个样本的预测值
- $n$ 是样本数量

### 向量形式

$$
\text{MSE} = \frac{1}{n} \|y - \hat{y}\|^2 = \frac{1}{n} (y - \hat{y})^T(y - \hat{y})
$$

### 梯度计算

MSE关于预测值的梯度：

$$
\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

在线性回归中 $\hat{y} = Xw + b$，关于参数的梯度：

$$
\begin{aligned}
\frac{\partial \text{MSE}}{\partial w} &= \frac{2}{n} X^T(Xw + b - y) \\
\frac{\partial \text{MSE}}{\partial b} &= \frac{2}{n} \sum_{i=1}^{n}(\hat{y}_i - y_i)
\end{aligned}
$$

## 特点与性质

### 优点

- ✅ **数学性质好**：处处可导，便于优化
- ✅ **凸函数**：对于线性模型，MSE是凸函数，保证全局最优
- ✅ **唯一解**：在正则化线性回归中有唯一闭式解
- ✅ **统计意义**：与最大似然估计（高斯噪声假设）等价
- ✅ **梯度明确**：梯度计算简单，易于实现

### 缺点

- ❌ **对异常值敏感**：平方操作放大了大误差的影响
- ❌ **量纲问题**：MSE的单位是原始单位的平方，不易解释
- ❌ **不对称性**：过高预测和过低预测受到相同惩罚

## 相关指标

### 1. RMSE (均方根误差)

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

**优点**：
- 与目标变量同量纲，更易解释
- 对大误差的惩罚介于MSE和MAE之间

### 2. MAE (平均绝对误差)

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**优点**：
- 对异常值不敏感
- 与目标变量同量纲
- 所有误差等权重

**缺点**：
- 在零点不可导
- 优化较困难

### 3. R² (决定系数)

$$
R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y)} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
$$

**解释**：
- 取值范围：(-∞, 1]
- 1表示完美拟合
- 0表示模型等同于预测均值
- 负值表示模型比预测均值还差

## MSE vs MAE

| 特性 | MSE | MAE |
|------|-----|-----|
| 公式 | $(y - \hat{y})^2$ | $|y - \hat{y}|$ |
| 可导性 | 处处可导 | 零点不可导 |
| 异常值敏感度 | 高（平方放大） | 低（线性） |
| 优化难度 | 容易 | 较难 |
| 统计假设 | 高斯噪声 | 拉普拉斯噪声 |
| 解释性 | 较差（量纲平方） | 好（同量纲） |

## 代码实现

### 基础MSE计算

```python
def mse(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)
```

### MSE梯度

```python
def mse_gradient(y_true, y_pred):
    """计算MSE梯度"""
    n = len(y_true)
    return (2.0 / n) * (y_pred - y_true)
```

### 线性回归（使用MSE）

```python
class LinearRegressionMSE:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # 前向传播
            y_pred = np.dot(X, self.weights) + self.bias

            # 计算梯度
            dw = (2.0 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2.0 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

## 使用示例

### 基础计算

```python
import numpy as np

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.3, 2.8, 4.2, 4.9])

# 计算MSE
mse_value = mse(y_true, y_pred)
print(f"MSE: {mse_value:.4f}")

# 计算RMSE
rmse_value = np.sqrt(mse_value)
print(f"RMSE: {rmse_value:.4f}")
```

### 线性回归示例

```python
from mse import LinearRegressionMSE

# 生成数据
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.randn(100) * 0.5

# 训练模型
model = LinearRegressionMSE(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 评估
mse_val = mse(y, y_pred)
print(f"MSE: {mse_val:.4f}")
```

## 应用场景

### 1. 回归问题

MSE是回归任务的标准损失函数：
- 房价预测
- 股票价格预测
- 温度预测
- 销量预测

### 2. 神经网络训练

作为输出层的损失函数：
- 全连接网络
- 卷积神经网络（图像回归）
- 循环神经网络（时间序列预测）

### 3. 模型评估

作为模型性能的评价指标：
- 交叉验证
- 模型选择
- 超参数调优

### 4. 图像处理

- 图像去噪
- 图像重建
- 图像超分辨率

## 优化方法

### 1. 闭式解（线性回归）

对于线性回归，MSE有解析解（正规方程）：

$$
w = (X^TX)^{-1}X^Ty
$$

**优点**：一步求解，无需迭代
**缺点**：矩阵求逆复杂度O(n³)，特征数量大时不适用

### 2. 梯度下降

迭代优化，适用于大规模问题：

```python
# 批量梯度下降
for iteration in range(n_iterations):
    y_pred = np.dot(X, w) + b
    dw = (2.0 / n) * np.dot(X.T, (y_pred - y))
    db = (2.0 / n) * np.sum(y_pred - y)
    w -= learning_rate * dw
    b -= learning_rate * db
```

### 3. 随机梯度下降

每次使用单个样本更新：

```python
for iteration in range(n_iterations):
    for i in range(n):
        y_pred_i = np.dot(X[i], w) + b
        dw = 2 * X[i] * (y_pred_i - y[i])
        db = 2 * (y_pred_i - y[i])
        w -= learning_rate * dw
        b -= learning_rate * db
```

### 4. Mini-Batch梯度下降

每次使用小批量样本：

```python
batch_size = 32
for iteration in range(n_iterations):
    indices = np.random.choice(n, batch_size)
    X_batch = X[indices]
    y_batch = y[indices]
    # 计算梯度并更新
```

## 实践技巧

### 1. 数据预处理

- **标准化**：将特征缩放到相似范围
- **归一化**：将数据映射到[0, 1]或[-1, 1]
- **处理异常值**：考虑使用Huber损失或MAE

### 2. 正则化

避免过拟合：

**L2正则化（Ridge）**：
$$
L = \text{MSE} + \lambda \|w\|^2
$$

**L1正则化（Lasso）**：
$$
L = \text{MSE} + \lambda \|w\|_1
$$

### 3. 学习率选择

- 从小学习率开始（如0.01）
- 使用学习率衰减
- 尝试自适应学习率（Adam、RMSprop）

### 4. 监控训练

- 绘制损失曲线
- 使用验证集评估
- 早停（Early Stopping）

## 变体和扩展

### 1. Weighted MSE

为不同样本赋予不同权重：

$$
\text{WMSE} = \frac{1}{n} \sum_{i=1}^{n} w_i(y_i - \hat{y}_i)^2
$$

### 2. Huber Loss

结合MSE和MAE的优点：

$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

### 3. Log-Cosh Loss

$$
L(y, \hat{y}) = \sum_{i=1}^{n} \log(\cosh(\hat{y}_i - y_i))
$$

## 常见问题

### 1. MSE为什么除以n？

- 保持损失值的尺度稳定
- 使梯度不依赖于样本数量
- 便于不同数据集间的比较

### 2. 为什么梯度公式有系数2？

- 来自求导：$\frac{d}{dx}x^2 = 2x$
- 通常在实现中可以省略（调整学习率）

### 3. 何时使用MSE而非MAE？

**使用MSE**：
- 大误差需要严重惩罚
- 数据符合高斯分布
- 需要梯度下降优化

**使用MAE**：
- 数据有异常值
- 需要鲁棒性
- 所有误差等权重

### 4. 如何处理MSE的量纲问题？

- 使用RMSE代替MSE
- 使用相对误差（MAPE）
- 使用R²等无量纲指标

## 统计解释

在统计学中，MSE与最大似然估计密切相关：

假设噪声服从高斯分布：
$$
y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

最大化似然函数等价于最小化MSE：
$$
\max_w \prod_{i=1}^{n} p(y_i|x_i, w) \Leftrightarrow \min_w \sum_{i=1}^{n}(y_i - f(x_i; w))^2
$$

这解释了为什么MSE在假设高斯噪声时是自然的选择。

## 参考文献

1. Lehmann, E. L., & Casella, G. (1998). Theory of point estimation. Springer.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning. Springer.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

## 文件说明

- `mse.py`: Python实现（包含MSE、RMSE、MAE和线性回归）
- `mse.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档
