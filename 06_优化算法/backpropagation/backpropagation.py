"""
反向传播算法实现

反向传播(Backpropagation)是训练神经网络的核心算法，通过链式法则高效计算损失函数
关于每个参数的梯度。本模块实现了从零开始的反向传播算法，并构建了简单的神经网络。

核心思想：
1. 前向传播：输入数据通过网络层层计算得到输出
2. 计算损失：比较输出与真实标签的差异
3. 反向传播：从输出层向输入层逐层计算梯度
4. 参数更新：使用梯度下降更新参数
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class Node:
    """
    计算图中的节点

    用于表示计算图中的操作，支持前向传播和反向传播。
    """

    def __init__(self, inputs=[]):
        """
        初始化节点

        Args:
            inputs: 输入节点列表
        """
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        # 将当前节点添加到输入节点的输出列表中
        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        """前向传播：计算节点的值"""
        raise NotImplementedError

    def backward(self):
        """反向传播：计算梯度"""
        raise NotImplementedError


class Input(Node):
    """输入节点"""

    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        """设置输入值"""
        if value is not None:
            self.value = value

    def backward(self):
        """输入节点不需要计算梯度"""
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


class Linear(Node):
    """线性变换节点：y = Wx + b"""

    def __init__(self, X, W, b):
        """
        Args:
            X: 输入节点
            W: 权重节点
            b: 偏置节点
        """
        Node.__init__(self, [X, W, b])

    def forward(self):
        """前向传播：计算 y = Wx + b"""
        X = self.inputs[0].value
        W = self.inputs[1].value
        b = self.inputs[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        """
        反向传播：计算梯度

        dL/dX = dL/dy * W^T
        dL/dW = X^T * dL/dy
        dL/db = sum(dL/dy)
        """
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            # 获取后继节点传来的梯度
            grad_cost = n.gradients[self]

            # 计算各个输入的梯度
            self.gradients[self.inputs[0]] += np.dot(grad_cost, self.inputs[1].value.T)
            self.gradients[self.inputs[1]] += np.dot(self.inputs[0].value.T, grad_cost)
            self.gradients[self.inputs[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    """Sigmoid激活函数节点"""

    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """Sigmoid函数：1 / (1 + exp(-x))"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self):
        """前向传播"""
        self.value = self._sigmoid(self.inputs[0].value)

    def backward(self):
        """
        反向传播

        sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        """
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inputs[0]] += sigmoid * (1 - sigmoid) * grad_cost


class MSE(Node):
    """均方误差损失节点"""

    def __init__(self, y_pred, y_true):
        """
        Args:
            y_pred: 预测值节点
            y_true: 真实值节点
        """
        Node.__init__(self, [y_pred, y_true])

    def forward(self):
        """计算MSE损失"""
        y_pred = self.inputs[0].value
        y_true = self.inputs[1].value
        self.diff = y_pred - y_true
        self.value = np.mean(self.diff ** 2)

    def backward(self):
        """
        反向传播

        dL/dy_pred = 2 * (y_pred - y_true) / n
        """
        self.gradients[self.inputs[0]] = 2 * self.diff / self.diff.shape[0]
        self.gradients[self.inputs[1]] = -2 * self.diff / self.diff.shape[0]


def topological_sort(feed_dict):
    """
    拓扑排序计算图

    Args:
        feed_dict: 输入节点和值的字典

    Returns:
        排序后的节点列表
    """
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outputs:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        if isinstance(n, Input):
            n.value = feed_dict[n]
        L.append(n)
        for m in n.outputs:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)

    return L


def forward_and_backward(graph):
    """
    执行前向传播和反向传播

    Args:
        graph: 拓扑排序后的计算图
    """
    # 前向传播
    for n in graph:
        n.forward()

    # 反向传播
    for n in graph[::-1]:
        n.backward()


class NeuralNetwork:
    """
    简单的全连接神经网络

    支持多层全连接网络，使用Sigmoid激活函数和MSE损失。
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        """
        初始化神经网络

        Args:
            layer_sizes: 每层的神经元数量，例如 [2, 4, 1] 表示输入2个，隐藏层4个，输出1个
            learning_rate: 学习率
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # He初始化
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            X: 输入数据，形状为 (batch_size, input_dim)

        Returns:
            网络输出，形状为 (batch_size, output_dim)
        """
        self.activations = [X]

        # 逐层前向传播
        for i in range(len(self.weights)):
            # 线性变换
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]

            # 激活函数
            if i < len(self.weights) - 1:
                # 隐藏层使用Sigmoid
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            else:
                # 输出层直接输出（对于回归任务）
                a = z

            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X: np.ndarray, y: np.ndarray) -> Tuple[List, List]:
        """
        反向传播计算梯度

        Args:
            X: 输入数据
            y: 真实标签

        Returns:
            权重梯度和偏置梯度的列表
        """
        m = X.shape[0]  # 样本数量

        # 输出层的梯度（MSE损失）
        delta = self.activations[-1] - y

        # 存储梯度
        weight_grads = []
        bias_grads = []

        # 从输出层向输入层反向传播
        for i in range(len(self.weights) - 1, -1, -1):
            # 计算权重和偏置的梯度
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            weight_grads.insert(0, dW)
            bias_grads.insert(0, db)

            # 计算前一层的梯度
            if i > 0:
                # Sigmoid的导数
                delta = np.dot(delta, self.weights[i].T) * self.activations[i] * (1 - self.activations[i])

        return weight_grads, bias_grads

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        执行一步训练

        Args:
            X: 输入数据
            y: 真实标签

        Returns:
            当前损失值
        """
        # 前向传播
        y_pred = self.forward(X)

        # 计算损失
        loss = np.mean((y_pred - y) ** 2)

        # 反向传播
        weight_grads, bias_grads = self.backward(X, y)

        # 更新参数
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_grads[i]
            self.biases[i] -= self.learning_rate * bias_grads[i]

        return loss

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000,
              verbose: bool = True) -> List[float]:
        """
        训练网络

        Args:
            X: 训练数据
            y: 训练标签
            epochs: 训练轮数
            verbose: 是否打印训练信息

        Returns:
            每个epoch的损失列表
        """
        losses = []

        for epoch in range(epochs):
            loss = self.train_step(X, y)
            losses.append(loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入数据

        Returns:
            预测结果
        """
        return self.forward(X)


# 示例使用
if __name__ == "__main__":
    print("=" * 60)
    print("反向传播算法演示")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(42)

    # ========== 1. 计算图示例 ==========
    print("\n1. 计算图示例")
    print("-" * 60)

    # 创建简单的计算图: y = sigmoid(W*x + b)
    X = Input()
    W = Input()
    b = Input()

    # 前向传播
    linear = Linear(X, W, b)
    output = Sigmoid(linear)

    # 设置输入值
    feed_dict = {
        X: np.array([[1.0, 2.0]]),
        W: np.array([[0.5], [0.3]]),
        b: np.array([[0.1]])
    }

    # 拓扑排序
    graph = topological_sort(feed_dict)

    # 前向传播
    for node in graph:
        node.forward()

    print(f"输入 X: {feed_dict[X]}")
    print(f"权重 W: {feed_dict[W].flatten()}")
    print(f"偏置 b: {feed_dict[b].flatten()}")
    print(f"线性输出: {linear.value}")
    print(f"Sigmoid输出: {output.value}")

    # ========== 2. 简单神经网络训练 ==========
    print("\n" + "=" * 60)
    print("2. 简单神经网络训练示例")
    print("=" * 60)

    # 生成XOR数据
    X_train = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    y_train = np.array([[0],
                       [1],
                       [1],
                       [0]])

    print("\nXOR问题数据：")
    print("输入:", X_train)
    print("输出:", y_train.flatten())

    # 创建神经网络 [2, 4, 1]
    nn = NeuralNetwork([2, 4, 1], learning_rate=0.5)

    print("\n网络结构: 输入层(2) -> 隐藏层(4) -> 输出层(1)")
    print("开始训练...")

    # 训练网络
    losses = nn.train(X_train, y_train, epochs=2000, verbose=True)

    # 测试
    print("\n训练后的预测结果：")
    predictions = nn.predict(X_train)
    for i in range(len(X_train)):
        print(f"输入: {X_train[i]}, 真实值: {y_train[i][0]}, "
              f"预测值: {predictions[i][0]:.4f}, "
              f"预测: {1 if predictions[i][0] > 0.5 else 0}")

    # ========== 3. 可视化训练过程 ==========
    print("\n" + "=" * 60)
    print("3. 可视化训练过程")
    print("=" * 60)

    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 决策边界
    plt.subplot(1, 2, 2)
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(),
               s=200, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('决策边界')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # ========== 4. 回归任务示例 ==========
    print("\n" + "=" * 60)
    print("4. 回归任务示例 (拟合sin函数)")
    print("=" * 60)

    # 生成sin函数数据
    X_reg = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y_reg = np.sin(X_reg)

    # 创建回归网络
    nn_reg = NeuralNetwork([1, 10, 10, 1], learning_rate=0.1)

    print("网络结构: 输入层(1) -> 隐藏层(10) -> 隐藏层(10) -> 输出层(1)")
    print("训练中...")

    # 训练
    losses_reg = nn_reg.train(X_reg, y_reg, epochs=1000, verbose=False)

    # 预测
    y_pred = nn_reg.predict(X_reg)

    # 可视化
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses_reg, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(X_reg, y_reg, 'b-', label='真实值', linewidth=2)
    plt.plot(X_reg, y_pred, 'r--', label='预测值', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sin函数拟合')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n最终损失: {losses_reg[-1]:.6f}")
