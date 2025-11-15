"""
梯度下降优化算法实现

梯度下降是深度学习中最基础的优化算法，通过迭代地沿着损失函数梯度的反方向更新参数来最小化损失函数。
本模块实现了几种常见的梯度下降变体：SGD、Momentum、Adam等。

核心思想：
1. 计算损失函数关于参数的梯度
2. 沿着梯度的反方向更新参数
3. 重复上述过程直到收敛
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


class SGD:
    """
    随机梯度下降 (Stochastic Gradient Descent)

    最基础的优化算法，每次迭代直接沿着梯度的反方向更新参数。
    更新规则：θ = θ - lr * ∇L(θ)
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        初始化SGD优化器

        Args:
            learning_rate: 学习率，控制每次更新的步长
        """
        self.learning_rate = learning_rate

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        更新参数

        Args:
            params: 当前参数
            grads: 参数的梯度

        Returns:
            更新后的参数
        """
        return params - self.learning_rate * grads


class Momentum:
    """
    动量法 (Momentum)

    在SGD的基础上增加了动量项，可以加速收敛并减少震荡。
    动量项累积了之前梯度的指数加权平均，使得参数更新更加平滑。

    更新规则：
    v = β * v + ∇L(θ)
    θ = θ - lr * v
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        初始化Momentum优化器

        Args:
            learning_rate: 学习率
            momentum: 动量系数，通常设为0.9
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        使用动量更新参数

        Args:
            params: 当前参数
            grads: 参数的梯度

        Returns:
            更新后的参数
        """
        # 初始化速度
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        # 更新速度：累积历史梯度信息
        self.velocity = self.momentum * self.velocity + grads

        # 更新参数
        return params - self.learning_rate * self.velocity


class RMSprop:
    """
    RMSprop优化器

    自适应学习率方法，对每个参数使用不同的学习率。
    通过梯度平方的指数加权平均来调整学习率。

    更新规则：
    v = β * v + (1 - β) * (∇L(θ))²
    θ = θ - lr * ∇L(θ) / (√v + ε)
    """

    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.9, epsilon: float = 1e-8):
        """
        初始化RMSprop优化器

        Args:
            learning_rate: 学习率
            decay_rate: 衰减率，控制历史梯度的权重
            epsilon: 防止除零的小常数
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        使用RMSprop更新参数

        Args:
            params: 当前参数
            grads: 参数的梯度

        Returns:
            更新后的参数
        """
        # 初始化缓存
        if self.cache is None:
            self.cache = np.zeros_like(params)

        # 更新梯度平方的移动平均
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * grads ** 2

        # 自适应学习率更新
        return params - self.learning_rate * grads / (np.sqrt(self.cache) + self.epsilon)


class Adam:
    """
    Adam优化器 (Adaptive Moment Estimation)

    结合了Momentum和RMSprop的优点，是目前最常用的优化算法之一。
    同时维护梯度的一阶矩估计（均值）和二阶矩估计（未中心化的方差）。

    更新规则：
    m = β1 * m + (1 - β1) * ∇L(θ)          # 一阶矩估计
    v = β2 * v + (1 - β2) * (∇L(θ))²      # 二阶矩估计
    m̂ = m / (1 - β1^t)                    # 偏差修正
    v̂ = v / (1 - β2^t)                    # 偏差修正
    θ = θ - lr * m̂ / (√v̂ + ε)
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        初始化Adam优化器

        Args:
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            epsilon: 防止除零的小常数
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩估计
        self.v = None  # 二阶矩估计
        self.t = 0     # 时间步

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        使用Adam更新参数

        Args:
            params: 当前参数
            grads: 参数的梯度

        Returns:
            更新后的参数
        """
        # 初始化矩估计
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # 更新时间步
        self.t += 1

        # 更新一阶矩估计（动量）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        # 更新二阶矩估计（梯度平方的移动平均）
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 更新参数
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def visualize_optimization(optimizers: dict, loss_fn: Callable, grad_fn: Callable,
                          init_params: np.ndarray, num_iterations: int = 100,
                          x_range: Tuple[float, float] = (-5, 5),
                          y_range: Tuple[float, float] = (-5, 5)) -> None:
    """
    可视化不同优化器的优化过程

    Args:
        optimizers: 优化器字典，格式为 {名称: 优化器对象}
        loss_fn: 损失函数
        grad_fn: 梯度函数
        init_params: 初始参数
        num_iterations: 迭代次数
        x_range: x轴范围
        y_range: y轴范围
    """
    # 准备绘图
    fig, axes = plt.subplots(1, len(optimizers), figsize=(6 * len(optimizers), 5))
    if len(optimizers) == 1:
        axes = [axes]

    # 创建网格用于绘制等高线
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = loss_fn(np.array([X[i, j], Y[i, j]]))

    # 对每个优化器进行优化并可视化
    for idx, (name, optimizer) in enumerate(optimizers.items()):
        ax = axes[idx]

        # 绘制损失函数等高线
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)

        # 优化过程
        params = init_params.copy()
        trajectory = [params.copy()]

        for _ in range(num_iterations):
            grads = grad_fn(params)
            params = optimizer.update(params, grads)
            trajectory.append(params.copy())

        # 绘制优化轨迹
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=2, markersize=8,
                alpha=0.7, label='优化路径')
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='起点')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='终点')

        ax.set_xlabel('参数 x')
        ax.set_ylabel('参数 y')
        ax.set_title(f'{name} 优化过程')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_convergence(optimizers: dict, loss_fn: Callable, grad_fn: Callable,
                       init_params: np.ndarray, num_iterations: int = 100) -> None:
    """
    比较不同优化器的收敛速度

    Args:
        optimizers: 优化器字典
        loss_fn: 损失函数
        grad_fn: 梯度函数
        init_params: 初始参数
        num_iterations: 迭代次数
    """
    plt.figure(figsize=(10, 6))

    for name, optimizer in optimizers.items():
        params = init_params.copy()
        losses = []

        for _ in range(num_iterations):
            # 记录损失
            losses.append(loss_fn(params))

            # 更新参数
            grads = grad_fn(params)
            params = optimizer.update(params, grads)

        # 绘制损失曲线
        plt.plot(losses, label=name, linewidth=2)

    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('不同优化器的收敛速度对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标
    plt.show()


# 示例使用
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)

    print("=" * 60)
    print("梯度下降优化算法演示")
    print("=" * 60)

    # 定义一个简单的二次函数作为损失函数
    # L(x, y) = x^2 + y^2
    def simple_loss(params):
        """简单的二次损失函数"""
        return params[0] ** 2 + params[1] ** 2

    def simple_grad(params):
        """损失函数的梯度"""
        return 2 * params

    # 定义一个更复杂的函数 (Rosenbrock函数)
    # L(x, y) = (1 - x)^2 + 100(y - x^2)^2
    def rosenbrock_loss(params):
        """Rosenbrock函数（香蕉函数）"""
        x, y = params
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def rosenbrock_grad(params):
        """Rosenbrock函数的梯度"""
        x, y = params
        dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
        dy = 200 * (y - x ** 2)
        return np.array([dx, dy])

    # 初始化参数
    init_params = np.array([4.0, 4.0])

    # 创建不同的优化器
    optimizers = {
        'SGD': SGD(learning_rate=0.01),
        'Momentum': Momentum(learning_rate=0.01, momentum=0.9),
        'RMSprop': RMSprop(learning_rate=0.1, decay_rate=0.9),
        'Adam': Adam(learning_rate=0.1, beta1=0.9, beta2=0.999)
    }

    print("\n1. 简单二次函数优化")
    print("-" * 60)

    # 测试简单函数
    for name, optimizer in optimizers.items():
        params = init_params.copy()
        print(f"\n{name}:")
        print(f"  初始参数: {params}")

        # 优化10步
        for i in range(10):
            grads = simple_grad(params)
            params = optimizer.update(params, grads)

        print(f"  最终参数: {params}")
        print(f"  最终损失: {simple_loss(params):.6f}")

    print("\n" + "=" * 60)
    print("2. 可视化优化过程（简单二次函数）")
    print("=" * 60)

    # 重新创建优化器（重置状态）
    optimizers_viz = {
        'SGD': SGD(learning_rate=0.1),
        'Momentum': Momentum(learning_rate=0.1, momentum=0.9),
        'Adam': Adam(learning_rate=0.3, beta1=0.9, beta2=0.999)
    }

    visualize_optimization(optimizers_viz, simple_loss, simple_grad,
                          init_params, num_iterations=50)

    print("\n" + "=" * 60)
    print("3. 收敛速度对比")
    print("=" * 60)

    # 重新创建优化器
    optimizers_conv = {
        'SGD': SGD(learning_rate=0.01),
        'Momentum': Momentum(learning_rate=0.01, momentum=0.9),
        'RMSprop': RMSprop(learning_rate=0.1, decay_rate=0.9),
        'Adam': Adam(learning_rate=0.1, beta1=0.9, beta2=0.999)
    }

    compare_convergence(optimizers_conv, simple_loss, simple_grad,
                       init_params, num_iterations=100)

    print("\n" + "=" * 60)
    print("4. Rosenbrock函数优化（更具挑战性）")
    print("=" * 60)

    # 使用Rosenbrock函数测试
    init_params_rb = np.array([-1.0, 2.0])
    optimizers_rb = {
        'SGD': SGD(learning_rate=0.001),
        'Momentum': Momentum(learning_rate=0.001, momentum=0.9),
        'Adam': Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
    }

    visualize_optimization(optimizers_rb, rosenbrock_loss, rosenbrock_grad,
                          init_params_rb, num_iterations=200,
                          x_range=(-2, 2), y_range=(-1, 3))
