"""
均方误差 (Mean Squared Error, MSE) 实现

MSE是机器学习中最常用的损失函数之一，用于衡量预测值与真实值之间的差异。
本模块实现了MSE的计算、梯度计算，并展示了在线性回归中的应用。

核心思想：
1. 计算预测值与真实值的差值
2. 对差值进行平方
3. 求所有样本的平均值
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差

    Args:
        y_true: 真实值，形状为 (n_samples,) 或 (n_samples, 1)
        y_pred: 预测值，形状与y_true相同

    Returns:
        均方误差值
    """
    return np.mean((y_true - y_pred) ** 2)


def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    计算MSE关于预测值的梯度

    对于MSE = (1/n) * Σ(y_true - y_pred)^2
    梯度为: ∂MSE/∂y_pred = (2/n) * (y_pred - y_true)

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        梯度值
    """
    n = len(y_true)
    return (2.0 / n) * (y_pred - y_true)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (Root Mean Squared Error)

    RMSE = sqrt(MSE)，与目标变量同量纲，更易解释。

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        均方根误差值
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)

    MAE对异常值不如MSE敏感。

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        平均绝对误差值
    """
    return np.mean(np.abs(y_true - y_pred))


class LinearRegressionMSE:
    """
    使用MSE作为损失函数的线性回归模型

    模型形式：y = wx + b
    损失函数：MSE = (1/n) * Σ(y_true - (wx + b))^2
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        初始化线性回归模型

        Args:
            learning_rate: 学习率
            n_iterations: 训练迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'LinearRegressionMSE':
        """
        训练线性回归模型

        使用梯度下降优化MSE损失函数。

        Args:
            X: 训练数据，形状为 (n_samples, n_features)
            y: 目标值，形状为 (n_samples,)
            verbose: 是否打印训练信息

        Returns:
            self
        """
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降
        for i in range(self.n_iterations):
            # 前向传播：计算预测值
            y_pred = np.dot(X, self.weights) + self.bias

            # 计算损失
            loss = mse(y, y_pred)
            self.loss_history.append(loss)

            # 计算梯度
            # dL/dw = (2/n) * X^T * (y_pred - y_true)
            # dL/db = (2/n) * Σ(y_pred - y_true)
            dw = (2.0 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2.0 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 打印训练信息
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Loss: {loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入数据

        Returns:
            预测值
        """
        return np.dot(X, self.weights) + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算R²得分

        R² = 1 - (MSE / Var(y))

        Args:
            X: 输入数据
            y: 真实值

        Returns:
            R²得分
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class PolynomialRegressionMSE:
    """
    使用MSE的多项式回归

    将特征扩展到多项式空间，然后应用线性回归。
    """

    def __init__(self, degree: int = 2, learning_rate: float = 0.01,
                 n_iterations: int = 1000):
        """
        初始化多项式回归模型

        Args:
            degree: 多项式的度数
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        self.degree = degree
        self.linear_model = LinearRegressionMSE(learning_rate, n_iterations)

    def _polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        生成多项式特征

        对于输入x，生成 [x, x^2, x^3, ..., x^degree]

        Args:
            X: 输入数据

        Returns:
            多项式特征
        """
        n_samples = X.shape[0]
        X_poly = np.zeros((n_samples, self.degree))

        for i in range(self.degree):
            X_poly[:, i] = (X.flatten() ** (i + 1))

        return X_poly

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'PolynomialRegressionMSE':
        """
        训练多项式回归模型

        Args:
            X: 训练数据
            y: 目标值
            verbose: 是否打印训练信息

        Returns:
            self
        """
        X_poly = self._polynomial_features(X)
        self.linear_model.fit(X_poly, y, verbose)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入数据

        Returns:
            预测值
        """
        X_poly = self._polynomial_features(X)
        return self.linear_model.predict(X_poly)

    @property
    def loss_history(self):
        """返回损失历史"""
        return self.linear_model.loss_history


def compare_loss_functions(y_true: np.ndarray, y_pred_range: np.ndarray) -> None:
    """
    可视化比较不同损失函数

    Args:
        y_true: 真实值（标量）
        y_pred_range: 预测值的范围
    """
    mse_losses = [(y_true - y_pred) ** 2 for y_pred in y_pred_range]
    mae_losses = [np.abs(y_true - y_pred) for y_pred in y_pred_range]

    plt.figure(figsize=(10, 6))

    plt.plot(y_pred_range, mse_losses, 'b-', linewidth=2, label='MSE (平方误差)')
    plt.plot(y_pred_range, mae_losses, 'r-', linewidth=2, label='MAE (绝对误差)')

    plt.axvline(x=y_true, color='g', linestyle='--', alpha=0.5, label=f'真实值 = {y_true}')
    plt.xlabel('预测值')
    plt.ylabel('损失值')
    plt.title('MSE vs MAE 损失函数比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 示例使用
if __name__ == "__main__":
    print("=" * 60)
    print("均方误差 (MSE) 演示")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(42)

    # ========== 1. 基础MSE计算 ==========
    print("\n1. 基础MSE计算")
    print("-" * 60)

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.3, 2.8, 4.2, 4.9])

    mse_value = mse(y_true, y_pred)
    rmse_value = rmse(y_true, y_pred)
    mae_value = mae(y_true, y_pred)

    print(f"真实值: {y_true}")
    print(f"预测值: {y_pred}")
    print(f"\nMSE:  {mse_value:.6f}")
    print(f"RMSE: {rmse_value:.6f}")
    print(f"MAE:  {mae_value:.6f}")

    # ========== 2. MSE梯度计算 ==========
    print("\n" + "=" * 60)
    print("2. MSE梯度计算")
    print("=" * 60)

    gradient = mse_gradient(y_true, y_pred)
    print(f"MSE关于预测值的梯度: {gradient}")
    print(f"梯度均值: {np.mean(gradient):.6f}")

    # ========== 3. 线性回归示例 ==========
    print("\n" + "=" * 60)
    print("3. 线性回归示例")
    print("=" * 60)

    # 生成线性数据
    X_train = np.linspace(0, 10, 100).reshape(-1, 1)
    y_train = 2 * X_train.flatten() + 1 + np.random.randn(100) * 0.5

    # 训练模型
    model = LinearRegressionMSE(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train, verbose=True)

    # 预测
    y_pred_train = model.predict(X_train)

    print(f"\n学习到的参数:")
    print(f"  权重 w: {model.weights[0]:.4f}")
    print(f"  偏置 b: {model.bias:.4f}")
    print(f"  真实参数: w=2.0, b=1.0")

    print(f"\n训练集性能:")
    print(f"  MSE:  {mse(y_train, y_pred_train):.6f}")
    print(f"  RMSE: {rmse(y_train, y_pred_train):.6f}")
    print(f"  R²:   {model.score(X_train, y_train):.6f}")

    # 可视化
    plt.figure(figsize=(12, 5))

    # 拟合结果
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.5, label='训练数据')
    plt.plot(X_train, y_pred_train, 'r-', linewidth=2, label='拟合直线')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history, linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('MSE损失')
    plt.title('训练损失曲线')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    # ========== 4. 多项式回归示例 ==========
    print("\n" + "=" * 60)
    print("4. 多项式回归示例")
    print("=" * 60)

    # 生成非线性数据
    X_poly = np.linspace(0, 10, 100).reshape(-1, 1)
    y_poly = 0.5 * X_poly.flatten() ** 2 - 3 * X_poly.flatten() + 5 + np.random.randn(100) * 2

    # 测试不同度数的多项式
    degrees = [1, 2, 3, 5]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, degree in enumerate(degrees):
        ax = axes[idx]

        # 训练模型
        poly_model = PolynomialRegressionMSE(degree=degree, learning_rate=0.001, n_iterations=2000)
        poly_model.fit(X_poly, y_poly)

        # 预测
        y_pred_poly = poly_model.predict(X_poly)

        # 计算误差
        mse_val = mse(y_poly, y_pred_poly)
        r2_val = 1 - (np.sum((y_poly - y_pred_poly) ** 2) / np.sum((y_poly - np.mean(y_poly)) ** 2))

        # 绘图
        ax.scatter(X_poly, y_poly, alpha=0.5, label='数据')
        ax.plot(X_poly, y_pred_poly, 'r-', linewidth=2, label='拟合曲线')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(f'度数={degree}, MSE={mse_val:.2f}, R²={r2_val:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========== 5. 损失函数对比 ==========
    print("\n" + "=" * 60)
    print("5. MSE vs MAE 损失函数对比")
    print("=" * 60)

    y_true_single = 0.0
    y_pred_range = np.linspace(-5, 5, 100)

    compare_loss_functions(y_true_single, y_pred_range)

    # ========== 6. 异常值影响分析 ==========
    print("\n" + "=" * 60)
    print("6. 异常值对MSE的影响")
    print("=" * 60)

    # 正常数据
    y_normal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_normal = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

    # 添加异常值
    y_outlier = np.append(y_normal, 100.0)
    y_pred_outlier = np.append(y_pred_normal, 1.0)

    mse_normal = mse(y_normal, y_pred_normal)
    mae_normal = mae(y_normal, y_pred_normal)

    mse_outlier = mse(y_outlier, y_pred_outlier)
    mae_outlier = mae(y_outlier, y_pred_outlier)

    print("正常数据：")
    print(f"  MSE: {mse_normal:.6f}")
    print(f"  MAE: {mae_normal:.6f}")

    print("\n添加异常值后：")
    print(f"  MSE: {mse_outlier:.6f} (增加 {mse_outlier/mse_normal:.2f}倍)")
    print(f"  MAE: {mae_outlier:.6f} (增加 {mae_outlier/mae_normal:.2f}倍)")

    print("\n结论: MSE对异常值更敏感（平方放大了误差）")

    # ========== 7. 梯度下降可视化 ==========
    print("\n" + "=" * 60)
    print("7. 梯度下降优化MSE可视化")
    print("=" * 60)

    # 简单的一维线性回归
    X_simple = np.array([[1], [2], [3], [4], [5]])
    y_simple = np.array([2, 4, 6, 8, 10])

    # 记录参数更新过程
    w_history = []
    b_history = []

    w, b = 0.0, 0.0
    lr = 0.1
    n_iters = 50

    for _ in range(n_iters):
        w_history.append(w)
        b_history.append(b)

        y_pred = w * X_simple.flatten() + b
        dw = (2.0 / len(X_simple)) * np.dot(X_simple.T, (y_pred - y_simple))[0]
        db = (2.0 / len(X_simple)) * np.sum(y_pred - y_simple)

        w -= lr * dw
        b -= lr * db

    print(f"最终参数: w={w:.4f}, b={b:.4f}")
    print(f"真实参数: w=2.0, b=0.0")

    # 可视化参数更新
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(w_history, linewidth=2, label='权重 w')
    plt.plot(b_history, linewidth=2, label='偏置 b')
    plt.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='真实 w')
    plt.axhline(y=0.0, color='g', linestyle='--', alpha=0.5, label='真实 b')
    plt.xlabel('迭代次数')
    plt.ylabel('参数值')
    plt.title('参数更新过程')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    losses = [mse(y_simple, w_h * X_simple.flatten() + b_h)
             for w_h, b_h in zip(w_history, b_history)]
    plt.plot(losses, linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('MSE损失')
    plt.title('损失下降曲线')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()
