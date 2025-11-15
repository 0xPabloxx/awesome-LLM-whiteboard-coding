"""
K-Means聚类算法实现

K-Means是最经典的无监督学习算法之一，通过迭代优化将数据点分配到K个簇中。
本模块实现了标准K-Means算法，并提供了可视化和评估工具。

核心思想：
1. 随机初始化K个聚类中心
2. 分配每个数据点到最近的聚类中心
3. 更新聚类中心为簇内所有点的均值
4. 重复2-3步直到收敛
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class KMeans:
    """
    K-Means聚类算法

    采用EM算法的思想，通过E步（分配）和M步（更新）交替迭代优化目标函数。
    """

    def __init__(self, n_clusters: int = 3, max_iters: int = 100,
                 init_method: str = 'random', random_state: int = None):
        """
        初始化K-Means聚类器

        Args:
            n_clusters: 聚类数量K
            max_iters: 最大迭代次数
            init_method: 初始化方法，可选 'random' 或 'kmeans++'
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init_method = init_method
        self.random_state = random_state

        self.centroids = None  # 聚类中心
        self.labels = None     # 数据点的簇标签
        self.inertia_ = None   # 簇内误差平方和

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        初始化聚类中心

        Args:
            X: 输入数据，形状为 (n_samples, n_features)

        Returns:
            初始聚类中心，形状为 (n_clusters, n_features)
        """
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.init_method == 'random':
            # 随机选择K个数据点作为初始中心
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices]

        elif self.init_method == 'kmeans++':
            # K-Means++ 初始化：选择彼此距离较远的初始中心
            centroids = np.zeros((self.n_clusters, n_features))

            # 随机选择第一个中心
            centroids[0] = X[np.random.randint(n_samples)]

            # 依次选择其他中心
            for i in range(1, self.n_clusters):
                # 计算每个点到最近中心的距离
                distances = np.array([
                    min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]])
                    for x in X
                ])

                # 以距离为概率选择下一个中心
                probabilities = distances / distances.sum()
                cumulative_probs = probabilities.cumsum()
                r = np.random.rand()

                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids[i] = X[j]
                        break

        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        return centroids

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        E步：将每个数据点分配到最近的聚类中心

        Args:
            X: 输入数据
            centroids: 当前聚类中心

        Returns:
            每个数据点的簇标签
        """
        # 计算每个点到所有中心的距离
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i in range(self.n_clusters):
            # 欧氏距离
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)

        # 分配到最近的簇
        labels = np.argmin(distances, axis=1)

        return labels

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        M步：更新聚类中心为簇内所有点的均值

        Args:
            X: 输入数据
            labels: 当前簇标签

        Returns:
            更新后的聚类中心
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for i in range(self.n_clusters):
            # 获取属于簇i的所有点
            cluster_points = X[labels == i]

            if len(cluster_points) > 0:
                # 计算均值
                centroids[i] = cluster_points.mean(axis=0)
            else:
                # 如果簇为空，随机重新初始化
                centroids[i] = X[np.random.randint(X.shape[0])]

        return centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray,
                        centroids: np.ndarray) -> float:
        """
        计算簇内误差平方和（Inertia）

        Args:
            X: 输入数据
            labels: 簇标签
            centroids: 聚类中心

        Returns:
            簇内误差平方和
        """
        inertia = 0.0

        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i]) ** 2)

        return inertia

    def fit(self, X: np.ndarray, verbose: bool = False) -> 'KMeans':
        """
        训练K-Means模型

        Args:
            X: 训练数据，形状为 (n_samples, n_features)
            verbose: 是否打印训练信息

        Returns:
            self
        """
        # 初始化聚类中心
        self.centroids = self._init_centroids(X)

        # 存储每次迭代的中心，用于可视化
        self.centroids_history = [self.centroids.copy()]
        self.inertia_history = []

        for iteration in range(self.max_iters):
            # E步：分配簇
            labels = self._assign_clusters(X, self.centroids)

            # M步：更新中心
            new_centroids = self._update_centroids(X, labels)

            # 计算Inertia
            inertia = self._compute_inertia(X, labels, new_centroids)
            self.inertia_history.append(inertia)

            if verbose:
                print(f"Iteration {iteration + 1}/{self.max_iters}, Inertia: {inertia:.4f}")

            # 检查收敛
            if np.allclose(self.centroids, new_centroids, rtol=1e-6):
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                self.centroids = new_centroids
                self.labels = labels
                break

            self.centroids = new_centroids
            self.centroids_history.append(self.centroids.copy())

        # 最终分配
        self.labels = self._assign_clusters(X, self.centroids)
        self.inertia_ = self._compute_inertia(X, self.labels, self.centroids)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据的簇标签

        Args:
            X: 输入数据

        Returns:
            簇标签
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self._assign_clusters(X, self.centroids)

    def fit_predict(self, X: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        训练并预测

        Args:
            X: 输入数据
            verbose: 是否打印训练信息

        Returns:
            簇标签
        """
        self.fit(X, verbose)
        return self.labels


def elbow_method(X: np.ndarray, max_k: int = 10, random_state: int = 42) -> List[float]:
    """
    肘部法则选择最佳K值

    通过绘制K与Inertia的关系曲线，在曲线"肘部"处选择K值。

    Args:
        X: 输入数据
        max_k: 最大K值
        random_state: 随机种子

    Returns:
        每个K值对应的Inertia列表
    """
    inertias = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    return inertias


def visualize_clustering(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                        title: str = "K-Means聚类结果") -> None:
    """
    可视化聚类结果（仅支持2D数据）

    Args:
        X: 输入数据
        labels: 簇标签
        centroids: 聚类中心
        title: 图表标题
    """
    if X.shape[1] != 2:
        print("只支持2维数据的可视化")
        return

    plt.figure(figsize=(8, 6))

    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                         alpha=0.6, edgecolors='k', s=50)

    # 绘制聚类中心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X',
               s=200, edgecolors='k', linewidths=2, label='聚类中心')

    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(title)
    plt.colorbar(scatter, label='簇标签')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_clustering_process(X: np.ndarray, kmeans: KMeans,
                                frames: List[int] = None) -> None:
    """
    可视化K-Means迭代过程

    Args:
        X: 输入数据
        kmeans: 已训练的K-Means模型
        frames: 要展示的迭代步骤列表，None表示展示全部
    """
    if X.shape[1] != 2:
        print("只支持2维数据的可视化")
        return

    if frames is None:
        # 选择几个关键帧
        n_frames = min(6, len(kmeans.centroids_history))
        frames = np.linspace(0, len(kmeans.centroids_history) - 1, n_frames, dtype=int)

    n_frames = len(frames)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, frame in enumerate(frames):
        if idx >= len(axes):
            break

        ax = axes[idx]
        centroids = kmeans.centroids_history[frame]

        # 分配当前中心对应的标签
        labels = kmeans._assign_clusters(X, centroids)

        # 绘制数据点
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                           alpha=0.6, edgecolors='k', s=50)

        # 绘制聚类中心
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X',
                  s=200, edgecolors='k', linewidths=2)

        ax.set_xlabel('特征 1')
        ax.set_ylabel('特征 2')
        ax.set_title(f'迭代 {frame + 1}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 示例使用
if __name__ == "__main__":
    print("=" * 60)
    print("K-Means聚类算法演示")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(42)

    # ========== 1. 生成测试数据 ==========
    print("\n1. 生成测试数据")
    print("-" * 60)

    # 生成3个高斯分布的簇
    n_samples = 300
    centers = np.array([[1, 1], [5, 5], [9, 1]])

    X1 = np.random.randn(n_samples // 3, 2) + centers[0]
    X2 = np.random.randn(n_samples // 3, 2) + centers[1]
    X3 = np.random.randn(n_samples // 3, 2) + centers[2]

    X = np.vstack([X1, X2, X3])

    print(f"数据形状: {X.shape}")
    print(f"数据范围: X1=[{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], "
          f"X2=[{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")

    # ========== 2. 训练K-Means模型 ==========
    print("\n" + "=" * 60)
    print("2. 训练K-Means模型 (K=3)")
    print("=" * 60)

    kmeans = KMeans(n_clusters=3, max_iters=100, init_method='kmeans++', random_state=42)
    kmeans.fit(X, verbose=True)

    print(f"\n最终Inertia: {kmeans.inertia_:.4f}")
    print(f"聚类中心:\n{kmeans.centroids}")

    # 可视化结果
    visualize_clustering(X, kmeans.labels, kmeans.centroids)

    # ========== 3. 可视化迭代过程 ==========
    print("\n" + "=" * 60)
    print("3. 可视化K-Means迭代过程")
    print("=" * 60)

    visualize_clustering_process(X, kmeans)

    # ========== 4. 肘部法则选择K值 ==========
    print("\n" + "=" * 60)
    print("4. 肘部法则选择最佳K值")
    print("=" * 60)

    max_k = 10
    inertias = elbow_method(X, max_k=max_k, random_state=42)

    print("\nK值与Inertia:")
    for k, inertia in enumerate(inertias, 1):
        print(f"K={k}: Inertia={inertia:.4f}")

    # 绘制肘部曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('聚类数量 K')
    plt.ylabel('Inertia (簇内误差平方和)')
    plt.title('肘部法则选择K值')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max_k + 1))
    plt.show()

    # ========== 5. 对比不同初始化方法 ==========
    print("\n" + "=" * 60)
    print("5. 对比不同初始化方法")
    print("=" * 60)

    methods = ['random', 'kmeans++']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, method in enumerate(methods):
        kmeans_test = KMeans(n_clusters=3, init_method=method, random_state=42)
        kmeans_test.fit(X)

        ax = axes[idx]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans_test.labels,
                           cmap='viridis', alpha=0.6, edgecolors='k', s=50)
        ax.scatter(kmeans_test.centroids[:, 0], kmeans_test.centroids[:, 1],
                  c='red', marker='X', s=200, edgecolors='k', linewidths=2)

        ax.set_xlabel('特征 1')
        ax.set_ylabel('特征 2')
        ax.set_title(f'{method} 初始化 (Inertia={kmeans_test.inertia_:.2f})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========== 6. 测试预测功能 ==========
    print("\n" + "=" * 60)
    print("6. 测试预测新数据")
    print("=" * 60)

    # 生成新的测试点
    X_new = np.array([[2, 2], [6, 6], [8, 1]])
    predictions = kmeans.predict(X_new)

    print("新数据点:")
    for i, (point, label) in enumerate(zip(X_new, predictions)):
        print(f"  点 {i+1}: {point} -> 簇 {label}")

    # ========== 7. Inertia收敛曲线 ==========
    print("\n" + "=" * 60)
    print("7. Inertia收敛曲线")
    print("=" * 60)

    plt.figure(figsize=(8, 5))
    plt.plot(kmeans.inertia_history, 'b-', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('Inertia')
    plt.title('K-Means收敛过程')
    plt.grid(True, alpha=0.3)
    plt.show()
