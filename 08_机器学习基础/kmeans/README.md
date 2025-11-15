# K-Means聚类算法

## 概述

K-Means是最经典和广泛使用的无监督学习算法之一，由Stuart Lloyd在1957年提出。它通过迭代优化将数据点分配到K个簇中，使得簇内数据点尽可能相似，簇间数据点尽可能不同。

## 核心思想

K-Means的目标是将n个数据点划分为k个簇，使得每个数据点属于距离最近的簇中心所代表的簇。

**基本流程**：
1. **初始化**：随机选择K个数据点作为初始聚类中心
2. **分配步骤（E步）**：将每个数据点分配到最近的聚类中心
3. **更新步骤（M步）**：重新计算每个簇的中心点（簇内所有点的均值）
4. **迭代**：重复步骤2-3直到聚类中心不再变化或达到最大迭代次数

## 数学原理

### 目标函数

K-Means最小化簇内误差平方和（Within-Cluster Sum of Squares, WCSS），也称为Inertia：

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

其中：
- $k$ 是簇的数量
- $C_i$ 是第 $i$ 个簇
- $\mu_i$ 是第 $i$ 个簇的中心
- $\|x - \mu_i\|^2$ 是欧氏距离的平方

### 算法步骤

给定数据集 $X = \{x_1, x_2, ..., x_n\}$，要将其划分为 $k$ 个簇：

**初始化**：
$$
\mu_1, \mu_2, ..., \mu_k \leftarrow \text{random initialization}
$$

**E步（分配）**：
$$
C_i^{(t)} = \{x_p : \|x_p - \mu_i^{(t)}\|^2 \leq \|x_p - \mu_j^{(t)}\|^2 \text{ for all } j\}
$$

将每个点 $x_p$ 分配到最近的簇。

**M步（更新）**：
$$
\mu_i^{(t+1)} = \frac{1}{|C_i^{(t)}|} \sum_{x \in C_i^{(t)}} x
$$

更新每个簇的中心为簇内所有点的均值。

**收敛条件**：
$$
\mu_i^{(t+1)} = \mu_i^{(t)} \text{ for all } i
$$

## 算法特点

### 优点
- ✅ **简单直观**：算法思想容易理解，实现简单
- ✅ **效率高**：时间复杂度O(nkt)，n为数据点数，k为簇数，t为迭代次数
- ✅ **可扩展性好**：适用于大规模数据集
- ✅ **收敛保证**：保证收敛到局部最优解

### 缺点
- ❌ **需要预先指定K**：需要事先知道聚类数量
- ❌ **对初始值敏感**：不同的初始中心可能导致不同结果
- ❌ **局部最优**：只能保证收敛到局部最优，不一定是全局最优
- ❌ **对异常值敏感**：异常点会影响聚类中心的计算
- ❌ **假设簇是凸形的**：对非凸形状的簇效果不好
- ❌ **假设簇大小相似**：不同大小的簇可能被错误划分

## 时间复杂度

- **时间复杂度**：O(n × k × t × d)
  - n: 数据点数量
  - k: 簇数量
  - t: 迭代次数
  - d: 特征维度

- **空间复杂度**：O(n × d + k × d)

## K-Means++初始化

标准K-Means对初始中心点敏感，K-Means++改进了初始化策略：

1. 随机选择第一个中心点 $\mu_1$
2. 对于每个数据点 $x$，计算其到最近中心的距离 $D(x)$
3. 以概率 $\frac{D(x)^2}{\sum D(x)^2}$ 选择下一个中心点
4. 重复2-3直到选出k个中心点

**优势**：
- 初始中心点相距较远
- 收敛更快，结果更稳定
- 近似比为O(log k)

## 选择K值的方法

### 1. 肘部法则（Elbow Method）

绘制K与Inertia的关系曲线，在曲线"肘部"（拐点）处选择K值。

```python
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias)
```

### 2. 轮廓系数（Silhouette Score）

衡量每个样本与其簇内其他样本的相似度，以及与其他簇的样本的差异度。

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

其中：
- $a(i)$: 样本i与同簇其他样本的平均距离
- $b(i)$: 样本i与最近其他簇的样本的平均距离
- 取值范围：[-1, 1]，值越大越好

### 3. Gap统计量

比较真实数据的Inertia与随机数据的Inertia。

### 4. 领域知识

根据实际应用场景和领域知识确定K值。

## 代码实现

### 基础实现

```python
class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        # 1. 随机初始化聚类中心
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iters):
            # 2. E步：分配簇
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # 3. M步：更新中心
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0)
                for i in range(self.n_clusters)
            ])

            # 4. 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
```

### K-Means++初始化

```python
def init_kmeans_plus_plus(X, n_clusters):
    centroids = [X[np.random.randint(len(X))]]

    for _ in range(1, n_clusters):
        # 计算到最近中心的距离
        distances = np.array([
            min([np.linalg.norm(x - c)**2 for c in centroids])
            for x in X
        ])

        # 按距离平方加权选择下一个中心
        probabilities = distances / distances.sum()
        cumulative_probs = probabilities.cumsum()
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                centroids.append(X[j])
                break

    return np.array(centroids)
```

## 使用示例

### 基础聚类

```python
from kmeans import KMeans
import numpy as np

# 生成数据
X = np.random.randn(300, 2)

# 训练模型
kmeans = KMeans(n_clusters=3, init_method='kmeans++')
kmeans.fit(X)

# 预测
labels = kmeans.predict(X)

print(f"聚类中心: {kmeans.centroids}")
print(f"Inertia: {kmeans.inertia_:.4f}")
```

### 肘部法则选择K

```python
from kmeans import elbow_method

# 测试K=1到10
inertias = elbow_method(X, max_k=10)

# 绘制肘部曲线
import matplotlib.pyplot as plt
plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

### 可视化聚类结果

```python
from kmeans import visualize_clustering

visualize_clustering(X, kmeans.labels, kmeans.centroids)
```

## 变体和改进

### 1. Mini-Batch K-Means

每次迭代只使用一小批数据，加快训练速度。

**优点**：
- 速度更快
- 内存占用更小

**缺点**：
- 结果可能不如标准K-Means稳定

### 2. K-Medoids (PAM)

使用实际数据点作为中心，而不是均值点。

**优点**：
- 对异常值更鲁棒
- 中心点有实际意义

**缺点**：
- 计算成本更高

### 3. Fuzzy C-Means

每个点可以属于多个簇，有不同的隶属度。

### 4. 谱聚类

基于图论，可以处理非凸形状的簇。

## 应用场景

1. **客户分群**：将客户按行为特征分组
2. **图像分割**：将图像像素聚类实现分割
3. **文档聚类**：将相似主题的文档分组
4. **异常检测**：识别不属于任何簇的异常点
5. **数据压缩**：使用簇中心代表数据点
6. **特征工程**：将聚类结果作为新特征
7. **推荐系统**：对用户或物品进行分组

## 实践技巧

1. **数据预处理**
   - 标准化特征（特别是不同量纲的特征）
   - 处理缺失值
   - 移除异常值

2. **选择K值**
   - 使用肘部法则
   - 结合业务需求
   - 尝试多个K值

3. **多次运行**
   - 由于随机初始化，建议多次运行取最佳结果
   - 使用K-Means++减少初始化的影响

4. **评估质量**
   - 轮廓系数
   - Davies-Bouldin指数
   - Calinski-Harabasz指数

5. **可视化**
   - 降维后可视化（PCA, t-SNE）
   - 查看聚类中心的特征

## 评估指标

### 1. Inertia（簇内误差平方和）

越小越好，但会随着K增大而减小。

### 2. 轮廓系数（Silhouette Score）

取值[-1, 1]，越接近1越好。

### 3. Calinski-Harabasz指数

簇间离散度与簇内离散度的比值，越大越好。

### 4. Davies-Bouldin指数

簇内距离与簇间距离的比值，越小越好。

## 常见问题

### 1. K-Means不收敛怎么办？
- 增加最大迭代次数
- 尝试不同的初始化
- 检查数据是否有问题

### 2. 如何处理高维数据？
- 先进行降维（PCA）
- 使用其他距离度量
- 考虑子空间聚类

### 3. 如何处理不同大小的簇？
- 尝试谱聚类
- 使用DBSCAN等密度聚类
- 调整聚类数量

### 4. 如何处理非球形簇？
- 使用DBSCAN或谱聚类
- 核K-Means
- 层次聚类

## 参考文献

1. Lloyd, S. (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory, 28(2), 129-137.
2. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. SODA '07.
3. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Berkeley Symposium on Mathematical Statistics and Probability.

## 文件说明

- `kmeans.py`: Python实现（包含K-Means和K-Means++）
- `kmeans.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档
