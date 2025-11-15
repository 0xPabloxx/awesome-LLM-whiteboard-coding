"""
Soft Attention (软注意力) 实现

软注意力是最基础的注意力机制形式，对所有位置计算权重，整个过程可微分。
主要应用于seq2seq模型、图像描述生成等任务。

核心思想：
1. 计算查询(query)与所有键(key)的相似度得分
2. 对得分进行softmax归一化得到注意力权重
3. 用权重对值(value)进行加权求和
"""

import numpy as np


def softmax(x):
    """
    Softmax函数实现

    Args:
        x: 输入数组
    Returns:
        softmax归一化后的数组
    """
    # 减去最大值提高数值稳定性
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SoftAttention:
    """
    软注意力机制实现类

    特点：
    - 对所有位置计算注意力权重
    - 全程可微分，可以用梯度下降优化
    - 计算复杂度为O(n)，n为序列长度
    """

    def __init__(self, hidden_dim):
        """
        初始化软注意力层

        Args:
            hidden_dim: 隐藏层维度
        """
        self.hidden_dim = hidden_dim
        # 初始化权重矩阵
        self.W_a = np.random.randn(hidden_dim, hidden_dim) * 0.01  # 注意力权重矩阵
        self.W_c = np.random.randn(hidden_dim, 1) * 0.01  # 上下文向量投影
        self.b_a = np.zeros((hidden_dim,))  # 偏置项

    def forward(self, query, keys, values):
        """
        前向传播

        Args:
            query: 查询向量，形状为 (hidden_dim,)
            keys: 键向量序列，形状为 (seq_len, hidden_dim)
            values: 值向量序列，形状为 (seq_len, hidden_dim)

        Returns:
            context: 上下文向量，形状为 (hidden_dim,)
            attention_weights: 注意力权重，形状为 (seq_len,)
        """
        seq_len = keys.shape[0]

        # 步骤1: 计算注意力得分
        # 将query与每个key结合计算得分
        scores = []
        for i in range(seq_len):
            # 拼接query和key
            combined = query + keys[i]  # 简化版本，也可以用更复杂的计算
            # 通过网络计算得分
            hidden = np.tanh(np.dot(combined, self.W_a) + self.b_a)
            score = np.dot(hidden, self.W_c.flatten())
            scores.append(score)

        scores = np.array(scores)  # (seq_len,)

        # 步骤2: Softmax归一化得到注意力权重
        attention_weights = softmax(scores)  # (seq_len,)

        # 步骤3: 加权求和得到上下文向量
        context = np.zeros(self.hidden_dim)
        for i in range(seq_len):
            context += attention_weights[i] * values[i]

        return context, attention_weights


def soft_attention_simple(query, keys, values):
    """
    简化版软注意力实现（不使用可学习参数）

    Args:
        query: 查询向量，形状为 (d,)
        keys: 键向量，形状为 (n, d)
        values: 值向量，形状为 (n, d)

    Returns:
        output: 注意力输出，形状为 (d,)
        weights: 注意力权重，形状为 (n,)
    """
    # 计算注意力得分（使用点积）
    scores = np.dot(keys, query)  # (n,)

    # Softmax归一化
    weights = softmax(scores)  # (n,)

    # 加权求和
    output = np.dot(weights, values)  # (d,)

    return output, weights


# 示例使用
if __name__ == "__main__":
    # 设置随机种子以便复现
    np.random.seed(42)

    # 参数设置
    hidden_dim = 64
    seq_len = 10

    print("=" * 50)
    print("软注意力机制演示")
    print("=" * 50)

    # 创建软注意力层
    attention = SoftAttention(hidden_dim)

    # 生成示例数据
    query = np.random.randn(hidden_dim)
    keys = np.random.randn(seq_len, hidden_dim)
    values = np.random.randn(seq_len, hidden_dim)

    # 前向传播
    context, weights = attention.forward(query, keys, values)

    print(f"\n查询向量形状: {query.shape}")
    print(f"键向量形状: {keys.shape}")
    print(f"值向量形状: {values.shape}")
    print(f"\n上下文向量形状: {context.shape}")
    print(f"注意力权重形状: {weights.shape}")
    print(f"\n注意力权重（前5个）: {weights[:5]}")
    print(f"注意力权重总和: {np.sum(weights):.6f}")

    print("\n" + "=" * 50)
    print("简化版软注意力演示")
    print("=" * 50)

    # 使用简化版
    output, simple_weights = soft_attention_simple(query, keys, values)
    print(f"\n输出向量形状: {output.shape}")
    print(f"注意力权重（前5个）: {simple_weights[:5]}")
    print(f"注意力权重总和: {np.sum(simple_weights):.6f}")
