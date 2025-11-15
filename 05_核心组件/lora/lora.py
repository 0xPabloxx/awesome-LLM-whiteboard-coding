"""
LoRA (Low-Rank Adaptation) 实现

LoRA是一种参数高效的微调方法，通过在预训练模型的权重矩阵旁边添加
低秩分解矩阵来适配下游任务，大幅减少可训练参数量。

核心思想：
1. 冻结预训练模型的原始权重W
2. 添加低秩分解矩阵: ΔW = BA，其中B和A是低秩矩阵
3. 前向传播: h = Wx + BAx = Wx + ΔWx
4. 只训练A和B，参数量大幅减少

公式：
h = W₀x + ΔWx = W₀x + BAx
其中 W₀ ∈ R^(d×k)，B ∈ R^(d×r)，A ∈ R^(r×k)，r << min(d,k)

优势：
- 大幅减少可训练参数（通常减少10000倍）
- 不增加推理延迟
- 多任务切换方便（只需替换BA矩阵）
- 保持预训练模型的泛化能力

应用：GPT-3、LLaMA、Stable Diffusion等大模型的微调
论文：LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
"""

import numpy as np


class LoRALayer:
    """
    LoRA层实现

    在标准线性层的基础上添加低秩适配矩阵。
    原始权重冻结，只训练低秩矩阵A和B。
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16, pretrained_weight=None):
        """
        初始化LoRA层

        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            rank: 低秩矩阵的秩（r），越小参数越少
            alpha: 缩放因子，控制LoRA的贡献度
            pretrained_weight: 预训练权重矩阵 (out_features, in_features)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # 预训练权重（冻结，不更新）
        if pretrained_weight is not None:
            self.W = pretrained_weight
        else:
            # 如果没有提供，随机初始化一个模拟预训练权重
            self.W = np.random.randn(out_features, in_features) / np.sqrt(in_features)

        # LoRA的低秩矩阵
        # A: (rank, in_features) - 使用Kaiming初始化
        self.A = np.random.randn(rank, in_features) / np.sqrt(in_features)

        # B: (out_features, rank) - 初始化为0，开始时ΔW=0
        self.B = np.zeros((out_features, rank))

        # 缩放因子 (通常 alpha/rank)
        self.scaling = alpha / rank

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入，形状为 (..., in_features)

        Returns:
            output: 输出，形状为 (..., out_features)
        """
        # 标准线性变换（使用冻结的预训练权重）
        output = np.dot(x, self.W.T)  # (..., out_features)

        # LoRA的低秩适配
        # x @ A^T @ B^T = (x @ A^T) @ B^T
        lora_output = np.dot(np.dot(x, self.A.T), self.B.T)  # (..., out_features)

        # 合并，加上缩放
        output = output + lora_output * self.scaling

        return output

    def merge_weights(self):
        """
        将LoRA权重合并到原始权重中

        用于推理时的优化：W_merged = W + α/r * BA
        合并后不需要额外计算，推理速度与原模型相同
        """
        # ΔW = BA
        delta_W = np.dot(self.B, self.A) * self.scaling

        # 合并权重
        W_merged = self.W + delta_W

        return W_merged

    def get_num_trainable_params(self):
        """
        计算可训练参数量

        Returns:
            trainable_params: LoRA可训练参数数量
            total_params: 总参数数量（包括冻结的）
        """
        # LoRA参数：A + B
        lora_params = self.rank * self.in_features + self.out_features * self.rank

        # 总参数（包括冻结的W）
        total_params = self.out_features * self.in_features + lora_params

        return lora_params, total_params


class LinearLayer:
    """
    标准线性层（用于对比）
    """

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # 所有权重都可训练
        self.W = np.random.randn(out_features, in_features) / np.sqrt(in_features)
        self.b = np.zeros(out_features)

    def forward(self, x):
        return np.dot(x, self.W.T) + self.b

    def get_num_params(self):
        return self.out_features * self.in_features + self.out_features


class MultiHeadAttentionWithLoRA:
    """
    带LoRA的多头注意力

    在Q、K、V、O投影中应用LoRA，是最常见的应用场景。
    """

    def __init__(self, embed_dim, num_heads, rank=8, alpha=16, use_lora_on=['q', 'v']):
        """
        初始化带LoRA的多头注意力

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            rank: LoRA的秩
            alpha: LoRA的缩放因子
            use_lora_on: 在哪些投影上使用LoRA，可选 ['q', 'k', 'v', 'o']
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 创建预训练权重（模拟）
        pretrained_wq = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        pretrained_wk = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        pretrained_wv = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        pretrained_wo = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # Q、K、V、O投影，根据配置决定是否使用LoRA
        if 'q' in use_lora_on:
            self.W_q = LoRALayer(embed_dim, embed_dim, rank, alpha, pretrained_wq)
        else:
            self.W_q = pretrained_wq

        if 'k' in use_lora_on:
            self.W_k = LoRALayer(embed_dim, embed_dim, rank, alpha, pretrained_wk)
        else:
            self.W_k = pretrained_wk

        if 'v' in use_lora_on:
            self.W_v = LoRALayer(embed_dim, embed_dim, rank, alpha, pretrained_wv)
        else:
            self.W_v = pretrained_wv

        if 'o' in use_lora_on:
            self.W_o = LoRALayer(embed_dim, embed_dim, rank, alpha, pretrained_wo)
        else:
            self.W_o = pretrained_wo

    def get_num_trainable_params(self):
        """计算可训练参数量"""
        trainable = 0
        total = 0

        for name, layer in [('q', self.W_q), ('k', self.W_k), ('v', self.W_v), ('o', self.W_o)]:
            if isinstance(layer, LoRALayer):
                t, tot = layer.get_num_trainable_params()
                trainable += t
            else:
                # 冻结层，无可训练参数
                tot = self.embed_dim * self.embed_dim
            total += tot

        return trainable, total


def compare_parameter_efficiency(d_model=768, rank=8):
    """
    比较LoRA与全量微调的参数效率

    Args:
        d_model: 模型维度
        rank: LoRA的秩

    Returns:
        comparison: 参数量对比字典
    """
    results = {}

    # 1. 单个线性层
    linear_params = d_model * d_model
    lora_params = rank * d_model + d_model * rank
    results['single_layer'] = {
        'full_finetune': linear_params,
        'lora': lora_params,
        'ratio': lora_params / linear_params
    }

    # 2. 多头注意力（4个投影矩阵）
    mha_params = 4 * d_model * d_model
    mha_lora_params = 4 * lora_params  # 假设所有投影都用LoRA
    results['multi_head_attention'] = {
        'full_finetune': mha_params,
        'lora': mha_lora_params,
        'ratio': mha_lora_params / mha_params
    }

    # 3. Transformer层（包括Attention和FFN）
    # Attention: 4 * d * d
    # FFN: 2 * d * (4d) = 8 * d * d （假设FFN中间层是4d）
    # 总计: 12 * d * d
    transformer_params = 12 * d_model * d_model
    transformer_lora_params = 12 * lora_params
    results['transformer_layer'] = {
        'full_finetune': transformer_params,
        'lora': transformer_lora_params,
        'ratio': transformer_lora_params / transformer_params
    }

    return results


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("LoRA (Low-Rank Adaptation) 演示")
    print("=" * 70)

    # ========== 1. 基础LoRA层 ==========
    print("\n" + "=" * 70)
    print("1. 基础LoRA层")
    print("=" * 70)

    in_features = 512
    out_features = 512
    rank = 8

    # 创建LoRA层
    lora_layer = LoRALayer(in_features, out_features, rank=rank, alpha=16)

    print(f"\n配置:")
    print(f"  输入维度: {in_features}")
    print(f"  输出维度: {out_features}")
    print(f"  LoRA秩: {rank}")
    print(f"  缩放因子α: 16")

    # 参数量分析
    trainable, total = lora_layer.get_num_trainable_params()

    print(f"\n参数量分析:")
    print(f"  原始权重W: {out_features} × {in_features} = {out_features * in_features:,}")
    print(f"  LoRA矩阵A: {rank} × {in_features} = {rank * in_features:,}")
    print(f"  LoRA矩阵B: {out_features} × {rank} = {out_features * rank:,}")
    print(f"  LoRA总参数: {trainable:,}")
    print(f"  参数减少比例: {(1 - trainable / (out_features * in_features)) * 100:.2f}%")
    print(f"  压缩比: {(out_features * in_features) / trainable:.1f}x")

    # 前向传播测试
    print("\n" + "=" * 70)
    print("2. 前向传播测试")
    print("=" * 70)

    batch_size, seq_len = 4, 10
    x = np.random.randn(batch_size, seq_len, in_features)

    output = lora_layer.forward(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 分解计算过程
    W_output = np.dot(x, lora_layer.W.T)
    lora_output = np.dot(np.dot(x, lora_layer.A.T), lora_layer.B.T) * lora_layer.scaling

    print(f"\n计算分解:")
    print(f"  原始输出(Wx): {W_output.shape}")
    print(f"  LoRA输出(BAx): {lora_output.shape}")
    print(f"  最终输出: {output.shape}")

    # ========== 3. 权重合并 ==========
    print("\n" + "=" * 70)
    print("3. 权重合并（推理优化）")
    print("=" * 70)

    # 合并权重
    W_merged = lora_layer.merge_weights()

    print(f"\n合并前:")
    print(f"  需要计算: Wx + α/r·BAx")
    print(f"  两次矩阵乘法")

    print(f"\n合并后:")
    print(f"  W_merged = W + α/r·BA")
    print(f"  只需计算: W_merged·x")
    print(f"  一次矩阵乘法")

    # 验证合并正确性
    output_merged = np.dot(x, W_merged.T)
    print(f"\n验证合并正确性:")
    print(f"  原始输出与合并输出的差异: {np.max(np.abs(output - output_merged)):.10f}")

    # ========== 4. 不同秩的影响 ==========
    print("\n" + "=" * 70)
    print("4. 不同秩(rank)的参数量对比")
    print("=" * 70)

    print(f"\n原始层参数量: {out_features * in_features:,}")
    print(f"\n{'Rank':<8} {'LoRA参数':<15} {'压缩比':<10} {'参数占比':<10}")
    print("-" * 50)

    for r in [1, 2, 4, 8, 16, 32, 64]:
        lora_params = r * in_features + out_features * r
        compression = (out_features * in_features) / lora_params
        ratio = (lora_params / (out_features * in_features)) * 100

        print(f"{r:<8} {lora_params:<15,} {compression:<10.1f}x {ratio:<10.2f}%")

    # ========== 5. 多头注意力中的LoRA ==========
    print("\n" + "=" * 70)
    print("5. 多头注意力中的LoRA应用")
    print("=" * 70)

    embed_dim = 768  # BERT-base的维度
    num_heads = 12
    rank = 8

    # 只在Q和V投影上使用LoRA（常见配置）
    mha_lora = MultiHeadAttentionWithLoRA(
        embed_dim, num_heads, rank=rank, alpha=16,
        use_lora_on=['q', 'v']
    )

    trainable_mha, total_mha = mha_lora.get_num_trainable_params()

    print(f"\n配置:")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  LoRA应用于: Q和V投影")

    print(f"\n参数量:")
    print(f"  全量微调（4个投影）: {4 * embed_dim * embed_dim:,}")
    print(f"  LoRA微调（Q和V）: {trainable_mha:,}")
    print(f"  参数减少: {(1 - trainable_mha / (4 * embed_dim * embed_dim)) * 100:.2f}%")

    # ========== 6. 完整模型的参数对比 ==========
    print("\n" + "=" * 70)
    print("6. 完整Transformer模型的参数对比")
    print("=" * 70)

    d_model = 768
    num_layers = 12  # BERT-base有12层

    print(f"\n模型配置（BERT-base风格）:")
    print(f"  隐藏维度: {d_model}")
    print(f"  层数: {num_layers}")
    print(f"  FFN中间维度: {d_model * 4}")

    # 每层的参数量
    # Attention: 4 * d * d (Q, K, V, O)
    # FFN: d * 4d + 4d * d = 8 * d * d
    # 总计: 12 * d * d
    params_per_layer = 12 * d_model * d_model

    # 全量微调
    total_params = num_layers * params_per_layer

    # LoRA微调（只在Attention的Q、V投影上，rank=8）
    lora_params_per_layer = 2 * (rank * d_model + d_model * rank)
    total_lora_params = num_layers * lora_params_per_layer

    print(f"\n全量微调:")
    print(f"  每层参数: {params_per_layer:,}")
    print(f"  总可训练参数: {total_params:,}")

    print(f"\nLoRA微调 (rank={rank}, 仅Q/V投影):")
    print(f"  每层LoRA参数: {lora_params_per_layer:,}")
    print(f"  总可训练参数: {total_lora_params:,}")

    print(f"\n效率提升:")
    print(f"  参数减少: {(1 - total_lora_params / total_params) * 100:.2f}%")
    print(f"  压缩比: {total_params / total_lora_params:.0f}x")

    # ========== 7. 不同rank对效果的影响 ==========
    print("\n" + "=" * 70)
    print("7. Rank选择建议")
    print("=" * 70)

    print("\n常用rank配置:")
    print("  • rank=1~4: 极度参数高效，适合资源极度受限场景")
    print("  • rank=8: 最常用，平衡效果和效率（LoRA论文推荐）")
    print("  • rank=16: 稍大，可能提升效果")
    print("  • rank=32~64: 较大，接近全量微调效果")

    print("\n应用建议:")
    print("  • 小任务/少样本: rank=4~8")
    print("  • 中等任务: rank=8~16")
    print("  • 复杂任务: rank=16~32")
    print("  • 领域差异大: 考虑更大的rank或全量微调")

    # ========== 8. LoRA的实际应用 ==========
    print("\n" + "=" * 70)
    print("8. LoRA的实际应用场景")
    print("=" * 70)

    print("\n成功应用案例:")
    print("  • GPT-3微调: 使用LoRA减少99.9%可训练参数")
    print("  • Stable Diffusion: LoRA用于风格迁移和人物定制")
    print("  • LLaMA微调: Alpaca、Vicuna等都使用LoRA")
    print("  • 多任务学习: 为每个任务训练独立的LoRA模块")

    print("\n优势:")
    print("  ✓ 大幅减少显存需求（可在消费级GPU上微调大模型）")
    print("  ✓ 训练速度更快（更少的参数需要梯度）")
    print("  ✓ 模型切换方便（只需加载不同的LoRA权重）")
    print("  ✓ 保持预训练模型的泛化能力")

    print("\n局限:")
    print("  ✗ 可能不如全量微调效果好（在某些任务上）")
    print("  ✗ 需要选择合适的rank（需要实验）")
    print("  ✗ 只适用于线性层（卷积层需要特殊处理）")

    # ========== 9. 参数效率对比总结 ==========
    print("\n" + "=" * 70)
    print("9. 参数效率总结")
    print("=" * 70)

    comparison = compare_parameter_efficiency(d_model=768, rank=8)

    print(f"\n层级\t\t\t全量微调\t\tLoRA\t\t比例")
    print("-" * 70)
    for key, val in comparison.items():
        print(f"{key:<20}\t{val['full_finetune']:>12,}\t{val['lora']:>12,}\t{val['ratio']*100:>6.2f}%")

    print("\n" + "=" * 70)
    print("LoRA的关键特性:")
    print("=" * 70)
    print("✓ 通过低秩分解大幅减少可训练参数（通常99%以上）")
    print("✓ 冻结原始权重，保持预训练模型的知识")
    print("✓ 推理时可合并权重，无额外开销")
    print("✓ 多任务场景下可快速切换不同的LoRA模块")
    print("✓ 广泛应用于GPT-3、LLaMA、Stable Diffusion等大模型微调")
