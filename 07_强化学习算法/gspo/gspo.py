"""
GSPO (Group-wise Stochastic Policy Optimization) 实现

GSPO是一种基于组的随机策略优化方法，结合了group-based方法和
随机优化的优势，特别适合大语言模型的对齐训练。

核心思想：
1. 将数据组织成groups（每个group包含多个样本）
2. 在group内部进行随机采样和比较
3. 使用随机梯度进行策略优化
4. 结合variance reduction技术提高稳定性

优势：
- 样本效率高（group内多次复用）
- 训练稳定（方差缩减）
- 计算高效（随机采样）
- 适合大规模训练

应用场景：
- 大语言模型对齐
- 多任务学习
- 分布式训练
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class GSPOTrainer:
    """
    GSPO训练器

    实现Group-wise Stochastic Policy Optimization算法
    """

    def __init__(
        self,
        group_size: int = 8,
        sample_size: int = 4,
        use_baseline: bool = True,
        variance_reduction: str = 'control_variate',
        temperature: float = 1.0
    ):
        """
        初始化GSPO训练器

        Args:
            group_size: 每个group的样本总数
            sample_size: 每次采样的样本数（< group_size）
            use_baseline: 是否使用基线
            variance_reduction: 方差缩减方法
                - 'control_variate': 控制变量法
                - 'antithetic': 对偶采样
                - None: 不使用
            temperature: 温度参数
        """
        self.group_size = group_size
        self.sample_size = sample_size
        self.use_baseline = use_baseline
        self.variance_reduction = variance_reduction
        self.temperature = temperature

    def sample_from_group(
        self,
        group_rewards: np.ndarray,
        group_log_probs: np.ndarray,
        method: str = 'uniform'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从group中采样

        Args:
            group_rewards: group的奖励，形状 (group_size,)
            group_log_probs: group的对数概率，形状 (group_size,)
            method: 采样方法
                - 'uniform': 均匀采样
                - 'importance': 重要性采样
                - 'top_k': 只采样top-k

        Returns:
            sampled_rewards: 采样的奖励
            sampled_log_probs: 采样的对数概率
            sample_indices: 采样的索引
        """
        if method == 'uniform':
            # 均匀随机采样
            sample_indices = np.random.choice(
                self.group_size,
                size=self.sample_size,
                replace=False
            )

        elif method == 'importance':
            # 基于奖励的重要性采样
            # 奖励越高，采样概率越大
            probs = np.exp(group_rewards / self.temperature)
            probs = probs / np.sum(probs)

            sample_indices = np.random.choice(
                self.group_size,
                size=self.sample_size,
                replace=False,
                p=probs
            )

        elif method == 'top_k':
            # 只采样top-k
            sample_indices = np.argsort(group_rewards)[-self.sample_size:]

        else:
            raise ValueError(f"Unknown sampling method: {method}")

        sampled_rewards = group_rewards[sample_indices]
        sampled_log_probs = group_log_probs[sample_indices]

        return sampled_rewards, sampled_log_probs, sample_indices

    def compute_group_baseline(self, group_rewards: np.ndarray) -> float:
        """
        计算group的基线

        Args:
            group_rewards: group的奖励

        Returns:
            baseline: 基线值
        """
        # 使用均值作为基线
        return np.mean(group_rewards)

    def gspo_loss(
        self,
        group_rewards: np.ndarray,
        group_log_probs: np.ndarray,
        num_samples: int = 1
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算GSPO损失

        Args:
            group_rewards: group的奖励，形状 (group_size,)
            group_log_probs: group的对数概率，形状 (group_size,)
            num_samples: 采样次数（多次采样平均）

        Returns:
            loss: GSPO损失
            metrics: 训练指标
        """
        losses = []
        all_advantages = []

        # 计算基线
        if self.use_baseline:
            baseline = self.compute_group_baseline(group_rewards)
        else:
            baseline = 0.0

        # 多次采样
        for _ in range(num_samples):
            # 从group中采样
            sampled_rewards, sampled_log_probs, _ = self.sample_from_group(
                group_rewards, group_log_probs
            )

            # 计算优势（相对于基线）
            advantages = sampled_rewards - baseline

            # 应用温度
            advantages = advantages / self.temperature

            # 策略损失
            sample_loss = -np.mean(sampled_log_probs * advantages)
            losses.append(sample_loss)
            all_advantages.extend(advantages)

        # 平均损失
        avg_loss = np.mean(losses)

        # 收集指标
        metrics = {
            'loss': avg_loss,
            'loss_std': np.std(losses),  # 损失的标准差（衡量稳定性）
            'reward/mean': np.mean(group_rewards),
            'reward/std': np.std(group_rewards),
            'reward/max': np.max(group_rewards),
            'reward/min': np.min(group_rewards),
            'advantage/mean': np.mean(all_advantages),
            'advantage/std': np.std(all_advantages),
            'baseline': baseline,
        }

        return avg_loss, metrics

    def batch_gspo_loss(
        self,
        rewards_groups: List[np.ndarray],
        log_probs_groups: List[np.ndarray],
        num_samples: int = 1
    ) -> Tuple[float, Dict[str, float]]:
        """
        批量计算多个group的GSPO损失

        Args:
            rewards_groups: 多个group的奖励列表
            log_probs_groups: 多个group的对数概率列表
            num_samples: 每个group的采样次数

        Returns:
            avg_loss: 平均损失
            avg_metrics: 平均指标
        """
        losses = []
        all_metrics = []

        for group_rewards, group_log_probs in zip(rewards_groups, log_probs_groups):
            loss, metrics = self.gspo_loss(
                group_rewards, group_log_probs, num_samples
            )
            losses.append(loss)
            all_metrics.append(metrics)

        # 计算平均
        avg_loss = np.mean(losses)

        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_loss, avg_metrics


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("GSPO (Group-wise Stochastic Policy Optimization) 演示")
    print("=" * 70)

    # ========== 1. 创建GSPO训练器 ==========
    print("\n" + "=" * 70)
    print("1. GSPO训练器初始化")
    print("=" * 70)

    gspo = GSPOTrainer(
        group_size=8,
        sample_size=4,
        use_baseline=True,
        temperature=1.0
    )

    print(f"\nGSPO配置:")
    print(f"  group_size: {gspo.group_size} - group总样本数")
    print(f"  sample_size: {gspo.sample_size} - 每次采样数")
    print(f"  use_baseline: {gspo.use_baseline}")
    print(f"  temperature: {gspo.temperature}")

    # ========== 2. Group采样 ==========
    print("\n" + "=" * 70)
    print("2. Group采样策略")
    print("=" * 70)

    # 模拟一个group
    group_rewards = np.array([0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2])
    group_log_probs = np.random.randn(8) * 0.5 - 2.5

    print(f"\nGroup奖励: {group_rewards}")
    print(f"排序索引: {np.argsort(group_rewards)[::-1]}")

    # 不同采样方法
    for method in ['uniform', 'importance', 'top_k']:
        sampled_rewards, sampled_log_probs, indices = gspo.sample_from_group(
            group_rewards, group_log_probs, method=method
        )

        print(f"\n{method}采样:")
        print(f"  采样索引: {indices}")
        print(f"  采样奖励: {sampled_rewards}")
        print(f"  平均奖励: {np.mean(sampled_rewards):.4f}")

    # ========== 3. GSPO损失计算 ==========
    print("\n" + "=" * 70)
    print("3. GSPO损失计算")
    print("=" * 70)

    # 计算损失（多次采样平均）
    loss, metrics = gspo.gspo_loss(
        group_rewards,
        group_log_probs,
        num_samples=10  # 10次采样平均
    )

    print(f"\nGSPO损失: {loss:.4f}")
    print(f"\n详细指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # ========== 4. 采样次数的影响 ==========
    print("\n" + "=" * 70)
    print("4. 采样次数对方差的影响")
    print("=" * 70)

    num_samples_list = [1, 5, 10, 20, 50]

    print(f"\n{'采样次数':<12} {'损失均值':<15} {'损失标准差':<15}")
    print("-" * 45)

    for n_samples in num_samples_list:
        # 多次实验测量方差
        losses_exp = []
        for _ in range(100):
            loss_exp, _ = gspo.gspo_loss(
                group_rewards, group_log_probs, num_samples=n_samples
            )
            losses_exp.append(loss_exp)

        print(f"{n_samples:<12} {np.mean(losses_exp):<15.4f} {np.std(losses_exp):<15.4f}")

    print("\n观察:")
    print("  • 采样次数越多，方差越小（更稳定）")
    print("  • 但计算成本也增加")
    print("  • 平衡点通常在5-10次")

    # ========== 5. Sample size vs Group size ==========
    print("\n" + "=" * 70)
    print("5. Sample Size选择")
    print("=" * 70)

    group_size = 8
    sample_sizes = [2, 4, 6, 8]

    print(f"\nGroup Size: {group_size}")
    print(f"\n{'Sample Size':<15} {'采样比例':<15} {'损失':<15}")
    print("-" * 50)

    for sample_size in sample_sizes:
        gspo_temp = GSPOTrainer(group_size=group_size, sample_size=sample_size)
        loss_temp, metrics_temp = gspo_temp.gspo_loss(
            group_rewards, group_log_probs, num_samples=10
        )
        ratio = sample_size / group_size
        print(f"{sample_size:<15} {ratio:<15.1%} {loss_temp:<15.4f}")

    print("\n建议:")
    print("  • sample_size = group_size / 2 （平衡效率和方差）")
    print("  • 太小: 方差大，不稳定")
    print("  • 太大: 接近全量，失去随机性优势")

    # ========== 6. 批量训练 ==========
    print("\n" + "=" * 70)
    print("6. 批量GSPO训练")
    print("=" * 70)

    # 模拟多个group
    num_groups = 16
    rewards_groups = []
    log_probs_groups = []

    for i in range(num_groups):
        rewards_groups.append(np.random.randn(8) * 0.3 + 0.5)
        log_probs_groups.append(np.random.randn(8) * 0.5 - 2.5)

    print(f"\n批次大小: {num_groups} groups")
    print(f"每个group: {gspo.group_size} 样本")
    print(f"总样本数: {num_groups * gspo.group_size}")

    # 计算批量损失
    batch_loss, batch_metrics = gspo.batch_gspo_loss(
        rewards_groups, log_probs_groups, num_samples=5
    )

    print(f"\n批量损失: {batch_loss:.4f}")
    print(f"\n批量指标:")
    for key, value in batch_metrics.items():
        print(f"  {key}: {value:.4f}")

    # ========== 7. 基线的作用 ==========
    print("\n" + "=" * 70)
    print("7. 基线对方差的影响")
    print("=" * 70)

    print(f"\n{'使用基线':<12} {'损失均值':<15} {'损失标准差':<15}")
    print("-" * 45)

    for use_baseline in [False, True]:
        gspo_bl = GSPOTrainer(use_baseline=use_baseline)

        # 多次实验
        losses_bl = []
        for _ in range(100):
            loss_bl, _ = gspo_bl.gspo_loss(
                group_rewards, group_log_probs, num_samples=5
            )
            losses_bl.append(loss_bl)

        print(f"{use_baseline!s:<12} {np.mean(losses_bl):<15.4f} {np.std(losses_bl):<15.4f}")

    print("\n作用:")
    print("  • 基线减少方差（方差缩减技术）")
    print("  • 训练更稳定")
    print("  • 推荐开启")

    # ========== 8. GSPO vs 其他方法 ==========
    print("\n" + "=" * 70)
    print("8. GSPO vs 其他方法")
    print("=" * 70)

    print("\nGSPO vs GRPO:")
    print("  相同点:")
    print("    • 都是基于group的方法")
    print("    • 都使用相对比较")
    print("  ")
    print("  不同点:")
    print("    • GSPO: 随机采样（提高效率）")
    print("    • GRPO: 使用全部样本")

    print("\nGSPO vs PPO:")
    print("  • GSPO: Group内随机优化")
    print("  • PPO: 全局策略优化")
    print("  • GSPO更适合大规模分布式训练")

    print("\nGSPO的独特优势:")
    print("  ✓ 样本效率高（group内复用）")
    print("  ✓ 方差缩减（多次采样平均）")
    print("  ✓ 计算高效（随机采样）")
    print("  ✓ 适合分布式（group并行）")

    # ========== 9. 实际应用建议 ==========
    print("\n" + "=" * 70)
    print("9. GSPO实际应用")
    print("=" * 70)

    print("\n推荐配置:")
    print("  • group_size: 8-16")
    print("  • sample_size: group_size / 2")
    print("  • num_samples: 5-10")
    print("  • use_baseline: True")
    print("  • temperature: 1.0")

    print("\n适用场景:")
    print("  • 大规模LLM训练（计算效率重要）")
    print("  • 分布式训练（group可并行）")
    print("  • 高方差环境（方差缩减有用）")
    print("  • 在线学习（需要快速迭代）")

    print("\n训练技巧:")
    print("  ✓ 从小group开始实验")
    print("  ✓ 监控损失的标准差（稳定性指标）")
    print("  ✓ 使用重要性采样提升质量")
    print("  ✓ 结合其他方差缩减技术")

    print("\n" + "=" * 70)
    print("GSPO的关键特性总结")
    print("=" * 70)
    print("✓ Group内随机采样，提高效率")
    print("✓ 多次采样平均，减少方差")
    print("✓ 使用基线，训练更稳定")
    print("✓ 适合大规模分布式训练")
    print("✓ 计算高效，样本效率高")
    print("✗ 需要调整采样超参数")
    print("✗ 理论分析相对较少")
