"""
GRPO (Group Relative Policy Optimization) 实现

GRPO是DeepSeek提出的强化学习算法，专门用于大语言模型的对齐训练。
核心创新是在一个group（同一prompt的多个响应）内部进行相对比较和优化。

核心思想：
1. 对每个prompt采样多个响应（形成一个group）
2. 在group内部对响应进行相对排序
3. 使用相对优势（而非绝对奖励）更新策略
4. 无需显式的奖励模型或参考模型

优势：
- 无需奖励模型（简化pipeline）
- 无需参考模型（减少计算开销）
- 相对比较更稳定（减少奖励尺度问题）
- 自适应基线（group内平均作为基线）

GRPO损失函数：
L_GRPO = -E[log π_θ(y|x) · A_relative(y)]

其中：
- A_relative(y) = r(x,y) - mean(r(x,y')) for y' in group
- r(x,y): 响应的得分（可以来自简单规则或轻量模型）

应用：DeepSeek-V2、DeepSeek-Coder等模型的对齐训练
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class GRPOTrainer:
    """
    GRPO训练器

    实现Group Relative Policy Optimization算法。
    """

    def __init__(
        self,
        group_size: int = 4,
        temperature: float = 1.0,
        advantage_normalization: bool = True,
        top_k: Optional[int] = None,
        use_ranking_loss: bool = False
    ):
        """
        初始化GRPO训练器

        Args:
            group_size: 每个prompt采样的响应数量
            temperature: softmax温度，控制优势的锐度
            advantage_normalization: 是否标准化优势函数
            top_k: 只使用top-k的响应进行训练（可选）
            use_ranking_loss: 是否使用排序损失
        """
        self.group_size = group_size
        self.temperature = temperature
        self.advantage_normalization = advantage_normalization
        self.top_k = top_k
        self.use_ranking_loss = use_ranking_loss

    def compute_group_advantages(
        self,
        rewards: np.ndarray,
        method: str = 'mean_baseline'
    ) -> np.ndarray:
        """
        计算group内的相对优势

        Args:
            rewards: 奖励数组，形状 (group_size,)
            method: 基线方法
                - 'mean_baseline': 使用group均值作为基线
                - 'min_baseline': 使用group最小值作为基线
                - 'median_baseline': 使用group中位数作为基线

        Returns:
            advantages: 相对优势，形状 (group_size,)
        """
        if method == 'mean_baseline':
            baseline = np.mean(rewards)
        elif method == 'min_baseline':
            baseline = np.min(rewards)
        elif method == 'median_baseline':
            baseline = np.median(rewards)
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        # 相对优势 = 奖励 - 基线
        advantages = rewards - baseline

        # 标准化（可选）
        if self.advantage_normalization and len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    def grpo_loss(
        self,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        baseline_method: str = 'mean_baseline'
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算GRPO损失

        对于一个group的响应，计算相对优势并优化策略。

        Args:
            log_probs: 对数概率，形状 (group_size,)
            rewards: 奖励，形状 (group_size,)
            baseline_method: 基线计算方法

        Returns:
            loss: GRPO损失
            metrics: 训练指标
        """
        # 计算相对优势
        advantages = self.compute_group_advantages(rewards, baseline_method)

        # 应用温度缩放
        advantages = advantages / self.temperature

        # 如果使用top-k，只保留top-k的样本
        if self.top_k is not None and self.top_k < len(rewards):
            top_k_indices = np.argsort(rewards)[-self.top_k:]
            log_probs = log_probs[top_k_indices]
            advantages = advantages[top_k_indices]
            rewards = rewards[top_k_indices]

        # GRPO损失：-E[log π(y|x) · A_relative]
        policy_loss = -np.mean(log_probs * advantages)

        # 排序损失（可选）
        ranking_loss = 0.0
        if self.use_ranking_loss:
            ranking_loss = self._compute_ranking_loss(log_probs, rewards)

        # 总损失
        total_loss = policy_loss + ranking_loss

        # 收集指标
        metrics = {
            'loss/policy': policy_loss,
            'loss/ranking': ranking_loss,
            'loss/total': total_loss,
            'reward/mean': np.mean(rewards),
            'reward/std': np.std(rewards),
            'reward/max': np.max(rewards),
            'reward/min': np.min(rewards),
            'advantage/mean': np.mean(advantages),
            'advantage/std': np.std(advantages),
            'advantage/positive_ratio': np.mean(advantages > 0),
        }

        return total_loss, metrics

    def _compute_ranking_loss(
        self,
        log_probs: np.ndarray,
        rewards: np.ndarray
    ) -> float:
        """
        计算排序损失（可选）

        鼓励高奖励的响应有更高的概率。

        Args:
            log_probs: 对数概率
            rewards: 奖励

        Returns:
            ranking_loss: 排序损失
        """
        # 计算所有配对的排序损失
        n = len(rewards)
        loss = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if rewards[i] != rewards[j]:
                    # 如果 r_i > r_j，则希望 log p_i > log p_j
                    if rewards[i] > rewards[j]:
                        margin = log_probs[i] - log_probs[j]
                    else:
                        margin = log_probs[j] - log_probs[i]

                    # Hinge loss: max(0, -margin)
                    loss += np.maximum(0, -margin)
                    count += 1

        return loss / count if count > 0 else 0.0

    def batch_grpo_loss(
        self,
        log_probs_groups: List[np.ndarray],
        rewards_groups: List[np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """
        批量计算多个group的GRPO损失

        Args:
            log_probs_groups: 多个group的对数概率列表
            rewards_groups: 多个group的奖励列表

        Returns:
            avg_loss: 平均损失
            avg_metrics: 平均指标
        """
        losses = []
        all_metrics = []

        for log_probs, rewards in zip(log_probs_groups, rewards_groups):
            loss, metrics = self.grpo_loss(log_probs, rewards)
            losses.append(loss)
            all_metrics.append(metrics)

        # 计算平均
        avg_loss = np.mean(losses)

        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_loss, avg_metrics


class GRPODataset:
    """
    GRPO数据集

    管理group形式的数据：每个prompt对应多个响应。
    """

    def __init__(self, group_size: int = 4):
        """
        初始化数据集

        Args:
            group_size: 每个prompt的响应数量
        """
        self.group_size = group_size
        self.groups = []

    def add_group(
        self,
        prompt: str,
        responses: List[str],
        rewards: List[float],
        log_probs: Optional[List[float]] = None
    ):
        """
        添加一个group

        Args:
            prompt: 输入提示
            responses: 响应列表
            rewards: 奖励列表
            log_probs: 对数概率列表（可选）
        """
        assert len(responses) == self.group_size
        assert len(rewards) == self.group_size

        self.groups.append({
            'prompt': prompt,
            'responses': responses,
            'rewards': np.array(rewards),
            'log_probs': np.array(log_probs) if log_probs else None,
        })

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]


def compare_grpo_with_other_methods():
    """
    对比GRPO与其他方法

    Returns:
        comparison: 对比字典
    """
    comparison = {
        'PPO': {
            'reward_model': '需要',
            'reference_model': '需要',
            'sampling': '需要多次采样',
            'complexity': '高',
            'stability': '高',
            'group_comparison': '否'
        },
        'DPO': {
            'reward_model': '不需要',
            'reference_model': '需要',
            'sampling': '使用现有数据',
            'complexity': '低',
            'stability': '高',
            'group_comparison': '配对比较'
        },
        'GRPO': {
            'reward_model': '不需要（或轻量）',
            'reference_model': '不需要',
            'sampling': '每个prompt多次采样',
            'complexity': '中',
            'stability': '高',
            'group_comparison': '是（group内）'
        }
    }

    return comparison


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("GRPO (Group Relative Policy Optimization) 演示")
    print("=" * 70)

    # ========== 1. 创建GRPO训练器 ==========
    print("\n" + "=" * 70)
    print("1. GRPO训练器初始化")
    print("=" * 70)

    grpo = GRPOTrainer(
        group_size=4,
        temperature=1.0,
        advantage_normalization=True
    )

    print(f"\nGRPO配置:")
    print(f"  group_size: {grpo.group_size} - 每个prompt采样的响应数")
    print(f"  temperature: {grpo.temperature} - 优势缩放温度")
    print(f"  advantage_normalization: {grpo.advantage_normalization}")

    # ========== 2. Group优势计算 ==========
    print("\n" + "=" * 70)
    print("2. Group内相对优势计算")
    print("=" * 70)

    # 模拟一个group的奖励
    rewards = np.array([0.8, 0.5, 0.9, 0.3])

    print(f"\nGroup奖励: {rewards}")
    print(f"奖励排序: {np.argsort(rewards)[::-1]} (从高到低)")

    # 不同基线方法
    for method in ['mean_baseline', 'min_baseline', 'median_baseline']:
        advantages = grpo.compute_group_advantages(rewards, method=method)
        print(f"\n{method}:")
        print(f"  优势: {advantages}")
        print(f"  正优势比例: {np.sum(advantages > 0) / len(advantages):.2%}")

    # ========== 3. GRPO损失计算 ==========
    print("\n" + "=" * 70)
    print("3. GRPO损失计算")
    print("=" * 70)

    # 模拟对数概率（对应4个响应）
    log_probs = np.array([-2.5, -3.0, -2.3, -3.5])

    print(f"\n对数概率: {log_probs}")
    print(f"奖励: {rewards}")

    # 计算损失
    loss, metrics = grpo.grpo_loss(log_probs, rewards)

    print(f"\nGRPO损失: {loss:.4f}")
    print(f"\n详细指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # ========== 4. 不同温度的影响 ==========
    print("\n" + "=" * 70)
    print("4. 温度参数的影响")
    print("=" * 70)

    temperatures = [0.5, 1.0, 2.0, 5.0]

    print(f"\n{'温度':<8} {'损失':<12} {'优势标准差':<15}")
    print("-" * 40)

    for temp in temperatures:
        grpo_temp = GRPOTrainer(temperature=temp)
        loss_temp, metrics_temp = grpo_temp.grpo_loss(log_probs, rewards)
        print(f"{temp:<8.1f} {loss_temp:<12.4f} {metrics_temp['advantage/std']:<15.4f}")

    print("\n解释:")
    print("  • 温度越小: 优势差异越大，训练信号越强")
    print("  • 温度越大: 优势差异越小，训练更平滑")
    print("  • 类似DPO的β参数")

    # ========== 5. Top-K选择 ==========
    print("\n" + "=" * 70)
    print("5. Top-K响应选择")
    print("=" * 70)

    print(f"\n原始奖励: {rewards}")
    print(f"排序索引: {np.argsort(rewards)[::-1]}")

    for k in [2, 3, 4]:
        grpo_topk = GRPOTrainer(group_size=4, top_k=k)
        loss_topk, metrics_topk = grpo_topk.grpo_loss(log_probs, rewards)

        print(f"\nTop-{k}:")
        print(f"  损失: {loss_topk:.4f}")
        print(f"  平均奖励: {metrics_topk['reward/mean']:.4f}")

    print("\nTop-K作用:")
    print("  • 只使用质量最高的k个响应")
    print("  • 过滤低质量样本")
    print("  • 提高训练效率")

    # ========== 6. 批量训练 ==========
    print("\n" + "=" * 70)
    print("6. 批量GRPO训练")
    print("=" * 70)

    # 模拟多个group
    num_groups = 8
    log_probs_groups = []
    rewards_groups = []

    for i in range(num_groups):
        log_probs_groups.append(np.random.randn(4) * 0.5 - 2.5)
        rewards_groups.append(np.random.randn(4) * 0.3 + 0.5)

    print(f"\n批次大小: {num_groups} groups")
    print(f"每个group: {grpo.group_size} 响应")
    print(f"总样本数: {num_groups * grpo.group_size}")

    # 计算批量损失
    batch_loss, batch_metrics = grpo.batch_grpo_loss(
        log_probs_groups,
        rewards_groups
    )

    print(f"\n批量损失: {batch_loss:.4f}")
    print(f"\n批量指标:")
    for key, value in batch_metrics.items():
        print(f"  {key}: {value:.4f}")

    # ========== 7. GRPO数据集 ==========
    print("\n" + "=" * 70)
    print("7. GRPO数据集示例")
    print("=" * 70)

    dataset = GRPODataset(group_size=4)

    # 添加示例数据
    examples = [
        {
            'prompt': '解释什么是机器学习',
            'responses': [
                '机器学习是AI的一个分支，让计算机从数据中学习。',
                '机器学习就是训练模型。',
                '机器学习是一种人工智能技术，通过算法从数据中学习规律，无需显式编程。',
                '机器学习是让电脑自己学东西。'
            ],
            'rewards': [0.7, 0.4, 0.9, 0.2],
            'log_probs': [-2.3, -3.5, -2.0, -4.0]
        },
        {
            'prompt': '如何保持健康？',
            'responses': [
                '多运动，健康饮食。',
                '保持健康需要均衡饮食、规律运动、充足睡眠和良好心态。',
                '睡觉。',
                '健康生活方式包括：每周150分钟运动、多吃蔬果、避免熬夜。'
            ],
            'rewards': [0.6, 0.8, 0.3, 0.85],
            'log_probs': [-2.8, -2.2, -4.2, -2.1]
        }
    ]

    for ex in examples:
        dataset.add_group(
            ex['prompt'],
            ex['responses'],
            ex['rewards'],
            ex['log_probs']
        )

    print(f"\n数据集大小: {len(dataset)} groups")
    print(f"\n示例group:")
    group = dataset[0]
    print(f"  Prompt: {group['prompt']}")
    print(f"  响应数: {len(group['responses'])}")
    print(f"  奖励: {group['rewards']}")
    print(f"  最佳响应索引: {np.argmax(group['rewards'])}")
    print(f"  最佳响应: {group['responses'][np.argmax(group['rewards'])]}")

    # ========== 8. GRPO vs 其他方法 ==========
    print("\n" + "=" * 70)
    print("8. GRPO vs 其他对齐方法")
    print("=" * 70)

    comparison = compare_grpo_with_other_methods()

    print(f"\n{'方法':<10} {'奖励模型':<15} {'参考模型':<15} {'复杂度':<10}")
    print("-" * 60)
    for method, props in comparison.items():
        print(f"{method:<10} {props['reward_model']:<15} "
              f"{props['reference_model']:<15} {props['complexity']:<10}")

    print("\nGRPO的独特优势:")
    print("  ✓ 无需显式奖励模型（可用简单规则）")
    print("  ✓ 无需参考模型（减少计算）")
    print("  ✓ Group内相对比较（更稳定）")
    print("  ✓ 自适应基线（自动调整）")

    # ========== 9. 实际应用场景 ==========
    print("\n" + "=" * 70)
    print("9. GRPO的实际应用")
    print("=" * 70)

    print("\n成功案例:")
    print("  • DeepSeek-V2: 使用GRPO进行对齐")
    print("  • DeepSeek-Coder: 代码生成模型对齐")
    print("  • 其他内部模型: 多个场景验证")

    print("\n典型工作流程:")
    print("  1. 对每个prompt采样K个响应（K=4~8）")
    print("  2. 使用简单规则或轻量模型评分")
    print("     - 代码任务: 执行测试用例")
    print("     - 对话任务: 长度、流畅度等规则")
    print("     - 数学任务: 答案正确性")
    print("  3. 在group内计算相对优势")
    print("  4. 使用GRPO损失更新策略")

    print("\n配置建议:")
    print("  • group_size: 4-8（平衡质量和效率）")
    print("  • temperature: 1.0（起点）")
    print("  • top_k: group_size的一半（可选）")
    print("  • 学习率: 1e-6（类似PPO）")

    # ========== 10. 优势分析 ==========
    print("\n" + "=" * 70)
    print("10. GRPO的优势分析")
    print("=" * 70)

    print("\n相比PPO:")
    print("  ✓ 无需价值函数（简化模型）")
    print("  ✓ 无需复杂的RL训练循环")
    print("  ✓ 可用简单奖励（不需要精确奖励模型）")
    print("  ✓ 计算效率更高")

    print("\n相比DPO:")
    print("  ✓ 不局限于偏好对（可用多个响应）")
    print("  ✓ 更灵活的奖励信号（不限于比较）")
    print("  ✓ 无需参考模型")
    print("  ✗ 需要在线采样（DPO可用离线数据）")

    print("\n适用场景:")
    print("  • 有明确评价指标的任务（代码、数学）")
    print("  • 需要在线采样和迭代优化")
    print("  • 计算资源有限（不想训练奖励模型）")
    print("  • 需要快速实验和迭代")

    print("\n" + "=" * 70)
    print("GRPO的关键特性总结")
    print("=" * 70)
    print("✓ 无需奖励模型和参考模型")
    print("✓ Group内相对比较，训练更稳定")
    print("✓ 自适应基线，自动调整优势")
    print("✓ 实现简单，易于调试")
    print("✓ DeepSeek验证，效果优秀")
    print("✗ 需要在线采样（计算开销）")
    print("✗ 需要合理的评分函数")
