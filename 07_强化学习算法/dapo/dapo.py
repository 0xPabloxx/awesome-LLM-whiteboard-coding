"""
DAPO (Distributional Advantage Policy Optimization) 实现

DAPO是一种考虑优势函数分布而非点估计的策略优化方法。
与传统方法只使用优势的期望值不同，DAPO建模整个优势分布。

核心思想：
1. 将优势函数建模为分布而非标量
2. 使用分布式强化学习的思想
3. 更准确地捕捉不确定性
4. 提供更稳定的策略更新

优势：
- 更准确的价值估计（考虑不确定性）
- 更稳定的训练（分布平滑噪声）
- 更好的风险意识（区分均值相同但方差不同的情况）
- 适合高方差环境

应用场景：
- 奖励信号不确定的环境
- 需要风险意识的决策
- 高方差的强化学习任务
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class DistributionalAdvantage:
    """
    分布式优势函数

    将优势建模为分位数分布（Quantile Distribution）
    """

    def __init__(self, num_quantiles: int = 51):
        """
        初始化分布式优势

        Args:
            num_quantiles: 分位数数量（奇数，中间为中位数）
        """
        self.num_quantiles = num_quantiles
        # 分位数位置：τ_i = (i + 0.5) / N
        self.quantile_midpoints = (np.arange(num_quantiles) + 0.5) / num_quantiles

    def compute_quantile_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray
    ) -> np.ndarray:
        """
        计算优势的分位数分布

        Args:
            rewards: 奖励序列
            values: 价值估计（分位数形式）

        Returns:
            quantile_advantages: 优势的分位数，形状 (T, num_quantiles)
        """
        T = len(rewards)
        quantile_advantages = np.zeros((T, self.num_quantiles))

        # 对每个时间步计算分位数优势
        for t in range(T):
            # 简化：使用value的分位数
            quantile_advantages[t] = rewards[t] - values[t]

        return quantile_advantages

    def get_mean_advantage(self, quantile_advantages: np.ndarray) -> float:
        """获取优势的期望（所有分位数的平均）"""
        return np.mean(quantile_advantages)

    def get_variance(self, quantile_advantages: np.ndarray) -> float:
        """获取优势的方差"""
        return np.var(quantile_advantages)

    def get_quantile(self, quantile_advantages: np.ndarray, q: float) -> float:
        """获取特定分位数的值"""
        idx = int(q * self.num_quantiles)
        idx = min(idx, self.num_quantiles - 1)
        return quantile_advantages[idx]


class DAPOTrainer:
    """
    DAPO训练器

    使用分布式优势进行策略优化
    """

    def __init__(
        self,
        num_quantiles: int = 51,
        risk_sensitivity: float = 0.0,
        use_quantile_huber: bool = True,
        kappa: float = 1.0
    ):
        """
        初始化DAPO训练器

        Args:
            num_quantiles: 优势分布的分位数数量
            risk_sensitivity: 风险敏感度 [-1, 1]
                - < 0: 风险规避（偏好低方差）
                - = 0: 风险中性（只看期望）
                - > 0: 风险追求（偏好高方差）
            use_quantile_huber: 是否使用分位数Huber损失
            kappa: Huber损失的阈值
        """
        self.num_quantiles = num_quantiles
        self.risk_sensitivity = risk_sensitivity
        self.use_quantile_huber = use_quantile_huber
        self.kappa = kappa

        self.dist_advantage = DistributionalAdvantage(num_quantiles)

    def compute_risk_adjusted_advantage(
        self,
        quantile_advantages: np.ndarray
    ) -> float:
        """
        计算风险调整后的优势

        Args:
            quantile_advantages: 优势的分位数分布

        Returns:
            risk_adjusted_adv: 风险调整后的优势标量
        """
        # 期望
        mean_adv = np.mean(quantile_advantages)

        # 方差
        var_adv = np.var(quantile_advantages)

        # 风险调整：adv' = mean + risk_sensitivity * sqrt(var)
        risk_adjusted_adv = mean_adv + self.risk_sensitivity * np.sqrt(var_adv)

        return risk_adjusted_adv

    def dapo_loss(
        self,
        log_probs: np.ndarray,
        quantile_advantages: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算DAPO损失

        Args:
            log_probs: 对数概率，形状 (batch_size,)
            quantile_advantages: 优势分布，形状 (batch_size, num_quantiles)

        Returns:
            loss: DAPO损失
            metrics: 训练指标
        """
        batch_size = len(log_probs)
        total_loss = 0.0

        # 对每个样本计算风险调整后的优势
        risk_adjusted_advantages = np.zeros(batch_size)
        mean_advantages = np.zeros(batch_size)
        var_advantages = np.zeros(batch_size)

        for i in range(batch_size):
            risk_adjusted_advantages[i] = self.compute_risk_adjusted_advantage(
                quantile_advantages[i]
            )
            mean_advantages[i] = np.mean(quantile_advantages[i])
            var_advantages[i] = np.var(quantile_advantages[i])

        # 策略损失：使用风险调整后的优势
        policy_loss = -np.mean(log_probs * risk_adjusted_advantages)

        # 分布损失（可选）：确保分位数估计准确
        if self.use_quantile_huber:
            dist_loss = self._quantile_regression_loss(quantile_advantages)
        else:
            dist_loss = 0.0

        # 总损失
        total_loss = policy_loss + 0.1 * dist_loss

        # 收集指标
        metrics = {
            'loss/policy': policy_loss,
            'loss/distribution': dist_loss,
            'loss/total': total_loss,
            'advantage/mean': np.mean(mean_advantages),
            'advantage/std': np.mean(np.sqrt(var_advantages)),
            'advantage/risk_adjusted': np.mean(risk_adjusted_advantages),
            'advantage/min_quantile': np.mean(quantile_advantages[:, 0]),
            'advantage/max_quantile': np.mean(quantile_advantages[:, -1]),
        }

        return total_loss, metrics

    def _quantile_regression_loss(self, quantile_values: np.ndarray) -> float:
        """
        分位数回归损失（Quantile Huber Loss）

        确保分位数估计的准确性
        """
        # 简化版本：检查分位数是否单调递增
        batch_size = quantile_values.shape[0]
        loss = 0.0

        for i in range(batch_size):
            # 计算相邻分位数的差异
            diffs = np.diff(quantile_values[i])
            # 惩罚非单调的情况
            loss += np.sum(np.maximum(0, -diffs))

        return loss / batch_size


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("DAPO (Distributional Advantage Policy Optimization) 演示")
    print("=" * 70)

    # ========== 1. 创建分布式优势 ==========
    print("\n" + "=" * 70)
    print("1. 分布式优势函数")
    print("=" * 70)

    num_quantiles = 51
    dist_adv = DistributionalAdvantage(num_quantiles)

    print(f"\n分位数数量: {num_quantiles}")
    print(f"分位数位置（前5个）: {dist_adv.quantile_midpoints[:5]}")

    # 模拟一个优势分布
    advantage_quantiles = np.sort(np.random.randn(num_quantiles) * 0.5)

    print(f"\n优势分布样本:")
    print(f"  最小值 (0%分位): {advantage_quantiles[0]:.4f}")
    print(f"  25%分位: {advantage_quantiles[12]:.4f}")
    print(f"  中位数 (50%分位): {advantage_quantiles[25]:.4f}")
    print(f"  75%分位: {advantage_quantiles[38]:.4f}")
    print(f"  最大值 (100%分位): {advantage_quantiles[-1]:.4f}")

    mean = dist_adv.get_mean_advantage(advantage_quantiles)
    var = dist_adv.get_variance(advantage_quantiles)

    print(f"\n统计量:")
    print(f"  均值: {mean:.4f}")
    print(f"  方差: {var:.4f}")
    print(f"  标准差: {np.sqrt(var):.4f}")

    # ========== 2. DAPO训练器 ==========
    print("\n" + "=" * 70)
    print("2. DAPO训练器初始化")
    print("=" * 70)

    dapo = DAPOTrainer(
        num_quantiles=51,
        risk_sensitivity=0.0,
        use_quantile_huber=True
    )

    print(f"\nDAPO配置:")
    print(f"  num_quantiles: {dapo.num_quantiles}")
    print(f"  risk_sensitivity: {dapo.risk_sensitivity}")
    print(f"  use_quantile_huber: {dapo.use_quantile_huber}")

    # ========== 3. 风险调整 ==========
    print("\n" + "=" * 70)
    print("3. 风险敏感度分析")
    print("=" * 70)

    # 创建两个不同的优势分布
    # 分布1: 低方差
    quantiles_low_var = np.linspace(-0.2, 0.2, num_quantiles)
    # 分布2: 高方差
    quantiles_high_var = np.linspace(-1.0, 1.0, num_quantiles)

    print(f"\n分布1 (低方差):")
    print(f"  均值: {np.mean(quantiles_low_var):.4f}")
    print(f"  标准差: {np.std(quantiles_low_var):.4f}")

    print(f"\n分布2 (高方差):")
    print(f"  均值: {np.mean(quantiles_high_var):.4f}")
    print(f"  标准差: {np.std(quantiles_high_var):.4f}")

    print(f"\n不同风险敏感度下的调整值:")
    print(f"{'风险敏感度':<12} {'低方差分布':<15} {'高方差分布':<15}")
    print("-" * 45)

    for risk_sens in [-0.5, 0.0, 0.5]:
        dapo_temp = DAPOTrainer(risk_sensitivity=risk_sens)
        adj_low = dapo_temp.compute_risk_adjusted_advantage(quantiles_low_var)
        adj_high = dapo_temp.compute_risk_adjusted_advantage(quantiles_high_var)
        print(f"{risk_sens:<12.1f} {adj_low:<15.4f} {adj_high:<15.4f}")

    print("\n解释:")
    print("  • risk < 0: 风险规避，惩罚高方差")
    print("  • risk = 0: 风险中性，只看均值")
    print("  • risk > 0: 风险追求，偏好高方差")

    # ========== 4. DAPO损失计算 ==========
    print("\n" + "=" * 70)
    print("4. DAPO损失计算")
    print("=" * 70)

    batch_size = 4
    log_probs = np.random.randn(batch_size) * 0.5 - 2.0

    # 每个样本的优势分布
    quantile_advantages = np.random.randn(batch_size, num_quantiles) * 0.3

    # 确保分位数单调递增
    for i in range(batch_size):
        quantile_advantages[i] = np.sort(quantile_advantages[i])

    print(f"\n批次大小: {batch_size}")
    print(f"优势分布形状: {quantile_advantages.shape}")

    # 计算损失
    loss, metrics = dapo.dapo_loss(log_probs, quantile_advantages)

    print(f"\nDAPO损失: {loss:.4f}")
    print(f"\n详细指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # ========== 5. 对比标准方法 ==========
    print("\n" + "=" * 70)
    print("5. DAPO vs 标准方法")
    print("=" * 70)

    # 标准方法：只用均值
    standard_advantages = np.mean(quantile_advantages, axis=1)
    standard_loss = -np.mean(log_probs * standard_advantages)

    print(f"\n标准方法损失: {standard_loss:.4f}")
    print(f"DAPO损失: {loss:.4f}")
    print(f"差异: {abs(loss - standard_loss):.4f}")

    print("\nDAPO的优势:")
    print("  ✓ 考虑了不确定性")
    print("  ✓ 可以调整风险偏好")
    print("  ✓ 更稳定的估计")

    # ========== 6. 分位数可视化 ==========
    print("\n" + "=" * 70)
    print("6. 优势分布可视化")
    print("=" * 70)

    sample_idx = 0
    sample_quantiles = quantile_advantages[sample_idx]

    print(f"\n样本 {sample_idx} 的优势分布:")
    print(f"  分位数范围: [{sample_quantiles[0]:.4f}, {sample_quantiles[-1]:.4f}]")
    print(f"  四分位距 (IQR): {sample_quantiles[38] - sample_quantiles[12]:.4f}")
    print(f"  均值: {np.mean(sample_quantiles):.4f}")
    print(f"  中位数: {sample_quantiles[25]:.4f}")

    # ========== 7. 风险意识决策 ==========
    print("\n" + "=" * 70)
    print("7. 风险意识决策示例")
    print("=" * 70)

    # 两个动作的优势分布
    action1_quantiles = np.linspace(0.0, 0.2, num_quantiles)  # 低方差
    action2_quantiles = np.linspace(-0.1, 0.3, num_quantiles)  # 高方差

    print(f"\n动作1 (保守):")
    print(f"  均值: {np.mean(action1_quantiles):.4f}")
    print(f"  标准差: {np.std(action1_quantiles):.4f}")

    print(f"\n动作2 (激进):")
    print(f"  均值: {np.mean(action2_quantiles):.4f}")
    print(f"  标准差: {np.std(action2_quantiles):.4f}")

    print(f"\n不同风险偏好下的选择:")
    for risk_sens in [-0.5, 0.0, 0.5]:
        dapo_risk = DAPOTrainer(risk_sensitivity=risk_sens)
        adj1 = dapo_risk.compute_risk_adjusted_advantage(action1_quantiles)
        adj2 = dapo_risk.compute_risk_adjusted_advantage(action2_quantiles)

        preferred = "动作1 (保守)" if adj1 > adj2 else "动作2 (激进)"
        print(f"  风险敏感度 {risk_sens:+.1f}: 偏好 {preferred}")

    # ========== 8. 应用场景 ==========
    print("\n" + "=" * 70)
    print("8. DAPO的应用场景")
    print("=" * 70)

    print("\n适用任务:")
    print("  • 金融交易: 需要风险管理")
    print("  • 机器人控制: 安全关键任务")
    print("  • 医疗决策: 需要谨慎评估")
    print("  • 游戏AI: 不同风格的策略（保守/激进）")

    print("\n优势:")
    print("  ✓ 捕捉不确定性")
    print("  ✓ 支持风险偏好")
    print("  ✓ 更稳定的估计")
    print("  ✓ 区分相同均值但不同方差的情况")

    print("\n劣势:")
    print("  ✗ 计算开销更大（需要估计分布）")
    print("  ✗ 实现更复杂")
    print("  ✗ 需要更多样本来估计分布")

    print("\n" + "=" * 70)
    print("DAPO的关键特性总结")
    print("=" * 70)
    print("✓ 将优势建模为分布而非标量")
    print("✓ 考虑不确定性和风险")
    print("✓ 支持风险敏感的决策")
    print("✓ 更准确的价值估计")
    print("✗ 计算成本较高")
    print("✗ 需要更多数据")
