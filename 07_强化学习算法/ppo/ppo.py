"""
PPO (Proximal Policy Optimization) 实现

PPO是OpenAI提出的策略梯度算法，也是RLHF中最常用的强化学习算法。
通过限制策略更新步长，PPO在样本效率和训练稳定性之间取得了良好平衡。

核心思想：
1. 收集当前策略的经验数据
2. 使用clip目标函数限制策略更新
3. 多次重用数据进行优化（提高样本效率）
4. 保证每次更新不会偏离旧策略太远（稳定性）

PPO-Clip目标函数：
L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

其中：
- r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t): 重要性采样比率
- Â_t: 优势函数估计
- ε: clip范围，通常为0.2
- clip函数: 将r_t限制在[1-ε, 1+ε]范围内

在LLM对齐中的应用：
- 状态s: 输入提示
- 动作a: 生成的token
- 奖励r: 来自奖励模型的评分
- 策略π: 语言模型

优势：
- 稳定性好，超参数鲁棒
- 样本效率高（可以重用数据）
- 实现简单，易于调试
- 在RLHF中表现优秀

应用：ChatGPT、GPT-4、Claude等所有主流对话模型的对齐训练
论文：Proximal Policy Optimization Algorithms (Schulman et al., 2017)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class PPOTrainer:
    """
    PPO训练器

    实现PPO-Clip算法，用于强化学习策略优化。
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1,
        gamma: float = 1.0,
        lam: float = 0.95,
        max_grad_norm: float = 0.5
    ):
        """
        初始化PPO训练器

        Args:
            clip_epsilon: PPO clip范围，控制策略更新幅度
            value_loss_coef: 价值函数损失的权重
            entropy_coef: 熵正则化的权重（鼓励探索）
            kl_coef: KL散度惩罚的权重
            gamma: 折扣因子，用于计算回报
            lam: GAE的λ参数，用于优势函数估计
            max_grad_norm: 梯度裁剪的最大范数
        """
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm

    def compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用GAE (Generalized Advantage Estimation) 计算优势函数

        GAE公式：
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            rewards: 奖励序列，形状 (T,)
            values: 价值函数估计，形状 (T+1,)
            dones: episode结束标志，形状 (T,)

        Returns:
            advantages: 优势函数，形状 (T,)
            returns: 回报，形状 (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)

        if dones is None:
            dones = np.zeros(T)

        # 从后向前计算
        gae = 0
        for t in reversed(range(T)):
            # TD误差: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            if t == T - 1:
                next_value = 0 if dones[t] else values[t + 1]
            else:
                next_value = values[t + 1] * (1 - dones[t])

            delta = rewards[t] + self.gamma * next_value - values[t]

            # GAE累积
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

            # 回报: R_t = A_t + V(s_t)
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def ppo_loss(
        self,
        old_log_probs: np.ndarray,
        new_log_probs: np.ndarray,
        advantages: np.ndarray,
        old_values: np.ndarray,
        new_values: np.ndarray,
        returns: np.ndarray,
        entropy: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算PPO损失

        总损失 = 策略损失 + 价值损失系数 * 价值损失 - 熵系数 * 熵

        Args:
            old_log_probs: 旧策略的对数概率，形状 (batch_size,)
            new_log_probs: 新策略的对数概率，形状 (batch_size,)
            advantages: 优势函数，形状 (batch_size,)
            old_values: 旧价值函数，形状 (batch_size,)
            new_values: 新价值函数，形状 (batch_size,)
            returns: 回报，形状 (batch_size,)
            entropy: 熵，形状 (batch_size,)

        Returns:
            total_loss: 总损失
            metrics: 包含各项指标的字典
        """
        # 标准化优势函数（提高训练稳定性）
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # 1. 策略损失（PPO-Clip）
        # 重要性采样比率: r_t = π_new / π_old = exp(log π_new - log π_old)
        ratio = np.exp(new_log_probs - old_log_probs)

        # Clip版本的策略目标
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

        # 取最小值（保守估计）
        policy_loss = -np.mean(np.minimum(surr1, surr2))

        # 2. 价值函数损失
        # Clip价值函数更新（可选，提高稳定性）
        value_pred_clipped = old_values + np.clip(
            new_values - old_values,
            -self.clip_epsilon,
            self.clip_epsilon
        )
        value_loss_unclipped = (new_values - returns) ** 2
        value_loss_clipped = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * np.mean(np.maximum(value_loss_unclipped, value_loss_clipped))

        # 3. 熵损失（鼓励探索）
        if entropy is None:
            entropy = np.zeros_like(advantages)
        entropy_loss = -np.mean(entropy)

        # 4. KL散度（用于监控，不加入损失）
        kl = np.mean(old_log_probs - new_log_probs)

        # 总损失
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss +
            self.entropy_coef * entropy_loss
        )

        # 计算额外指标
        metrics = {
            'loss/total': total_loss,
            'loss/policy': policy_loss,
            'loss/value': value_loss,
            'loss/entropy': entropy_loss,
            'policy/approx_kl': kl,
            'policy/clip_fraction': np.mean(np.abs(ratio - 1.0) > self.clip_epsilon),
            'policy/ratio_mean': np.mean(ratio),
            'policy/ratio_std': np.std(ratio),
            'policy/advantages_mean': np.mean(advantages),
            'returns/mean': np.mean(returns),
            'returns/std': np.std(returns),
            'value/mean': np.mean(new_values),
            'value/explained_variance': self._explained_variance(returns, new_values),
        }

        return total_loss, metrics

    def _explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算解释方差（评估价值函数的拟合质量）

        EV = 1 - Var(y_true - y_pred) / Var(y_true)
        """
        var_y = np.var(y_true)
        return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)


class PPORolloutBuffer:
    """
    PPO经验回放缓冲区

    存储trajectory数据用于PPO训练。
    """

    def __init__(self, buffer_size: int):
        """
        初始化缓冲区

        Args:
            buffer_size: 缓冲区大小
        """
        self.buffer_size = buffer_size
        self.reset()

    def reset(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool = False
    ):
        """添加一个时间步的数据"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.ptr += 1

    def get(self) -> Dict[str, np.ndarray]:
        """获取缓冲区中的所有数据"""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones),
        }

    def __len__(self):
        return self.ptr


class RewardModel:
    """
    简单的奖励模型

    在实际RLHF中，这是一个从人类偏好数据训练的神经网络。
    这里用简化版本进行演示。
    """

    def __init__(self, reward_fn=None):
        """
        初始化奖励模型

        Args:
            reward_fn: 奖励函数，接受(state, action)返回reward
        """
        self.reward_fn = reward_fn or self._default_reward

    def _default_reward(self, state: np.ndarray, action: int) -> float:
        """默认奖励函数（示例）"""
        # 简单示例：奖励与action成正比，带随机噪声
        return float(action) * 0.1 + np.random.randn() * 0.01

    def __call__(self, state: np.ndarray, action: int) -> float:
        """计算奖励"""
        return self.reward_fn(state, action)


def compute_kl_penalty(
    current_logprobs: np.ndarray,
    reference_logprobs: np.ndarray,
    kl_coef: float = 0.1
) -> np.ndarray:
    """
    计算KL散度惩罚

    在RLHF中，我们希望策略不要偏离参考策略太远。
    KL惩罚: -kl_coef * (log π_θ - log π_ref)

    Args:
        current_logprobs: 当前策略的对数概率
        reference_logprobs: 参考策略的对数概率
        kl_coef: KL系数

    Returns:
        kl_penalty: KL惩罚（加到奖励上）
    """
    kl = current_logprobs - reference_logprobs
    return -kl_coef * kl


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("PPO (Proximal Policy Optimization) 演示")
    print("=" * 70)

    # ========== 1. 创建PPO训练器 ==========
    print("\n" + "=" * 70)
    print("1. PPO训练器初始化")
    print("=" * 70)

    ppo = PPOTrainer(
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        kl_coef=0.1,
        gamma=1.0,
        lam=0.95
    )

    print(f"\nPPO配置:")
    print(f"  clip_epsilon: {ppo.clip_epsilon} - Clip范围")
    print(f"  value_loss_coef: {ppo.value_loss_coef} - 价值损失权重")
    print(f"  entropy_coef: {ppo.entropy_coef} - 熵正则化权重")
    print(f"  kl_coef: {ppo.kl_coef} - KL散度惩罚权重")
    print(f"  gamma: {ppo.gamma} - 折扣因子")
    print(f"  lam: {ppo.lam} - GAE λ参数")

    # ========== 2. GAE优势函数计算 ==========
    print("\n" + "=" * 70)
    print("2. GAE (Generalized Advantage Estimation) 计算")
    print("=" * 70)

    # 模拟一个trajectory
    T = 10
    rewards = np.random.randn(T) * 0.5 + 0.5  # 奖励
    values = np.random.randn(T + 1) * 0.3  # 价值函数估计（需要T+1个）

    print(f"\nTrajectory长度: {T}")
    print(f"奖励样本: {rewards[:5]}")
    print(f"价值估计样本: {values[:5]}")

    # 计算优势函数
    advantages, returns = ppo.compute_advantages(rewards, values)

    print(f"\n优势函数: {advantages[:5]}")
    print(f"回报: {returns[:5]}")
    print(f"\n优势函数均值: {np.mean(advantages):.4f}")
    print(f"优势函数标准差: {np.std(advantages):.4f}")

    # ========== 3. PPO损失计算 ==========
    print("\n" + "=" * 70)
    print("3. PPO-Clip损失计算")
    print("=" * 70)

    batch_size = 32

    # 旧策略的对数概率
    old_log_probs = np.random.randn(batch_size) * 0.5 - 2.0

    # 新策略的对数概率（略有不同）
    new_log_probs = old_log_probs + np.random.randn(batch_size) * 0.1

    # 优势函数
    advantages_batch = np.random.randn(batch_size)

    # 价值函数
    old_values = np.random.randn(batch_size)
    new_values = old_values + np.random.randn(batch_size) * 0.1

    # 回报
    returns_batch = old_values + advantages_batch

    # 熵
    entropy_batch = -new_log_probs

    print(f"\n批次大小: {batch_size}")
    print(f"旧策略log prob样本: {old_log_probs[:3]}")
    print(f"新策略log prob样本: {new_log_probs[:3]}")

    # 计算损失
    loss, metrics = ppo.ppo_loss(
        old_log_probs,
        new_log_probs,
        advantages_batch,
        old_values,
        new_values,
        returns_batch,
        entropy_batch
    )

    print(f"\n总损失: {loss:.4f}")
    print(f"\n详细指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # ========== 4. Clip机制可视化 ==========
    print("\n" + "=" * 70)
    print("4. PPO Clip机制分析")
    print("=" * 70)

    # 生成不同的ratio值
    ratios = np.linspace(0.5, 1.5, 100)
    advantage_pos = 1.0  # 正优势
    advantage_neg = -1.0  # 负优势

    # 计算目标函数值
    surr1_pos = ratios * advantage_pos
    surr2_pos = np.clip(ratios, 1 - ppo.clip_epsilon, 1 + ppo.clip_epsilon) * advantage_pos
    objective_pos = np.minimum(surr1_pos, surr2_pos)

    surr1_neg = ratios * advantage_neg
    surr2_neg = np.clip(ratios, 1 - ppo.clip_epsilon, 1 + ppo.clip_epsilon) * advantage_neg
    objective_neg = np.minimum(surr1_neg, surr2_neg)

    print(f"\nClip范围: [{1-ppo.clip_epsilon:.2f}, {1+ppo.clip_epsilon:.2f}]")
    print(f"\n当优势为正时:")
    print(f"  - ratio > {1+ppo.clip_epsilon:.2f}: 目标被clip，停止增长")
    print(f"  - ratio < {1-ppo.clip_epsilon:.2f}: 正常更新")
    print(f"\n当优势为负时:")
    print(f"  - ratio < {1-ppo.clip_epsilon:.2f}: 目标被clip，停止下降")
    print(f"  - ratio > {1+ppo.clip_epsilon:.2f}: 正常更新")

    print(f"\n解释:")
    print(f"  • Clip机制防止策略更新过大")
    print(f"  • 当更新方向正确但幅度过大时，限制其增益")
    print(f"  • 提高训练稳定性")

    # ========== 5. 不同clip_epsilon的影响 ==========
    print("\n" + "=" * 70)
    print("5. 不同clip_epsilon的影响")
    print("=" * 70)

    epsilons = [0.1, 0.2, 0.3, 0.5]

    print(f"\n{'ε':<8} {'损失':<12} {'Clip比例':<12} {'KL':<12}")
    print("-" * 50)

    for eps in epsilons:
        ppo_temp = PPOTrainer(clip_epsilon=eps)
        loss_temp, metrics_temp = ppo_temp.ppo_loss(
            old_log_probs,
            new_log_probs,
            advantages_batch,
            old_values,
            new_values,
            returns_batch,
            entropy_batch
        )
        print(f"{eps:<8.1f} {loss_temp:<12.4f} {metrics_temp['policy/clip_fraction']:<12.4f} "
              f"{metrics_temp['policy/approx_kl']:<12.4f}")

    print("\n观察:")
    print("  • ε越小: 更保守，clip比例越高")
    print("  • ε越大: 更激进，允许更大更新")
    print("  • 典型值: ε = 0.2（论文推荐）")

    # ========== 6. PPO经验缓冲区 ==========
    print("\n" + "=" * 70)
    print("6. PPO Rollout Buffer示例")
    print("=" * 70)

    buffer = PPORolloutBuffer(buffer_size=1000)

    # 模拟收集数据
    num_steps = 50
    state_dim = 8

    print(f"\n收集{num_steps}步经验...")
    for step in range(num_steps):
        state = np.random.randn(state_dim)
        action = np.random.randint(0, 10)
        reward = np.random.randn() * 0.5
        value = np.random.randn() * 0.3
        log_prob = np.random.randn() * 0.5 - 2.0
        done = (step % 10 == 9)  # 每10步一个episode

        buffer.add(state, action, reward, value, log_prob, done)

    print(f"缓冲区大小: {len(buffer)}")

    # 获取数据
    data = buffer.get()
    print(f"\n数据形状:")
    for key, value in data.items():
        print(f"  {key}: {value.shape}")

    # ========== 7. RLHF中的PPO ==========
    print("\n" + "=" * 70)
    print("7. PPO在RLHF中的应用")
    print("=" * 70)

    print("\nRLHF训练流程:")
    print("  1. SFT: 在指令数据上微调基础模型")
    print("  2. RM: 训练奖励模型（从人类偏好数据）")
    print("  3. PPO: 使用奖励模型优化策略")
    print("     - 采样: 从当前策略生成响应")
    print("     - 评分: 使用奖励模型打分")
    print("     - 优化: PPO更新策略")
    print("     - 约束: KL散度限制偏离参考策略")

    print("\n在LLM中的映射:")
    print("  • 状态(s): 输入提示")
    print("  • 动作(a): 生成的token")
    print("  • 策略(π): 语言模型")
    print("  • 奖励(r): 奖励模型评分")
    print("  • 参考策略(π_ref): SFT模型")

    # ========== 8. KL散度惩罚 ==========
    print("\n" + "=" * 70)
    print("8. KL散度惩罚")
    print("=" * 70)

    # 模拟当前策略和参考策略的对数概率
    current_logprobs = np.random.randn(10) * 0.5 - 2.0
    reference_logprobs = current_logprobs + np.random.randn(10) * 0.3

    kl_penalty = compute_kl_penalty(current_logprobs, reference_logprobs, kl_coef=0.1)

    print(f"\n当前策略log prob: {current_logprobs[:3]}")
    print(f"参考策略log prob: {reference_logprobs[:3]}")
    print(f"KL惩罚: {kl_penalty[:3]}")
    print(f"\nKL惩罚均值: {np.mean(kl_penalty):.4f}")

    print(f"\n作用:")
    print(f"  • 防止策略偏离参考模型太远")
    print(f"  • 保持预训练知识")
    print(f"  • 避免奖励模型过度优化（reward hacking）")

    # ========== 9. PPO vs 其他RL算法 ==========
    print("\n" + "=" * 70)
    print("9. PPO vs 其他RL算法")
    print("=" * 70)

    comparison = {
        'PPO': {
            'stability': 'High',
            'sample_efficiency': 'Medium-High',
            'implementation': 'Medium',
            'hyperparameters': 'Few',
        },
        'TRPO': {
            'stability': 'Very High',
            'sample_efficiency': 'Medium',
            'implementation': 'Complex',
            'hyperparameters': 'Medium',
        },
        'A3C': {
            'stability': 'Medium',
            'sample_efficiency': 'Medium',
            'implementation': 'Medium',
            'hyperparameters': 'Many',
        },
        'DQN': {
            'stability': 'Medium',
            'sample_efficiency': 'Low',
            'implementation': 'Simple',
            'hyperparameters': 'Many',
        }
    }

    print(f"\n{'算法':<10} {'稳定性':<15} {'样本效率':<15} {'实现难度':<15}")
    print("-" * 60)
    for algo, props in comparison.items():
        print(f"{algo:<10} {props['stability']:<15} {props['sample_efficiency']:<15} "
              f"{props['implementation']:<15}")

    print("\nPPO优势:")
    print("  ✓ 训练稳定（Clip机制）")
    print("  ✓ 样本效率高（可以重用数据）")
    print("  ✓ 超参数鲁棒")
    print("  ✓ 实现相对简单")
    print("  ✓ 在RLHF中表现优秀")

    # ========== 10. 实际应用 ==========
    print("\n" + "=" * 70)
    print("10. PPO的实际应用")
    print("=" * 70)

    print("\n成功案例:")
    print("  • ChatGPT: 使用PPO进行对齐训练")
    print("  • GPT-4: RLHF中使用PPO")
    print("  • Claude: Anthropic的对话模型")
    print("  • Llama 2-Chat: Meta的对话模型")
    print("  • Sparrow: DeepMind的研究项目")

    print("\n典型配置（LLM对齐）:")
    print("  • clip_epsilon: 0.2")
    print("  • value_loss_coef: 0.5")
    print("  • entropy_coef: 0.01")
    print("  • kl_coef: 0.1")
    print("  • 学习率: 1e-6")
    print("  • batch_size: 64-512")
    print("  • PPO epochs: 4")
    print("  • 序列长度: 512-2048")

    print("\n训练技巧:")
    print("  ✓ 从SFT模型开始")
    print("  ✓ 使用较小学习率")
    print("  ✓ 监控KL散度（避免偏离太远）")
    print("  ✓ 监控奖励（避免reward hacking）")
    print("  ✓ 使用梯度裁剪")
    print("  ✓ 标准化优势函数")

    print("\n" + "=" * 70)
    print("PPO的关键特性总结")
    print("=" * 70)
    print("✓ Clip机制保证训练稳定性")
    print("✓ 可以重用数据，提高样本效率")
    print("✓ 超参数鲁棒，易于调优")
    print("✓ RLHF的标准算法，广泛应用")
    print("✓ 在ChatGPT等主流模型中使用")
    print("✗ 比DPO复杂（需要奖励模型和RL训练）")
    print("✗ 训练时间较长")
