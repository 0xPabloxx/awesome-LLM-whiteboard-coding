"""
DPO (Direct Preference Optimization) 实现

DPO是一种无需奖励模型的对齐算法，直接从人类偏好数据优化策略。
相比传统RLHF，DPO避免了奖励模型训练和强化学习的复杂性。

核心思想：
1. 使用Bradley-Terry模型表示人类偏好
2. 将奖励模型参数化为策略的隐式表示
3. 直接优化策略以最大化偏好概率

公式：
L_DPO(π_θ; π_ref) = -E[(x,y_w,y_l)~D] [
    log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))
]

其中：
- y_w: 偏好的响应（赢家）
- y_l: 不偏好的响应（输家）
- β: 温度参数，控制偏离参考策略的程度
- σ: sigmoid函数
- π_ref: 参考策略（通常是SFT模型）

优势：
- 无需训练奖励模型
- 更稳定（避免RL的不稳定性）
- 计算高效（简单的监督学习）
- 效果可以媲美PPO

应用：Llama 2、Zephyr、Mistral等模型的对齐训练
论文：Direct Preference Optimization: Your Language Model is Secretly a Reward Model
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def log_sigmoid(x):
    """数值稳定的log sigmoid"""
    return -np.log1p(np.exp(-np.clip(x, -500, 500)))


class DPOTrainer:
    """
    DPO训练器

    实现Direct Preference Optimization算法，从偏好对中学习最优策略。
    """

    def __init__(self, beta: float = 0.1, reference_free: bool = False):
        """
        初始化DPO训练器

        Args:
            beta: KL散度惩罚系数，控制策略偏离参考模型的程度
                  - 较大的beta: 更保守，更接近参考策略
                  - 较小的beta: 更激进，可能偏离参考策略
            reference_free: 是否使用reference-free版本（不需要参考策略）
        """
        self.beta = beta
        self.reference_free = reference_free

    def compute_log_probs(self, logits: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """
        计算序列的对数概率

        Args:
            logits: 模型输出logits，形状 (seq_len, vocab_size)
            tokens: token序列，形状 (seq_len,)

        Returns:
            log_prob: 序列的总对数概率（标量）
        """
        # Softmax归一化
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)

        # 提取对应token的概率
        log_prob = 0.0
        for i, token in enumerate(tokens):
            log_prob += np.log(probs[i, token] + 1e-10)

        return log_prob

    def dpo_loss(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        reference_chosen_logps: Optional[np.ndarray] = None,
        reference_rejected_logps: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算DPO损失

        Args:
            policy_chosen_logps: 策略模型对偏好响应的对数概率，形状 (batch_size,)
            policy_rejected_logps: 策略模型对拒绝响应的对数概率，形状 (batch_size,)
            reference_chosen_logps: 参考模型对偏好响应的对数概率
            reference_rejected_logps: 参考模型对拒绝响应的对数概率

        Returns:
            loss: DPO损失（标量）
            metrics: 包含各项指标的字典
        """
        if self.reference_free:
            # Reference-free DPO：不使用参考策略
            logits = self.beta * (policy_chosen_logps - policy_rejected_logps)
        else:
            # 标准DPO：相对于参考策略的对数比率
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            logits = self.beta * (pi_logratios - ref_logratios)

        # DPO损失: -log sigmoid(logits)
        losses = -log_sigmoid(logits)
        loss = np.mean(losses)

        # 计算隐式奖励
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps) \
                         if not self.reference_free else self.beta * policy_chosen_logps
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps) \
                          if not self.reference_free else self.beta * policy_rejected_logps

        # 收集指标
        metrics = {
            'loss': loss,
            'reward_margin': np.mean(chosen_rewards - rejected_rewards),
            'reward_chosen': np.mean(chosen_rewards),
            'reward_rejected': np.mean(rejected_rewards),
            'accuracy': np.mean((chosen_rewards > rejected_rewards).astype(float)),
            'logits': np.mean(logits),
        }

        return loss, metrics

    def compute_preference_probability(
        self,
        policy_chosen_logps: float,
        policy_rejected_logps: float,
        reference_chosen_logps: Optional[float] = None,
        reference_rejected_logps: Optional[float] = None
    ) -> float:
        """
        计算偏好概率 P(y_w > y_l | x)

        根据Bradley-Terry模型，偏好概率为：
        P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

        在DPO中，奖励定义为：
        r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))

        Args:
            policy_chosen_logps: log π_θ(y_w|x)
            policy_rejected_logps: log π_θ(y_l|x)
            reference_chosen_logps: log π_ref(y_w|x)
            reference_rejected_logps: log π_ref(y_l|x)

        Returns:
            preference_prob: 偏好概率，范围 [0, 1]
        """
        if self.reference_free:
            logit = self.beta * (policy_chosen_logps - policy_rejected_logps)
        else:
            pi_ratio = policy_chosen_logps - policy_rejected_logps
            ref_ratio = reference_chosen_logps - reference_rejected_logps
            logit = self.beta * (pi_ratio - ref_ratio)

        return sigmoid(logit)


class DPODataset:
    """
    DPO数据集

    管理偏好对数据：(prompt, chosen_response, rejected_response)
    """

    def __init__(self):
        self.data = []

    def add_preference(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        prompt_tokens: Optional[List[int]] = None,
        chosen_tokens: Optional[List[int]] = None,
        rejected_tokens: Optional[List[int]] = None
    ):
        """
        添加一个偏好对

        Args:
            prompt: 输入提示
            chosen: 偏好的响应
            rejected: 不偏好的响应
            prompt_tokens: 提示的token序列（可选）
            chosen_tokens: 偏好响应的token序列（可选）
            rejected_tokens: 拒绝响应的token序列（可选）
        """
        self.data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'prompt_tokens': prompt_tokens,
            'chosen_tokens': chosen_tokens,
            'rejected_tokens': rejected_tokens,
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, batch_size: int, shuffle: bool = True):
        """获取一个批次的数据"""
        indices = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(self.data), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield [self.data[idx] for idx in batch_indices]


def compare_dpo_with_rlhf():
    """
    对比DPO与传统RLHF的差异

    Returns:
        comparison: 对比结果字典
    """
    comparison = {
        'RLHF (PPO)': {
            'steps': [
                '1. 训练奖励模型 r(x, y)',
                '2. 使用PPO优化策略 π_θ',
                '3. 最大化 E[r(x, y)] - β·KL(π_θ || π_ref)'
            ],
            'pros': [
                '理论基础成熟',
                '可以处理复杂的奖励信号',
                '灵活（奖励模型可复用）'
            ],
            'cons': [
                '需要训练奖励模型（计算开销大）',
                'RL训练不稳定',
                '超参数调优困难',
                '需要采样大量响应'
            ],
            'complexity': 'High'
        },
        'DPO': {
            'steps': [
                '1. 直接从偏好数据优化策略',
                '2. 最大化 log σ(β·log(π/π_ref)_chosen - β·log(π/π_ref)_rejected)'
            ],
            'pros': [
                '无需奖励模型（简化pipeline）',
                '稳定（监督学习）',
                '计算高效',
                '超参数少'
            ],
            'cons': [
                '只能使用偏好对',
                '可能不如RLHF灵活',
                '理论性质还在研究中'
            ],
            'complexity': 'Low'
        }
    }

    return comparison


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("DPO (Direct Preference Optimization) 演示")
    print("=" * 70)

    # ========== 1. 创建DPO训练器 ==========
    print("\n" + "=" * 70)
    print("1. DPO训练器初始化")
    print("=" * 70)

    beta = 0.1
    dpo_trainer = DPOTrainer(beta=beta)

    print(f"\nDPO配置:")
    print(f"  β (KL惩罚系数): {beta}")
    print(f"  作用: 控制策略偏离参考模型的程度")
    print(f"    - β越大: 策略越保守，更接近参考策略")
    print(f"    - β越小: 策略越激进，可能偏离参考策略")

    # ========== 2. 创建偏好数据集 ==========
    print("\n" + "=" * 70)
    print("2. 偏好数据集示例")
    print("=" * 70)

    dataset = DPODataset()

    # 添加示例偏好对
    examples = [
        {
            'prompt': '解释什么是机器学习',
            'chosen': '机器学习是一种人工智能技术，让计算机从数据中学习规律，无需显式编程。它包括监督学习、无监督学习和强化学习等方法。',
            'rejected': '机器学习就是让电脑自己学东西。'
        },
        {
            'prompt': '如何保持健康？',
            'chosen': '保持健康需要：1) 均衡饮食，2) 规律运动，3) 充足睡眠，4) 心理健康，5) 定期体检。建议每周运动150分钟以上。',
            'rejected': '多吃点就行了。'
        },
        {
            'prompt': '什么是Python？',
            'chosen': 'Python是一种高级编程语言，以简洁易读著称。它广泛应用于数据科学、Web开发、自动化等领域，拥有丰富的第三方库生态。',
            'rejected': 'Python是一种蛇。'
        }
    ]

    for ex in examples:
        dataset.add_preference(ex['prompt'], ex['chosen'], ex['rejected'])

    print(f"\n数据集大小: {len(dataset)}")
    print(f"\n示例偏好对:")
    for i, item in enumerate(dataset.data[:2]):
        print(f"\n例子 {i+1}:")
        print(f"  提示: {item['prompt']}")
        print(f"  偏好响应: {item['chosen'][:50]}...")
        print(f"  拒绝响应: {item['rejected']}")

    # ========== 3. DPO损失计算 ==========
    print("\n" + "=" * 70)
    print("3. DPO损失计算")
    print("=" * 70)

    # 模拟一个批次的对数概率
    batch_size = 4

    # 策略模型的对数概率
    policy_chosen_logps = np.array([-2.5, -3.0, -2.8, -2.6])
    policy_rejected_logps = np.array([-4.5, -5.0, -4.8, -4.6])

    # 参考模型的对数概率
    reference_chosen_logps = np.array([-3.0, -3.5, -3.2, -3.1])
    reference_rejected_logps = np.array([-4.0, -4.5, -4.3, -4.2])

    print(f"\n批次大小: {batch_size}")
    print(f"\n策略模型:")
    print(f"  log π_θ(y_chosen|x): {policy_chosen_logps}")
    print(f"  log π_θ(y_rejected|x): {policy_rejected_logps}")

    print(f"\n参考模型:")
    print(f"  log π_ref(y_chosen|x): {reference_chosen_logps}")
    print(f"  log π_ref(y_rejected|x): {reference_rejected_logps}")

    # 计算损失
    loss, metrics = dpo_trainer.dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps
    )

    print(f"\nDPO损失: {loss:.4f}")
    print(f"\n详细指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # ========== 4. 偏好概率计算 ==========
    print("\n" + "=" * 70)
    print("4. 偏好概率分析")
    print("=" * 70)

    print("\n根据Bradley-Terry模型:")
    print("P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))")
    print("其中 r(x, y) = β · log(π_θ(y|x) / π_ref(y|x))")

    for i in range(min(3, batch_size)):
        pref_prob = dpo_trainer.compute_preference_probability(
            policy_chosen_logps[i],
            policy_rejected_logps[i],
            reference_chosen_logps[i],
            reference_rejected_logps[i]
        )
        print(f"\n样本 {i+1}:")
        print(f"  偏好概率 P(chosen > rejected): {pref_prob:.4f}")
        print(f"  解释: 模型认为chosen比rejected好的概率是 {pref_prob*100:.1f}%")

    # ========== 5. 不同beta值的影响 ==========
    print("\n" + "=" * 70)
    print("5. 不同β值的影响")
    print("=" * 70)

    betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    print(f"\n{'β':<8} {'损失':<12} {'奖励差':<12} {'准确率':<12}")
    print("-" * 50)

    for beta_val in betas:
        trainer = DPOTrainer(beta=beta_val)
        loss_val, metrics_val = trainer.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        print(f"{beta_val:<8.2f} {loss_val:<12.4f} {metrics_val['reward_margin']:<12.4f} {metrics_val['accuracy']:<12.2f}")

    print("\n观察:")
    print("  • β越大，损失越小（更保守的更新）")
    print("  • β控制了KL散度的权重")
    print("  • 典型值：β ∈ [0.1, 0.5]")

    # ========== 6. DPO vs RLHF对比 ==========
    print("\n" + "=" * 70)
    print("6. DPO vs 传统RLHF对比")
    print("=" * 70)

    comparison = compare_dpo_with_rlhf()

    for method, details in comparison.items():
        print(f"\n{method}:")
        print(f"  复杂度: {details['complexity']}")

        print(f"\n  训练步骤:")
        for step in details['steps']:
            print(f"    {step}")

        print(f"\n  优势:")
        for pro in details['pros']:
            print(f"    ✓ {pro}")

        print(f"\n  劣势:")
        for con in details['cons']:
            print(f"    ✗ {con}")

    # ========== 7. Reference-Free DPO ==========
    print("\n" + "=" * 70)
    print("7. Reference-Free DPO变体")
    print("=" * 70)

    dpo_ref_free = DPOTrainer(beta=0.1, reference_free=True)

    print("\nReference-Free DPO:")
    print("  • 不需要参考模型π_ref")
    print("  • 损失: -log σ(β · (log π_θ(y_w|x) - log π_θ(y_l|x)))")
    print("  • 适用场景: 没有合适的参考模型时")

    loss_ref_free, metrics_ref_free = dpo_ref_free.dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps
    )

    print(f"\n标准DPO损失: {loss:.4f}")
    print(f"Reference-Free DPO损失: {loss_ref_free:.4f}")

    # ========== 8. 实际应用场景 ==========
    print("\n" + "=" * 70)
    print("8. DPO的实际应用")
    print("=" * 70)

    print("\n成功案例:")
    print("  • Zephyr-7B: 使用DPO对齐Mistral-7B")
    print("  • Llama 2: Meta使用RLHF (PPO)，但DPO可达到类似效果")
    print("  • Tulu 2: 结合SFT和DPO的指令微调")
    print("  • OpenHermes: 使用DPO进行偏好对齐")

    print("\n数据需求:")
    print("  • 偏好对数据：(prompt, chosen, rejected)")
    print("  • 数据来源：")
    print("    - 人工标注")
    print("    - 从奖励模型排序")
    print("    - 从AI反馈 (RLAIF)")
    print("    - 从现有数据集（如HH-RLHF、Anthropic-HH）")

    print("\n典型训练配置:")
    print("  • β: 0.1 ~ 0.5")
    print("  • 学习率: 5e-7 ~ 1e-6（比SFT小）")
    print("  • 训练轮数: 1-3 epoch")
    print("  • 批次大小: 32-128")

    # ========== 9. 优化技巧 ==========
    print("\n" + "=" * 70)
    print("9. DPO训练技巧")
    print("=" * 70)

    print("\n数据质量:")
    print("  ✓ 确保偏好对有明确区分度")
    print("  ✓ 过滤模糊的偏好对")
    print("  ✓ 平衡不同类型的提示")

    print("\n训练稳定性:")
    print("  ✓ 从SFT模型初始化")
    print("  ✓ 使用较小的学习率")
    print("  ✓ 监控奖励差距和准确率")
    print("  ✓ 避免过拟合（early stopping）")

    print("\n超参数选择:")
    print("  ✓ β: 从0.1开始，根据任务调整")
    print("    - 安全性敏感任务: 较大β（更保守）")
    print("    - 创造性任务: 较小β（更灵活）")

    # ========== 10. DPO的理论洞察 ==========
    print("\n" + "=" * 70)
    print("10. DPO的理论洞察")
    print("=" * 70)

    print("\n关键发现:")
    print("  1. 奖励模型可以参数化为策略")
    print("     r(x, y) = β · log π_θ(y|x) / π_ref(y|x) + β · log Z(x)")
    print("  ")
    print("  2. Bradley-Terry偏好模型")
    print("     P(y_w > y_l | x) = exp(r(w)) / (exp(r(w)) + exp(r(l)))")
    print("  ")
    print("  3. DPO直接优化这个概率")
    print("     无需显式的奖励模型")

    print("\n数学等价性:")
    print("  • DPO的最优解与RLHF相同")
    print("  • 但训练过程更简单稳定")

    print("\n" + "=" * 70)
    print("DPO的关键特性总结")
    print("=" * 70)
    print("✓ 无需奖励模型，简化RLHF pipeline")
    print("✓ 稳定的监督学习，避免RL不稳定性")
    print("✓ 计算高效，训练速度快")
    print("✓ 效果可媲美PPO，甚至更好")
    print("✓ 广泛应用于开源LLM对齐（Zephyr、Tulu等）")
    print("✗ 需要高质量的偏好对数据")
    print("✗ 不如RLHF灵活（只能用偏好对）")
