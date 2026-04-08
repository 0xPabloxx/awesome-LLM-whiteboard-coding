# Roadmap

基于 2026 年主流开源模型架构（Llama 3/Qwen3/DeepSeek-V3/Mixtral）整理。

## P0 — Attention

必须手写通过测试，面试核心考点。

| # | 算法 | 说明 | 状态 |
|---|------|------|------|
| 1 | Scaled Dot-Product Attention | 基础中的基础 | |
| 2 | Multi-Head Attention (MHA) | 标准实现 | |
| 3 | Masked / Causal Attention | decoder-only 的核心 | |
| 4 | Multi-Query Attention (MQA) | 共享 KV head | |
| 5 | Grouped-Query Attention (GQA) | MHA 和 MQA 的折中，Llama 3 / Qwen3 在用 | |
| 6 | Multi-head Latent Attention (MLA) | DeepSeek-V2/V3 的杀手锏，低秩压缩 KV | |
| 7 | Flash Attention | 理解 tiling 思路即可，不用手写 CUDA kernel | |

## P0 — Transformer

必须手写通过测试，现代 LLM 的完整 building blocks。

| # | 算法 | 说明 | 状态 |
|---|------|------|------|
| 1 | Vanilla Transformer Block | LayerNorm + Attention + FFN + Residual | |
| 2 | Pre-Norm vs Post-Norm | 现代模型都用 Pre-Norm (RMSNorm) | |
| 3 | RoPE | 旋转位置编码，几乎所有开源模型标配 | |
| 4 | SwiGLU FFN | 替代传统 ReLU FFN，Llama/Qwen/DeepSeek 都在用 | |
| 5 | KV Cache | 推理时必须理解的机制 | |
| 6 | Mixture of Experts (MoE) | Mixtral / Qwen3-235B / DeepSeek-V3 的路由机制 | |

## P1 — 值得了解

优先级低一档，能聊清楚很加分。

| # | 算法 | 说明 | 状态 |
|---|------|------|------|
| 1 | Linear Attention / Gated DeltaNet | Qwen3.5 hybrid 架构验证了这个方向 | |
| 2 | Mamba / SSM | Nemotron 3 Nano 用 Mamba-2 做主要序列建模 | |
| 3 | Multi-Token Prediction (MTP) | DeepSeek-V3 / Nemotron 在用 | |

## P2 — 训练与优化

| # | 算法 | 说明 | 状态 |
|---|------|------|------|
| 1 | LoRA | 低秩适应，参数高效微调 | |
| 2 | Muon Optimizer | Momentum + Newton-Schulz 正交化 | |
| 3 | PPO | RLHF 主流训练算法 | |
| 4 | DPO | 无需 reward model 的对齐方法 | |
| 5 | GRPO | DeepSeek 提出的组相对策略优化 | |
