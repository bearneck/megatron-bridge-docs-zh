# 通信重叠

通信重叠通过将集合通信或点对点传输与有效计算重叠，减少分布式训练中暴露的通信开销。Megatron Bridge 支持跨多个并行维度的重叠，但不同模式下的可用行为并不完全相同。

本文档是关于通信重叠是什么、何时使用以及哪些约束是持久性的稳定概述。关于操作设置、代码锚点和验证命令，请参阅：

- [skills/perf-techniques/tp-dp-comm-overlap/SKILL.md](../skills/perf-techniques/tp-dp-comm-overlap/SKILL.md)
- [skills/perf-techniques/expert-parallel-overlap/SKILL.md](../skills/perf-techniques/expert-parallel-overlap/SKILL.md)

## 概述

在 Bridge 中，通信重叠涵盖了几个相关的子功能：

- 数据并行（Data Parallel）重叠：用于梯度归约分散（reduce-scatter）和参数全收集（all-gather）
- 张量并行（Tensor Parallel）重叠：用于 GEMM 工作下的 TP 通信
- 管道并行（Pipeline Parallel）重叠：用于 PP 发送和接收行为
- 上下文并行（Context Parallel）重叠：内置于上下文并行执行路径中
- MoE 专家并行（MoE Expert Parallel）重叠：用于专家令牌分发通信

这些是相关的性能优化技术，但它们不共享相同的启用开关、默认值或操作风险。

## 何时使用

通信重叠适用于以下情况：

- 模型已经需要 TP、DP、PP、CP 或 EP 来实现扩展
- 通信在单步训练时间中占显著部分
- 正确性已经确立，并且您正在针对吞吐量进行调优

在以下情况下不太适用：

- 您仍在搭建新的训练路径，希望保持最少的可变部分
- 功能组合对分支敏感或验证不足
- 启动时的环境调优可能与其他技术冲突

## 各模式的稳定指导

### 数据并行

DP 重叠与分布式优化器路径绑定。它是分片优化器状态训练的自然重叠机制，应与分布式优化器行为一起考虑，而不是作为一个独立的开关。

### 张量并行

TP 重叠在概念上与序列并行（Sequence Parallelism）绑定。如果序列并行不可用或未启用，不应假定 TP 重叠会保持激活状态。

### 管道并行

PP 重叠并非所有管道并行训练的通用属性。在实践中，交错式（interleaved）流水线调度是最重要的正面案例。

### 上下文并行

CP 重叠是 Bridge 上下文并行执行模型的一部分，而不是一个独立的独立技术页面。关于分层或 `a2a+p2p` CP 的指导，请参阅 `docs/training/hybrid-context-parallel.md`。

### MoE 专家并行

MoE 专家并行重叠通过将令牌分发/组合的全对全（all-to-all）通信与专家 FFN 计算重叠，隐藏其开销。可选地，延迟的专家权重梯度计算（`moe_delay_wgrad_compute`）提供了额外的重叠。

MoE 重叠应与通用的 TP、DP 和 PP 重叠分开对待。其约束取决于分发器选择（`alltoall` 或 `flex`）、专家并行度、精度（BF16/FP16）和运行时支持。当使用管道并行时，需要虚拟管道并行（virtual pipeline parallelism）才能使重叠调度正确交错。

## 稳定的约束和注意事项

最持久的注意事项是：

1. 并非所有重叠模式在所有情况下都会自动启用。
2. 一些与重叠相关的精度设置由混合精度配置控制，而不是仅由独立的重叠调优控制。
3. 启动时的环境设置在实践中是技术的一部分，特别是对于 TP、CP 和 MoE 重叠路径。
4. 配方默认值通常是保守的；功能的存在并不意味着每个公开的配方都启用了相应的重叠路径。

## 推荐级别

将通信重叠视为在可工作的分布式配置之上的调优层，而不是在基本正确性尚不确定时首先使用的开关。

对于大多数团队，正确的顺序是：

1. 建立一个正确的分布式配置
2. 选择必要的并行策略
3. 针对特定的通信瓶颈启用或调优重叠

## 相关文档

- [docs/performance-guide.md](../performance-guide.md)
- [docs/training/hybrid-context-parallel.md](hybrid-context-parallel.md)
- [skills/perf-techniques/tp-dp-comm-overlap/SKILL.md](../skills/perf-techniques/tp-dp-comm-overlap/SKILL.md)
- [skills/perf-techniques/expert-parallel-overlap/SKILL.md](../skills/perf-techniques/expert-parallel-overlap/SKILL.md)
- [skills/perf-techniques/moe-comm-overlap/SKILL.md](../skills/perf-techniques/moe-comm-overlap/SKILL.md)
- [skills/perf-techniques/moe-comm-overlap/card.yaml](../skills/perf-techniques/moe-comm-overlap/card.yaml)