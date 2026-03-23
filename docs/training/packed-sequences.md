# 打包序列（Packed Sequences）

打包序列是一种微调技术，通过将多个样本连接成一个数据包（pack）来减少填充（padding）浪费，同时为注意力机制保留序列边界。在 Megatron Bridge 中，这主要是一种监督微调（SFT）和参数高效微调（PEFT）的优化技术，而非通用的预训练功能。

本页是关于打包序列是什么、何时使用以及哪些约束是持久性的稳定概述。关于操作设置、代码锚点和验证命令，请参阅 [skills/perf-techniques/sequence-packing/SKILL.md](../skills/perf-techniques/sequence-packing/SKILL.md)。

## 它是什么

微调数据集通常包含长度变化很大的样本。当这些样本按常规方式批处理时，每个批次中的许多标记（token）只是填充。打包序列通过将多个样本构建成更长的数据包，并将边界元数据传递到注意力路径中，来减少这种浪费。

目前，在 Bridge 中，有两条不同的打包路径，以及通过上下文并行（Context Parallelism）实现的长上下文支持：

| 路径 | 使用场景 | 关键配置 |
|---|---|---|
| 离线打包 SFT | 纯文本微调 | `packed_sequence_specs` |
| VLM 批内打包 | VLM 微调 | `pack_sequences_in_batch=True` |
| 长上下文（CP） | 16K-128K+ 序列长度的预训练/微调 | `context_parallel_size > 1` |

这些技术相关，但它们并非同一个开关。离线打包 SFT 和 VLM 批内打包解决填充浪费问题；长上下文训练主要解决更大序列长度下的激活内存和通信权衡问题。

## 何时使用

当以下所有条件都满足时，打包序列是一个很好的选择：

- 你正在进行 SFT、PEFT 或 VLM 微调（所有三种打包路径都支持；参见上表）
- 你的样本长度不一，且填充浪费显著
- 你可以容忍打包训练带来的微批次（micro-batch）约束

在以下情况下，打包序列通常不是正确的答案：

- 你正在进行标准的 Megatron 风格预训练，它已经在采样过程中连接了文档
- 你通常想要进行长上下文训练，此时上下文并行通常是主要技术
- 你的模型系列或配方（recipe）明确选择不支持打包序列

## 稳定约束

Bridge 中打包序列的持久性约束是：

- 打包 SFT 要求 `micro_batch_size == 1`
- 当使用上下文并行时，序列长度必须满足标准的 CP 可除性约束
- 对于启用 CP 的微调，每个标记的损失行为和归约设置很重要
- 支持 CUDA 图的打包元数据需要额外的填充约束

模型系列的支持并非普遍。一些模型系列和配方路径明确选择不支持打包序列或相关的打包模式。

## 与长序列训练的关系

打包序列和长序列训练经常被一起提及，因为它们都影响序列布局和内存行为，但它们解决不同的问题：

- 打包序列主要减少微调数据集中的填充浪费
- 长序列训练主要解决更大序列长度下的激活内存和通信权衡问题

关于长序列训练的指导，请参阅：

- `docs/performance-guide.md`
- `docs/training/hybrid-context-parallel.md`

## 实际注意事项

需要记住的最稳定的注意事项是：

1. 打包序列的支持是特定于配方和模型系列的。
2. 不应假设微调序列打包能与所有其他训练功能协同工作。
3. 打包序列主要通过减少填充浪费来提高效率，而不是替代长上下文并行或内存规划技术。

## 相关文档

- [docs/training/multi-token-prediction.md](multi-token-prediction.md)
- [docs/performance-guide.md](../performance-guide.md)
- [docs/training/hybrid-context-parallel.md](hybrid-context-parallel.md)
- [skills/perf-techniques/sequence-packing/SKILL.md](../skills/perf-techniques/sequence-packing/SKILL.md)
- [skills/perf-techniques/sequence-packing/card.yaml](../skills/perf-techniques/sequence-packing/card.yaml)
- [skills/perf-techniques/packed-sequences-long-context/SKILL.md](../skills/perf-techniques/packed-sequences-long-context/SKILL.md)
- [skills/perf-techniques/packed-sequences-long-context/card.yaml](../skills/perf-techniques/packed-sequences-long-context/card.yaml)