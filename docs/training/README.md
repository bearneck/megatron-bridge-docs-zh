# 训练与定制

本目录包含使用 Megatron Bridge 进行模型训练和定制的全面文档。学习如何配置训练、优化性能以及定制训练工作流。

## 快速导航

### 我想要

**🚀 开始训练**
→ 从 [配置容器概述](config-container-overview.md) 开始，了解训练设置

**⚙️ 配置训练参数**
→ 查看 [训练循环设置](training-loop-settings.md) 和 [优化器与调度器](optimizer-scheduler.md)

**📊 监控和分析训练**
→ 查看 [日志记录](logging.md) 和 [性能分析](profiling.md) 指南

**💾 管理检查点**
→ 阅读 [检查点](checkpointing.md) 以了解如何保存和恢复训练

**⚡ 优化性能**
→ 探索 [性能指南](../performance-guide.md) 和 [性能摘要](../performance-summary.md)

**🔧 定制训练**
→ 查看 [PEFT](peft.md)、[蒸馏](distillation.md)、[入口点](entry-points.md) 和 [回调函数](callbacks.md)

## 核心训练文档

### 配置与设置

| 文档 | 目的 | 何时阅读 |
|----------|---------|--------------|
| **[配置容器概述](config-container-overview.md)** | 所有训练设置的中央配置对象 | 首次设置训练时 |
| **[入口点](entry-points.md)** | 训练入口点与执行流程 | 理解训练如何启动时 |
| **[训练循环设置](training-loop-settings.md)** | 训练循环参数与配置 | 配置批次大小、迭代次数、验证时 |

### 优化与性能

| 文档 | 目的 | 何时阅读 |
|----------|---------|--------------|
| **[优化器与调度器](optimizer-scheduler.md)** | 优化器与学习率调度器配置 | 设置优化时 |
| **[混合精度](mixed-precision.md)** | 用于内存效率的混合精度训练 | 减少内存使用时 |
| **[通信重叠](communication-overlap.md)** | 通信与计算重叠 | 优化分布式训练时 |
| **[混合上下文并行](hybrid-context-parallel.md)** | 分层 `a2a+p2p` 上下文并行指南 | 高级长序列扩展时 |
| **[注意力优化](attention-optimizations.md)** | 优化注意力机制 | 提高训练速度时 |
| **[激活重计算](activation-recomputation.md)** | 梯度检查点策略 | 减少内存占用时 |
| **[CPU 卸载](cpu-offloading.md)** | 卸载到 CPU 进行内存管理 | GPU 内存有限时 |

### 监控与调试

| 文档 | 目的 | 何时阅读 |
|----------|---------|--------------|
| **[日志记录](logging.md)** | 日志配置与 TensorBoard/WandB 集成 | 监控训练进度时 |
| **[性能分析](profiling.md)** | 性能分析与剖析 | 识别瓶颈时 |
| **[弹性恢复](resiliency.md)** | 处理故障与恢复 | 构建健壮的训练流水线时 |

### 高级功能

| 文档 | 目的 | 何时阅读 |
|----------|---------|--------------|
| **[PEFT](peft.md)** | 参数高效微调（LoRA 等） | 资源有限时进行微调 |
| **[打包序列](packed-sequences.md)** | 用于效率的序列打包 | 优化数据加载时 |
| **[Megatron FSDP](megatron-fsdp.md)** | Megatron FSDP 的稳定概述 | 选择 FSDP 路径时 |
| **[蒸馏](distillation.md)** | 知识蒸馏技术 | 在模型间转移知识时 |
| **[检查点](checkpointing.md)** | 检查点保存、加载与恢复 | 管理训练状态时 |
| **[回调函数](callbacks.md)** | 向训练循环注入自定义逻辑 | 自定义日志记录、指标、第三方集成时 |

## 训练工作流

典型的训练工作流包括：

1.  **配置训练** - 使用模型、数据和训练参数设置 `ConfigContainer`
2.  **准备数据** - 配置数据集加载和预处理
3.  **设置优化** - 配置优化器、调度器和混合精度
4.  **启用监控** - 设置日志记录和性能分析
5.  **配置检查点** - 设置检查点保存和恢复
6.  **启动训练** - 使用配置的入口点开始训练
7.  **监控进度** - 通过日志记录和性能分析跟踪指标
8.  **必要时恢复** - 使用检查点从保存的状态恢复

## 相关文档

-   **[主文档索引](../index.md)** - 返回主文档
-   **[性能指南](../performance-guide.md)** - 全面的性能优化指南
-   **[性能摘要](../performance-summary.md)** - 快速性能参考
-   **[配方使用](../recipe-usage.md)** - 使用训练配方
-   **[并行策略](../parallelisms.md)** - 理解分布式训练策略
-   **[Bridge 指南](../bridge-guide.md)** - 使用 Hugging Face 模型

## 常见训练场景

### 🆕 首次训练设置

1. [配置容器概述](config-container-overview.md) - 了解配置系统
2. [入口点](entry-points.md) - 学习如何启动训练
3. [训练循环设置](training-loop-settings.md) - 配置基本训练参数
4. [日志记录](logging.md) - 设置监控

### ⚡ 性能优化

1. [性能指南](../performance-guide.md) - 全面的优化策略
2. [混合精度](mixed-precision.md) - 启用混合精度训练
3. [通信重叠](communication-overlap.md) - 优化分布式训练
4. [激活重计算](activation-recomputation.md) - 减少内存使用
5. [性能剖析](profiling.md) - 识别瓶颈

### 💾 生产训练

1. [检查点](checkpointing.md) - 可靠的检查点管理
2. [弹性恢复](resiliency.md) - 优雅地处理故障
3. [日志记录](logging.md) - 全面的监控
4. [性能剖析](profiling.md) - 性能分析

### 🔧 自定义

1. [PEFT](peft.md) - 参数高效微调
2. [蒸馏](distillation.md) - 知识蒸馏
3. [入口点](entry-points.md) - 自定义训练工作流
4. [回调函数](callbacks.md) - 注入自定义逻辑（第三方集成）

---

**准备开始训练了吗？** 从 [配置容器概述](config-container-overview.md) 开始，或返回 [主文档](../README.md)。