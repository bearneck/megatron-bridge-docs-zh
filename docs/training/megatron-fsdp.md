# Megatron FSDP

Megatron FSDP 是目前 Megatron Bridge 中实用的全分片数据并行路径。它在数据并行进程间分片参数、梯度和优化器状态，与普通的分布式数据并行（DDP）或分布式优化器路径相比，可以显著减少模型状态内存。

本页是关于 Megatron FSDP 是什么、何时使用它以及需要注意哪些限制的稳定概述。关于操作启用、代码锚点和验证命令，请参阅 [skills/perf-techniques/megatron-fsdp/SKILL.md](../skills/perf-techniques/megatron-fsdp/SKILL.md)。

## 它是什么

Megatron FSDP 是 Megatron-Core 自定义的 FSDP 实现，通过 `use_megatron_fsdp` 在 Bridge 中暴露。

与其他数据并行策略的比较：

| 特性 | DDP | 分布式优化器 | Megatron FSDP |
|---|---|---|---|
| 参数存储 | 复制 | 复制 | 分片 |
| 优化器状态 | 复制 | 分片 | 分片 |
| 梯度通信 | 全归约 | 归约-分散 | 归约-分散 |
| 参数通信 | 无 | 全收集（更新后） | 全收集（按需） |
| 内存效率 | 基线 | 高 | 最高 |
| 通信开销 | 低 | 中等 | 中高 |

实际结果是，当模型状态内存（而非激活内存）是主要瓶颈时，Megatron FSDP 最为有用。

## 何时使用它

当以下所有条件都成立时，Megatron FSDP 是一个很好的选择：

- 模型对于普通 DDP 或分布式优化器来说太大
- 您想要 Bridge 中当前支持的最强大的 FSDP 路径
- 您愿意用更多的通信来换取更低的内存
- 您可以采用所需的 FSDP 检查点格式

在以下情况下，请选择其他路径：

- DDP 已经能轻松满足需求，且简单性最重要
- 分布式优化器在不完全分片的情况下提供了足够的内存缓解
- 您正在评估 PyTorch FSDP2 在此分支上的生产使用

## 稳定要求

Bridge 中的 Megatron FSDP 要求：

- 启用 `use_megatron_fsdp`
- 检查点格式为 `fsdp_dtensor`
- 标准的进程初始化顺序

`fsdp_dtensor` 格式使用 PyTorch DTensor 和 `torch.distributed.checkpoint`（DCP）来存储分片的参数和优化器状态。它**不能**与 `torch_dist` 或 `zarr` 检查点互换——您无法将 `fsdp_dtensor` 检查点加载到非 FSDP 的运行中，反之亦然。

`fsdp_dtensor` 与 5D 并行（TP + PP + DP + CP + EP）兼容。因为 DCP 存储了 DTensor 放置元数据，所以在一个并行布局下保存的检查点可以在不同的布局下加载（例如，在运行之间更改 TP 或 PP 大小）——DCP 会自动处理分片重映射。唯一不支持的组合是 `use_tp_pp_dp_mapping=True`，它使用一种与 FSDP 分片冲突的替代进程初始化顺序。

重要的稳定限制：

- `use_megatron_fsdp` 和 `use_torch_fsdp2` 互斥
- `use_tp_pp_dp_mapping` 不支持与 Megatron FSDP 一起使用
- 旧版检查点格式（如 `torch_dist` 和 `zarr`）对 Megatron FSDP 的保存/加载无效

当启用 Megatron FSDP 时，Bridge 也会自动调整一些设置，包括禁用 `average_in_collective` 以及一些与 FSDP 路径不匹配的缓冲区重用优化。

## 兼容性与注意事项

在配置层面，Megatron FSDP 旨在与以下技术协同工作：

- 张量并行（Tensor Parallelism）
- 管道并行（Pipeline Parallelism）
- 上下文并行（Context Parallelism）
- 专家并行（Expert Parallelism）
- BF16 或 FP16 混合精度

然而，并非所有组合都具有相同程度的仓库内验证或性能证据。应将广泛的兼容性视为首先由代码支持，而不是针对每种组合都经过完全基准测试验证。

两个最实际的注意事项是：

1. 公共配方（recipes）可能会暴露 `use_megatron_fsdp`，但仍默认使用非 FSDP 检查点格式。即使配方的易用性有所滞后，检查点要求也是稳定且强制性的。
2. FSDP 减少的是模型状态内存，而不是激活内存。对于长序列或受激活内存限制的工作负载，可能仍然需要其他技术，如上下文并行、激活重计算或 CPU 卸载。

## Torch FSDP2 状态

Megatron Bridge 也通过 `use_torch_fsdp2` 暴露了 PyTorch FSDP2 路径，但在本分支上，该路径仍应被视为实验性的。

目前的稳定建议是：

- 如果您在 Bridge 中需要一个 FSDP 路径，请使用 Megatron FSDP
- 不要将 FSDP2 视为可与 Megatron FSDP 互换

## 相关文档

- [docs/training/checkpointing.md](checkpointing.md)
- [docs/training/cpu-offloading.md](cpu-offloading.md)
- [docs/performance-guide.md](../performance-guide.md)
- [skills/perf-techniques/megatron-fsdp/SKILL.md](../skills/perf-techniques/megatron-fsdp/SKILL.md)
- [skills/perf-techniques/megatron-fsdp/card.yaml](../skills/perf-techniques/megatron-fsdp/card.yaml)