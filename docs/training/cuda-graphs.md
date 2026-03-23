# CUDA 图

CUDA 图可以一次性捕获一系列 GPU 操作，并以最小的主机开销重放它们，从而消除每个训练步骤中重复的内核启动和驱动程序开销。Megatron Bridge 支持两种捕获实现和细粒度的范围选择，以平衡性能提升与内存成本。

本页是关于 CUDA 图是什么、何时使用它们以及哪些限制是持久性的稳定概述。关于操作设置、代码锚点和验证命令，请参阅 [skills/perf-techniques/cuda-graphs/SKILL.md](../skills/perf-techniques/cuda-graphs/SKILL.md)。

## 它是什么

CUDA 图通过在捕获阶段将一系列 GPU 操作（内核、内存拷贝等）记录到一个图中，然后在后续步骤中重放该图来工作。这消除了每个步骤的主机端开销，例如内核启动延迟和驱动程序 API 调用。

在 Bridge 中，有两种捕获实现：

| `cuda_graph_impl` | 机制 | 支持的范围 |
|---|---|---|
| `"local"` | MCore `CudaGraphManager` / `FullCudaGraphWrapper` | `full_iteration` (整个前向+反向) |
| `"transformer_engine"` | TE `make_graphed_callables()` 每层 | `attn`, `mlp`, `moe`, `moe_router`, `moe_preprocess`, `mamba` |
| `"none"` (默认) | 禁用 | — |

## 何时使用

CUDA 图在以下情况下最有效：

- **张量形状在训练步骤间是静态的**（固定序列长度，固定微批次大小）。可变长度序列会破坏图重放的假设。
- **主机开销相对于 GPU 计算而言是显著的** —— 较小的模型或高步进率受益最大。
- **内存预算允许** —— 图捕获会分配静态缓冲区，通常会增加几 GB 内存。使用 `PP > 1` 的模型可能会消耗超过 10 GB 的额外内存。

### 本地全迭代图

将整个前向-反向传播过程捕获为一个图。提供最高的主机开销减少，但需要禁用 NaN 检查，并且内存占用最大。

### Transformer Engine 范围图

通过 TE 捕获各个层组件（注意力、MLP、MoE 路由器等）。更灵活，适用于只有密集模块可以被图化的 MoE 模型，并支持选择性范围组合。

## 配置

```python
cfg.model.cuda_graph_impl = "transformer_engine"        # 或 "local"
cfg.model.cuda_graph_scope = ["attn", "moe_router"]     # 范围列表
cfg.model.cuda_graph_warmup_steps = 3                   # 捕获前的预热步骤
cfg.rng.te_rng_tracker = True                           # 必需
```

### 关键限制

- 当 `cuda_graph_impl != "none"` 时，`cfg.rng.te_rng_tracker` 必须为 `True`。
- `full_iteration` 范围要求 `cuda_graph_impl = "local"` 并且 `rerun_state_machine.check_for_nan_in_loss = False`。
- 使用无令牌丢弃路由的 MoE 模型对图的支持有限（仅密集模块）。
- `cuda_graph_impl = "none"` 会自动清除 `cuda_graph_scope`。

## MoE 注意事项

由于动态令牌路由，MoE 模型通常无法对完整的专家调度路径进行图化。常见做法是：

- 对 `moe_router` 和 `moe_preprocess`（静态部分）进行图化。
- 为密集注意力块添加 `attn` 范围。
- 让专家调度保持为即时执行模式。

不要将 `moe` 范围与 `moe_router` 范围结合使用 —— 它们是互斥的。

## 内存影响

CUDA 图会分配在训练期间持续存在的静态缓冲区。预计会增加几 GB 的内存。当 `PP > 1` 时，由于流水线阶段缓冲，内存开销可能超过 10 GB。请相应地规划激活内存。

## 相关文档

- [docs/performance-guide.md](../performance-guide.md)
- [docs/training/communication-overlap.md](communication-overlap.md)
- [skills/perf-techniques/cuda-graphs/SKILL.md](../skills/perf-techniques/cuda-graphs/SKILL.md)