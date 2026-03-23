# 训练循环配置

{py:class}`bridge.training.config.TrainingConfig` 包含了与训练循环边界、退出条件、验证、批次大小和内存管理相关的设置。

## 关键参数

配置这些参数以控制核心训练行为、资源利用率和分布式设置中的监控。

### 批次配置
定义训练期间数据如何在设备上进行分批和分发。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `micro_batch_size` | `Optional[int]` | `None` | 每个模型实例的批次大小（本地批次大小） |
| `global_batch_size` | `Optional[int]` | `None` | 所有设备上的训练批次大小 |
| `rampup_batch_size` | `Optional[list[int]]` | `None` | 批次大小预热：`[起始大小, 增量, 预热样本数]` |
| `decrease_batch_size_if_needed` | `bool` | `False` | 如果需要，自动减小批次大小以实现容错 |

批次大小之间的关系：
- **全局批次大小** = `micro_batch_size` × `data_parallel_size` × `gradient_accumulation_steps`
- 如果未设置 `global_batch_size`，则默认为 `micro_batch_size` × `data_parallel_size`

### 训练时长

使用迭代次数、样本数或基于时间的限制来控制训练何时停止。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `train_iters` | `Optional[int]` | `None` | 训练的总迭代次数 |
| `train_samples` | `Optional[int]` | `None` | 训练的总样本数 |
| `exit_interval` | `Optional[int]` | `None` | 当迭代次数可被此值整除时退出 |
| `exit_duration_in_mins` | `Optional[int]` | `None` | 经过这么多分钟后退出 |

**训练模式选择**

Megatron-Bridge 支持两种指定训练时长的模式：

1.  **基于迭代的训练**：指定 `train_iters` 来控制训练的总迭代次数。
2.  **基于样本的训练**：指定 `train_samples` 来控制训练的总样本数。

**重要约束：**
- 您必须指定 `train_iters` 或 `train_samples` 中的**恰好一个**，不能同时指定。
- 使用 `train_samples` 时，训练迭代次数会自动计算为 `train_samples // global_batch_size`。
- 批次大小预热（`rampup_batch_size`）目前不支持基于样本的训练。
- 您的调度器配置应与您的训练模式匹配（参见[学习率调度](optimizer-scheduler.md#learning-rate-scheduling)）。

### 验证
配置验证频率、时长和仅评估模式。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `eval_iters` | `int` | `100` | 验证/测试评估的迭代次数 |
| `eval_interval` | `Optional[int]` | `1000` | 两次验证运行之间的间隔 |
| `skip_train` | `bool` | `False` | 跳过训练，仅进行评估并退出 |

**注意：** 要控制验证行为：
- 将 `eval_iters` 设置为 `0` 以完全禁用验证（训练期间和训练后）。
- 将 `eval_interval` 设置为 `None` 以跳过训练期间的验证，但在训练完成后仍会运行验证。

### 内存管理
控制 GPU 内存清理和垃圾回收，以防止训练期间出现内存问题。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `empty_unused_memory_level` | `Literal[0, 1, 2]` | `0` | 每次迭代调用 `torch.cuda.empty_cache()` (0=关闭, 1=中等, 2=激进) |
| `manual_gc` | `bool` | `False` | 跨进程同步 Python 垃圾回收以避免掉队者 |
| `manual_gc_interval` | `int` | `0` | 手动垃圾回收的训练步长间隔 (0=禁用) |
| `manual_gc_eval` | `bool` | `True` | 在使用手动垃圾回收时，在评估期间启用垃圾回收 |

### 信号处理和退出条件
为基于信号的中断设置自动检查点保存和干净退出过程。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `exit_signal_handler` | `bool` | `False` | 检测到信号时保存检查点并优雅关闭 |
| `exit_signal` | `int` | `signal.SIGTERM` | 用于优雅关闭处理的信号 |
| `exit_signal_handler_for_dataloader` | `bool` | `False` | 为数据加载器工作进程使用信号处理器 |

### 性能监控
监控分布式进程中训练的一致性和同步性。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `check_weight_hash_across_dp_replicas_interval` | `Optional[int]` | `None` | 检查数据并行副本间的权重哈希一致性 |
| `train_sync_interval` | `Optional[int]` | `None` | CPU-GPU 同步间隔，以防止 CPU 运行超前 |