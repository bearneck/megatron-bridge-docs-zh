# 检查点

{py:class}`bridge.training.config.CheckpointConfig` 控制模型检查点的行为，包括保存和加载检查点、检查点格式以及各种优化功能。

```{note}
本文档涵盖训练期间使用的 **Megatron 格式检查点**。关于 🤗 Hugging Face 和 Megatron 格式之间的转换，请参阅 {doc}`../bridge-guide`。
```

## 概述

Megatron Bridge 使用 Megatron Core 的分布式检查点系统，该系统专为跨多个 GPU 和节点的大规模训练而设计。分布式检查点方法通过将检查点数据分片到多个文件中来保存分布式训练作业的状态，从而减少内存开销并提高保存/加载操作期间的 GPU 利用率。

### 分布式检查点的优势

**内存效率**：分布式检查点不是将所有模型参数和优化器状态收集到单个 rank 上，而是直接从每个 rank 保存数据，从而显著减少了检查点期间的内存需求。

**并行灵活性**：该系统提供了使用不同并行策略恢复训练的灵活性。您可以在检查点保存和加载操作之间更改张量并行、管道并行或数据并行的大小。

**可扩展性**：处理所有类型的并行，包括：
- **数据并行（DP）**：在不同数据批次上跨多个 GPU 复制模型
- **张量并行（TP）**：将单个层的参数分布在 GPU 上
- **管道并行（PP）**：将连续层分配给不同的 GPU
- **上下文并行（CP）**：沿序列维度对长序列的张量进行分片
- **专家并行（EP）**：将 MoE 专家权重分布在 GPU 上

**性能**：分布式优化器将优化器状态和主参数分片到数据并行 rank 上，而不是复制它们，从而减少了内存使用和通信开销。

## 保存配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `save` | `Optional[str]` | `None` | 用于保存检查点 **（Megatron 格式）** 的输出目录 |
| `save_interval` | `Optional[int]` | `None` | 持久化检查点保存之间的迭代次数 |
| `save_optim` | `bool` | `True` | 是否保存优化器状态 |
| `save_rng` | `bool` | `True` | 是否保存随机数生成器状态 |
| `save_tokenizer_assets` | `bool` | `True` | 是否将分词器文件（词汇表、配置、特殊标记）保存到检查点 |

### 异步保存

异步保存允许在检查点数据在后台持久化到磁盘的同时继续训练，从而减少检查点对训练吞吐量的影响。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `async_save` | `bool` | `False` | 启用异步检查点保存（需要 `torch_dist` 格式） |

## 加载配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `load` | `Optional[str]` | `None` | 包含要加载的模型检查点 **（Megatron 格式）** 的目录 |
| `load_optim` | `bool` | `True` | 是否从检查点加载优化器状态 |
| `load_rng` | `bool` | `True` | 是否从检查点加载随机数生成器状态 |
| `load_main_params_from_ckpt` | `bool` | `False` | 从检查点加载主参数（与 `load_optim=False` 一起使用） |
| `ckpt_step` | `Optional[int]` | `None` | 要加载的特定检查点迭代（覆盖跟踪器中的最新检查点） |
| `exit_on_missing_checkpoint` | `bool` | `False` | 如果未找到指定的检查点则退出，而不是进行随机初始化 |
| `dist_ckpt_strictness` | `Literal[...]` | `"assume_ok_unexpected"` | 分布式检查点加载期间对键不匹配的处理方式 |

### 加载特定检查点迭代

默认情况下，Megatron Bridge 通过读取跟踪器文件（`latest_train_state.pt`）来加载指定目录中**可用的最新检查点**。但是，您可以使用 `ckpt_step` 参数显式加载特定的检查点迭代。

**Python API：**
```python
from megatron.bridge.training.config import CheckpointConfig

# 加载最新检查点
checkpoint = CheckpointConfig(
    load="/path/to/checkpoint_dir"
)

# 加载特定迭代
checkpoint = CheckpointConfig(
    load="/path/to/checkpoint_dir",
    ckpt_step=5000  # 覆盖跟踪器，加载 iter_0005000
)
```

```{note}
`load` 参数应始终指向基础检查点目录（而不是 `iter_N` 子目录）。`ckpt_step` 参数覆盖从该目录加载哪个迭代。
```

**重要提示：** 如果指定了 `ckpt_step` 但检查点目录不存在，训练将**立即失败**并抛出 `FileNotFoundError`。这是有意为之，以防止您本意是从特定检查点恢复训练时，却意外地从头开始训练。

**PEFT 注意事项：** `ckpt_step` 参数**仅适用于 `load` 路径**（适配器检查点），不适用于 `pretrained_checkpoint`（冻结的基础模型）。恢复 PEFT 训练时：
- `pretrained_checkpoint`：始终加载最新/发布的检查点（基础模型）
- `load` + `ckpt_step`：可以加载特定的适配器检查点迭代步数

### 检查点加载严格性

加载分布式检查点时，保存的检查点中的键可能与当前模型期望的键不匹配。当使用不同的并行设置、模型配置或软件版本恢复训练时，可能会发生这种情况。`dist_ckpt_strictness` 参数控制如何处理这些不匹配：

- **`assume_ok_unexpected`**：假设意外的键是可接受的（默认，最宽松）
- **`log_unexpected`**：记录意外的键但继续加载
- **`log_all`**：记录所有键不匹配以进行调试
- **`raise_unexpected`**：遇到意外的键时引发错误（更严格的验证）
- **`raise_all`**：遇到任何键不匹配时引发错误（最严格的验证）
- **`return_unexpected`**：返回有关意外键的信息
- **`return_all`**：返回有关所有键不匹配的信息
- **`ignore_all`**：完全忽略所有键不匹配

## 微调配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `pretrained_checkpoint` | `Optional[str]` | `None` | 包含用于微调的**Megatron 格式**预训练模型检查点的目录 |

## 检查点格式

Megatron Bridge 支持针对不同用例优化的多种检查点格式：

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `ckpt_format` | `Literal["torch_dist", "zarr", "fsdp_dtensor"]` | `"torch_dist"` | 要使用的检查点格式 |

### 可用格式

**`torch_dist`** (默认)
- PyTorch 分布式检查点格式
- 兼容大多数并行策略（DP, TP, PP, CP, EP）
- 当 `async_save=True` 时支持异步保存
- 推荐用于一般用途

**`zarr`**
- 基于 Zarr 的检查点格式
- 在某些用例中可作为 `torch_dist` 的替代方案
- 兼容分布式并行策略

**`fsdp_dtensor`**
- 专用于 Megatron FSDP（全分片数据并行）的格式
- **当使用 `use_megatron_fsdp=True` 时必须使用**
- 针对分片参数布局进行了优化
- 与其他 FSDP 实现不兼容

### 格式选择

根据您的训练配置选择检查点格式：

```python
from megatron.bridge.training.config import CheckpointConfig

# 标准分布式训练 (DDP, TP, PP)
checkpoint = CheckpointConfig(
    ckpt_format="torch_dist",  # 默认值，适用于大多数情况
    save="/path/to/checkpoints",
)

# Megatron FSDP 训练
checkpoint = CheckpointConfig(
    ckpt_format="fsdp_dtensor",  # FSDP 必需
    save="/path/to/checkpoints",
)
```

### 格式兼容性

| 格式 | DDP | 分布式优化器 | Megatron FSDP | Torch FSDP2 | 异步保存 |
|--------|-----|----------------------|---------------|-------------|------------|
| `torch_dist` | ✅ | ✅ | ❌ | ✅ | ✅ |
| `zarr` | ✅ | ✅ | ❌ | ✅ | ❌ |
| `fsdp_dtensor` | ❌ | ❌ | ✅ | ❌ | ❌ |

**重要提示**：当使用 Megatron FSDP (`use_megatron_fsdp=True`) 时，必须设置 `ckpt_format="fsdp_dtensor"`。其他格式与 FSDP 的分片参数布局不兼容。有关完整的 FSDP 配置详情，请参阅 {doc}`megatron-fsdp`。

## 性能优化

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `fully_parallel_save` | `bool` | `True` | 跨数据并行秩应用完全保存并行化 |
| `fully_parallel_load` | `bool` | `False` | 跨数据并行秩应用完全加载并行化 |
| `ckpt_assume_constant_structure` | `bool` | `False` | 为性能优化，假设连续检查点保存之间的模型/优化器结构是恒定的 |

## 检查点内容

当使用 `torch_dist` 检查点格式时，检查点包含以下组件：
- **模型参数和优化器状态**：存储在 `.distcp` 文件中以支持分布式训练。
- **训练状态**：捕获当前迭代次数、已消耗样本数以及学习率调度器的状态。
- **配置**：序列化为 YAML 文件 (`run_config.yaml`)，包含完整的 `ConfigContainer`。
- **分词器文件**：所有分词器工件（词汇表、特殊标记、配置），用于自包含的检查点。
- **数据加载器状态**：确保数据迭代的确定性恢复。

- **元数据**：用于验证和正确加载检查点。

Megatron Bridge 创建具有以下目录结构的检查点：

```
checkpoint_dir/
├── latest_train_state.pt                      # 最新的训练状态（顶层）
├── iter_N/                                    # 第 N 次迭代的检查点
│   ├── __0_0.distcp                          # 分布式检查点分片：映射到 PyTorch DCP 权重格式
│   ├── __0_1.distcp                          # 包含模型参数、优化器状态
│   ├── __1_0.distcp
│   ├── __1_1.distcp
│   ├── ...
│   ├── .metadata                             # PyTorch DCP 检查点元数据
│   ├── common.pt                             # 由 rank 0 保存的 MCore 分布式检查点状态
│   ├── metadata.json                         # MCore 分布式检查点元数据
│   ├── run_config.yaml                       # 序列化的 ConfigContainer
│   ├── train_state.pt                        # 步数、已消耗样本数等
│   ├── tokenizer/                            # 分词器文件（默认保存）
│   │   ├── tokenizer.json                   # 完整的分词器词汇表
│   │   ├── tokenizer_config.json            # 分词器配置
│   │   ├── special_tokens_map.json          # 特殊令牌定义
│   │   └── ...                              # 其他分词器文件
│   ├── dataloader_state/                     # 数据迭代器状态
│   │   ├── train_dataloader_dprank000.pt    # DP rank 0 数据加载器状态
│   │   ├── train_dataloader_dprank001.pt    # DP rank 1 数据加载器状态
│   │   ├── train_dataloader_dprank002.pt    # DP rank 2 数据加载器状态
│   │   └── ...                              # 每个 DP rank 一个文件
```

### 分词器资源

默认情况下，Megatron Bridge 会将所有分词器文件保存到检查点目录中，使检查点自包含且可移植。这对于以下情况尤为重要：
- **推理和评估**：直接访问分词器以计算对数概率
- **可移植性**：不依赖于原始分词器文件位置
- **可复现性**：精确的分词器状态得以保留

保存的分词器文件取决于分词器类型：
- **HuggingFace 分词器**：`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json` 以及词汇文件
- **SentencePiece 分词器**：`tokenizer.model` 文件
- **GPT2 BPE 分词器**：`vocab.json` 和 `merges.txt`
- **BERT 分词器**：`vocab.txt`
- **Tiktoken 分词器**：`tokenizer.json`

要在性能敏感的场景中禁用分词器资源保存：

```python
from megatron.bridge.training.config import CheckpointConfig

checkpoint = CheckpointConfig(
    save_tokenizer_assets=False,  # 跳过分词器文件保存
    ...
)
```

或在 YAML 中配置：

```yaml
checkpoint:
  save_tokenizer_assets: false
```

## 本地检查点

本地检查点将模型检查点直接保存到每个节点的本地存储（例如，本地 SSD 或 RAM 磁盘），而不是仅仅依赖共享的网络文件系统。这种方法可以显著加快保存过程，并减少共享存储基础设施的负载。

本地检查点利用了 [NVIDIA Resiliency Extension](https://nvidia.github.io/nvidia-resiliency-ext/checkpointing/local/index.html) 并提供几个关键特性：

- **本地保存**：每个节点将其部分检查点保存在本地，减少网络 I/O 并提高保存性能。
- **同步和异步支持**：保存可以同步或异步进行，镜像全局检查点使用的配置。
- **自动清理**：自动处理过时或不完整的本地检查点的移除。
- **可选复制**：对于多节点作业，检查点会被复制到其他节点，以便在节点保存后发生故障时仍能恢复。单节点作业不使用复制。
- **自动加载**：恢复时，框架会自动找到最新的有效检查点，比较本地和全局检查点，并在节点间检索所需的任何部分。

### 非持久化检查点配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `non_persistent_save_interval` | `Optional[int]` | `None` | 非持久化保存之间的迭代间隔 |
| `non_persistent_ckpt_type` | `Optional[Literal["global", "local", "in_memory", "None"]]` | `None` | 非持久化检查点类型 |
| `non_persistent_global_ckpt_dir` | `Optional[str]` | `None` | 全局非持久化检查点目录 |
| `non_persistent_local_ckpt_dir` | `Optional[str]` | `None` | 本地非持久化检查点目录 |
| `non_persistent_local_ckpt_algo` | `Literal["fully_parallel", "atomic"]` | `"fully_parallel"` | 本地非持久化检查点算法 |

### 复制与容错

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|

| `replication` | `bool` | `False` | 启用本地检查点在各个 rank 间的复制 |
| `replication_jump` | `Optional[int]` | `None` | 存储副本的 rank 之间的间隔 |
| `replication_factor` | `int` | `2` | 存储每个 rank 数据副本的机器数量 |

### 分布式优化器检查点

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `dist_ckpt_optim_fully_reshardable` | `bool` | `False` | 使优化器分布式检查点完全可重分片（TP/PP/EP/DP），而不是普通的 DP 可重分片性 |
| `distrib_optim_fully_reshardable_mem_efficient` | `bool` | `False` | 在保存和加载期间通过使用 Gloo 来尽可能少地使用内存。仅在设置了 `dist_ckpt_optim_fully_reshardable` 标志时生效 |

## 相关文档

- {doc}`megatron-fsdp` - Megatron FSDP 配置和 `fsdp_dtensor` 格式要求
- {doc}`../parallelisms` - 理解数据和模型并行策略
- {doc}`config-container-overview` - 完整的配置参考