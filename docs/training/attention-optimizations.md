# 注意力优化

Megatron Bridge 提供了多种注意力优化技术，以提高 Transformer 模型的效率和性能。这些优化包括用于内存效率的 Flash Attention，以及用于计算效率的多查询注意力（MQA）和分组查询注意力（GQA）。

## Flash Attention

### 概述

Flash Attention 是一种旨在提高 GPT 和 BERT 等 Transformer 模型中注意力机制效率的算法。注意力机制在序列长度上具有二次时间和内存复杂度，对于较长的序列会带来显著的运行时和内存挑战。

与标准的非 Flash 算法相比，Flash Attention 应用了两种技术来降低内存需求并提高计算效率：

1.  **分块技术**：根据共享内存大小分解输入，并一次计算一个分块的 softmax。它不是一次性处理整个查询、键和值张量，而是对这些张量进行多次遍历，然后在后续步骤中组合结果。

2.  **重计算技术**：存储 softmax 归一化因子（与序列长度呈线性关系），而不是存储 softmax 结果（与序列长度呈二次关系），并使用这些归一化因子重新计算注意力分数。这减少了需要写入全局内存的数据量，并降低了全局内存和共享内存之间的 I/O 流量。

Flash Attention 将内存占用和计算复杂度从二次降低到线性，极大地扩展了大语言模型中允许的序列长度范围。

### 配置 Flash Attention

在 Megatron Bridge 中，Flash Attention 通过模型配置中的 `attention_backend` 参数进行配置。该框架通过 Transformer Engine 集成支持多种注意力后端：

```python
from megatron.bridge.models import GPTModelProvider
from megatron.core.transformer.enums import AttnBackend

# 使用 Flash Attention 配置模型（默认）
model_config = GPTModelProvider(
    attention_backend=AttnBackend.auto,  # 让 TE 选择最佳后端（默认）
    # ... 其他模型参数
)

# 或者显式指定 Flash Attention
model_config = GPTModelProvider(
    attention_backend=AttnBackend.flash_attn,  # 显式使用 Flash Attention
    # ... 其他模型参数
)
```

### 注意力后端选项

Megatron Bridge 通过 `attention_backend` 配置支持多种注意力后端：

- `AttnBackend.auto`：自动选择最佳可用后端（推荐）
- `AttnBackend.flash_attn`：显式使用 Flash Attention 实现
- `AttnBackend.fused_attn`：使用 cuDNN 融合注意力（当可用时）
- `AttnBackend.local`：使用本地 PyTorch 实现（用于调试）

### 环境变量控制

要进行细粒度控制，您仍然可以使用环境变量来禁用特定的实现：

```bash
# 禁用 Flash Attention
export NVTE_FLASH_ATTN=0

# 禁用 cuDNN Flash Attention
export NVTE_FUSED_ATTN=0
```

但是，推荐的方法是使用 `attention_backend` 配置参数。

## 多查询注意力（MQA）和分组查询注意力（GQA）

**多查询注意力（MQA）** 和 **分组查询注意力（GQA）** 是对 Transformer 模型中传统多头注意力机制的改进。这些方法提高了注意力机制的效率和效果。

### 概述

**多查询注意力（MQA）**

MQA 将所有注意力头视为一个组，降低了计算复杂度并加速了训练时间。当关注模型可扩展性或计算资源有限时，它非常有益。

**分组查询注意力（GQA）**

GQA 将注意力头分组为簇，每个簇独立处理一个查询子集。这种方法平衡了传统多头注意力的细致关注和 MQA 的广泛方法，增强了对细微输入数据的处理能力。

这些注意力变体提供了：

- **降低计算负载**：两种方法都减少了计算量，对大型模型有益
- **提高处理速度**：简化注意力机制可以带来更快的训练和推理
- **灵活性和适应性**：可以根据任务需求或硬件限制进行调整

### 启用 MQA 和 GQA

要在 Megatron Bridge 中使用 MQA 或 GQA，请调整模型配置中的 `num_query_groups` 参数：

#### 多查询注意力（MQA）
将 `num_query_groups` 设置为 1，将所有注意力头视为一个组：

```python
from megatron.bridge.models import GPTModelProvider

model_config = GPTModelProvider(
    num_attention_heads=32,
    num_query_groups=1,  # 启用多查询注意力
    # ... 其他模型参数
)
```

#### 分组查询注意力（GQA）
将 `num_query_groups` 设置为一个数字，该数字是总注意力头数的除数（大于 1 但小于总头数）：

```python

model_config = GPTModelProvider(
    num_attention_heads=32,
    num_query_groups=8,  # 启用分组查询注意力（每组4个头）
    # ... 其他模型参数
)
```

#### 常规多头注意力
对于常规注意力，将此参数设置为 `None` 或使其与头数匹配：

```python
model_config = GPTModelProvider(
    num_attention_heads=32,
    num_query_groups=None,  # 常规多头注意力的默认设置
    # 或者等价地：
    # num_query_groups=32,  # 每个头一个组
    # ... 其他模型参数
)
```

## 资源

- [Megatron Core 注意力实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [Transformer Engine 注意力机制](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html)