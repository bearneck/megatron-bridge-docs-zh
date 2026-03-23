# 激活重计算

网络层的输入激活存储在设备内存中，并在反向传播期间用于计算梯度。当使用长序列长度或大微批次大小训练大语言模型（LLM）时，这些输入激活会迅速耗尽设备内存。对部分激活进行检查点保存并重新计算其余部分，是减少设备内存使用量的常用技术。

Megatron Bridge 中的激活重计算通过模型提供者的重计算参数进行配置，这些参数基于 Megatron Core 的 `TransformerConfig`。

## Transformer 层重计算

Megatron Bridge 支持 Transformer 层重计算，该方法会保存每个 Transformer 层的输入检查点，并重新计算其余层的激活。该技术显著减少了激活内存的使用。然而，由于需要重新执行整个层的前向计算，它会使每个 Transformer 层的计算成本增加 30%。

Megatron Bridge 还支持部分 Transformer 层重计算，当重计算少数 Transformer 层有助于释放足够的 GPU 内存以使模型能够容纳时，这种方法非常有益。这种方法避免了重新计算其余层的需要。

### 配置

Transformer 层重计算通过模型提供者的重计算参数进行配置：

```python
from megatron.bridge.models import GPTModelProvider

# 完全重计算 - 重计算所有层
model_config = GPTModelProvider(
    recompute_granularity="full",  # 启用全层重计算
    recompute_method="uniform",    # 跨层均匀分布
    recompute_num_layers=4,        # 每个重计算块的层数
    # ... 其他模型参数
)
```

### 重计算方法

#### 块方法
每个流水线阶段重计算特定数量的 Transformer 层：

```python
model_config = GPTModelProvider(
    recompute_granularity="full",
    recompute_method="block",      # 块状重计算
    recompute_num_layers=4,        # 每个流水线阶段重计算 4 层
)
```

#### 均匀方法
均匀划分 Transformer 层的总数，并为每个划分的块重计算输入激活：

```python
model_config = GPTModelProvider(
    recompute_granularity="full",
    recompute_method="uniform",    # 均匀分布
    recompute_num_layers=8,        # 每个重计算块的层数
)
```

### 流水线并行注意事项

使用流水线并行进行训练时：
- `recompute_num_layers` 表示每个流水线阶段的层数
- 当使用虚拟流水线时，`recompute_num_layers` 指定每个虚拟流水线阶段的层数
- 框架会自动处理跨流水线阶段的重计算协调

![激活重计算方法](images/activation-recomputation-example-1.jpg)
*图 1：均匀和块检查点方法的示意图（完全检查点粒度）*

## 自注意力重计算

Megatron Bridge 支持选择性自注意力重计算，该方法会保存每个自注意力块的输入检查点，并重新计算中间输入激活。这种高性价比的方法以最小的重计算成本实现了高内存节省。

自注意力块的中间层占据了激活内存的大部分，因为 softmax、dropout 和 QKV 点积注意力层的输入大小具有与序列长度平方成正比的内存复杂度。然而，它们的重计算成本相对于其他与隐藏层大小平方成比例的线性投影层来说相对较小。

![激活重计算粒度](images/activation-recomputation-example-2.jpg)
*图 2：完全和选择性检查点粒度的示意图*

### 配置

使用选择性粒度启用自注意力重计算：

```python
from megatron.bridge.models import GPTModelProvider

model_config = GPTModelProvider(
    recompute_granularity="selective",  # 启用选择性重计算
    recompute_modules=["core_attn"],    # 重计算注意力模块（默认）
    # ... 其他模型参数
)
```

### 重计算模块

Megatron Bridge 支持对各种模块进行选择性重计算：

```python
model_config = GPTModelProvider(
    recompute_granularity="selective",
    recompute_modules=[
        "core_attn",      # 核心注意力计算（默认）
        "mlp",            # MLP 层
        "layernorm",      # 层归一化
        "moe",            # 专家混合层
        "moe_act",        # MoE 激活函数
        "shared_experts", # 共享专家层
        "mla_up_proj",    # 多潜在注意力向上投影
    ],
)
```

### Flash Attention 集成

当通过 Transformer Engine 使用 Flash Attention 时，自注意力重计算会自动启用。Flash Attention 本质上通过重计算注意力分数而非存储它们来提供内存效率，因此通常不需要额外的显式重计算。

## 高级重计算配置

### 分布式激活检查点

对于使用模型并行的模型，您可以将保存的激活分布在模型并行组中：

```python
model_config = GPTModelProvider(
    recompute_granularity="selective",
    distribute_saved_activations=True,  # 在模型并行组中分布
    # 注意：不能与 sequence_parallel=True 同时使用
)
```

### 内存与计算权衡

不同的重计算策略提供不同的内存-计算权衡：

- **选择性重计算**：通过针对注意力等内存密集型操作，以最小的重计算成本提供高内存节省
- **完全重计算**：显著减少激活内存使用，但每个 Transformer 层的计算成本增加约 30%
- **无重计算**：将所有激活保留在内存中，需要更多 GPU 内存但无需额外计算

### MoE 特定重计算

对于专家混合（Mixture of Experts）模型，提供了专门的重计算选项：

```python
model_config = GPTModelProvider(
    # MoE 配置
    num_moe_experts=8,
    expert_model_parallel_size=2,
    
    # MoE 重计算
    recompute_granularity="selective",
    recompute_modules=["moe", "moe_act"],  # 重计算 MoE 特定模块
)
```