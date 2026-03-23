# 参数高效微调（PEFT）

本指南解释了如何在 Megatron Bridge 中配置和使用 PEFT——涵盖 LoRA 和 DoRA、所需的检查点、示例配置以及内部设计和训练工作流程——以便您能够高效地集成、扩展和检查点适配器。

## 模型定制
定制模型使您能够将通用的预训练模型适配到特定的用例或领域。这个过程会产生一个微调模型，它保留了预训练阶段的广泛知识，同时针对特定的下游任务提供更准确的输出。

模型定制通常通过监督微调实现，主要分为两种方法：全参数微调（称为监督微调，SFT）和参数高效微调（PEFT）。

在 SFT 中，所有模型参数都会被更新，以使模型的输出符合特定任务的要求。这种方法通常能获得最高的性能，但计算量可能很大。

相比之下，PEFT 只更新一小部分参数，这些参数被插入到基础模型的关键位置。基础模型的权重保持冻结，只训练适配器模块。这显著减少了可训练参数的数量——通常不到 1%——同时仍能达到接近 SFT 的准确度水平。

随着语言模型规模的持续增长，PEFT 因其高效性和最低的硬件需求而越来越受欢迎，使其成为许多实际应用的实用选择。

## PEFT 配置

PEFT 在 `ConfigContainer` 中配置为一个可选属性：

```python
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.peft.lora import LoRA

config = ConfigContainer(
    # ... 其他必需的配置
    peft=LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=16,
        alpha=32,
        dropout=0.1,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/pretrained/checkpoint",  # PEFT 必需
        save="/path/to/peft/checkpoints",
    ),
)
```

```{note}
**要求**：PEFT 需要设置 `checkpoint.pretrained_checkpoint` 来加载基础模型权重。
```

## 支持的 PEFT 方法

### [LoRA：大语言模型的低秩适配](https://arxiv.org/abs/2106.09685)

LoRA 通过使用两个低秩分解矩阵表示权重更新，使微调变得高效。原始模型权重保持冻结，而低秩分解矩阵被更新以适应新数据，从而保持可训练参数数量较低。与适配器相比，原始模型权重和适配后的权重可以在推理过程中合并，避免了推理时模型的任何架构更改或额外延迟。

在 Megatron Bridge 中，您可以配置适配器的瓶颈维度以及应用 LoRA 的目标模块。LoRA 支持任何线性层，在 Transformer 模型中通常包括：

1. 查询、键和值（QKV）注意力投影
2. 注意力输出投影
3. 一个或两个 MLP 层

Megatron Bridge 将 QKV 投影融合到单个线性层中。因此，LoRA 为组合的 QKV 表示学习统一的低秩适配。

```python
from megatron.bridge.peft.lora import LoRA

lora_config = LoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=16,                    # 适配的秩
    alpha=32,                  # 缩放参数
    dropout=0.1,               # 丢弃率
)
```

#### 关键参数
下表列出了配置 DoRA 的关键超参数，这些参数控制其模块定位、适配秩、缩放行为和正则化策略。
| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `target_modules` | `List[str]` | 所有线性层 | 应用 DoRA 的模块 |
| `dim` | `int` | `32` | 低秩适配的秩 |
| `alpha` | `float` | `16` | DoRA 的缩放参数 |
| `dropout` | `float` | `0.0` | DoRA 层的丢弃率 |

#### 目标模块
下表列出了 Transformer 架构中通常作为 LoRA 目标的特定子模块，以实现注意力和前馈组件的高效微调：
| 模块        | 描述                                 |
|---------------|---------------------------------------------|
| `linear_qkv`  | 注意力中的查询、键、值投影  |
| `linear_proj` | 注意力输出投影                 |
| `linear_fc1`  | 第一个 MLP 层                             |
| `linear_fc2`  | 第二个 MLP 层                            |

#### 通配符目标模块
为了更精细地定位，适配器可以针对单个层。
```python
# 仅针对特定层
lora_config = LoRA(
    target_modules=[

```
        "*.layers.0.*.linear_qkv",   # 仅第一层
        "*.layers.1.*.linear_qkv",   # 仅第二层
    ]
)
```

### 标准 LoRA：性能版 vs 标准版变体

Megatron Bridge 中实现了两种 LoRA 变体："性能版 LoRA" (`LoRA`) 和 "标准版 LoRA" (`CanonicalLoRA`)。

这种区别源于 Megatron Core 通过将多个线性层融合为一个层来优化以下两个线性模块的实现。当这些层使用 LoRA 进行适配时，性能版也只为该线性模块使用一个适配器。这两个线性模块是：

1. `linear_qkv`：自注意力中将隐藏状态转换为查询、键和值的投影矩阵。Megatron Core 将这三个投影矩阵融合为单个矩阵，以高效并行化矩阵乘法。因此，性能版 LoRA 对 qkv 投影矩阵应用单个适配器，而标准版 LoRA 应用三个适配器。
2. `linear_fc1`：MLP 模块中在中间激活之前的第一个线性层。对于门控线性激活，Megatron Core 将 up 和 gate 投影矩阵融合为单个矩阵以实现高效并行化。因此，性能版 LoRA 对 up 和 gate 投影矩阵应用单个适配器，而标准版 LoRA 应用两个适配器。

以下两图以 `linear_qkv` 层为例说明了标准版和性能版 LoRA 之间的区别。标准版 LoRA 顺序运行三个适配器，而性能版 LoRA 运行一个适配器。

```{image} images/canonical_lora.png
:width: 640
:align: center
```

```{image} images/performant_lora.png
:width: 400
:align: center
```

标准版 LoRA 更贴近参考实现，但如上所述，由于它顺序执行多个矩阵乘法，因此相比之下速度较慢。性能版 LoRA 的参数比标准版 LoRA 少，并且通常能达到与标准版 LoRA 相同的准确度水平。

虽然不明显，但当 `linear_qkv` 中的 $A_q$、$A_k$、$A_v$ 矩阵被绑定（即在训练期间强制共享相同权重）时，性能版 LoRA 在数学上等同于标准版 LoRA；类似地，当 `linear_fc1` 中的 $A_{up}$、$A_{gate}$ 矩阵被绑定时也是如此。

```{admonition} 数学证明：权重绑定时性能版 LoRA 等同于标准版 LoRA
:class: dropdown

令 $[x \quad y]$ 表示矩阵拼接。（在 Megatron Bridge 中，这种拼接是以交错方式完成的，但这不影响下面的证明。）

令 $A_q = A_k = A_v = A_{qkv}$ （权重绑定）

则

$$
\begin{align}
& [query \quad key \quad value] \\
= & [W_q x + B_q A_q x \quad W_k x + B_k A_k x \quad W_v x + B_v A_v x] \quad\quad \text{(标准版公式)} \\
= & [W_q x + B_q (A_{qkv} x) \quad W_k x + B_k (A_{qkv} x) \quad W_v x + B_v (A_{qkv} x)] \\
= & [W_q \quad W_k \quad W_v] x + [B_q \quad B_k \quad B_v]A_{qkv} x \\
= & W_{qkv} x + B_{qkv} A_{qkv} x  \quad\quad \text{(性能版公式)}
\end{align}
$$

注意：权重矩阵的维度如下：

$$
\begin{align}
W_q:     &\ h \times n_q d          \qquad & A_q:     &\ h \times r \qquad  & B_q:     &\ r \times n_q d \\
W_k:     &\ h \times n_{kv} d       \qquad & A_k:     &\ h \times r \qquad  & B_k:     &\ r \times n_{kv} d \\
W_v:     &\ h \times n_{kv} d       \qquad & A_v:     &\ h \times r \qquad  & B_v:     &\ r \times n_{kv} d \\
W_{qkv}: &\ h \times (n_q+2n_{kv})d \qquad & A_{qkv}: &\ h \times r \qquad  & B_{qkv}: &\ r \times (n_q+2n_{kv})d
\end{align}
$$

其中：
- $n_q$：注意力头数量 (`num_attention_heads`)。
- $n_{kv}$：键值头数量 (`num_query_groups`)。注意，如果不使用分组查询注意力（GQA），则 $n_{kv} = n_q$。
- $h$：Transformer 隐藏层大小 (`hidden_size`)。
- $d$：Transformer 头维度 (`kv_channels`)。
- $r$：LoRA 秩。

```

#### 使用标准版 LoRA

```python
from megatron.bridge.peft.canonical_lora import CanonicalLoRA

canonical_lora_config = CanonicalLoRA(
    target_modules=[
        "linear_q", "linear_k", "linear_v",      # 独立的 Q、K、V 投影
        "linear_proj",                           # 注意力输出投影
        "linear_fc1_up", "linear_fc1_gate",     # 独立的 up 和 gate 投影
        "linear_fc2"                             # 第二个 MLP 层
    ],
    dim=16,                    # 适配的秩
    alpha=32,                  # 缩放参数
    dropout=0.1,               # 丢弃率
)
```

#### 关键参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `target_modules` | `List[str]` | 所有标准线性层 | 应用标准版 LoRA 的模块 |
| `dim` | `int` | `32` | 低秩适配的秩 |
| `alpha` | `float` | `32` | LoRA 的缩放参数 |
| `dropout` | `float` | `0.0` | LoRA 层的丢弃率 |

| `dropout_position` | `Literal["pre", "post"]` | `"pre"` | 应用 dropout 的位置 |
| `lora_A_init_method` | `str` | `"xavier"` | LoRA A 矩阵的初始化方法 |
| `lora_B_init_method` | `str` | `"zero"` | LoRA B 矩阵的初始化方法 |

#### 标准 LoRA 的目标模块

下表列出了标准 LoRA 在 Transformer 架构中针对的具体子模块：

| 模块 | 描述 |
|--------|-------------|
| `linear_q` | 注意力机制中的查询投影 |
| `linear_k` | 注意力机制中的键投影 |
| `linear_v` | 注意力机制中的值投影 |
| `linear_proj` | 注意力输出投影 |
| `linear_fc1_up` | MLP 中的向上投影 |
| `linear_fc1_gate` | MLP 中的门控投影 |
| `linear_fc2` | 第二个 MLP 层 |

```{note}
标准 LoRA 不支持 `linear_qkv` 或 `linear_fc1` 目标。请使用单独的组件目标（对于 QKV 使用 `linear_q`、`linear_k`、`linear_v`，对于 FC1 使用 `linear_fc1_up`、`linear_fc1_gate`）替代。
```

### [DoRA: 权重分解的低秩自适应](https://arxiv.org/abs/2402.09353)

DoRA 将预训练权重分解为幅度和方向。它学习一个单独的幅度参数，同时使用 LoRA 进行方向更新，从而高效地最小化可训练参数的数量。DoRA 增强了 LoRA 的学习能力和训练稳定性，同时避免了任何额外的推理开销。DoRA 已被证明在各种下游任务上持续优于 LoRA。

在 Megatron Bridge 中，DoRA 利用了与 LoRA 相同的适配器结构。Megatron Bridge 为 DoRA 添加了对张量并行（Tensor Parallelism）和管道并行（Pipeline Parallelism）的支持，使得 DoRA 能够扩展到更大的模型变体。

```python
from megatron.bridge.peft.dora import DoRA

dora_config = DoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=16,                    # 自适应秩
    alpha=32,                  # 缩放参数
    dropout=0.1,               # Dropout 率
)
```

#### 关键参数

以下参数定义了如何将 LoRA 应用于您的模型。它们控制目标模块、自适应秩、缩放行为和 dropout 配置：

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `target_modules` | `List[str]` | 所有线性层 | 应用 DoRA 的模块 |
| `dim` | `int` | `32` | 低秩自适应的秩 |
| `alpha` | `float` | `16` | DoRA 的缩放参数 |
| `dropout` | `float` | `0.0` | DoRA 层的 Dropout 率 |

## 完整配置示例

```python
from megatron.bridge.training.config import (
    ConfigContainer, TrainingConfig, CheckpointConfig
)
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.peft.lora import LoRA
from megatron.core.optimizer import OptimizerConfig

# 配置 PEFT 微调
config = ConfigContainer(
    model=model_provider,
    train=TrainingConfig(
        train_iters=1000,
        global_batch_size=64,
        micro_batch_size=1,  # 如果使用打包序列，则为必需项
        eval_interval=100,
    ),
    optimizer=OptimizerConfig(
        optimizer="adam",
        lr=1e-4,  # 微调时使用较低学习率
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=True,
    ),
    scheduler=SchedulerConfig(
        lr_decay_style="cosine",
        lr_warmup_iters=100,
        lr_decay_iters=1000,
    ),
    dataset=HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=process_squad_example,
        seq_length=512,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/pretrained/model",  # 必需项
        save="/path/to/peft/checkpoints",
        save_interval=200,
    ),
    peft=LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=16,
        alpha=32,
        dropout=0.1,
    ),
    # ... 其他配置
)
```

## Megatron Bridge 中的 PEFT 设计

本节描述了 PEFT 如何集成到 Megatron Bridge 中的内部设计和架构。

### 架构概述

PEFT 框架引入了一种模块化设计，用于将适配器集成到大规模模型中。其架构由以下组件组成：

1.  **基础 PEFT 类**：所有 PEFT 方法都继承自抽象的 {py:class}`bridge.peft.base.PEFT` 基类，该类定义了模块转换的核心接口。
2.  **模块转换**：PEFT 遍历模型结构，以单独识别和转换目标模块。
3.  **适配器集成**：适配器在模型初始化期间使用预包装钩子注入到选定的模块中。
4.  **检查点集成**：在检查点保存和加载期间，仅保存和加载适配器参数；基础模型权重保持冻结且不变。

### 训练中的 PEFT 工作流程

PEFT 的训练工作流程遵循一个结构化的序列，以确保以最小的开销进行高效的微调：
1.  **模型加载**：从指定的预训练检查点初始化基础模型。
2.  **PEFT 应用**：在 Megatron Core 模型初始化之后、分布式包装之前应用适配器转换。
3.  **参数冻结**：冻结基础模型参数以降低训练复杂度；仅更新适配器参数。
4.  **适配器权重加载**：恢复训练时，从检查点恢复适配器权重。
5.  **检查点保存**：仅保存适配器状态，从而生成显著更小的检查点文件。

### 主要优势

PEFT 为可扩展且高效的模型微调提供了多项优势：

-   **减少检查点大小**：仅包含适配器的检查点比完整模型检查点小得多。
-   **内存效率**：由于仅计算适配器参数的梯度，内存使用量显著减少。
-   **恢复训练支持**：可以使用仅包含适配器的检查点无缝恢复训练，而无需重新加载完整的模型权重。