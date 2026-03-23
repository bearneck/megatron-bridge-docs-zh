# 混合精度训练

混合精度训练通过以低精度格式执行运算显著提升计算效率，同时选择性地在关键网络区域保持少量数据为单精度，以保留关键信息。Megatron Bridge 通过 {py:class}`bridge.training.mixed_precision.MixedPrecisionConfig` 配置，支持在大多数模型中通过 Transformer Engine (TE) 实现 FP16、BF16 和 FP8 训练。

## 配置概述

在 Megatron Bridge 中，混合精度通过 {py:class}`bridge.training.config.ConfigContainer` 中的 `mixed_precision` 字段进行配置，该字段接受以下任一形式：
- 引用预定义配方的字符串名称（例如 `"bf16_mixed"`）
- 用于自定义配置的 {py:class}`bridge.training.mixed_precision.MixedPrecisionConfig` 对象

混合精度配置会自动使用适当的精度参数更新模型、优化器和分布式数据并行设置。

## 半精度训练

Megatron Bridge 通过 Megatron Core 和分布式优化器支持半精度 FP16 和 BF16 计算训练。此训练配方在所有层计算中使用半精度，同时将模型状态（优化器状态和主参数）保持在单精度。为了避免在每层计算中重复进行数据类型转换，Megatron Core 会保留一份单独的半精度参数副本，并在每次优化器步骤后更新。

### 使用预定义配方

启用混合精度最简单的方法是使用预定义的配方名称：

```python
from megatron.bridge.training.config import ConfigContainer

# 使用 BF16 混合精度进行配置
config = ConfigContainer(
    mixed_precision="bf16_mixed",
    # ... 其他配置参数
)

# 使用 FP16 混合精度进行配置
config = ConfigContainer(
    mixed_precision="fp16_mixed",
    # ... 其他配置参数
)
```

### 自定义混合精度配置

如需更多控制，可以创建自定义的 {py:class}`bridge.training.mixed_precision.MixedPrecisionConfig`：

```python
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
import torch

# 自定义 BF16 配置
bf16_config = MixedPrecisionConfig(
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    autocast_enabled=False,
    grad_reduce_in_fp32=True,
)

config = ConfigContainer(
    mixed_precision=bf16_config,
    # ... 其他配置参数
)
```

## FP8 训练

NVIDIA H100 GPU 引入了对新数据类型 FP8（8 位浮点数）的支持，从而提高了矩阵乘法和卷积的吞吐量。Megatron Bridge 使用 NVIDIA TransformerEngine (TE) 来利用 FP8 带来的加速。更详细的概述，请参阅 [TE 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)，特别是关于 FP8 格式和配方的部分。

### FP8 配置参数

{py:class}`bridge.training.mixed_precision.MixedPrecisionConfig` 提供了几个 FP8 特定的参数：

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `fp8` | `Optional[str]` | `None` | FP8 格式：`"hybrid"`（激活/权重使用 E4M3，梯度使用 E5M2）或 `"e4m3"` |
| `fp8_recipe` | `str` | `"tensorwise"` | FP8 配方类型：`"tensorwise"`、`"delayed"`、`"blockwise"`、`"mxfp8"`（仅限 Blackwell） |
| `first_last_layers_bf16` | `bool` | `False` | 如果为 True，则将首尾 N 个 TransformerBlocks 保留为 BF16 而非 FP8 |
| `num_layers_at_start_in_bf16` | `int` | `0` | 当 `first_last_layers_bf16` 为 True 时，模型开头保持 BF16 精度的层数 |
| `num_layers_at_end_in_bf16` | `int` | `0` | 当 `first_last_layers_bf16` 为 True 时，模型末尾保持 BF16 精度的层数 |
| `fp8_margin` | `int` | `0` | 缩放因子偏移 $2^{margin}$ |
| `fp8_amax_history_len` | `int` | `1` | amax 历史记录的窗口大小 |
| `fp8_amax_compute_algo` | `str` | `"most_recent"` | Amax 选择算法：`"max"` 或 `"most_recent"` |
| `fp8_param` | `Optional[bool]` | `None` | 在 FP8 中存储模块级参数 |
| `fp8_param_gather` | `bool` | `False` | 启用 FP8 参数收集 |

### FP8 配方示例

使用 `mixed_precision` 参数配合任何预定义的 FP8 配方名称：

```python
# 示例：使用 FP8 当前缩放的 BF16
config = ConfigContainer(
    mixed_precision="bf16_with_fp8_current_scaling_mixed",
    # ... 其他配置参数
)
```

## 可用的混合精度配方

Megatron Bridge 为不同使用场景提供了众多预定义的混合精度方案。你可以使用 {py:func}`~megatron.bridge.training.mixed_precision.get_mixed_precision_config` 工具函数将字符串简称转换为类实例。有关可用方案的完整列表及其具体配置，请参阅 {py:mod}`megatron.bridge.training.mixed_precision` 模块。

### 自定义 FP8 配置

对于高级用例，可以创建自定义的 FP8 配置：

```python
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
import torch

# 自定义 FP8 配置
fp8_config = MixedPrecisionConfig(
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    fp8="hybrid",
    fp8_recipe="tensorwise", 
    fp8_margin=0,
    fp8_amax_history_len=1024,
    fp8_amax_compute_algo="max",
    fp8_param_gather=True,
)

config = ConfigContainer(
    mixed_precision=fp8_config,
    # ... 其他配置参数
)
```

### 注册自定义混合精度方案

你也可以注册自己的自定义混合精度配置，以便与简称系统配合使用。在返回 `MixedPrecisionConfig` 对象的函数上使用 {py:func}`~megatron.bridge.training.mixed_precision.register` 装饰器：

```python
from megatron.bridge.training.mixed_precision import register, MixedPrecisionConfig

@register
def my_custom_fp8_recipe() -> MixedPrecisionConfig:
    """针对我的用例的特定自定义 FP8 方案。"""
    return MixedPrecisionConfig(
        bf16=True,
        fp8="hybrid",
        fp8_recipe="tensorwise",
        fp8_param_gather=True,
        # ... 其他自定义设置
    )

# 现在你可以通过工具函数使用它
config = get_mixed_precision_config("my_custom_fp8_recipe")
```

常见的方案类别包括：
- **半精度方案**：基本的 BF16 和 FP16 混合精度
- **FP8 方案**：各种 FP8 缩放策略（延迟、当前、子通道）
- **架构特定方案**：针对特定 GPU 架构（Hopper、Blackwell）优化
- **模型特定方案**：针对特定模型系列调优

## 配置同步

当提供混合精度配置时，它会自动在模型、优化器和分布式数据并行（DDP）配置之间同步与精度相关的设置。这确保了整个训练流程中精度行为的一致性。

**重要提示**：混合精度设置将覆盖可能直接在模型、优化器或 DDP 配置上设置的任何冲突的精度参数。混合精度配置是所有精度相关参数的权威来源。

例如，如果你同时指定了：
```python
# 这将被覆盖
model_config.bf16 = False
optimizer_config.bf16 = False

config = ConfigContainer(
    model=model_config,
    optimizer=optimizer_config,
    mixed_precision="bf16_mixed",  # 在训练期间，此配置具有优先权
    # ... 其他配置
)
```

混合精度配置将在模型和优化器配置上设置 `bf16=True`，覆盖显式设置的 `False` 值。这种同步可以防止配置不匹配，从而避免导致训练问题。

## 性能考量

- **FP8 方案是实验性的**，尚未对所有模型的收敛性进行完全验证
- 为了获得更好的数值稳定性，通常**推荐使用 BF16 而非 FP16**
- **FP8** 在 H100 GPU 上提供最佳性能，但需要仔细调优
- **MXFP8** 方案仅在 Blackwell 架构的 GPU 上受支持
- **分块缩放**方案针对 Hopper 架构的 GPU 进行了优化

## 资源

- [Transformer Engine 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [FP8、浮点格式和混合精度训练简介](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Introduction-to-FP8)
- [通过启用 TE 的 FP8 训练，Megatron Bridge 原生支持的性能优化](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced_optimizations.html)