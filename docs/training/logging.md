# 日志记录与监控

本指南介绍如何在 Megatron Bridge 中配置日志记录。它介绍了高级别的 `LoggerConfig`，解释了如何将实验日志记录到 TensorBoard 和 Weights & Biases (W&B)，并记录了控制台日志记录行为。

## LoggerConfig 概述

{py:class}`~bridge.training.config.LoggerConfig` 是一个数据类，封装了训练相关的日志记录设置。它位于整体的 {py:class}`bridge.training.config.ConfigContainer` 内部，该容器代表一次训练运行的完整配置。

### 计时器配置选项

使用以下选项来控制训练期间收集哪些计时指标以及如何聚合和记录它们。

#### `timing_log_level`
控制执行期间记录哪些计时器：

- **级别 0**：仅记录整体迭代时间。
- **级别 1**：包含每次迭代执行一次的操作，例如梯度全归约。
- **级别 2**：捕获频繁执行的操作，提供更详细的洞察，但会增加开销。

#### `timing_log_option`
指定如何跨不同进程（rank）聚合计时器值。有效选项：

- `"max"`：记录所有进程中的最大值。
- `"minmax"`：同时记录最小值和最大值。
- `"all"`：记录所有进程的所有值。

#### `log_timers_to_tensorboard`
启用后，框架会将计时器指标记录到支持的日志后端，例如 TensorBoard。

### 诊断选项

框架提供了几个可选开关，用于增强监控和诊断：

- **损失缩放（Loss Scale）**：为混合精度训练启用动态损失缩放。
- **验证困惑度（Validation Perplexity）**：跟踪验证期间的模型困惑度。
- **CUDA 内存统计（CUDA Memory Statistics）**：报告详细的 GPU 内存使用情况。
- **世界大小（World Size）**：显示分布式进程的总数。

### 日志记录选项

使用以下选项在训练期间启用额外的诊断和性能监控。

- **`log_params_norm`**：计算并记录模型参数的 L2 范数。如果可用，还会记录梯度范数。
- **`log_energy`**：激活能耗监控器，记录每个 GPU 的能耗和瞬时功耗。
- **`log_memory`**：记录来自 `torch.cuda.memory_stats()` 的模型内存使用情况。
- **`log_throughput_to_tensorboard`**：计算训练吞吐量和利用率。
- **`log_runtime_to_tensorboard`**：估算训练结束前的剩余总时间。
- **`log_l2_norm_grad_to_tensorboard`**：计算并记录每个模型层的梯度 L2 范数。

## 实验日志记录
支持将指标记录到 TensorBoard 和 W&B。使用 W&B 时，建议同时启用 TensorBoard，以确保所有标量指标在所有日志后端中都能被一致地记录。

### TensorBoard

#### 记录内容

TensorBoard 捕获一系列训练和系统指标，包括：

- **学习率**，包括适用时的解耦学习率
- 用于详细分析的**每个损失的标量值**
- **批次大小**和**损失缩放因子**
- **CUDA 内存使用情况**和**世界大小**（如果启用）
- **验证损失**，以及可选的**困惑度**
- **计时器**，当计时功能启用时
- **能耗**和**瞬时功耗**，如果启用了能耗日志记录

#### 启用 TensorBoard 日志记录
  1) 安装 TensorBoard（如果尚未安装）：
  ```bash
  pip install tensorboard
  ```
  2) **在训练设置中配置日志记录**。在这些示例中，`cfg` 指的是一个 `ConfigContainer` 实例（例如由配方生成的实例），它包含一个代表 `LoggerConfig` 的 `logger` 属性：
  
  ```python
  from megatron.bridge.training.config import LoggerConfig

  cfg.logger = LoggerConfig(
      tensorboard_dir="./runs/tensorboard",
      tensorboard_log_interval=10,
      log_timers_to_tensorboard=True,   # 可选
      log_memory_to_tensorboard=False,  # 可选
  )
  ```

  ```{note}
  当设置了 `tensorboard_dir` 时，写入器（writer）会在最后一个进程上延迟创建。
  ```

#### 设置输出目录

TensorBoard 事件文件保存到 `tensorboard_dir` 指定的目录。

**启用额外指标的示例：**
```python
cfg.logger.tensorboard_dir = "./logs/tb"
cfg.logger.tensorboard_log_interval = 5
cfg.logger.log_loss_scale_to_tensorboard = True
cfg.logger.log_validation_ppl_to_tensorboard = True
cfg.logger.log_world_size_to_tensorboard = True
cfg.logger.log_timers_to_tensorboard = True
```

### Weights & Biases (W&B)

#### 记录内容

启用后，W&B 会自动镜像记录到 TensorBoard 的标量指标。
此外，完整的运行配置会在初始化时同步，以实现可复现性和实验跟踪。

#### 启用 W&B 日志记录

  1) 安装 W&B（如果尚未安装）：
  ```bash
  pip install wandb
  ```
  2) 使用以下方法之一通过 W&B 进行身份验证：
  - 在运行前在环境中设置 `WANDB_API_KEY`，或者
  - 在机器上运行一次 `wandb login`。

2) **配置日志记录** 在你的训练设置中。在这些示例中，`cfg` 指的是一个 `ConfigContainer` 实例（例如由配方生成的那个），它包含一个代表 `LoggerConfig` 的 `logger` 属性：

```python
from megatron.bridge.training.config import LoggerConfig

cfg.logger = LoggerConfig(
    tensorboard_dir="./runs/tensorboard",   # 推荐：启用共享日志门
    wandb_project="my_project",
    wandb_exp_name="my_experiment",
    wandb_entity="my_team",                 # 可选
    wandb_save_dir="./runs/wandb",          # 可选
)
```

```{note}
当设置了 `wandb_project` 且 `wandb_exp_name` 非空时，W&B 会在最后一个 rank 上延迟初始化。
```

#### 使用 NeMo Run 启动时的 W&B 配置

对于使用 NeMo Run 启动训练脚本的用户，可以选择使用 {py:class}`bridge.recipes.run_plugins.WandbPlugin` 来配置 W&B。

该插件会自动转发 `WANDB_API_KEY`，并默认注入以下日志记录器参数的 CLI 覆盖：

- `logger.wandb_project`
- `logger.wandb_entity`
- `logger.wandb_exp_name`
- `logger.wandb_save_dir`

这允许将 W&B 日志记录无缝集成到你的训练工作流中，无需手动配置。

### MLFlow

Megatron Bridge 可以按照与 W&B 集成相同的模式，将指标和工件记录到 MLFlow。

#### 记录内容

启用后，MLFlow 会接收：

- 训练配置作为运行参数
- 标量指标（损失、学习率、批次大小、吞吐量、计时器、内存、运行时、范数、能耗等）
- 每次迭代保存在实验特定工件路径下的检查点工件

#### 启用 MLFlow 日志记录

1) 安装 MLFlow（Megatron Bridge 默认已安装）：

```bash
pip install mlflow / uv add mlflow
```

2) 配置跟踪服务器（可选）：
   - 在环境中设置 `MLFLOW_TRACKING_URI`，或者
   - 在日志记录器配置中传递显式的 `mlflow_tracking_uri`。

3) 在你的训练设置中配置日志记录。

```python
from megatron.bridge.training.config import LoggerConfig

cfg.logger = LoggerConfig(
    tensorboard_dir="./runs/tensorboard",
    mlflow_experiment="my_megatron_experiment",
    mlflow_run_name="llama32_1b_pretrain_run",
    mlflow_tracking_uri="http://mlflow:5000",  # 可选
    mlflow_tags={                              # 可选
        "project": "llama32",
        "phase": "pretrain",
    },
)
```

### Comet ML

Megatron Bridge 可以按照与 W&B 和 MLFlow 集成相同的模式，将指标和实验元数据记录到 Comet ML。

#### 记录内容

启用后，Comet ML 会接收：

- 训练配置作为实验参数
- 标量指标（损失、学习率、批次大小、吞吐量、计时器、内存、运行时、范数、能耗等）
- 验证损失和困惑度指标
- 检查点保存/加载元数据

#### 启用 Comet ML 日志记录

1) 安装 Comet ML：

```bash
pip install comet-ml
```

2) 身份验证：
   - 在环境中设置 `COMET_API_KEY`，或者
   - 在日志记录器配置中传递显式的 `comet_api_key`。

3) 在你的训练设置中配置日志记录。

```python
from megatron.bridge.training.config import LoggerConfig

cfg.logger = LoggerConfig(
    tensorboard_dir="./runs/tensorboard",
    comet_project="my_project",
    comet_experiment_name="llama32_1b_pretrain_run",
    comet_workspace="my_workspace",          # 可选
    comet_tags=["pretrain", "llama32"],       # 可选
)
```

```{note}
当设置了 `comet_project` 且 `comet_experiment_name` 非空时，Comet ML 会在最后一个 rank 上延迟初始化。
```

#### 使用 NeMo Run 启动时的 Comet ML 配置

对于使用 NeMo Run 启动训练脚本的用户，可以选择使用 {py:class}`bridge.recipes.run_plugins.CometPlugin` 来配置 Comet ML。

该插件会自动转发 `COMET_API_KEY`，并默认注入以下日志记录器参数的 CLI 覆盖：

- `logger.comet_project`
- `logger.comet_workspace`
- `logger.comet_experiment_name`

#### 进度日志

当启用 `logger.log_progress` 时，框架会在检查点保存目录中生成一个 `progress.txt` 文件。

该文件包括：
- **作业级元数据**，例如时间戳和 GPU 数量
- **训练过程中的周期性进度条目**

在每个检查点边界，日志会更新以下内容：
- **作业吞吐量**（TFLOP/s/GPU）
- **累计吞吐量**
- **总浮点运算次数**
- **已处理的令牌数**

这提供了一个轻量级的、基于文本的训练进度审计跟踪，有助于跨重启跟踪性能。

## 张量检查

Megatron Bridge 通过 NVIDIA DLFW Inspect 与 TransformerEngine 的张量检查功能集成。此集成由 {py:class}`~bridge.training.config.TensorInspectConfig` 控制，能够在训练期间启用高级调试和张量统计分析。启用后，框架会自动处理初始化、步骤跟踪和清理。

```{note}
**当前限制：** 张量检查目前仅支持 TransformerEngine 中的线性模块（例如 `fc1`、`fc2`、`layernorm_linear`）。不支持注意力等操作。
```

```{note}
本节涵盖 Megatron Bridge 配置。有关功能、配置语法和高级用法的完整文档，请参阅：

- [TransformerEngine 调试文档](https://github.com/NVIDIA/TransformerEngine/tree/af2a0c16ec11363c0af84690cd877a59f898820e/docs/debug)
- [NVIDIA DLFW Inspect 文档](https://github.com/NVIDIA/nvidia-dlfw-inspect/tree/4118044cc84f0183714a2ab1bc215fa49f9aaa82/docs)
```

### 安装

如果尚未安装，请安装 NVIDIA DLFW Inspect：
```bash
pip install nvdlfw-inspect
```

### 可用功能

TransformerEngine 提供以下调试功能：

- **LogTensorStats** – 记录高精度张量统计信息：`min`、`max`、`mean`、`std`、`l1_norm`、`l2_norm`、`cur_amax`、`dynamic_range`。
- **LogFp8TensorStats** – 记录 FP8 量化方案的量化张量统计信息：`underflows%`、`scale_inv_min`、`scale_inv_max`、`mse`。支持模拟替代方案（例如，在逐张量当前缩放训练期间跟踪 `mxfp8_underflows%`）
- **DisableFP8GEMM** – 以高精度运行特定的 GEMM 操作
- **DisableFP8Layer** – 为整个层禁用 FP8
- **PerTensorScaling** – 为特定张量启用逐张量当前缩放
- **FakeQuant** – 实验性量化测试

有关完整参数列表和使用详情，请参阅 [TransformerEngine 调试功能](https://github.com/NVIDIA/TransformerEngine/tree/af2a0c16ec11363c0af84690cd877a59f898820e/transformer_engine/debug/features)。

### 配置

使用 {py:class}`~bridge.training.config.TensorInspectConfig` 通过 YAML 文件或内联字典配置张量检查。

#### YAML 配置

```yaml
tensor_inspect:
  enabled: true
  features: ./conf/fp8_tensor_stats.yaml
  log_dir: ./logs/tensor_inspect
```

**功能配置文件示例：**

```yaml
fp8_tensor_stats:
  enabled: true
  layers:
    layer_name_regex_pattern: ".*(fc2)"
  transformer_engine:
    LogFp8TensorStats:
      enabled: true
      tensors: [weight,activation,gradient]
      stats: ["underflows%", "mse"]
      freq: 5
      start_step: 0
      end_step: 100
```

#### Python 配置

```python
from bridge.training.config import TensorInspectConfig

# 选项 1：内联 python 字典
cfg.tensor_inspect = TensorInspectConfig(
    enabled=True,
    features={
        "fp8_gradient_stats": {
            "enabled": True,
            "layers": {"layer_name_regex_pattern": ".*(fc1|fc2)"},
            "transformer_engine": {
                "LogFp8TensorStats": {
                    "enabled": True,
                    "tensors": ["weight","activation","gradient"],
                    "stats": ["underflows%", "mse"],
                    "freq": 5,
                    "start_step": 0,
                    "end_step": 100,
                },
            },
        }
    },
    log_dir="./logs/tensor_inspect",
)

# 选项 2：引用外部 YAML 文件
cfg.tensor_inspect = TensorInspectConfig(
    enabled=True,
    features="./conf/fp8_inspect.yaml",
    log_dir="./logs/tensor_inspect",
)

```

#### 层选择

功能应用于 `layers` 部分中匹配选择器的线性模块：

- `layer_name_regex_pattern: .*` – 所有支持的线性层
- `layer_name_regex_pattern: .*layers\.(0|1|2).*(fc1|fc2|layernorm_linear)` – 前三个 Transformer 层中的线性模块
- `layer_name_regex_pattern: .*(fc1|fc2)` – 仅 MLP 投影层
- `layer_types: [layernorm_linear, fc1]` – 字符串匹配（正则表达式的替代方案）

张量级选择器（`tensors`、`tensors_struct`）控制记录哪些张量角色：`activation`、`gradient`、`weight`、`output`、`wgrad`、`dgrad`。

### 输出与监控

张量统计信息写入 `tensor_inspect.log_dir`，并在启用时转发到 TensorBoard/W&B。

**日志位置：**
- 文本日志：`<log_dir>/nvdlfw_inspect_statistics_logs/`
- TensorBoard
- W&B

### 性能考虑

- 使用 `freq > 1` 以减少开销。对于大型模型，统计信息收集成本很高。
- 使用特定的正则表达式模式而非 `.*` 来缩小层选择范围

## 控制台日志记录

Megatron Bridge 使用标准的 Python 日志记录子系统进行控制台输出。

### 配置控制台日志记录

要控制控制台日志记录行为，请使用以下配置选项：

- `logging_level` 设置默认的日志详细级别。可通过 `MEGATRON_BRIDGE_LOGGING_LEVEL` 环境变量覆盖此设置。
- `filter_warnings` 抑制 WARNING 级别的消息。
- `modules_to_filter` 指定要从输出中排除的日志记录器名称前缀。
- `set_level_for_all_loggers` 决定日志级别是应用于所有日志记录器还是仅应用于一个子集，具体取决于当前实现。

### 监控日志记录频率与内容

为了定期监控训练进度，框架会每隔 `log_interval` 次迭代打印一行摘要信息。

每条摘要包括：
- **时间戳**
- **迭代计数器**
- **已消耗和已跳过的样本数**
- **迭代时间（毫秒）**
- **学习率**
- **全局批次大小**
- **各损失平均值**
- **损失缩放因子**

启用后，还会打印以下额外指标：
- **梯度范数**
- **梯度中的零值**
- **参数范数**
- **每个 GPU 的能耗与功率**

掉队者计时报告遵循相同的 `log_interval` 频率，有助于识别跨不同进程的性能瓶颈。

### 最小化计时开销

为了减少性能影响，请将 `timing_log_level` 设置为 `0`。
仅在需要更详细的计时指标时，才将其增加到 `1` 或 `2`，因为更高的级别会引入额外的日志记录开销。