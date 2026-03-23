# 训练入口点

Megatron Bridge 为预训练、监督微调（SFT）和参数高效微调（PEFT）提供了统一的训练入口点。所有训练模式共享相同的基础训练循环架构，主要区别在于数据处理和模型配置。

## 主要入口点

{py:func}`bridge.training.pretrain.pretrain` 和 {py:func}`bridge.training.finetune.finetune` 函数是预训练模型的主要入口点——无论是从头开始训练还是通过微调。每个函数都接受一个 {py:class}`bridge.training.config.ConfigContainer` 以及一个定义如何运行训练循环的 `forward_step_func`。

## 前向步骤函数

`forward_step_func` 定义了每个训练步骤的执行方式。它应遵循以下签名：

```python
def forward_step_func(
    global_state: GlobalState,
    data_iterator: Iterable,
    model: MegatronModule,
    return_schedule_plan: bool = False,
) -> tuple[Any, Callable]:
    """前向步骤函数。
    
    Args:
        global_state: 包含配置和实用工具的训练状态对象
        data_iterator: 训练/评估数据的迭代器
        model: 执行前向步骤的模型
        return_schedule_plan: 是否返回调度计划（用于 MoE 重叠）
        
    Returns:
        包含以下内容的元组：
        - output: 前向传递输出（张量或张量集合）
        - loss_func: 从输出计算损失值的函数
    """
```

### 职责

前向步骤函数有三个主要职责：

1.  **获取批次**：从数据迭代器中检索并处理下一个批次。
2.  **运行前向传递**：在批次上执行模型的前向传递。
3.  **返回损失函数**：提供一个从输出计算损失值的函数。

### 状态访问

Megatron Bridge 自动提供包含以下内容的 {py:class}`bridge.training.state.GlobalState` 对象：
- **配置**：完整的训练配置（`global_state.cfg`）。
- **计时器**：性能监控实用工具（`global_state.timers`）。
- **训练进度**：当前步骤、已消耗样本数（`global_state.train_state`）。
- **日志记录器**：用于指标跟踪的 TensorBoard 和 WandB 日志记录器。

所有配置和状态信息都可以通过注入的 `state` 对象访问。

有关完整的实现示例，请参阅 {py:func}`bridge.training.gpt_step.forward_step`。

## 损失计算与归约

前向步骤返回的损失函数可以根据您的需求遵循不同的模式：

### 损失函数模式

1.  **标准模式**：返回 `(loss, metadata_dict)`
    - 损失值会自动在微批次间取平均
    - 元数据字典包含用于日志记录的命名损失分量
    - 标准训练中最常见的模式

2.  **词元感知模式**：返回 `(loss, num_tokens, metadata_dict)`
    - 损失值会在微批次和词元两个维度上取平均
    - 当您希望按词元进行损失平均时很有用
    - 推荐用于可变长度序列

3.  **推理模式**：返回任意数据结构
    - 与 `collect_non_loss_data=True` 和 `forward_only=True` 配合使用
    - 适用于推理、评估指标或自定义数据收集
    - 不应用自动损失处理

### 自动损失处理

训练循环自动处理：
- **微批次归约**：聚合全局批次中所有微批次的损失。
- **分布式归约**：跨数据并行秩执行 all-reduce 操作。
- **流水线协调**：仅最后一个流水线阶段计算并归约损失。
- **日志集成**：自动将损失分量记录到 TensorBoard/WandB。

有关实现细节，请参阅 {py:func}`bridge.training.train.train_step` 和 {py:func}`bridge.training.losses.masked_token_loss` 作为示例。

## 自定义

### 何时需要自定义

当您需要以下情况时，可以自定义前向步骤函数：

- **自定义损失函数**：超越标准的语言建模损失（例如，添加正则化、多目标训练）。
- **多任务学习**：在多个任务上同时训练模型，每个任务有不同的损失分量。
- **自定义数据处理**：针对特定领域数据格式的专用批次预处理。
- **额外指标**：在训练期间计算额外的评估指标。
- **模型特定逻辑**：针对自定义模型架构或训练过程的特殊处理。