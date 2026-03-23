# CPU 卸载

## 概述

Megatron Bridge 中的 CPU 卸载功能通过将激活值和未使用的权重卸载到 CPU 存储来降低 GPU 的峰值内存使用量。Megatron Bridge 支持在 Transformer 层级别进行卸载，允许用户指定其语言模型中需要 CPU 卸载的 Transformer 层数量。在前向传播过程中，Megatron Bridge 会在最佳时机卸载激活值，并在反向传播过程中根据需要重新加载它们。

## 特性

- 通过高效管理激活内存，支持训练长序列模型
- 通过卸载激活内存，实现每个 GPU 的高批次大小
- 在卸载和重新加载过程中，实现计算与数据传输（Host2Device 和 Device2Host）的重叠

## 配置

CPU 卸载通过模型提供者参数进行配置：

```python
from megatron.bridge.models import GPTModelProvider

# 基本 CPU 卸载配置
model_config = GPTModelProvider(
    # 模型架构
    hidden_size=4096,
    num_layers=32,
    
    # CPU 卸载设置
    cpu_offloading=True,              # 启用 CPU 卸载
    cpu_offloading_num_layers=16,     # 要卸载的层数 (0 到 num_layers-1)
    cpu_offloading_activations=True,  # 卸载激活值
    cpu_offloading_weights=True,      # 卸载权重
    
    # ... 其他模型参数
)
```

### 配置参数

- **`cpu_offloading`**: 设置为 `True` 以启用 CPU 卸载
- **`cpu_offloading_num_layers`**: 要卸载的 Transformer 层数（值在 0 到总层数减一之间）
- **`cpu_offloading_activations`**: 是否将激活值卸载到 CPU 内存（默认：`True`）
- **`cpu_offloading_weights`**: 是否将未使用的权重卸载到 CPU 内存（默认：`False`）
- **`cpu_offloading_double_buffering`**: 在从 CPU 重新加载激活值时启用跨层双缓冲（默认：`False`）

### 卸载策略

您可以根据内存需求配置不同的卸载组合：

#### 仅卸载激活值
```python
model_config = GPTModelProvider(
    cpu_offloading=True,
    cpu_offloading_num_layers=8,
    cpu_offloading_activations=True,   # 卸载激活值
    cpu_offloading_weights=False,      # 权重保留在 GPU 上
)
```

#### 仅卸载权重
```python
model_config = GPTModelProvider(
    cpu_offloading=True,
    cpu_offloading_num_layers=8,
    cpu_offloading_activations=False,  # 激活值保留在 GPU 上
    cpu_offloading_weights=True,       # 卸载权重
)
```

#### 同时卸载激活值和权重
```python
model_config = GPTModelProvider(
    cpu_offloading=True,
    cpu_offloading_num_layers=8,
    cpu_offloading_activations=True,   # 卸载激活值
    cpu_offloading_weights=True,       # 卸载权重
)
```