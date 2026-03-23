# Nemotron H 与 Nemotron Nano v2

[Nemotron H](https://huggingface.co/collections/nvidia/nemotron-h) 和 [Nemotron Nano v2](https://huggingface.co/collections/nvidia/nvidia-nemotron-v2) 是 **NVIDIA** 推出的 **混合 SSM-注意力模型** 系列，它们将 Mamba（状态空间模型）层与传统注意力层相结合。这些模型通过其混合架构，在保持计算效率的同时实现了强大的性能。

Nemotron H 系列包含参数量从 4B 到 56B、上下文长度为 8K 的模型，而 Nemotron Nano v2 模型（9B 和 12B）则针对边缘部署进行了优化，并支持扩展的 128K 上下文长度。

## 模型系列

### Nemotron H
- **4B**: 52 层，3072 隐藏大小，8K 上下文
- **8B**: 52 层，4096 隐藏大小，8K 上下文
- **47B**: 98 层，8192 隐藏大小，8K 上下文
- **56B**: 118 层，8192 隐藏大小，8K 上下文

### Nemotron Nano v2
- **9B**: 56 层，4480 隐藏大小，128K 上下文
- **12B**: 62 层，5120 隐藏大小，128K 上下文

所有模型都通过 Bridge 系统获得支持，并针对混合 SSM-注意力架构提供了专门的配置。

## 模型架构

### 所有模型的共同特性
- **架构**: 混合 SSM-注意力（Mamba + 多查询注意力）
- **SSM**: Mamba-2 选择性状态空间层
- **注意力**: 带有 QK LayerNorm 和 RoPE 的多查询注意力
- **激活函数**: Squared ReLU（FFN 中使用 SwiGLU）
- **归一化**: RMSNorm
- **位置嵌入**: RoPE（旋转位置嵌入）
- **混合模式**: 可配置的 Mamba（"M"）层和注意力（"*"）层的逐层混合

### Nemotron H 4B 规格
- **参数量**: 4B
- **层数**: 52（混合模式：`M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-`）
- **隐藏大小**: 3072
- **FFN 隐藏大小**: 12288
- **注意力头**: 32 个查询头，8 个键值组
- **KV 通道数**: 128
- **Mamba 头数**: 112
- **Mamba 头维度**: 64
- **Mamba 状态维度**: 128
- **上下文长度**: 8K 词元

### Nemotron H 8B 规格
- **参数量**: 8B
- **层数**: 52（混合模式：`M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-`）
- **隐藏大小**: 4096
- **FFN 隐藏大小**: 21504
- **注意力头**: 32 个查询头，8 个键值组
- **KV 通道数**: 128
- **Mamba 头数**: 128
- **Mamba 头维度**: 64
- **Mamba 状态维度**: 128
- **上下文长度**: 8K 词元

### Nemotron H 47B 规格
- **参数量**: 47B
- **层数**: 98
- **隐藏大小**: 8192
- **FFN 隐藏大小**: 30720
- **注意力头**: 64 个查询头，8 个键值组
- **KV 通道数**: 128
- **Mamba 头数**: 256
- **Mamba 头维度**: 64
- **Mamba 状态维度**: 256
- **上下文长度**: 8K 词元

### Nemotron H 56B 规格
- **参数量**: 56B
- **层数**: 118
- **隐藏大小**: 8192
- **FFN 隐藏大小**: 32768
- **注意力头**: 64 个查询头，8 个键值组
- **KV 通道数**: 128
- **Mamba 头数**: 256
- **Mamba 头维度**: 64
- **Mamba 状态维度**: 256
- **上下文长度**: 8K 词元

### Nemotron Nano 9B v2 规格
- **参数量**: 9B
- **层数**: 56（混合模式：`M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-`）
- **隐藏大小**: 4480
- **FFN 隐藏大小**: 15680
- **注意力头**: 40 个查询头，8 个键值组
- **KV 通道数**: 128
- **Mamba 头数**: 128
- **Mamba 头维度**: 80
- **Mamba 状态维度**: 128
- **上下文长度**: 128K 词元
- **词表大小**: 131,072

### Nemotron Nano 12B v2 规格
- **参数量**: 12B
- **层数**: 62（混合模式：`M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-`）
- **隐藏大小**: 5120
- **FFN 隐藏大小**: 20480
- **注意力头**: 40 个查询头，8 个键值组
- **KV 通道数**: 128
- **Mamba 头数**: 128
- **Mamba 头维度**: 80
- **Mamba 状态维度**: 128
- **上下文长度**: 128K 词元
- **词表大小**: 131,072

## 主要特性

### 混合 SSM-注意力架构
- **Mamba 层（M）**: 用于高效长程建模的状态空间模型层
- **注意力层（*）**: 用于复杂推理的标准多查询注意力层
- **可配置模式**: 每个模型都有一个预定义的混合模式，以平衡效率和性能

### 高级优化
- **Squared ReLU 激活函数**: 增强的非线性，以获得更好的梯度流
- **QK LayerNorm**: 对查询和键投影应用 LayerNorm，以提高训练稳定性
- **RoPE**: 基数为 10000 的旋转位置嵌入
- **多查询注意力**: 具有共享键值头的高效注意力机制
- **选择性状态空间**: 带有选择性门控的 Mamba-2 架构

### 扩展上下文（Nano v2）
- **128K 上下文窗口**: Nemotron Nano v2 模型支持高达 128K 词元
- **高效长程建模**: 针对长序列优化的混合架构

## 使用 🤗 Hugging Face 进行转换

### 加载 HF → Megatron

#### Nemotron H 模型
```python
from megatron.bridge import AutoBridge

# 示例：Nemotron H 8B

bridge = AutoBridge.from_hf_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
provider = bridge.to_megatron_provider()

# 在实例化模型之前配置并行策略
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1
provider.context_parallel_size = 1
provider.sequence_parallel = True

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)

# 其他模型：
# bridge = AutoBridge.from_hf_pretrained("nvidia/Nemotron-H-4B-Base-8K", trust_remote_code=True)
# bridge = AutoBridge.from_hf_pretrained("nvidia/Nemotron-H-47B-Base-8K", trust_remote_code=True)
# bridge = AutoBridge.from_hf_pretrained("nvidia/Nemotron-H-56B-Base-8K", trust_remote_code=True)
```

#### Nemotron Nano v2 模型
```python
from megatron.bridge import AutoBridge

# 示例：Nemotron Nano 9B v2
bridge = AutoBridge.from_hf_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", trust_remote_code=True)
provider = bridge.to_megatron_provider()

# 配置并行策略
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1
provider.context_parallel_size = 1
provider.sequence_parallel = True

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)

# 对于指令调优变体：
# bridge = AutoBridge.from_hf_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2", trust_remote_code=True)

# 对于 12B 模型：
# bridge = AutoBridge.from_hf_pretrained("nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base", trust_remote_code=True)
```

### 导出 Megatron → HF
```python
# 从 Megatron 检查点目录转换为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/nemotronh_8b/checkpoints/iter_0500000",
    hf_path="./nemotronh-8b-hf-export",
)
```

## 示例

- 检查点转换：[examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- 训练脚本：[examples/models/train_any_basic.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/train_any_basic.py)

## 微调配方

### Nemotron H 4B 微调

#### LoRA 微调
```python
from megatron.bridge.recipes.nemotronh import nemotronh_4b_peft_config

cfg = nemotronh_4b_peft_config(
    tokenizer_path="nvidia/Nemotron-H-4B-Base-8K",
    name="nemotronh_4b_lora",
    pretrained_checkpoint="path/to/nemotronh/4b/checkpoint",
    peft_scheme="lora",  # 或 "dora" 用于 DoRA
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
)
```

#### 全量监督微调 (SFT)
```python
from megatron.bridge.recipes.nemotronh import nemotronh_4b_sft_config

cfg = nemotronh_4b_sft_config(
    tokenizer_path="nvidia/Nemotron-H-4B-Base-8K",
    name="nemotronh_4b_sft",
    pretrained_checkpoint="path/to/nemotronh/4b/checkpoint",
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=5e-6,  # 全量 SFT 使用较低的学习率
)
```

### Nemotron H 8B 微调

```python
from megatron.bridge.recipes.nemotronh import nemotronh_8b_peft_config

# LoRA 微调
cfg = nemotronh_8b_peft_config(
    tokenizer_path="nvidia/Nemotron-H-8B-Base-8K",
    name="nemotronh_8b_lora",
    pretrained_checkpoint="path/to/nemotronh/8b/checkpoint",
    peft_scheme="lora",
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
)
```

### Nemotron H 47B 微调

```python
from megatron.bridge.recipes.nemotronh import nemotronh_47b_peft_config

# LoRA 微调（推荐用于 47B）
cfg = nemotronh_47b_peft_config(
    tokenizer_path="nvidia/Nemotron-H-47B-Base-8K",
    name="nemotronh_47b_lora",
    pretrained_checkpoint="path/to/nemotronh/47b/checkpoint",
    peft_scheme="lora",
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
) 
```

### Nemotron H 56B 微调

```python
from megatron.bridge.recipes.nemotronh import nemotronh_56b_peft_config

# LoRA 微调（推荐用于 56B）
cfg = nemotronh_56b_peft_config(
    tokenizer_path="nvidia/Nemotron-H-56B-Base-8K",
    name="nemotronh_56b_lora",
    pretrained_checkpoint="path/to/nemotronh/56b/checkpoint",
    peft_scheme="lora",
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
)
```

### Nemotron Nano 9B v2 微调

```python
from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_peft_config

# LoRA 微调
cfg = nemotron_nano_9b_v2_peft_config(
    tokenizer_path="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base",
    name="nano_9b_v2_lora",
    pretrained_checkpoint="path/to/nano/9b/v2/checkpoint",
    peft_scheme="lora",
    train_iters=1000,
    global_batch_size=128,
    seq_length=2048,  # 可使用高达 128K
    finetune_lr=1e-4,
)
```

### Nemotron Nano 12B v2 微调

```python
from megatron.bridge.recipes.nemotronh import nemotron_nano_12b_v2_peft_config

# LoRA 微调
cfg = nemotron_nano_12b_v2_peft_config(
    tokenizer_path="nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base",
    name="nano_12b_v2_lora",

pretrained_checkpoint="path/to/nano/12b/v2/checkpoint",
    peft_scheme="lora",
    train_iters=1000,
    global_batch_size=128,
    seq_length=2048,  # 可使用高达 128K
    finetune_lr=1e-4,
)
```

## 默认配置

### Nemotron H 模型

#### 4B - LoRA (1 节点, 8 GPU)
- TP=1, PP=1, CP=1, LR=1e-4
- 序列并行：False
- 精度：BF16 混合精度
- 针对单 GPU 微调优化

#### 4B - 全量 SFT (1 节点, 8 GPU)
- TP=1, PP=1, CP=1, LR=5e-6
- 序列并行：False
- 精度：BF16 混合精度

#### 8B - LoRA (1 节点, 8 GPU)
- TP=1, PP=1, CP=1, LR=1e-4
- 序列并行：False
- 精度：BF16 混合精度

#### 8B - 全量 SFT (1 节点, 8 GPU)
- TP=2, PP=1, CP=1, LR=5e-6
- 序列并行：True
- 精度：BF16 混合精度

#### 47B - LoRA (2+ 节点)
- TP=4, PP=1, CP=1, LR=1e-4
- 序列并行：False
- 精度：FP8 混合精度（推荐）

#### 47B - 全量 SFT (4+ 节点)
- TP=8, PP=1, CP=1, LR=5e-6
- 序列并行：True
- 精度：FP8 混合精度

#### 56B - LoRA (2+ 节点)
- TP=4, PP=1, CP=1, LR=1e-4
- 序列并行：False
- 精度：FP8 混合精度（推荐）

#### 56B - 全量 SFT (4+ 节点)
- TP=8, PP=1, CP=1, LR=5e-6
- 序列并行：True
- 精度：FP8 混合精度

### Nemotron Nano v2 模型

#### 9B - LoRA (1 节点, 8 GPU)
- TP=2, PP=1, CP=1, LR=1e-4
- 序列并行：True
- 精度：BF16 混合精度
- 上下文长度：最高 128K 词元

#### 9B - 全量 SFT (1 节点, 8 GPU)
- TP=2, PP=1, CP=1, LR=1e-4
- 序列并行：True
- 精度：BF16 混合精度

#### 12B - LoRA (2 节点, 16 GPU)
- TP=4, PP=1, CP=1, LR=1e-4
- 序列并行：True
- 精度：FP8 混合精度（推荐）
- 上下文长度：最高 128K 词元

#### 12B - 全量 SFT (2 节点, 16 GPU)
- TP=4, PP=1, CP=1, LR=1e-4
- 序列并行：True
- 精度：FP8 混合精度

## API 参考

### Nemotron H
- Nemotron H 配方：[bridge.recipes.nemotronh](../../apidocs/bridge/bridge.recipes.nemotronh.md)
- Nemotron H 模型提供者：[bridge.models.nemotronh](../../apidocs/bridge/bridge.models.nemotronh.md)

### Nemotron Nano v2
- Nemotron Nano v2 配方：[bridge.recipes.nemotronh.nemotron_nano_v2](../../apidocs/bridge/bridge.recipes.nemotronh.md)
- Nemotron Nano v2 模型提供者：[bridge.models.nemotronh.NemotronNanoModelProvider9Bv2](../../apidocs/bridge/bridge.models.nemotronh.md)

## 性能优化

### 内存效率
- **选择性重计算**：减少大型模型的激活内存
- **序列并行**：将序列维度分布到多个 GPU 上（8B+ 模型启用）
- **上下文并行**：支持超长序列（Nano v2）
- **手动垃圾回收**：积极的垃圾回收以稳定内存使用
- **精度感知优化器**：BF16/FP8 梯度配合 FP32 主权重

### 计算效率
- **Mamba-2 优化**：高效的选择性状态空间计算
- **混合架构**：Mamba 层和注意力层的平衡组合
- **平方 ReLU**：具有良好梯度特性的高效激活函数
- **RoPE 融合**：位置嵌入的可选优化
- **多查询注意力**：减少 KV 缓存内存和计算

### 混合模式优化
混合覆盖模式决定了哪些层使用 Mamba (M) 与注意力 (*)：
- **Mamba 层**：快速、内存高效，擅长处理长程依赖
- **注意力层**：更擅长复杂推理和多词元关系
- **最优模式**：基于大量实验，为每个模型大小预配置

## 管道并行布局

Nemotron H 模型支持多种 PP 配置及预定义布局：
- **PP=1**：无流水线并行（大多数配置的默认值）
- **PP=2**：支持对称层分割
- **PP=4**：支持大型模型（47B, 56B）
- **VP（虚拟流水线）**：支持以减少流水线气泡

## Hugging Face 模型卡片

### Nemotron H 模型
- **4B Base**: [nvidia/Nemotron-H-4B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-4B-Base-8K)
- **8B Base**: [nvidia/Nemotron-H-8B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K)
- **47B Base**: [nvidia/Nemotron-H-47B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-47B-Base-8K)
- **56B Base**: [nvidia/Nemotron-H-56B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-56B-Base-8K)

### Nemotron Nano v2 模型
- **9B Base**: [nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base)
- **9B Instruct**: [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- **12B Base**: [nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base)
- **12B Instruct**: [nvidia/NVIDIA-Nemotron-Nano-12B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2)

## 技术资源

### 研究论文
- **Nemotron 技术报告**: [arXiv:2508.14444](https://arxiv.org/abs/2508.14444)

- **Mamba-2**: [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)

## 相关文档

- 配方使用与自定义：[配方使用](../../recipe-usage.md)
- 训练配置：[配置概述](../../training/config-container-overview.md)
- 训练入口点：[入口点](../../training/entry-points.md)
- PEFT 方法（LoRA, DoRA）：[PEFT 指南](../../training/peft.md)