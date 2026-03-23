# Qwen

[Qwen](https://huggingface.co/Qwen) 是由阿里云开发的一系列大语言模型，包括稠密模型（Qwen2、Qwen2.5、Qwen3）和混合专家模型（Qwen3 MoE、Qwen3-Next）。这些模型采用了 QK 层归一化（QK layernorm）、门控增量网络（Gated-Delta Networks）和零中心 RMSNorm（Zero-Centered RMSNorm）等创新技术，以提高训练稳定性和性能。

Qwen 系列模型通过 Bridge 系统得到支持，该系统能够自动检测配置并进行权重映射。

## 可用模型

Megatron Bridge 支持以下 Qwen 模型变体：

### 稠密模型
- **Qwen2**: 0.5B, 1.5B, 7B, 72B
- **Qwen2.5**: 0.5B, 1.5B, 7B, 14B, 32B, 72B
- **Qwen3**: 0.6B, 1.7B, 4B, 8B, 14B, 32B

### MoE 模型
- **Qwen3 MoE**: 30B (激活 3B), 235B (激活 22B)
- **Qwen3-Next**: 80B (激活 3B)

## 模型架构特性

### 通用特性
- **预归一化（Pre-normalization）**: 每个 Transformer 子层前使用 RMSNorm
- **SwiGLU 激活函数**: 前馈网络中使用门控线性单元
- **旋转位置编码（Rotary Positional Embeddings, RoPE）**: 相对位置编码
- **分组查询注意力（Grouped Query Attention, GQA）**: 内存高效的注意力机制

### Qwen3-Next 特有特性
- **门控增量网络（Gated-Delta Networks）**: 用于改进学习的高级门控机制
- **零中心 RMSNorm（Zero-Centered RMSNorm）**: 用于训练稳定性的中心化归一化
- **多令牌预测（Multi-Token Prediction, MTP）**: 辅助训练目标

### Qwen3 特有特性
- **QK 层归一化（QK Layernorm）**: 在查询和键投影上应用层归一化
- **QK 层归一化权重衰减（QK Layernorm Weight Decay）**: 训练期间应用的权重衰减

### Qwen2 特有特性
- **QKV 中的偏置项（Bias in QKV）**: 查询、键、值投影中的偏置项

---

## Qwen3-Next

### 使用 🤗 Hugging Face 进行转换

#### 加载 HF → Megatron

```python
from megatron.bridge import AutoBridge

# 示例：Qwen3-Next-80B-A3B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
provider = bridge.to_megatron_provider()

# 可选地在实例化模型前配置并行策略
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 8
provider.expert_model_parallel_size = 16

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

#### 从 HF 导入检查点

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --megatron-path /checkpoints/qwen3_next_80b_megatron
```

#### 导出 Megatron → HF

```python
from megatron.bridge import AutoBridge

# 从 HF 模型 ID 加载 bridge
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

# 将训练好的 Megatron 检查点导出为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/qwen3_next_80b/checkpoints/iter_0000500",
    hf_path="/exports/qwen3_next_80b_hf",
)
```

#### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --megatron_model_path /checkpoints/qwen3_next_80b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2 \
  --pp 8 \
  --ep 16
```

### 配方

#### 可用配方
- `qwen3_next_80b_a3b_pretrain_config`: Qwen3-Next-80B-A3B 的预训练配置
- `qwen3_next_80b_a3b_sft_config`: Qwen3-Next-80B-A3B 的微调配置（仅全参数微调）

#### 并行配置

| 模型 | 模式 | TP | PP | EP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|----|-----------:|----------|
| **Qwen3-Next-80B** | 预训练 | 2 | 8 | 16 | 256 | 预训练 (32 个节点) |
| **Qwen3-Next-80B** | 全参数微调 | 2 | 8 | 16 | 256 | 全监督微调 (32 个节点) |

#### 预训练示例

```python
from megatron.bridge.recipes.qwen import qwen3_next_80b_a3b_pretrain_config

config = qwen3_next_80b_a3b_pretrain_config(
    name="qwen3_next_80b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen3_next_80b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # 自动使用 TP=2, PP=8, EP=16 (256 GPUs)
)
```

#### 微调示例

```python
from megatron.bridge.recipes.qwen import qwen3_next_80b_a3b_sft_config

config = qwen3_next_80b_a3b_sft_config(
    name="qwen3_next_80b_full_sft",
    pretrained_checkpoint="/results/qwen3_next_80b/checkpoints/iter_0500000",
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # 自动使用 TP=2, PP=8, EP=16 (256 GPUs)
)
```

**注意**: Qwen3-Next 模型目前不支持 PEFT（LoRA/DoRA）微调。

### Hugging Face 模型卡片

- Qwen3-Next-80B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3-Next-80B-A3B-Thinking: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking

---

## Qwen3 MoE

### 使用 🤗 Hugging Face 进行转换

#### 加载 HF → Megatron

```python
from megatron.bridge import AutoBridge

# 示例：Qwen3-30B-A3B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-30B-A3B")
provider = bridge.to_megatron_provider()
```

# 在实例化模型前可选配置并行度
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 8

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

#### 从 HF 导入检查点

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-30B-A3B \
  --megatron-path /checkpoints/qwen3_30b_a3b_megatron
```

#### 导出 Megatron → HF

```python
from megatron.bridge import AutoBridge

# 从 HF 模型 ID 加载桥接器
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-30B-A3B")

# 将训练好的 Megatron 检查点导出为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/qwen3_30b_a3b/checkpoints/iter_0000500",
    hf_path="/exports/qwen3_30b_a3b_hf",
)
```

#### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path Qwen/Qwen3-30B-A3B \
  --megatron_model_path /checkpoints/qwen3_30b_a3b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --ep 8
```

### 配方

#### 可用配方
- `qwen3_30b_a3b_pretrain_config`: Qwen3-30B-A3B 的预训练（300 亿参数，30 亿激活）
- `qwen3_235b_a22b_pretrain_config`: Qwen3-235B-A22B 的预训练（2350 亿参数，220 亿激活）
- `qwen3_30b_a3b_sft_config`: Qwen3-30B-A3B 的完整 SFT
- `qwen3_30b_a3b_peft_config`: Qwen3-30B-A3B 的 PEFT（LoRA, DoRA）
- `qwen3_235b_a22b_sft_config`: Qwen3-235B-A22B 的完整 SFT
- `qwen3_235b_a22b_peft_config`: Qwen3-235B-A22B 的 PEFT（LoRA, DoRA）

#### 并行度配置

| 模型 | 模式 | TP | PP | EP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|----|-----------:|----------|
| **Qwen3-30B-A3B** | 预训练 | 1 | 1 | 8 | 8 | 预训练（单节点） |
| **Qwen3-30B-A3B** | 完整 SFT | 1 | 1 | 8 | 8 | 完整监督微调 |
| **Qwen3-30B-A3B** | LoRA/DoRA | 1 | 1 | 8 | 8 | PEFT 微调（单节点） |
| **Qwen3-235B-A22B** | 预训练 | 2 | 8 | 32 | 512 | 预训练（64 节点） |
| **Qwen3-235B-A22B** | 完整 SFT | 2 | 8 | 32 | 512 | 完整监督微调（64 节点） |
| **Qwen3-235B-A22B** | LoRA/DoRA | 2 | 8 | 32 | 512 | PEFT 微调（64 节点） |

#### 预训练示例

**Qwen3-30B-A3B:**

```python
from megatron.bridge.recipes.qwen import qwen3_30b_a3b_pretrain_config

config = qwen3_30b_a3b_pretrain_config(
    name="qwen3_30b_a3b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen3_30b_a3b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # 自动使用 TP=1, PP=1, EP=8 (8 GPUs)
)
```

**Qwen3-235B-A22B**

```python
from megatron.bridge.recipes.qwen import qwen3_235b_a22b_pretrain_config

config = qwen3_235b_a22b_pretrain_config(
    name="qwen3_235b_a22b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen3_235b_a22b",
    train_iters=500_000,
    global_batch_size=4096,
    seq_length=4096,
    # 自动使用 TP=2, PP=8, EP=32 (512 GPUs)
)
```

#### 微调示例

**完整微调 (30B):**

```python
from megatron.bridge.recipes.qwen import qwen3_30b_a3b_sft_config

config = qwen3_30b_a3b_sft_config(
    name="qwen3_30b_a3b_full_sft",
    pretrained_checkpoint="/results/qwen3_30b_a3b/checkpoints/iter_0500000",
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # 自动使用 TP=1, PP=1, EP=8 (8 GPUs)
)
```

**LoRA 微调 (30B):**

```python
from megatron.bridge.recipes.qwen import qwen3_30b_a3b_peft_config

config = qwen3_30b_a3b_peft_config(
    name="qwen3_30b_a3b_lora",
    pretrained_checkpoint="/results/qwen3_30b_a3b/checkpoints/iter_0500000",
    peft_scheme="lora",  # 或 "dora"
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # 自动使用 TP=1, PP=1, EP=8 (8 GPUs)
)
```

### Hugging Face 模型卡片

- Qwen3-30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen3-235B-A22B: https://huggingface.co/Qwen/Qwen3-235B-A22B

---

## Qwen3

### 使用 🤗 Hugging Face 进行转换

#### 加载 HF → Megatron

```python
from megatron.bridge import AutoBridge

# 示例：Qwen3-8B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-8B")
provider = bridge.to_megatron_provider()

# 在实例化模型前可选配置并行度
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

#### 从 HF 导入检查点

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-8B \
  --megatron-path /checkpoints/qwen3_8b_megatron
```

#### 导出 Megatron → HF

```python
from megatron.bridge import AutoBridge

# 从 HF 模型 ID 加载桥接器
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-8B")

# 将训练好的 Megatron 检查点导出为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/qwen3_8b/checkpoints/iter_0000500",

hf_path="/exports/qwen3_8b_hf",
)
```

#### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path Qwen/Qwen3-8B \
  --megatron_model_path /checkpoints/qwen3_8b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2
```

### 配方

#### 可用配方
- **预训练配方**：`qwen3_600m_pretrain_config`, `qwen3_1p7b_pretrain_config`, `qwen3_4b_pretrain_config`, `qwen3_8b_pretrain_config`, `qwen3_14b_pretrain_config`, `qwen3_32b_pretrain_config`
- **监督微调配方**：`qwen3_600m_sft_config`, `qwen3_1p7b_sft_config`, `qwen3_4b_sft_config`, `qwen3_8b_sft_config`, `qwen3_14b_sft_config`, `qwen3_32b_sft_config`
- **参数高效微调配方**（LoRA, DoRA）：`qwen3_600m_peft_config`, `qwen3_1p7b_peft_config`, `qwen3_4b_peft_config`, `qwen3_8b_peft_config`, `qwen3_14b_peft_config`, `qwen3_32b_peft_config`

#### 并行配置

| 模型 | 模式 | TP | PP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|------------|----------|
| **Qwen3 (0.6B-4B)** | 预训练 | 1 | 1 | 8 | 预训练（单节点） |
| **Qwen3 (0.6B-4B)** | 全量监督微调 | 1 | 1 | 8 | 全量监督微调 |
| **Qwen3 (0.6B-4B)** | LoRA/DoRA | 1 | 1 | 8 | 参数高效微调（单节点） |
| **Qwen3 (8B-14B)** | 预训练 | 2 | 1 | 16 | 预训练（2节点） |
| **Qwen3 (8B-14B)** | 全量监督微调 | 2 | 1 | 16 | 全量监督微调（2节点） |
| **Qwen3 (8B-14B)** | LoRA/DoRA | 1 | 1 | 8 | 参数高效微调（单节点！） |
| **Qwen3-32B** | 预训练 | 4 | 1 | 32 | 预训练（4节点） |
| **Qwen3-32B** | 全量监督微调 | 4 | 1 | 32 | 全量监督微调（4节点） |
| **Qwen3-32B** | LoRA/DoRA | 2 | 1 | 16 | 参数高效微调（2节点） |

#### 预训练示例

```python
from megatron.bridge.recipes.qwen import qwen3_8b_pretrain_config

config = qwen3_8b_pretrain_config(
    name="qwen3_8b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen3_8b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # 自动使用 TP=2, PP=1 (16 GPUs)
)
```

#### 微调示例

**全量微调 (8B)：**

```python
from megatron.bridge.recipes.qwen import qwen3_8b_sft_config

config = qwen3_8b_sft_config(
    name="qwen3_8b_full_sft",
    pretrained_checkpoint="/results/qwen3_8b/checkpoints/iter_0500000",
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # 自动使用 TP=2, PP=1 (16 GPUs)
)
```

**LoRA 微调 (8B)：**

```python
from megatron.bridge.recipes.qwen import qwen3_8b_peft_config

config = qwen3_8b_peft_config(
    name="qwen3_8b_lora",
    pretrained_checkpoint="/results/qwen3_8b/checkpoints/iter_0500000",
    peft_scheme="lora",  # 或 "dora"
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # 自动使用 TP=1, PP=1 (8 GPUs)
)
```

### Hugging Face 模型卡片

- Qwen3 系列：https://huggingface.co/collections/Qwen/qwen3

---

## Qwen2 / Qwen2.5

### 使用 🤗 Hugging Face 进行转换

#### 加载 HF → Megatron

```python
from megatron.bridge import AutoBridge

# 示例：Qwen2.5-7B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-7B")
provider = bridge.to_megatron_provider()

# 可选：在实例化模型前配置并行策略
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

#### 从 HF 导入检查点

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen2.5-7B \
  --megatron-path /checkpoints/qwen25_7b_megatron
```

#### 导出 Megatron → HF

```python
from megatron.bridge import AutoBridge

# 从 HF 模型 ID 加载桥接器
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-7B")

# 将训练好的 Megatron 检查点导出为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/qwen25_7b/checkpoints/iter_0000500",
    hf_path="/exports/qwen25_7b_hf",
)
```

#### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path Qwen/Qwen2.5-7B \
  --megatron_model_path /checkpoints/qwen25_7b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2
```

### 配方

#### 可用配方
- **Qwen2 预训练**：`qwen2_500m_pretrain_config`, `qwen2_1p5b_pretrain_config`, `qwen2_7b_pretrain_config`, `qwen2_72b_pretrain_config`
- **Qwen2.5 预训练**：`qwen25_500m_pretrain_config`, `qwen25_1p5b_pretrain_config`, `qwen25_7b_pretrain_config`, `qwen25_14b_pretrain_config`, `qwen25_32b_pretrain_config`, `qwen25_72b_pretrain_config`
- **Qwen2 监督微调**：`qwen2_500m_sft_config`, `qwen2_1p5b_sft_config`, `qwen2_7b_sft_config`, `qwen2_72b_sft_config`
- **Qwen2 参数高效微调**（LoRA, DoRA）：`qwen2_500m_peft_config`, `qwen2_1p5b_peft_config`, `qwen2_7b_peft_config`, `qwen2_72b_peft_config`

- **Qwen2.5 SFT**：`qwen25_500m_sft_config`、`qwen25_1p5b_sft_config`、`qwen25_7b_sft_config`、`qwen25_14b_sft_config`、`qwen25_32b_sft_config`、`qwen25_72b_sft_config`
- **Qwen2.5 PEFT** (LoRA, DoRA)：`qwen25_500m_peft_config`、`qwen25_1p5b_peft_config`、`qwen25_7b_peft_config`、`qwen25_14b_peft_config`、`qwen25_32b_peft_config`、`qwen25_72b_peft_config`

#### 并行配置

| 模型 | 模式 | TP | PP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|------------|----------|
| **Qwen2/2.5 (0.5B-1.5B)** | 预训练 | 1 | 1 | 8 | 预训练（单节点） |
| **Qwen2/2.5 (0.5B-1.5B)** | 全量 SFT | 1 | 1 | 8 | 全量监督微调 |
| **Qwen2/2.5 (0.5B-1.5B)** | LoRA/DoRA | 1 | 1 | 8 | PEFT 微调（单节点） |
| **Qwen2/2.5 (7B-14B)** | 预训练 | 2 | 1 | 16 | 预训练（2 节点） |
| **Qwen2/2.5 (7B-14B)** | 全量 SFT | 2 | 1 | 16 | 全量监督微调（2 节点） |
| **Qwen2/2.5 (7B-14B)** | LoRA/DoRA | 1 | 1 | 8 | PEFT 微调（单节点！） |
| **Qwen2.5-32B** | 预训练 | 4 | 1 | 32 | 预训练（4 节点） |
| **Qwen2.5-32B** | 全量 SFT | 4 | 1 | 32 | 全量监督微调（4 节点） |
| **Qwen2.5-32B** | LoRA/DoRA | 2 | 1 | 16 | PEFT 微调（2 节点） |
| **Qwen2/2.5-72B** | 预训练 | 8 | 1 | 64 | 预训练（8 节点） |
| **Qwen2/2.5-72B** | 全量 SFT | 8 | 1 | 64 | 全量监督微调（8 节点） |
| **Qwen2/2.5-72B** | LoRA/DoRA | 4 | 1 | 32 | PEFT 微调（4 节点） |

#### 预训练示例

```python
from megatron.bridge.recipes.qwen import qwen25_7b_pretrain_config

config = qwen25_7b_pretrain_config(
    name="qwen25_7b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen25_7b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # 自动使用 TP=2, PP=1 (16 GPUs)
)
```

#### 微调示例

**全量微调 (7B)：**

```python
from megatron.bridge.recipes.qwen import qwen25_7b_sft_config

config = qwen25_7b_sft_config(
    name="qwen25_7b_full_sft",
    pretrained_checkpoint="/results/qwen25_7b/checkpoints/iter_0500000",
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # 自动使用 TP=2, PP=1 (16 GPUs)
)
```

**LoRA 微调 (7B)：**

```python
from megatron.bridge.recipes.qwen import qwen25_7b_peft_config

config = qwen25_7b_peft_config(
    name="qwen25_7b_lora",
    pretrained_checkpoint="/results/qwen25_7b/checkpoints/iter_0500000",
    peft_scheme="lora",  # 或 "dora"
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # 自动使用 TP=1, PP=1 (8 GPUs)
)
```

### Hugging Face 模型卡片

- Qwen2 合集：https://huggingface.co/collections/Qwen/qwen2
- Qwen2.5 合集：https://huggingface.co/collections/Qwen/qwen25

---

## 相关文档
- 配方使用：[Recipe usage](../../recipe-usage.md)
- 自定义训练配方配置：[Configuration overview](../../training/config-container-overview.md)
- 训练入口点：[Entry points](../../training/entry-points.md)