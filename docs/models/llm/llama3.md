# Llama 3

[Meta 的 Llama](https://www.llama.com/models/llama-3/) 建立在通用的 Transformer 解码器框架之上，并加入了一些关键改进，例如预归一化（pre-normalization）、SwiGLU 激活函数和旋转位置嵌入（Rotary Positional Embeddings, RoPE）。更多信息请参阅配套论文 ["Llama: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)。Llama 提供了多种模型尺寸，可以满足各种推理预算的需求。

Llama 系列模型通过 Bridge 系统得到支持，该系统能够自动检测配置和权重映射。

## 可用模型

Megatron Bridge 支持以下 Llama 模型变体：

- **Llama 3.2**: 1B, 3B
- **Llama 3**: 8B, 70B (包含 8K、16K、64K、128K 上下文长度变体)
- **Llama 3.1**: 8B, 70B, 405B (包含 128K 上下文长度)

所有模型都支持全参数更新的预训练和微调，以及 PEFT 方法（LoRA、DoRA）。

## 模型架构特性

- **预归一化（Pre-normalization）**: 在每个 Transformer 子层之前使用 RMSNorm，以提高训练稳定性
- **SwiGLU 激活函数（SwiGLU Activation）**: 在前馈网络中使用门控线性单元
- **旋转位置嵌入（Rotary Positional Embeddings, RoPE）**: 通过旋转矩阵进行相对位置编码
- **分组查询注意力（Grouped Query Attention, GQA）**: 内存高效的注意力机制（70B+ 模型）
- **扩展上下文（Extended Context）**: 原生支持长达 128K 个令牌的长序列（Llama 3.1）

## 使用 🤗 Hugging Face 进行转换

### 加载 HF → Megatron

```python
from megatron.bridge import AutoBridge

# 示例：Llama 3.1 8B
bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3.1-8B")
provider = bridge.to_megatron_provider()

# 可选地在实例化模型前配置并行策略
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 从 HF 导入检查点

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model meta-llama/Meta-Llama-3.1-8B \
  --megatron-path /checkpoints/llama31_8b_megatron
```

### 导出 Megatron → HF

```python
from megatron.bridge import AutoBridge

# 从 HF 模型 ID 加载 bridge
bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3.1-8B")

# 将训练/微调后的 Megatron 检查点导出为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/llama31_8b/checkpoints/iter_0000500",
    hf_path="/exports/llama31_8b_hf",
)
```

### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path meta-llama/Meta-Llama-3.1-8B \
  --megatron_model_path /checkpoints/llama31_8b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2
```

更多详情，请参阅 [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## 配方

参见：[bridge.recipes.llama.llama3](../../apidocs/bridge/bridge.recipes.llama.llama3.md)

### 可用配方

- **预训练配方（Pretrain recipes）**:
  - `llama32_1b_pretrain_config`, `llama32_3b_pretrain_config`: Llama 3.2 (1B, 3B)
  - `llama3_8b_pretrain_config`: Llama 3 8B，8K 上下文
  - `llama3_8b_16k_pretrain_config`, `llama3_8b_64k_pretrain_config`, `llama3_8b_128k_pretrain_config`: Llama 3 8B，扩展上下文 (16K/64K/128K)
  - `llama3_8b_low_precision_pretrain_config`: Llama 3 8B，低精度 (FP8/MXFP8/NVFP4)
  - `llama3_70b_pretrain_config`, `llama3_70b_16k_pretrain_config`, `llama3_70b_64k_pretrain_config`: Llama 3 70B (8K/16K/64K 上下文)
  - `llama31_8b_pretrain_config`, `llama31_70b_pretrain_config`, `llama31_405b_pretrain_config`: Llama 3.1 (8B/70B/405B, 128K 上下文)

- **监督微调配方（SFT recipes）**:
  - `llama32_1b_sft_config`, `llama32_3b_sft_config`: Llama 3.2 全参数 SFT
  - `llama3_8b_sft_config`, `llama31_8b_sft_config`: Llama 3/3.1 8B 全参数 SFT
  - `llama3_70b_sft_config`, `llama31_70b_sft_config`: Llama 3/3.1 70B 全参数 SFT
  - `llama31_405b_sft_config`: Llama 3.1 405B 全参数 SFT

- **参数高效微调配方（PEFT recipes）** (LoRA, DoRA):
  - `llama32_1b_peft_config`, `llama32_3b_peft_config`: Llama 3.2 PEFT
  - `llama3_8b_peft_config`, `llama31_8b_peft_config`: Llama 3/3.1 8B PEFT
  - `llama3_70b_peft_config`, `llama31_70b_peft_config`: Llama 3/3.1 70B PEFT
  - `llama31_405b_peft_config`: Llama 3.1 405B PEFT

### 并行配置

#### Llama 3.2 (1B, 3B)
| 模型 | 模式 | TP | PP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|------------|----------|
| **1B / 3B** | 预训练（Pretrain） | 1 | 1 | 8 | 预训练（单节点） |
| **1B / 3B** | 全参数 SFT（Full SFT） | 1 | 1 | 8 | 全参数监督微调 |
| **1B / 3B** | LoRA/DoRA | 1 | 1 | 8 | PEFT 微调 |

#### Llama 3 / 3.1 (8B)
| 模型 | 模式 | TP | PP | CP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|----|-----------:|----------|
| **8B** | 预训练（Pretrain） | 1 | 1 | 2 | 16 | 预训练 |
| **8B** | 全参数 SFT（Full SFT） | 2 | 1 | 1 | 16 | 全参数监督微调 |

| **8B** | LoRA/DoRA | 1 | 1 | 1 | 8 | PEFT 微调（单节点） |

#### Llama 3 / 3.1 (70B)
| 模型 | 模式 | TP | PP | VP | CP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|----|----|------------|----------|
| **70B** | 预训练 | 4 | 4 | 5 | 2 | 64 | 预训练 |
| **70B** | 全量 SFT | 8 | 4 | - | 1 | 256 | 全量监督微调（32 节点） |
| **70B** | LoRA/DoRA | 8 | 1 | - | 1 | 8 | PEFT 微调（单节点！） |

#### Llama 3.1 (405B)
| 模型 | 模式 | TP | PP | VP | CP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|----|----|------------|----------|
| **405B** | 预训练 | 8 | 8 | 2 | 4 | 512 | 预训练（64 节点） |
| **405B** | 全量 SFT | 8 | 16 | - | 1 | 2048 | 全量监督微调（256 节点） |
| **405B** | LoRA/DoRA | 4 | 8 | 8 | 1 | 256 | PEFT 微调（32 节点） |

**关键特性**：
- **上下文并行（Context Parallelism）**：为长上下文训练（16K/64K/128K 变体）启用
- **序列并行（Sequence Parallel）**：默认在较大模型（70B+）上启用以提高内存效率
- **低精度训练**：8B 模型支持 FP8、MXFP8、NVFP4 选项
- **虚拟管道（Virtual Pipeline）**：70B 和 405B 模型支持 VP 并行

### 预训练示例

```python
from megatron.bridge.recipes.llama import llama3_8b_pretrain_config

config = llama3_8b_pretrain_config(
    name="llama3_8b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/llama3_8b",
    train_iters=500_000,
    global_batch_size=512,
    seq_length=8192,
    # 自动使用 TP=1, PP=1, CP=2 (16 GPUs)
)
```

### 微调示例

**开始微调前**，请确保设置以下环境变量：
- `SAVE_DIR`：检查点和日志保存目录
- `HF_TOKEN`：用于从 HF Hub 下载模型（如果需要）
- `HF_HOME`：（可选）避免重复下载模型和数据集
- `WANDB_API_KEY`：（可选）启用 WandB 日志记录

#### 全量微调（Llama 3 8B）

```python
from megatron.bridge.recipes.llama import llama3_8b_sft_config

cfg = llama3_8b_sft_config(
    name="llama3_8b_full_sft",
    pretrained_checkpoint="/results/llama3_8b/checkpoints/iter_0500000",
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # 自动使用 TP=2, PP=1 (16 GPUs)
)
```

#### LoRA 微调（8B）

```python
from megatron.bridge.recipes.llama import llama3_8b_peft_config

cfg = llama3_8b_peft_config(
    name="llama3_8b_lora",
    pretrained_checkpoint="/results/llama3_8b/checkpoints/iter_0500000",
    peft_scheme="lora",  # 或 "dora" 用于 DoRA
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # 自动使用 TP=1, PP=1 (8 GPUs)
)
```

#### LoRA 微调（70B）

```python
from megatron.bridge.recipes.llama import llama3_70b_peft_config

cfg = llama3_70b_peft_config(
    name="llama3_70b_lora",
    pretrained_checkpoint="/results/llama3_70b/checkpoints/iter_0500000",
    peft_scheme="lora",
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # 自动使用 TP=8, PP=1 (8 GPUs)
)
```

## Hugging Face 模型卡片与参考文献

### Hugging Face 模型卡片
- Llama 3.2 1B: https://huggingface.co/meta-llama/Llama-3.2-1B
- Llama 3.2 3B: https://huggingface.co/meta-llama/Llama-3.2-3B
- Llama 3 8B: https://huggingface.co/meta-llama/Meta-Llama-3-8B
- Llama 3 70B: https://huggingface.co/meta-llama/Meta-Llama-3-70B
- Llama 3.1 8B: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
- Llama 3.1 70B: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B
- Llama 3.1 405B: https://huggingface.co/meta-llama/Meta-Llama-3.1-405B

### 技术论文
- Llama: Open and Efficient Foundation Language Models: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- The Llama 3 Herd of Models: [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

## 相关文档
- 配方使用：[配方使用](../../recipe-usage.md)
- 自定义训练配方配置：[配置概述](../../training/config-container-overview.md)
- 训练入口点：[入口点](../../training/entry-points.md)