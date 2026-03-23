# Gemma 3

[Google 的 Gemma 3](https://huggingface.co/collections/google/gemma-3-release) 是一个轻量级、最先进的开源模型系列，基于创建 Gemini 模型所用的相同研究和技术构建。Gemma 3 架构建立在 Transformer 解码器框架之上，并进行了增强，包括使用 RMSNorm 的预归一化、GeGLU 激活函数、旋转位置嵌入（RoPE）以及混合注意力模式（滑动窗口和全局注意力）。

Gemma 3 模型专为广泛的文本生成任务而设计，并提供多种尺寸以适应不同的计算预算。

Gemma 系列模型通过 Bridge 系统提供支持，具有自动检测的配置和权重映射。

## 可用模型

### 纯文本模型
- **Gemma 3 1B** (`google/gemma-3-1b-it`): 紧凑的 10 亿参数模型，针对效率进行了优化
  - 26 层，1152 隐藏层大小
  - 8 个注意力头，2 个查询组（GQA）
  - 序列长度：131,072 个令牌
  - 适用于单 GPU 部署

所有模型均支持 131,072 个令牌的序列长度，并使用混合注意力模式（滑动窗口 + 全局）。

## 模型架构特性

Gemma 3 引入了多项架构创新：

- **混合注意力模式**：在全局注意力和局部滑动窗口注意力之间交替，以实现高效的长上下文处理
- **GeGLU 激活函数**：使用带有 GELU 激活的门控线性单元以提高性能
- **RMSNorm**：无需均值中心化的层归一化，计算速度更快
- **旋转嵌入**：为局部和全局注意力层提供独立的 RoPE 配置
  - 局部注意力：使用滑动窗口，旋转基数为 10,000
  - 全局注意力：扩展的旋转基数，以获得更好的长程依赖关系

## 使用 🤗 Hugging Face 进行转换

### 加载 HF → Megatron
```python
from megatron.bridge import AutoBridge

# 示例：Gemma 3 1B
bridge = AutoBridge.from_hf_pretrained("google/gemma-3-1b-it")
provider = bridge.to_megatron_provider()

# 在实例化模型之前配置并行度
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 导入 HF → Megatron
要将 HF 模型导入到您期望的 Megatron 路径：
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model google/gemma-3-1b-it \
--megatron-path /models/gemma-3-1b-it
```

### 导出 Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model google/gemma-3-1b-it \
--megatron-path /results/gemma3_1b/checkpoints/iter_00001000 \
--hf-path ./gemma3-hf-export
```

### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
--hf_model_path google/gemma-3-1b-it \
--megatron_model_path /models/gemma-3-1b-it \
--prompt "What is artificial intelligence?" \
--max_new_tokens 100
```

注意：
- `--megatron_model_path` 是可选的。如果未指定，脚本将先转换模型，然后运行前向传播。

## 配方

参见：[bridge.recipes.gemma](../../apidocs/bridge/bridge.recipes.gemma.md)

### 可用配方

- **预训练配方**：
  - `gemma3_1b_pretrain_config`: Gemma 3 1B 的预训练

- **SFT 配方**：
  - `gemma3_1b_sft_config`: Gemma 3 1B 的完整 SFT

- **PEFT 配方**（LoRA, DoRA）：
  - `gemma3_1b_peft_config`: Gemma 3 1B 的 PEFT

**开始训练前**，请确保设置了以下环境变量：
- `SAVE_DIR`: 检查点和日志保存目录
- `HF_TOKEN`: 用于从 HF Hub 下载模型（如果需要）
- `HF_HOME`: （可选）避免重复下载模型和数据集
- `WANDB_API_KEY`: （可选）启用 WandB 日志记录

### 并行度配置

| 模型 | 模式 | TP | PP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|------------|----------|
| **Gemma 3 1B** | 预训练 | 1 | 1 | 8 | 预训练（单节点） |
| **Gemma 3 1B** | 完整 SFT | 1 | 1 | 8 | 完整监督微调 |
| **Gemma 3 1B** | LoRA/DoRA | 1 | 1 | 8 | PEFT 微调（单节点） |

### 预训练示例

```python
from megatron.bridge.recipes.gemma import gemma3_1b_pretrain_config

config = gemma3_1b_pretrain_config(
    name="gemma3_1b_pretrain",
    data_paths=["path/to/data"],
    train_iters=100000,
    global_batch_size=256,
    # 自动使用 TP=1, PP=1 (8 GPUs)
)
```

### 微调示例

#### 完整微调

```python
from megatron.bridge.recipes.gemma import gemma3_1b_sft_config

config = gemma3_1b_sft_config(
    name="gemma3_1b_full_finetune",
    pretrained_checkpoint="/models/gemma-3-1b-it",
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # 自动使用 TP=1, PP=1 (8 GPUs)
)
```

#### LoRA 微调

```python
from megatron.bridge.recipes.gemma import gemma3_1b_peft_config

config = gemma3_1b_peft_config(
    name="gemma3_1b_lora_finetune",
    pretrained_checkpoint="/models/gemma-3-1b-it",
    peft_scheme="lora",  # 或 "dora"
```

train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # 自动使用 TP=1, PP=1 (8 个 GPU)
)
```

### 命令行训练

**全量微调：**
```bash
torchrun --nproc-per-node=8 run/run_recipe.py \
  --pretrained-checkpoint /models/gemma-3-1b-it \
  --recipe gemma3_1b_sft_config \
  train.global_batch_size=64 \
  train.train_iters=1000 \
  checkpoint.save=$SAVE_DIR/gemma3_1b_finetune
```

**LoRA 微调：**
```bash
torchrun --nproc-per-node=8 run/run_recipe.py \
  --pretrained-checkpoint /models/gemma-3-1b-it \
  --recipe gemma3_1b_peft_config \
  --peft_scheme lora \
  train.global_batch_size=128 \
  checkpoint.save=$SAVE_DIR/gemma3_1b_lora
```

## 示例
- 检查点导入/导出：[examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- 生成文本（HF→Megatron）：[examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Hugging Face 模型卡片

- Gemma 3 1B：https://huggingface.co/google/gemma-3-1b-it

## 相关文档
- Gemma3 视觉语言模型：[Gemma 3 VL](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/vlm/gemma3_vl/README.md)
- 配方使用：[配方使用](../../recipe-usage.md)
- 自定义训练配方配置：[配置概述](../../training/config-container-overview.md)
- 训练入口点：[入口点](../../training/entry-points.md)