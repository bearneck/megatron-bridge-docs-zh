# Gemma 2

[Google 的 Gemma 2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) 是一个轻量级、开放模型系列，基于创建 Gemini 模型所用的相同研究和技术构建。Gemma 2 架构建立在 Transformer 解码器框架之上，并进行了增强，包括使用 RMSNorm 的预归一化、GeGLU 激活函数、旋转位置嵌入（RoPE）、注意力对数软上限和滑动窗口注意力。

Gemma 2 模型设计用于广泛的文本生成任务，并提供多种尺寸以适应不同的计算预算。

Gemma 系列模型通过 Bridge 系统得到支持，具有自动检测的配置和权重映射。

## 可用模型

### 纯文本模型
- **Gemma 2 2B** (`google/gemma-2-2b`): 紧凑的 20 亿参数模型，针对效率优化
  - 26 层，2304 隐藏层大小
  - 8 个注意力头，4 个查询组（GQA）
  - 序列长度：8,192 个词元
  - 适用于单 GPU 部署

- **Gemma 2 9B** (`google/gemma-2-9b`): 中等规模的 90 亿参数模型，平衡性能与效率
  - 42 层，3584 隐藏层大小
  - 16 个注意力头，8 个查询组（GQA）
  - 序列长度：8,192 个词元
  - 推荐配置：4-8 个 GPU 配合张量并行

- **Gemma 2 27B** (`google/gemma-2-27b`): 大规模 270 亿参数模型，追求极致性能
  - 46 层，4608 隐藏层大小
  - 32 个注意力头，16 个查询组（GQA）
  - 序列长度：8,192 个词元
  - 推荐配置：8-16 个 GPU 配合张量并行和管道并行

所有模型均支持 8,192 个词元的序列长度，并使用滑动窗口注意力进行高效的上下文处理。

## 模型架构特性

Gemma 2 引入了多项架构创新：

- **滑动窗口注意力**：使用 4,096 个词元窗口的局部注意力，用于高效的长上下文处理
- **注意力对数软上限**：将注意力对数限制在 50.0，以防止极端值
- **最终对数软上限**：将最终输出对数限制在 30.0，以实现稳定的生成
- **GeGLU 激活函数**：使用带有 GELU 激活的门控线性单元以提高性能
- **RMSNorm**：无需均值中心化的层归一化，计算速度更快
- **旋转嵌入**：基础频率为 10,000 的 RoPE
- **查询预注意力缩放**：在注意力计算前将查询缩放 224 倍

## 使用 🤗 Hugging Face 进行转换

### 加载 HF → Megatron
```python
from megatron.bridge import AutoBridge

# 示例：Gemma 2 9B
bridge = AutoBridge.from_hf_pretrained("google/gemma-2-9b")
provider = bridge.to_megatron_provider()

# 在实例化模型前配置并行策略
provider.tensor_model_parallel_size = 8
provider.pipeline_model_parallel_size = 1

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 导入 HF → Megatron
将 HF 模型导入到您期望的 Megatron 路径：
```bash
# Gemma 2 2B
python examples/conversion/convert_checkpoints.py import \
--hf-model google/gemma-2-2b \
--megatron-path /models/gemma-2-2b

# Gemma 2 9B
python examples/conversion/convert_checkpoints.py import \
--hf-model google/gemma-2-9b \
--megatron-path /models/gemma-2-9b

# Gemma 2 27B
python examples/conversion/convert_checkpoints.py import \
--hf-model google/gemma-2-27b \
--megatron-path /models/gemma-2-27b
```

### 导出 Megatron → HF
```bash
# Gemma 2 9B 示例
python examples/conversion/convert_checkpoints.py export \
--hf-model google/gemma-2-9b \
--megatron-path /results/gemma2_9b/checkpoints/iter_00001000 \
--hf-path ./gemma2-9b-hf-export
```

### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
--hf_model_path google/gemma-2-9b \
--megatron_model_path /models/gemma-2-9b \
--prompt "What is artificial intelligence?" \
--max_new_tokens 100
```

注意：
- `--megatron_model_path` 是可选的。如果未指定，脚本将先转换模型，然后运行前向传播。

## 预训练和微调配方

- 参见：[bridge.recipes.gemma](../../apidocs/bridge/bridge.recipes.gemma.md)
- 可用配方：
  - **预训练：**
    - `gemma2_2b_pretrain_config`: Gemma 2 2B 的预训练配置
    - `gemma2_9b_pretrain_config`: Gemma 2 9B 的预训练配置
    - `gemma2_27b_pretrain_config`: Gemma 2 27B 的预训练配置
  - **监督微调（SFT）：**
    - `gemma2_2b_sft_config`: Gemma 2 2B 的完整 SFT 配置
    - `gemma2_9b_sft_config`: Gemma 2 9B 的完整 SFT 配置
    - `gemma2_27b_sft_config`: Gemma 2 27B 的完整 SFT 配置
  - **参数高效微调（PEFT）**（LoRA, DoRA）:
    - `gemma2_2b_peft_config`: Gemma 2 2B 的 PEFT 配置
    - `gemma2_9b_peft_config`: Gemma 2 9B 的 PEFT 配置
    - `gemma2_27b_peft_config`: Gemma 2 27B 的 PEFT 配置

开始训练前，请确保设置了以下环境变量：
1. `SAVE_DIR`: 检查点和日志保存目录
2. `HF_TOKEN`: 用于从 HF Hub 下载模型（如果需要）
3. `HF_HOME`: （可选）用于避免重复下载模型和数据集

4. `WANDB_API_KEY`：（可选）用于启用 WandB 日志记录

### 预训练

#### Gemma 2 2B
```python
from megatron.bridge.recipes.gemma import gemma2_2b_pretrain_config

# 创建预训练配置
config = gemma2_2b_pretrain_config(
    name="my_gemma2_2b_pretrain",
    data_paths=["path/to/data"],
    train_iters=100000,
    global_batch_size=32,
)
```

#### Gemma 2 9B
```python
from megatron.bridge.recipes.gemma import gemma2_9b_pretrain_config

config = gemma2_9b_pretrain_config(
    name="my_gemma2_9b_pretrain",
    data_paths=["path/to/data"],
    train_iters=100000,
    global_batch_size=32,
)
```

#### Gemma 2 27B
```python
from megatron.bridge.recipes.gemma import gemma2_27b_pretrain_config

config = gemma2_27b_pretrain_config(
    name="my_gemma2_27b_pretrain",
    data_paths=["path/to/data"],
    train_iters=100000,
    global_batch_size=32,
)
```

### 全参数微调

#### Gemma 2 2B
```bash
torchrun --nproc-per-node=8 run/run_recipe.py \
--pretrained-checkpoint /models/gemma-2-2b \
--recipe gemma2_2b_sft_config \
train.global_batch_size=64 \
train.train_iters=1000 \
checkpoint.save=$SAVE_DIR/gemma2_2b_finetune
```

或者以编程方式：
```python
from megatron.bridge.recipes.gemma import gemma2_2b_sft_config

config = gemma2_2b_sft_config(
    name="gemma2_2b_full_finetune",
    pretrained_checkpoint="/models/gemma-2-2b",
    train_iters=1000,
    global_batch_size=64,
)
```

#### Gemma 2 9B
```bash
torchrun --nproc-per-node=8 run/run_recipe.py \
--pretrained-checkpoint /models/gemma-2-9b \
--recipe gemma2_9b_sft_config \
train.global_batch_size=64 \
train.train_iters=1000 \
checkpoint.save=$SAVE_DIR/gemma2_9b_finetune
```

#### Gemma 2 27B
```bash
torchrun --nproc-per-node=16 run/run_recipe.py \
--pretrained-checkpoint /models/gemma-2-27b \
--recipe gemma2_27b_sft_config \
train.global_batch_size=64 \
train.train_iters=1000 \
checkpoint.save=$SAVE_DIR/gemma2_27b_finetune
```

### 使用 LoRA 进行参数高效微调（PEFT）

#### Gemma 2 2B
```bash
torchrun --nproc-per-node=8 run/run_recipe.py \
--pretrained-checkpoint /models/gemma-2-2b \
--recipe gemma2_2b_peft_config \
--peft_scheme lora \
train.global_batch_size=128 \
checkpoint.save=$SAVE_DIR/gemma2_2b_lora
```

PEFT 选项：
- `--peft_scheme`：设置为 `lora` 使用 LoRA，或 `dora` 使用 DoRA。全参数微调请使用 `gemma2_*_sft_config`。

或者以编程方式：
```python
from megatron.bridge.recipes.gemma import gemma2_2b_peft_config

# LoRA 微调
config = gemma2_2b_peft_config(
    name="gemma2_2b_lora_finetune",
    pretrained_checkpoint="/models/gemma-2-2b",
    peft_scheme="lora",  # 或 "dora"
    train_iters=1000,
    global_batch_size=128,
)
```

#### Gemma 2 9B LoRA
```python
from megatron.bridge.recipes.gemma import gemma2_9b_peft_config

config = gemma2_9b_peft_config(
    name="gemma2_9b_lora_finetune",
    pretrained_checkpoint="/models/gemma-2-9b",
    peft_scheme="lora",
    train_iters=1000,
    global_batch_size=128,
)
```

#### Gemma 2 27B LoRA
```python
from megatron.bridge.recipes.gemma import gemma2_27b_peft_config

config = gemma2_27b_peft_config(
    name="gemma2_27b_lora_finetune",
    pretrained_checkpoint="/models/gemma-2-27b",
    peft_scheme="lora",
    train_iters=1000,
    global_batch_size=128,
)
```

### 推荐配置

| 模型 | 模式 | TP | PP | 全局批大小 | 学习率 |
|-------|------|----|----|-------------------|---------------|
| Gemma 2 2B | 全参数 SFT | 1 | 1 | 64-128 | 5e-6 |
| Gemma 2 2B | LoRA/DoRA | 1 | 1 | 128-256 | 1e-4 |
| Gemma 2 9B | 全参数 SFT | 4 | 1 | 64-128 | 5e-6 |
| Gemma 2 9B | LoRA/DoRA | 1 | 1 | 128-256 | 1e-4 |
| Gemma 2 27B | 全参数 SFT | 8 | 2 | 64-128 | 5e-6 |
| Gemma 2 27B | LoRA/DoRA | 4 | 1 | 128-256 | 1e-4 |

## 示例
- 检查点导入/导出：[examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- 生成文本（HF→Megatron）：[examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Hugging Face 模型卡片

- Gemma 2 2B: https://huggingface.co/google/gemma-2-2b
- Gemma 2 9B: https://huggingface.co/google/gemma-2-9b
- Gemma 2 27B: https://huggingface.co/google/gemma-2-27b
- Gemma 2 合集: https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315

## 相关文档
- 配方使用：[配方使用](../../recipe-usage.md)
- 自定义训练配方配置：[配置概述](../../training/config-container-overview.md)
- 训练入口点：[入口点](../../training/entry-points.md)