# 量化

本指南介绍使用 NVIDIA ModelOpt 在 Megatron Bridge 中进行模型量化，包括训练后量化（PTQ）和量化感知训练（QAT）。

## 目录

- [概述](#概述)
- [训练后量化（PTQ）](#训练后量化ptq)
- [量化感知训练（QAT）](#量化感知训练qat)

## 概述

量化是一种有效的模型优化技术，通过将精度从高精度格式（FP16/BF16）降低到低精度格式（NVFP4、FP8、INT8、INT4）来压缩模型。使用 Model Optimizer 进行量化可以将模型大小压缩 2-4 倍，在保持模型质量的同时加速推理。

在 Megatron Bridge 中，量化由 NVIDIA Model Optimizer（ModelOpt）启用——这是一个用于量化压缩深度学习模型以在 GPU 上进行优化推理的库。Model Optimizer 支持高性能的量化格式，包括 FP8、INT8、INT4 和 NVFP4，并通过易于使用的 Python API 支持 SmoothQuant 和 AWQ 等高级算法。

### 量化方法

Megatron Bridge 支持两种量化方法：

#### 训练后量化（PTQ）

PTQ 在训练后降低模型精度以提高推理效率，无需重新训练。这是最快的方法，适用于大多数模型。

**流程：**
1. 加载预训练模型检查点
2. 使用小型数据集（通常 128-512 个样本）校准模型以获取缩放因子
3. 生成量化后的检查点

#### 量化感知训练（QAT）

量化感知训练（QAT）有助于将模型精度提升到超越训练后量化（PTQ）的水平。QAT 可以在低精度（例如，INT4 或 NVIDIA Blackwell 平台上的 FP4）下进一步保持模型精度。

**流程：**

1. 以原始精度训练/微调模型，不进行量化
2. 使用 `mtq.quantize()` 量化步骤 1 中的模型
3. 使用较小的学习率（例如，Adam 优化器使用 1e-5）训练/微调量化后的模型

> **注意**：步骤 3 是实际的“量化感知训练”步骤。QAT 的最佳超参数设置可能因模型和训练数据集而异。

> **注意**：跳过原始精度训练/微调（即跳过步骤 1）的 QAT 会导致精度下降。因此，建议先进行未量化的原始精度训练/微调，然后再进行 QAT，以获得最佳精度。

---

## 训练后量化（PTQ）

PTQ 通过使用小型数据集运行校准来计算缩放因子，从而量化预训练模型。完整的工作流程包括：量化 → 恢复并生成 → 导出。

### 量化

使用 `examples/quantization/quantize.py` 脚本进行 LLM PTQ：

```bash
torchrun --nproc_per_node 2 examples/quantization/quantize.py \
    --hf-model-id meta-llama/Llama-3.2-1B \
    --export-quant-cfg fp8 \
    --tp 2 \
    --megatron-save-path ./llama3_2_1b_fp8
```

使用 `examples/quantization/quantize_vlm.py` 脚本进行 VLM PTQ：

```bash
torchrun --nproc_per_node 8 examples/quantization/quantize_vlm.py \
    --hf-model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
    --export-quant-cfg fp8 \
    --megatron-save-path ./Qwen3-VL-30B-A3B-Instruct_fp8 \
    --tp 4 \
    --etp 4 \
    --pp 2 \
    --calib-size 256
```

**关键参数：**
- `--hf-model-id` - HuggingFace 模型 ID 或本地路径
- `--export-quant-cfg` - 量化格式（`fp8`、`nvfp4` 等）
- `--megatron-save-path` - 输出检查点路径
- `--tp` - 张量并行大小
- `--pp` - 管道并行大小
- `--ep` - MoE 模型的专家并行
- `--etp` - MoE 模型的专家张量并行
- `--calib-size` - 用于量化的校准样本数（默认：512）

### 恢复并生成

使用 `examples/quantization/ptq_generate.py` 恢复量化后的检查点并进行文本生成测试（适用于 LLM）：

```bash
torchrun --nproc_per_node 2 examples/quantization/ptq_generate.py \
    --hf-model-id meta-llama/Llama-3.2-1B \
    --megatron-load-path ./llama3_2_1b_fp8 \
    --tp 2
```

使用 `examples/quantization/ptq_generate_vlm.py` 恢复量化后的检查点并进行文本生成测试（适用于 VLM）：

```bash
torchrun --nproc_per_node 8 examples/quantization/ptq_generate_vlm.py \
    --hf-model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
    --megatron-load-path ./Qwen3-VL-30B-A3B-Instruct_fp8 \
    --tp 8 \
    --ep 8 \
    --image-path ./demo.jpeg \
    --prompts "Describe this image."
```

**关键参数：**
- `--megatron-load-path` - 量化检查点路径
- `--hf-model-id` - HuggingFace 模型 ID 或本地路径（用于分词器）
- `--image-path` - 用于视觉语言模型提示生成的输入图像文件路径
- `--prompts` - 测试提示词

### 导出

使用 `examples/quantization/export.py` 将量化后的检查点导出为统一的 HuggingFace 格式：

```bash
torchrun --nproc_per_node 2 examples/quantization/export.py \
    --hf-model-id meta-llama/Llama-3.2-1B \

--megatron-load-path ./llama3_2_1b_fp8 \
    --export-dir ./llama3_2_1b_fp8_hf \
    --pp 2 \
    --dtype bfloat16
```

**关键参数：**
- `--export-dir` - 统一 HuggingFace 检查点的输出目录
- `--dtype` - 导出的数据类型

### 支持 PTQ 的模型

| 模型 | fp8 | nvfp4 |
|-------|-----|-------|
| Llama-3.2-1B | ✅ | ✅ |
| Qwen3-8B | ✅ | ✅ |
| Qwen3-30B-A3B | ✅ | ✅ |
| Nemotron-H-8B-Base-8K | ✅ | ✅ |
| Qwen3-VL-8B-Instruct | ✅ | ✅ |
| Qwen3-VL-30B-A3B-Instruct | ✅ | ✅ |

---

## 量化感知训练 (QAT)

在 QAT 中，使用 `mtq.quantize()` 量化的模型可以直接使用原始训练流程进行微调。在 QAT 期间，量化器内部的缩放因子被冻结，模型权重被微调。

### 完整的 QAT 工作流程

#### 步骤 1：创建初始量化检查点 (PTQ)

```bash
torchrun --nproc_per_node 8 examples/quantization/quantize.py \
    --hf-model-id meta-llama/Meta-Llama-3-8B \
    --export-quant-cfg fp8 \
    --tp 8 \
    --megatron-save-path /models/llama3_8b_fp8_init
```

#### 步骤 2：配置训练

创建一个 YAML 配置文件（例如 `conf/my_qat_config.yaml`）：

```yaml
model:
  tensor_model_parallel_size: 4
  gradient_accumulation_fusion: False

train:
  train_iters: 20
  global_batch_size: 8
  eval_iters: 0 

scheduler:
  lr_warmup_iters: 10

logger:
  log_interval: 1

checkpoint:
  pretrained_checkpoint: /models/llama3_8b_fp8_init
  save_interval: 20
  finetune: true
```

#### 步骤 3：运行 QAT 训练

使用 `examples/quantization/pretrain_quantized_llama3_8b.py`：

```bash
python pretrain_quantized_llama3_8b.py \
  --nproc-per-node=4 \
  --config-file=conf/my_qat_config.yaml \
  --hf-path=meta-llama/Meta-Llama-3-8B
```

**配置覆盖：**

你也可以使用命令行参数覆盖配置：

```bash
torchrun pretrain_quantized_llama3_8b.py \
    --nproc_per_node 4 \
    model.tensor_model_parallel_size=4 \
    model.gradient_accumulation_fusion=False \
    checkpoint.pretrained_checkpoint=/models/llama3_8b_fp8_init
```

### 支持 QAT 的模型

| 模型 | 支持 |
|-------|---------|
| Meta-Llama-3-8B | ✅ |