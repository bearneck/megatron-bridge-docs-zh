# Nemotron 3 Super
[Nemotron 3 Super](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) 是由 NVIDIA 训练的大型语言模型（LLM），旨在提供强大的智能体、推理和对话能力。它采用了混合的**潜在专家混合（Latent Mixture-of-Experts，LatentMoE）** 架构，利用交错的 Mamba-2 和 MoE 层，以及选定的注意力层。与 Nano 模型不同，Super 模型集成了**多令牌预测（Multi-Token Prediction，MTP）** 层以实现更快的文本生成和更高的质量，并且使用 **NVFP4** 量化进行训练以最大化计算效率。该模型拥有 **120 亿活跃参数** 和 **1200 亿总参数**。

NeMo Megatron Bridge 支持对该模型进行预训练、全参数微调和 LoRA 微调。微调后的模型可以转换回 🤗 Hugging Face 格式以进行下游评估。

```{important}
使用此模型时，请使用自定义容器 `nvcr.io/nvidia/nemo:26.02.nemotron_3_super`。

所有命令均在 `/opt/Megatron-Bridge` 目录下运行（例如 `docker run -w /opt/Megatron-Bridge ...`）。
```

## 获取最新代码

为了获得最佳体验，建议使用 `super-v3` 分支的最新代码。有两种方法可以实现：

### 选项 1：在容器内更新代码

启动容器并在原地更新代码：

```bash
# 从 super-v3 分支拉取最新更改
cd /opt/megatron
git pull origin super-v3
```

### 选项 2：从主机挂载仓库

此方法允许您在主机上处理代码，并在运行时将其挂载到容器中。

**步骤 1 — 在主机上拉取最新的 `super-v3` 分支：**

```bash
git checkout super-v3 && git pull origin super-v3
```

**步骤 2 — 启动容器时挂载仓库：**

```bash
MEGATRON_BRIDGE_PATH=/path/to/Megatron-Bridge  # 将其设置为您的本地克隆路径

docker run --rm -it \
  -v $MEGATRON_BRIDGE_PATH:/opt/Megatron-Bridge \
  -w /opt/Megatron-Bridge \
  nvcr.io/nvidia/nemo:26.02.nemotron_3_super \
  bash
```

---

## 与 🤗 Hugging Face 的转换

### 导入 HF → Megatron
要将 HF 模型导入到您期望的 `$MEGATRON_MODEL_PATH`，请使用分布式转换脚本，因为此模型使用了专家并行。单进程的 `examples/conversion/convert_checkpoints.py` 脚本仅限于单 GPU 转换，不支持模型并行。

```bash
HF_MODEL=/path/to/hf/model
MEGATRON_PATH=/path/to/output/megatron/ckpt

torchrun --nproc-per-node=8 examples/conversion/convert_checkpoints_multi_gpu.py import \
--hf-model $HF_MODEL \
--megatron-path $MEGATRON_PATH \
--tp 1 \
--ep 8
```

注意：
- 默认并行配置为 TP=1, EP=8（专家并行）
- 根据您可用的 GPU 数量调整 `--nproc-per-node`

### 导出 Megatron → HF
```bash
HF_MODEL=/path/to/hf/model
MEGATRON_PATH=/path/to/trained/megatron/ckpt
OUTPUT_PATH=/path/to/output/hf/ckpt

torchrun --nproc-per-node=8 examples/conversion/convert_checkpoints_multi_gpu.py export \
--hf-model $HF_MODEL \
--megatron-path $MEGATRON_PATH \
--hf-path $OUTPUT_PATH \
--tp 1 \
--ep 8
```

### 往返测试
为了验证导入/导出转换的正确性：

```bash
HF_MODEL=/path/to/hf/model
MEGATRON_PATH=/path/to/megatron/ckpt

torchrun --nproc-per-node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
--hf-model-id $HF_MODEL \
--megatron-load-path $MEGATRON_PATH \
--tp 1 \
--ep 8 \
--trust-remote-code
```

### 比较 HF 和 Megatron 输出
比较 HF 和 Megatron 模型之间的输出：

```bash
HF_MODEL=/path/to/hf/model
MEGATRON_PATH=/path/to/megatron/ckpt

torchrun --nproc-per-node=8 examples/conversion/compare_hf_and_megatron/compare.py \
--hf_model_path $HF_MODEL \
--megatron_model_path $MEGATRON_PATH \
--prompt "Hello who are " \
--tp 8 \
--ep 8 \
--trust_remote_code
```

## 预训练示例

### 使用真实数据进行预训练
```bash
BLEND_PATH=/path/to/dataset/blend.json
CHECKPOINT_DIR=/path/to/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/pretrain_nemotron_3_super.py \
--per-split-data-args-path=${BLEND_PATH} \
logger.wandb_project=your_project \
logger.wandb_entity=nvidia \
logger.log_interval=5 \
checkpoint.load=${CHECKPOINT_DIR} \
checkpoint.save=${CHECKPOINT_DIR} \
checkpoint.save_interval=100 \
train.global_batch_size=8 \
train.micro_batch_size=1 \
train.train_iters=1280 \
scheduler.lr_warmup_iters=128 \
scheduler.lr_decay_iters=1152 \
scheduler.lr_wsd_decay_iters=1152 \
model.tensor_model_parallel_size=4 \
model.context_parallel_size=1 \
model.expert_model_parallel_size=64 \
model.sequence_parallel=True
```

注意：
- **GPU 要求**：需要 B200 GPU 以支持 NVFP4。至少需要 8 个节点（64 个 GPU）
- 默认并行设置为 TP=4, EP=64, PP=1, CP=1，并启用了序列并行
- 针对 MoE 架构，专家并行（EP）设置为 64
- 根据您的训练需求调整批次大小和迭代次数

- 如果使用 WandB 日志记录，请确保设置 WandB 凭据

### 使用模拟数据进行预训练
为了在没有数据集的情况下快速测试：

```bash
CHECKPOINT_DIR=/path/to/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/pretrain_nemotron_3_super.py \
logger.wandb_project=your_project \
logger.wandb_entity=nvidia \
checkpoint.load=${CHECKPOINT_DIR} \
checkpoint.save=${CHECKPOINT_DIR} \
checkpoint.save_interval=100 \
train.global_batch_size=128 \
train.train_iters=100 \
scheduler.lr_warmup_iters=10 \
model.hybrid_override_pattern="MEME*ME" \
model.num_layers=7
```

注意事项：
- 如果未指定 `BLEND_PATH`，将使用模拟数据集
- `hybrid_override_pattern` 可用于自定义 MoE 层模式
- 适用于调试和测试训练流程

## 微调方案

### 全参数微调
```bash
MEGATRON_PATH=/path/to/pretrained/megatron/ckpt
CHECKPOINT_DIR=/path/to/finetuned/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_super.py \
logger.wandb_project=your_project \
logger.wandb_entity=nvidia \
logger.log_interval=5 \
checkpoint.load=${CHECKPOINT_DIR} \
checkpoint.save=${CHECKPOINT_DIR} \
checkpoint.save_interval=50 \
train.global_batch_size=16 \
train.train_iters=200 \
scheduler.lr_warmup_iters=10 \
model.tensor_model_parallel_size=4 \
model.sequence_parallel=True \
checkpoint.pretrained_checkpoint=$MEGATRON_PATH
```

注意事项：
- 默认并行配置为 TP=4, EP=8, PP=1, CP=1，并启用序列并行
- 默认使用 [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) 数据集。
- 微调需要一个预训练的 Megatron 检查点，可以从上面的“导入 HF → Megatron”部分获取
- 根据您的 GPU 内存和需求调整 `global_batch_size` 和并行设置

### LoRA 微调
要启用 LoRA 微调，请向脚本传递 `--peft lora`：

```bash
MEGATRON_PATH=/path/to/pretrained/megatron/ckpt
CHECKPOINT_DIR=/path/to/lora/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_super.py \
--peft lora \
logger.wandb_project=your_project \
logger.wandb_entity=nvidia \
logger.log_interval=5 \
checkpoint.load=${CHECKPOINT_DIR} \
checkpoint.save=${CHECKPOINT_DIR} \
checkpoint.save_interval=100 \
train.global_batch_size=4 \
train.train_iters=200 \
model.tensor_model_parallel_size=4 \
model.context_parallel_size=2 \
model.sequence_parallel=True \
scheduler.lr_warmup_iters=30 \
checkpoint.pretrained_checkpoint=$MEGATRON_PATH
```

注意事项：
- 默认情况下，目标模块是模型中的线性层 `["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]`
- LoRA 微调使用更少的内存，并且可以使用更小的批次大小
- 对于更长的序列，考虑使用上下文并行（Context Parallel, CP）

## 量化（PTQ 和 QAT）

```{important}
量化支持需要来自 `super-v3` 分支的最新代码。有关说明，请参阅[获取最新代码](#getting-the-latest-code)。
```

Nemotron 3 Super 支持四种量化配置：

| 配置名称 | 格式 | 描述 |
|---|---|---|
| `mamba_moe_fp8_aggressive` | FP8 | 针对 Mamba-MoE 的激进 FP8 量化 |
| `mamba_moe_fp8_conservative` | FP8 | 针对 Mamba-MoE 的保守 FP8 量化 |
| `mamba_moe_nvfp4_aggressive` | NVFP4 | 针对 Mamba-MoE 的激进 NVFP4 量化 |
| `mamba_moe_nvfp4_conservative` | NVFP4 | 针对 Mamba-MoE 的保守 NVFP4 量化 |

通过 `--export-quant-cfg` 将所需的配置名称传递给 `quantize.py`。

### 量化
```bash
export HF_MODEL=/path/to/hf/model
export MEGATRON_SAVE_PATH=/path/to/quantized/megatron/ckpt

torchrun --nproc_per_node=8 examples/quantization/quantize.py \
    --hf-model-id $HF_MODEL \
    --export-quant-cfg mamba_moe_nvfp4_conservative \
    --megatron-save-path $MEGATRON_SAVE_PATH \
    --pp 1 \
    --tp 8 \
    --ep 8 \
    --trust-remote-code
```

### 使用 PTQ 生成进行验证
```bash
torchrun --nproc_per_node=8 examples/quantization/ptq_generate.py \
    --hf-model-id $HF_MODEL \
    --megatron-load-path $MEGATRON_SAVE_PATH \
    --pp 1 \
    --tp 8 \
    --ep 8 \
    --trust-remote-code
```

注意事项：
- 对于多节点设置（例如，2 个节点，每个节点 8× H100），请相应增加 `--pp`（例如 `--pp 2`），并使用像 SLURM 这样的作业调度器在节点间启动。

### 导出量化后的 Megatron 检查点 → HF

量化后，将 Megatron 检查点导出回 Hugging Face 格式：

```bash
HF_MODEL=/path/to/hf/model
MEGATRON_LOAD_PATH=/path/to/quantized/megatron/ckpt
EXPORT_DIR=/path/to/output/hf/ckpt

torchrun --nproc_per_node=8 examples/quantization/export.py \
    --hf-model-id $HF_MODEL \
    --megatron-load-path $MEGATRON_LOAD_PATH \
    --export-dir $EXPORT_DIR \
    --pp 8 \
    --dtype bfloat16 \
    --trust-remote-code
```

### 量化感知训练（QAT）

量化后，可以通过从量化后的 Megatron 检查点继续训练，使用 QAT 进一步提高模型质量。

```bash
MEGATRON_PATH=/path/to/quantized/megatron/ckpt
CHECKPOINT_DIR=/path/to/qat/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/qat_nemotron_3_super.py \
--megatron-load-path=${MEGATRON_PATH} \
--seq-length=8192 \
--packed-sequence \
logger.wandb_project=your_project \
logger.wandb_entity=nvidia \
logger.log_interval=5 \
checkpoint.load=${CHECKPOINT_DIR} \
checkpoint.save=${CHECKPOINT_DIR} \
checkpoint.save_interval=50 \
train.global_batch_size=16 \
train.train_iters=200 \
scheduler.lr_warmup_iters=10 \
model.tensor_model_parallel_size=4 \
model.sequence_parallel=True
```