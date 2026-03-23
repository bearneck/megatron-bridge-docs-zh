# Nemotron 3 Nano
[Nemotron 3 Nano](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) 是 NVIDIA 从头开始训练的大型语言模型（LLM），设计为一个统一模型，适用于推理和非推理任务。该模型采用混合专家（Mixture-of-Experts, MoE）架构，包含 23 个 Mamba-2 和 MoE 层，以及 6 个注意力（Attention）层。每个 MoE 层包含 128 个专家和 1 个共享专家，每个令牌激活 5 个专家。该模型拥有 35 亿激活参数和总计 300 亿参数。

NeMo Megatron Bridge 支持对该模型进行预训练、全参数微调和 LoRA 微调。微调后的模型可以转换回 🤗 Hugging Face 格式以进行下游评估。

```{important}
使用此模型时，请使用自定义容器 `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`。

所有命令均在 `/opt/Megatron-Bridge` 目录下运行（例如 `docker run -w /opt/Megatron-Bridge ...`）。
```

```{tip}
我们在本页面中使用以下环境变量：
- `HF_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `MEGATRON_MODEL_PATH=/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`（可自行设置路径）
```

## 与 🤗 Hugging Face 的转换

### 导入 HF → Megatron
要将 HF 模型导入到您期望的 `$MEGATRON_MODEL_PATH`，请运行以下命令。
```bash
python examples/conversion/convert_checkpoints.py import  \
--hf-model $HF_MODEL_ID  \
--megatron-path /path/to/output/megatron/ckpt \
--trust-remote-code
```

### 导出 Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export  \
--hf-model $HF_MODEL_ID  \
--megatron-path /path/to/trained/megatron/ckpt \
--hf-path /path/to/output/hf/ckpt
```

## 预训练示例
```bash
BLEND_PATH=/path/to/dataset/blend
TOKENIZER_MODEL=/path/to/tiktok/tokenizer/model

torchrun --nproc-per-node=8 examples/models/nemotron_3/pretrain_nemotron_3_nano.py \
--per-split-data-args-path=${BLEND_PATH} \
--tokenizer-model=${TOKENIZER_MODEL} \
train.global_batch_size=3072 \
train.train_iters=39500 \
scheduler.lr_warmup_iters=350
```

注意事项：
- 默认并行设置是 TP=4, EP=8, PP=1, CP=1。建议在 4 个 H100 节点（32 个 GPU）上运行此预训练。
- 要启用 wandb 日志记录，可以附加 `logger.wandb_project=PROJECT_NAME`、`wandb_entity=ENTITY_NAME` 和 `wandb_exp_name=EXP_NAME` 参数。
- 如果未指定 `BLEND_PATH` 和 `TOKENIZER_MODEL`，将使用模拟数据集。

## 微调配方

### 全参数微调
```bash
torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_nano.py \
train.global_batch_size=128 \
train.train_iters=100 \
scheduler.lr_warmup_iters=10 \
checkpoint.pretrained_checkpoint=/path/to/output/megatron/ckpt
```

注意事项：
- 默认并行设置 TP=1, EP=8, PP=1, CP=1。运行此配方至少需要 2 个 H100 节点（16 个 GPU）。
- 默认使用 [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) 数据集。要使用自定义数据集，请参阅此[教程](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/tutorials/recipes/llama#quickstart)。
- 微调需要一个预训练的 Megatron 检查点，可以在上面的“导入 HF → Megatron”部分获取。

### LoRA 微调
要启用 LoRA 微调，请向脚本传递 `--peft lora` 参数。
```bash
torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_nano.py \
--peft lora \
train.global_batch_size=128 \
train.train_iters=100 \
scheduler.lr_warmup_iters=10 \
checkpoint.pretrained_checkpoint=/path/to/output/megatron/ckpt
```

注意事项：
- 默认情况下，目标模块是模型中的线性层 `["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]`。
- 其余设置与上面的全参数微调相同。

LoRA 检查点仅包含可学习的适配器权重。为了将 LoRA 检查点转换为 Hugging Face 格式以进行下游评估，需要将 LoRA 适配器合并回基础模型。

```bash
python examples/peft/merge_lora.py \
--hf-model-path $HF_MODEL_ID \
--lora-checkpoint /path/to/lora/ckpt/iter_xxxxxxx \
--output /path/to/merged/ckpt
```