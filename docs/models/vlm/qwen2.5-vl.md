# Qwen2.5-VL

Qwen2.5-VL 是由阿里云开发的一系列视觉语言模型，能够实现跨文本、图像和视频的多模态理解。这些模型支持多种视觉语言任务，包括图像理解、视觉问答和多模态推理。

NeMo Megatron Bridge 支持在单图像和多图像数据集上对 Qwen2.5-VL 模型（3B、7B、32B 和 72B 变体）进行微调。
微调后的模型可以转换回 🤗 Hugging Face 格式，用于下游评估。

```{tip}
我们在本页中全程使用以下环境变量
- `HF_MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct` (也可以设置为 `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`)
- `MEGATRON_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct` (可以自由设置你自己的路径)

除非明确说明，以下命令中的任何 megatron 模型路径都不应包含迭代编号 `iter_xxxxxx`。有关检查点的更多详细信息，请参阅
[此处](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html#checkpoint-contents)
```

## 使用 🤗 Hugging Face 进行转换

### 导入 HF → Megatron
要将 HF 模型导入到你期望的 `$MEGATRON_MODEL_PATH`，请运行以下命令。
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model $HF_MODEL_PATH \
--megatron-path $MEGATRON_MODEL_PATH
```

### 导出 Megatron → HF
你可以使用以下命令导出一个训练好的模型。
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model $HF_MODEL_PATH \
--megatron-path <trained megatron model path> \
--hf-path <output hf model path>
```

### 在转换后的检查点上运行框架内推理
你可以使用以下命令对转换后的检查点进行快速完整性检查。
```bash
python examples/conversion/hf_to_megatron_generate_vlm.py \
--hf_model_path $HF_MODEL_PATH \
--megatron_model_path $MEGATRON_MODEL_PATH \
--image_path <example image path> \
--prompt "Describe this image." \
--max_new_tokens 100
```

注意：
- `--megatron_model_path` 是可选的。如果未指定，脚本将转换模型然后运行前向传播。如果指定了，脚本将直接加载 megatron 模型。
- `--max_new_tokens` 控制生成的令牌数量。
- 你也可以使用图片 URL：`--image_path="https://example.com/image.jpg"`

## 微调配方
在训练之前，请确保设置了以下环境变量。
1. `SAVE_DIR`：用于指定检查点和日志的保存目录，在下面的命令中使用。
2. `HF_TOKEN`：用于从 HF Hub 下载模型（如果需要）。
3. `HF_HOME`：（可选）避免每次重新下载模型和数据集。
4. `WANDB_API_KEY`：（可选）启用 WandB 日志记录。

### 全参数微调

全参数微调的示例用法：

```bash
torchrun --nproc-per-node=8 examples/models/vlm/qwen_vl/finetune_qwen25_vl.py \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--recipe qwen25_vl_3b_finetune_config \
--dataset-type hf \
dataset.maker_name=make_cord_v2_dataset \
train.global_batch_size=<batch size> \
train.train_iters=<number of iterations> \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name>
```

注意：
- `--recipe` 参数选择模型大小的配置。可用选项：
  - `qwen25_vl_3b_finetune_config` - 用于 3B 模型
  - `qwen25_vl_7b_finetune_config` - 用于 7B 模型
  - `qwen25_vl_32b_finetune_config` - 用于 32B 模型
  - `qwen25_vl_72b_finetune_config` - 用于 72B 模型
- 配置文件 `examples/models/vlm/qwen_vl/conf/qwen25_vl_pretrain_override_example.yaml` 包含一个可以在命令中覆盖的参数列表。例如，你可以在命令中设置 `train.global_batch_size=<batch size>`。
- 数据集格式应为 JSONL，并采用对话格式（参见下面的数据集部分）。
- 训练后，你可以通过提供训练好的 megatron 检查点，使用 `hf_to_megatron_generate_vlm.py` 运行推理。你也可以将训练好的检查点导出为 Hugging Face 格式。

### 参数高效微调 (PEFT)
支持使用 LoRA 或 DoRA 进行参数高效微调 (PEFT)。你可以使用 `--peft_scheme` 参数来启用 PEFT 训练：

```bash
torchrun --nproc-per-node=8 examples/models/vlm/qwen_vl/finetune_qwen25_vl.py \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--recipe qwen25_vl_3b_finetune_config \
--peft_scheme lora \
--dataset-type hf \
dataset.maker_name=make_cord_v2_dataset \
train.global_batch_size=<batch size> \
checkpoint.save=$SAVE_DIR/<experiment name>
```

PEFT 选项：
- `--peft_scheme`：设置为 `lora` 使用 LoRA（低秩适应），或设置为 `dora` 使用 DoRA（权重分解低秩适应）。设置为 `None` 或省略则进行全参数微调。

你也可以将 PEFT 与冻结选项结合，以控制哪些组件是可训练的：

- `model.freeze_language_model`：设置为 `True` 以冻结语言模型
- `model.freeze_vision_model`：设置为 `True` 以冻结视觉编码器
- `model.freeze_vision_projection`：设置为 `True` 以冻结视觉投影层

使用 LoRA 和冻结选项的示例：
```bash
torchrun --nproc-per-node=8 examples/models/vlm/qwen_vl/finetune_qwen25_vl.py \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--recipe qwen25_vl_3b_finetune_config \
--peft_scheme lora \
model.freeze_language_model=True \
model.freeze_vision_model=False \
model.freeze_vision_projection=False \
checkpoint.save=$SAVE_DIR/<experiment name>
```


## 示例数据集

Megatron Bridge 支持多种视觉语言数据集示例，可用于微调 Qwen 2.5 VL：
| 数据集 | 创建器名称 | 描述 |
|---------|------------|-------------|
| [cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | `make_cord_v2_dataset` | OCR 收据：用于收据理解的单图像文本数据集，输出类似 XML 的标注文本。 |
| [MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) | `make_medpix_dataset` | 医学 VQA：涵盖临床医学图像和自由形式答案的单图像问答数据集。 |
| [The Cauldron (Raven 子集)](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | `make_raven_dataset` | 视觉推理：多图像视觉推理数据集，用于不同视觉布局中的类比推理。 |

要更改数据集，请指定 `dataset.maker_name=make_raven_dataset`


## Hugging Face 模型卡片
- Qwen2.5-VL-3B: `https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct`
- Qwen2.5-VL-7B: `https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct`
- Qwen2.5-VL-32B: `https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct`
- Qwen2.5-VL-72B: `https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct`