# Nemotron Nano V2 VL

NVIDIA Nemotron Nano v2 VL 是一个开源的 120 亿参数多模态推理模型，用于文档智能和视频理解。
它使[AI助手](https://www.nvidia.com/en-us/use-cases/ai-assistants)能够跨文本、图像、表格和视频提取、解释信息并采取行动。这使得该模型对于专注于数据分析、文档处理和视觉理解的智能体非常有价值，适用于生成报告、策划视频、媒体资产管理中的密集字幕生成以及检索增强搜索等应用。

NeMo Megatron Bridge 支持在单图像、多图像和视频数据集上对此模型进行微调（包括 LoRA 微调）。
微调后的模型可以转换回 🤗 Hugging Face 格式，用于下游评估。

```{important}
使用此模型时，请使用自定义容器 `nvcr.io/nvidia/nemo:25.09.nemotron_nano_v2_vl`。

所有命令均在 `/opt/Megatron-Bridge` 目录下运行（例如 `docker run -w /opt/Megatron-Bridge ...`）。
```

```{tip}
我们在本页面中使用以下环境变量
- `HF_MODEL_PATH=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`
- `MEGATRON_MODEL_PATH=/models/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`（可以设置为自己的路径）

除非明确说明，以下命令中的任何 Megatron 模型路径**不应**包含迭代编号 `iter_xxxxxx`。有关检查点的更多详细信息，请参阅[此处](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html#checkpoint-contents)。
```

## 与 🤗 Hugging Face 的转换

### 导入 HF → Megatron
要将 HF 模型导入到您期望的 `$MEGATRON_MODEL_PATH`，请运行以下命令。
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model $HF_MODEL_PATH \
--megatron-path $MEGATRON_MODEL_PATH \
--trust-remote-code
```

### 导出 Megatron → HF
您可以使用以下命令导出一个训练好的模型。
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model $HF_MODEL_PATH \
--megatron-path <训练好的 megatron 模型路径> \
--hf-path <输出的 hf 模型路径> \
--not-strict
```

注意：看到警告 `vision_model.radio_model.input_conditioner.norm_mean` 和 `vision_model.radio_model.input_conditioner.norm_std` 来自源但不在导出的检查点中是正常的。这两个权重在检查点中不需要。

### 在转换后的检查点上运行框架内推理
您可以使用以下命令对转换后的检查点进行快速完整性检查。
```bash
python examples/conversion/hf_to_megatron_generate_vlm.py \
--hf_model_path $HF_MODEL_PATH \
--megatron_model_path $MEGATRON_MODEL_PATH \
--image_path <示例图像路径> \
--prompt "描述这张图片。" \
--max_new_tokens 100 \
--use_llava_model
```

注意：
- `--megatron_model_path` 是可选的。如果未指定，脚本将转换模型然后运行前向传播。如果指定了，脚本将直接加载 Megatron 模型。
- `--max_new_tokens` 控制要生成的令牌数量。
- 对于多图像推理，传入一个逗号分隔的列表，例如 `--image_path="/path/to/example1.jpeg,/path/to/example2.jpeg"`。使用合适的提示，例如 `--prompt="详细描述这两张图片。"`。
- 对于视频推理，传入视频路径，例如 `--video_path="/path/to/demo.mp4"`。使用合适的提示，例如 `--prompt="描述你看到的内容。"`。

## 微调配方
在训练之前，请确保设置了以下环境变量。
1. `SAVE_DIR`：指定检查点和日志的保存目录，用于下面的命令。
2. `HF_TOKEN`：用于从 HF Hub 下载模型。
3. `HF_HOME`：（可选）避免每次重新下载模型和数据集。
4. `WANDB_API_KEY`：（可选）启用 WandB 日志记录。

### 全参数微调
使用 [Raven 数据集](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron/viewer/raven)进行全参数微调的示例用法：

```bash
torchrun --nproc-per-node=8 examples/models/vlm/nemotron_vl/finetune_nemotron_nano_v2_vl.py \
--hf-model-path $HF_MODEL_PATH \
--pretrained-checkpoint <megatron 模型路径> \
dataset.maker_name=make_raven_dataset \
logger.wandb_project=<可选的 wandb 项目名称> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<实验名称>
```

注意：
- 配置文件 `examples/models/vlm/nemotron_vl/conf/nemotron_nano_v2_vl_override_example.yaml` 包含可以在命令中覆盖的参数列表。例如，您可以在命令中设置 `train.global_batch_size=<批次大小>`。
- 要更改数据集，只需更改 `dataset.maker_name`。有关详细信息，请参阅下面的数据集部分。
- 训练后，您可以通过提供训练好的 Megatron 检查点，使用 `hf_to_megatron_generate_vlm.py` 运行推理。您也可以将训练好的检查点导出为 Hugging Face 格式。
- 此全参数微调配方至少需要 4 块 H100 (80G) GPU。

### 参数高效微调（PEFT）
支持使用 LoRA 进行参数高效微调（PEFT）。
LoRA 可以独立应用于视觉模型、视觉投影和语言模型。我们在示例脚本中开箱即用地支持两种常用设置：
1. 将 LoRA 应用于语言模型，并完全微调视觉模型和投影（当视觉分布与预训练分布显著不同时使用）。

```bash
torchrun --nproc-per-node=8 examples/models/vlm/nemotron_vl/finetune_nemotron_nano_v2_vl.py \
--hf-model-path $HF_MODEL_PATH \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--lora-on-language-model \
dataset.maker_name=make_raven_dataset \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name> \
model.freeze_language_model=True \
model.freeze_vision_model=False \
model.freeze_vision_projection=False
```

2. 将 LoRA 应用于视觉模型、视觉投影和语言模型中注意力（attention）和 MLP 模块的所有线性层。

```bash
torchrun --nproc-per-node=8 examples/models/vlm/nemotron_vl/finetune_nemotron_nano_v2_vl.py \
--hf-model-path $HF_MODEL_PATH \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--lora-on-language-model \
—-lora-on-vision-model \
dataset.maker_name=make_raven_dataset \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name> \
model.freeze_language_model=True \
model.freeze_vision_model=True \
model.freeze_vision_projection=True
```

这些 LoRA 微调方案至少需要 2 张 H100（80G）GPU。

LoRA 检查点仅包含可学习的适配器权重。为了将 LoRA 检查点转换为 Hugging Face 格式以进行下游评估，需要将 LoRA 适配器合并回基础模型。

```bash
python examples/peft/merge_lora.py \
--hf-model-path $HF_MODEL_PATH \
--lora-checkpoint <trained LoRA checkpoint>/iter_N \
--output <LoRA checkpoint merged>
```
现在，您可以通过提供合并后的 LoRA 检查点，使用 `hf_to_megatron_generate_vlm.py` 运行框架内推理。您也可以将合并后的 LoRA 检查点导出为 Hugging Face 格式。

## 示例数据集

Megatron Bridge 支持多种视觉语言数据集示例，可用于微调 Nemotron Nano V2 VL：
| 数据集 | 创建器名称 | 描述 |
|---------|------------|-------------|
| [cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | `make_cord_v2_dataset` | OCR 收据：用于收据理解的单图像文本数据集，输出类似 XML 的标注文本。 |
| [MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) | `make_medpix_dataset` | 医学 VQA：覆盖临床医学图像和自由形式答案的单图像问答数据集。 |
| [The Cauldron (Raven 子集)](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | `make_raven_dataset` | 视觉推理：多图像视觉推理数据集，用于不同视觉布局中的类比推理。 |
| [LLaVA-Video-178K (0_30_s_nextqa 子集)](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | `make_llava_video_178k_dataset` | 视频理解：覆盖日常场景的视频问答数据集。 |

`cord-v2` 是一个小型数据集，仅用于演示。不建议将此数据集用于该模型的 PEFT 微调，因为其 XML 输出格式会与特殊标记（special tokens）交互，导致意外结果。

关于视频训练示例的说明：
- 我们提供了一个视频配置文件 yaml 文件，而不是默认的配置文件 yaml 文件，它会覆盖一些命令。请传入 `--config-file "examples/models/vlm/nemotron_vl/conf/nemotron_nano_v2_vl_video.yaml"`。
- LLaVA 视频数据集需要事先手动下载。请将下载并解压的视频文件放在文件夹 `VIDEO_ROOT` 中，并通过 `dataset.maker_kwargs={"video_root_path":$VIDEO_ROOT}` 将其传递给创建器。在 nextqa 子集示例中，`VIDEO_ROOT` 应如下所示：
  ```
  $VIDEO_ROOT/
  ├── NextQA/
  │   └── NExTVideo/
  │       └── 0000/
  │           └── 2440175990.mp4
  │       └── 0001/
  │           └── ...
  └── ...
  ```

完整的视频训练示例命令：
```bash
torchrun --nproc-per-node=8 examples/models/vlm/nemotron_vl/finetune_nemotron_nano_v2_vl.py \
--hf-model-path $HF_MODEL_PATH \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--config-file "examples/models/vlm/nemotron_vl/conf/nemotron_nano_v2_vl_video.yaml" \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name> \
dataset.maker_kwargs={"video_root_path":$VIDEO_ROOT}
```