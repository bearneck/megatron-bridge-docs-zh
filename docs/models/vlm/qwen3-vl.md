# Qwen3-VL

Qwen3-VL 是阿里云最新一代的视觉语言模型，支持跨文本、图像和视频的多模态理解。Qwen3-VL 包含密集模型和混合专家（Mixture-of-Experts, MoE）变体，以提高效率。

NeMo Megatron Bridge 支持对 Qwen3-VL 模型（8B 密集模型和 30B MoE 变体）进行微调。

```{tip}
我们在本页面全程使用以下环境变量
- `HF_MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct` (对于 MoE 模型，使用 `Qwen/Qwen3-VL-30B-A3B-Instruct`)
- `MEGATRON_MODEL_PATH=/models/Qwen3-VL-8B-Instruct` (可以自由设置您自己的路径)
除非明确说明，以下命令中的任何 megatron 模型路径都不应包含迭代编号 `iter_xxxxxx`。有关检查点的更多详细信息，请参阅
[此处](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html#checkpoint-contents)
```

## 示例

关于检查点转换、推理、微调配方以及分步训练指南，请参阅 [Qwen3-VL 示例](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/vlm/qwen3_vl/README.md)。

## Hugging Face 模型卡片
- Qwen3-VL-8B: `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`
- Qwen3-VL-30B-A3B (MoE): `https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct`
- Qwen3-VL-235B-A22B (MoE): `https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct`