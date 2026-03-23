# Ministral 3

[Mistral AI 的 Ministral 3](https://huggingface.co/collections/mistralai/ministral-3) 是一个为在各种硬件配置上部署而优化的边缘视觉语言模型系列。Ministral 3 架构结合了强大的语言模型和视觉编码器，以实现多模态理解。

Ministral 3 模型支持多模态任务，包括图像描述、视觉问答、OCR 和通用视觉语言理解。尽管模型尺寸紧凑，但它们为设备端和边缘部署场景提供了强大的性能。

Ministral 系列模型通过 Bridge 系统得到支持，具有自动检测的配置和权重映射。

```{important}
请升级到 `transformers` v5 并升级 `mistral-common` 以使用 Ministral 3 模型。
```

## 可用模型

### 视觉语言模型
- **Ministral 3 3B** (`mistralai/Ministral-3-3B-Base-2512`): 34 亿参数的视觉语言模型
  - 26 层，3072 隐藏大小
  - 32 个注意力头，8 个查询组 (GQA)
  - 视觉编码器：约 4 亿参数
  - 推荐配置：1 个节点，8 个 GPU

- **Ministral 3 8B** (`mistralai/Ministral-3-8B-Base-2512`): 84 亿参数的视觉语言模型
  - 34 层，4096 隐藏大小
  - 32 个注意力头，8 个查询组 (GQA)
  - 视觉编码器：约 4 亿参数
  - 推荐配置：1 个节点，8 个 GPU

- **Ministral 3 14B** (`mistralai/Ministral-3-14B-Base-2512`): 约 140 亿参数的视觉语言模型
  - 40 层，5120 隐藏大小
  - 32 个注意力头，8 个查询组 (GQA)
  - 视觉编码器：约 4 亿参数
  - 推荐配置：1 个节点，8 个 GPU

所有模型都支持使用 YaRN RoPE 缩放技术，将上下文长度扩展到最多 256K 个令牌。

## 模型架构特性

Ministral 3 将高效的语言建模与多模态能力相结合：

**语言模型特性：**
- **YaRN RoPE 缩放**：先进的 rope 缩放技术，用于扩展上下文长度（最多 256K 个令牌）
- **分组查询注意力 (GQA)**：具有 8 个查询组的内存高效注意力机制
- **SwiGLU 激活**：使用 SiLU 激活的门控线性单元，以提高性能
- **RMSNorm**：无需均值中心化的层归一化，用于更快的计算
- **Llama 4 注意力缩放**：位置相关的注意力缩放，以改进长上下文处理

**视觉语言特性：**
- **视觉编码器**：预训练的视觉编码器，用于鲁棒的视觉理解
- **多模态投影器**：将视觉特征投影到语言模型空间
- **灵活的图像处理**：支持可变分辨率图像和每次对话中的多张图像

## 示例

关于检查点转换、推理、微调配方和分步训练指南，请参阅 [Ministral 3 示例](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/vlm/ministral3/README.md)。

## Hugging Face 模型卡片

- Ministral 3 3B Base: https://huggingface.co/mistralai/Ministral-3-3B-Base-2512
- Ministral 3 3B Instruct: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
- Ministral 3 8B Base: https://huggingface.co/mistralai/Ministral-3-8B-Base-2512
- Ministral 3 8B Instruct: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
- Ministral 3 14B Base: https://huggingface.co/mistralai/Ministral-3-14B-Base-2512
- Ministral 3 14B Instruct: https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512

## 相关文档
- 相关 LLM: [Mistral](../llm/mistral.md)
- 配方使用: [配方使用](../../recipe-usage.md)
- 自定义训练配方配置: [配置概述](../../training/config-container-overview.md)
- 训练入口点: [入口点](../../training/entry-points.md)