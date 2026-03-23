# Gemma 3 VL（视觉语言模型）

[Google 的 Gemma 3 VL](https://huggingface.co/collections/google/gemma-3-release) 是一个基于创建 Gemini 模型所用的相同研究和技术构建的视觉语言模型系列。Gemma 3 VL 架构结合了 Gemma 3 的文本生成能力和 SigLIP 视觉编码器，以实现强大的视觉理解。

Gemma 3 VL 模型支持多模态任务，包括图像描述、视觉问答、OCR 和通用视觉语言理解。

Gemma 系列模型通过 Bridge 系统提供支持，具有自动检测的配置和权重映射。

## 可用模型

### 视觉语言模型
- **Gemma 3 VL 4B** (`google/gemma-3-4b-it`): 40 亿参数的视觉语言模型
  - 34 层，2560 隐藏大小
  - 16 个注意力头，4 个查询组（GQA）
  - 视觉编码器：具有 7.29 亿参数的 SigLIP
  - 推荐配置：1 个节点，8 个 GPU
  
- **Gemma 3 VL 12B** (`google/gemma-3-12b-it`): 120 亿参数的视觉语言模型
  - 48 层，3840 隐藏大小
  - 24 个注意力头，8 个查询组（GQA）
  - 视觉编码器：具有 7.29 亿参数的 SigLIP
  - 推荐配置：1 个节点，8 个 GPU
  
- **Gemma 3 VL 27B** (`google/gemma-3-27b-it`): 270 亿参数的视觉语言模型
  - 62 层，5376 隐藏大小
  - 32 个注意力头，16 个查询组（GQA）
  - 视觉编码器：具有 7.29 亿参数的 SigLIP
  - 推荐配置：2 个节点，16 个 GPU

所有模型支持 131,072 个令牌的序列长度，并使用混合注意力模式（滑动窗口 + 全局）。

## 模型架构特性

Gemma 3 VL 基于 Gemma 3 架构构建，并增加了多模态能力：

**语言模型特性：**
- **混合注意力模式**：在全局和局部滑动窗口注意力之间交替，以实现高效的长上下文处理
- **GeGLU 激活**：使用带有 GELU 激活的门控线性单元以提高性能
- **RMSNorm**：无需均值中心化的层归一化，以加快计算速度
- **旋转嵌入**：为局部和全局注意力层提供独立的 RoPE 配置

**视觉语言特性：**
- **SigLIP 视觉编码器**：具有 7.29 亿参数的预训练视觉编码器，用于强大的视觉理解
- **多模态集成**：通过学习的投影层无缝集成视觉和文本信息
- **灵活的图像处理**：支持可变分辨率图像和每次对话中的多张图像

## 示例

关于检查点转换、推理、微调配方和分步训练指南，请参阅 [Gemma 3 VL 示例](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/vlm/gemma3_vl/README.md)。

## Hugging Face 模型卡片

- Gemma 3 VL 4B: https://huggingface.co/google/gemma-3-4b-it
- Gemma 3 VL 12B: https://huggingface.co/google/gemma-3-12b-it
- Gemma 3 VL 27B: https://huggingface.co/google/gemma-3-27b-it

## 相关文档
- 纯文本模型：[Gemma 3](../llm/gemma3.md)
- 配方使用：[配方使用](../../recipe-usage.md)
- 自定义训练配方配置：[配置概述](../../training/config-container-overview.md)
- 训练入口点：[入口点](../../training/entry-points.md)