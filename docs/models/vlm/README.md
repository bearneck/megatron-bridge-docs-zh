# 视觉语言模型

本目录包含 Megatron Bridge 支持的视觉语言模型（Vision Language Models，VLMs）的文档。这些模型结合了视觉和语言能力，适用于多模态人工智能应用。

## 可用模型

Megatron Bridge 支持以下 VLM 系列：

| 模型 | 文档 | 描述 |
|-------|---------------|-------------|
| **Gemma 3 VL** | [gemma3-vl.md](gemma3-vl.md) | Google Gemma 3 视觉语言模型 |
| **Ministral 3** | [ministral3.md](ministral3.md) | Ministral 3 视觉语言模型 |
| **Nemotron Nano V2 VL** | [nemotron-nano-v2-vl.md](nemotron-nano-v2-vl.md) | NVIDIA Nemotron Nano V2 视觉语言模型 |
| **Qwen2.5 VL** | [qwen2.5-vl.md](qwen2.5-vl.md) | 阿里云 Qwen2.5 视觉语言模型 |
| **Qwen3 VL** | [qwen3-vl.md](qwen3-vl.md) | 阿里云 Qwen3 视觉语言模型 |

## 快速导航

### 我想要

**🔍 查找特定的 VLM 模型**
→ 浏览上方的模型列表或使用 [索引页](index.md)

**🔄 在格式之间转换模型**
→ 每个模型页面都包含 Hugging Face ↔ Megatron Bridge 的转换示例

**🚀 开始训练**
→ 查看 [训练文档](../../training/README.md) 获取训练指南

**📚 理解 VLM 架构**
→ 每个模型页面都记录了视觉语言架构特性

**🔧 添加对新 VLM 的支持**
→ 参考 [添加新模型](../../adding-new-models.md)

## 相关文档

- **[模型概述](../README.md)** - 返回主模型文档
- **[大语言模型](../llm/README.md)** - LLM 模型文档
- **[训练文档](../../training/README.md)** - 训练和自定义指南
- **[Bridge 指南](../../bridge-guide.md)** - 使用 Hugging Face 模型
- **[添加新模型](../../adding-new-models.md)** - 扩展模型支持

## 视觉语言模型特性

VLM 通常支持：

- **图像理解** - 处理和理解视觉输入
- **多模态融合** - 结合视觉和语言表示
- **视觉语言任务** - 图像描述、视觉问答等
- **跨模态学习** - 学习视觉和文本数据之间的关系

---

**准备探索了吗？** 从上方列表中选择一个模型，或返回 [主文档](../../README.md)。