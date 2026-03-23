# 大语言模型

本目录包含 Megatron Bridge 支持的大语言模型（LLMs）的文档。每个模型文档都包含与 🤗 Hugging Face 格式相互转换的示例以及训练配方的链接。

## 可用模型

Megatron Bridge 支持以下 LLM 系列：

| 模型 | 文档 | 描述 |
|-------|---------------|-------------|
| **DeepSeek V2** | [deepseek-v2.md](deepseek-v2.md) | DeepSeek V2 模型系列 |
| **DeepSeek V3** | [deepseek-v3.md](deepseek-v3.md) | DeepSeek V3 模型系列 |
| **Gemma 2** | [gemma2.md](gemma2.md) | Google Gemma 2 模型 |
| **Gemma 3** | [gemma3.md](gemma3.md) | Google Gemma 3 模型 |
| **GLM-4.5** | [glm45.md](glm45.md) | GLM-4.5 模型系列 |
| **GPT-OSS** | [gpt-oss.md](gpt-oss.md) | 开源 GPT 风格模型 |
| **LLaMA 3** | [llama3.md](llama3.md) | Meta LLaMA 3 模型 |
| **LLaMA Nemotron** | [llama-nemotron.md](llama-nemotron.md) | NVIDIA LLaMA Nemotron 模型 |
| **Mistral** | [mistral.md](mistral.md) | Mistral AI 模型 |
| **Moonlight** | [moonlight.md](moonlight.md) | Moonlight 模型系列 |
| **Nemotron-3** | [nemotron3.md](nemotron3.md) | NVIDIA Nemotron-3 模型 |
| **Nemotron-3 Super** | [nemotron3-super.md](nemotron3-super.md) | NVIDIA Nemotron-3 Super 模型 |
| **Nemotron-H** | [nemotronh.md](nemotronh.md) | NVIDIA Nemotron-H 模型 |
| **OLMoE** | [olmoe.md](olmoe.md) | OLMoE（开放语言模型 - 专家混合） |
| **Qwen** | [qwen.md](qwen.md) | 阿里巴巴云 Qwen 模型系列 |

## 快速导航

### 我想要

**🔍 查找特定模型**
→ 浏览上方的模型列表或使用 [索引页](index.md)

**🔄 在格式之间转换模型**
→ 每个模型页面都包含 Hugging Face ↔ Megatron Bridge 的转换示例

**🚀 开始训练**
→ 查看 [训练文档](../../training/README.md) 获取训练指南

**📚 理解模型架构**
→ 每个模型页面都记录了特定于架构的功能和配置

**🔧 添加对新模型的支持**
→ 参考 [添加新模型](../../adding-new-models.md)

## 相关文档

- **[模型概述](../README.md)** - 返回主模型文档
- **[视觉语言模型](../vlm/README.md)** - VLM 模型文档
- **[训练文档](../../training/README.md)** - 训练和自定义指南
- **[Bridge 指南](../../bridge-guide.md)** - 使用 Hugging Face 模型
- **[添加新模型](../../adding-new-models.md)** - 扩展模型支持

## 模型文档结构

每个模型文档页面通常包含：

1.  **模型概述** - 架构和关键特性
2.  **可用变体** - 支持的模型大小和配置
3.  **转换示例** - 在 Hugging Face 和 Megatron 格式之间转换
4.  **训练配方** - 训练配置和示例的链接
5.  **架构详情** - 模型特定的功能和配置

---

**准备好探索了吗？** 从上方列表中选择一个模型或返回 [主文档](../../README.md)。