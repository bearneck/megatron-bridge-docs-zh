# 支持的模型

本目录包含 Megatron Bridge 支持的所有模型的文档，包括大语言模型（LLMs）和视觉语言模型（VLMs）。每个模型文档都包含架构详情、Hugging Face ↔ Megatron Bridge 的转换示例以及训练配方链接。

## 模型类别

Megatron Bridge 支持两大类模型：

### 🔤 大语言模型（LLMs）

仅处理文本的模型，用于语言理解和生成任务。

| 类别 | 模型数量 | 文档 |
|----------|-------------|---------------|
| **大语言模型** | 13 个模型 | [LLM 文档](llm/README.md) |

**支持的 LLM 系列：**

- DeepSeek (V2, V3)
- Gemma (2, 3)
- GLM-4.5
- GPT-OSS
- LLaMA (3, Nemotron)
- Mistral
- Moonlight
- Nemotron-H
- OLMoE
- Qwen (2, 2.5, 3, 3 MoE, 3-Next)

### 🖼️ 视觉语言模型（VLMs）

结合视觉和语言能力的多模态模型。

| 类别 | 模型数量 | 文档 |
|----------|-------------|---------------|
| **视觉语言模型** | 4 个模型 | [VLM 文档](vlm/README.md) |

**支持的 VLM 系列：**

- Gemma 3 VL
- Nemotron Nano V2 VL
- Qwen (2.5 VL, 3 VL)

---

## 快速导航

### 我想要

**🔍 查找特定的 LLM 模型**
→ 浏览[大语言模型](llm/README.md)文档

**🖼️ 查找特定的 VLM 模型**
→ 浏览[视觉语言模型](vlm/README.md)文档

**🔄 在格式之间转换模型**
→ 查看[桥接指南](../bridge-guide.md)了解 Hugging Face ↔ Megatron 转换

**🚀 开始训练**
→ 查看[训练文档](../training/README.md)获取训练指南

**📚 理解模型架构**
→ 每个模型页面都记录了特定于架构的特性和配置

**🔧 添加对新模型的支持**
→ 参考[添加新模型](../adding-new-models.md)

**📊 使用训练配方**
→ 阅读[配方使用](../recipe-usage.md)了解预配置的训练配方

---

## 模型文档结构

每个模型文档页面通常包括：

1.  **模型概述** - 架构和关键特性
2.  **可用变体** - 支持的模型大小和配置
3.  **转换示例** - 在 Hugging Face 和 Megatron 格式之间转换
4.  **训练配方** - 训练配置和示例的链接
5.  **架构详情** - 模型特定的特性和配置

---

## 按模型类型的常见任务

### 对于 LLM 模型

**训练：**

- 在大规模语料库上进行预训练
- 监督微调（SFT）
- 参数高效微调（PEFT/LoRA）
- 偏好优化（DPO）

**部署：**

- 导出为 Hugging Face 格式
- 与推理引擎集成
- 模型服务与部署

**用例：**

- 文本生成
- 问答
- 对话式 AI
- 代码生成

### 对于 VLM 模型

**训练：**

- 多模态预训练
- 视觉-语言对齐
- 在视觉任务上微调

**部署：**

- 导出为 Hugging Face 格式
- 多模态推理

**用例：**

- 图像描述
- 视觉问答
- 文档理解
- 多模态推理

---

## 相关文档

### 入门指南

-   **[主文档](../README.md)** - 返回主文档
-   **[桥接指南](../bridge-guide.md)** - Hugging Face ↔ Megatron 转换
-   **[桥接技术详情](../bridge-tech-details.md)** - 桥接系统的技术细节

### 训练资源

-   **[训练文档](../training/README.md)** - 全面的训练指南
-   **[配置容器](../training/config-container-overview.md)** - 训练配置
-   **[并行策略指南](../parallelisms.md)** - 数据和模型并行策略
-   **[性能指南](../performance-guide.md)** - 性能优化

### 高级主题

-   **[添加新模型](../adding-new-models.md)** - 扩展模型支持
-   **[配方使用](../recipe-usage.md)** - 使用预配置的训练配方
-   **[桥接 RL 集成](../bridge-rl-integration.md)** - 强化学习集成
-   **[PEFT](../training/peft.md)** - 参数高效微调

---

## 模型支持概览

### 按架构类型

**仅解码器（自回归）：**

- GPT 风格模型（GPT-OSS）
- LLaMA 系列（LLaMA 3, LLaMA Nemotron）
- Qwen 系列（Qwen 2, 2.5, 3, 3-Next）
- Gemma 系列（Gemma 2, 3）
- DeepSeek 系列（DeepSeek V2, V3）
- Mistral, Moonlight, Nemotron-H, GLM-4.5

**专家混合（MoE）：**

- Qwen 3 MoE, Qwen 3-Next
- DeepSeek V2, V3
- OLMoE

**视觉-语言（多模态）：**

- Gemma 3 VL
- Qwen 2.5 VL, Qwen 3 VL
- Nemotron Nano V2 VL

### 按提供商

**Meta/LLaMA:**

- LLaMA 3

**NVIDIA:**

- LLaMA Nemotron
- Nemotron-H
- Nemotron Nano V2 VL

**Alibaba Cloud:**

- Qwen (2, 2.5, 3, 3 MoE, 3-Next)
- Qwen VL (2.5, 3)

**Google:**

- Gemma (2, 3)
- Gemma 3 VL

**DeepSeek:**

- DeepSeek (V2, V3)

**其他：**

- Mistral AI (Mistral)
- GLM-4.5
- GPT-OSS
- Moonlight
- OLMoE

---

## 转换支持

所有模型均支持双向转换：

- **Hugging Face → Megatron Bridge**：加载预训练权重进行训练
- **Megatron Bridge → Hugging Face**：导出训练好的模型用于部署

转换特性：

- 自动架构检测
- 支持并行策略的转换（TP/PP/VPP/CP/EP）
- 流式处理和内存高效的传输
- 转换准确性的验证机制

有关详细的转换说明，请参阅 [桥接指南](../bridge-guide.md)。

---

**准备开始探索？** 选择一个模型类别：

- [大语言模型 (LLMs)](llm/README.md)
- [视觉语言模型 (VLMs)](vlm/README.md)

或返回 [主文档](../README.md)。