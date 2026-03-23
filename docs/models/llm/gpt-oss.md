# GPT OSS

GPT OSS 是一个专家混合（Mixture-of-Experts, MoE）语言模型系列，包含两个变体：**GPT OSS 20B** 和 **GPT OSS 120B**。这些模型采用先进的注意力机制和 MoE 架构设计，针对长上下文理解进行了优化。

GPT OSS 模型采用仅解码器架构，并包含路由专家层，通过 YaRN 位置嵌入支持高达 128K 令牌的上下文长度。两个变体都使用分组查询注意力（Grouped-Query Attention）和专门的注意力机制，包括带有可学习 softmax 的滑动窗口注意力。

GPT OSS 模型通过 Bridge 系统提供支持，并配有针对 MoE 优化和长上下文训练的特殊配置。

## 模型架构

### GPT OSS 20B
- **参数量**：总计 200 亿
- **层数**：24 个解码器层
- **专家数**：每层 32 个路由专家，采用 top-4 路由
- **隐藏层大小**：2880
- **FFN 隐藏层大小**：2880（密集层），2880（专家层）
- **注意力头数**：64 个查询头，8 个键值组（GQA）
- **KV 通道数**：64
- **词表大小**：201,088
- **上下文长度**：128K 令牌（通过 YaRN）
- **激活函数**：带有门控线性单元的 QuickGELU
- **归一化**：RMSNorm

### GPT OSS 120B
- **参数量**：总计 1200 亿
- **层数**：36 个解码器层
- **专家数**：每层 128 个路由专家，采用 top-4 路由
- **隐藏层大小**：2880
- **FFN 隐藏层大小**：2880（密集层），2880（专家层）
- **注意力头数**：64 个查询头，8 个键值组（GQA）
- **KV 通道数**：64
- **词表大小**：201,088
- **上下文长度**：128K 令牌（通过 YaRN）
- **激活函数**：带有门控线性单元的 QuickGELU
- **归一化**：RMSNorm

## 主要特性

- **YaRN 位置嵌入**：先进的旋转位置嵌入，缩放因子为 32.0，用于长上下文扩展
- **分组查询注意力（GQA）**：高效的注意力机制，包含 8 个键值组
- **滑动窗口注意力**：窗口大小为 128 个令牌，采用交替的全注意力/窗口注意力模式
- **可学习 Softmax**：新颖的 softmax 实现，带有可学习的偏移参数（sink attention）
- **QuickGELU 激活函数**：快速的近似 GELU，在 7.0 处进行截断以确保稳定性
- **MoE 路由**：Top-4 专家路由，无负载均衡损失
- **分组 GEMM**：针对专家计算优化的分组矩阵乘法
- **线性层偏置**：线性层包含偏置项
- **激活截断**：输出激活被截断至 [-7.0, 7.0] 以确保数值稳定性

## 示例

关于检查点转换、推理、微调配方和分步训练指南，请参阅 [GPT-OSS 示例](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/gpt_oss/README.md)。

## API 参考

- GPT OSS 配方：[bridge.recipes.gpt_oss](../../apidocs/bridge/bridge.recipes.gpt_oss.md)
- GPT OSS 模型提供器：[bridge.models.gpt_oss.GPTOSSProvider](../../apidocs/bridge/bridge.models.gpt_oss.md)

## Hugging Face 模型卡片

### GPT OSS 20B
- 基础模型：[openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

### GPT OSS 120B
- 基础模型：[openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)

## 相关文档

- 配方使用与自定义：[配方使用](../../recipe-usage.md)
- 训练配置：[配置概述](../../training/config-container-overview.md)
- 训练入口点：[入口点](../../training/entry-points.md)
- 注意力优化：[注意力优化](../../training/attention-optimizations.md)