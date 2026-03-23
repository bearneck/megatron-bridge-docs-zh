# Qwen 3.5

[Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) 是一个支持跨文本、图像和视频多模态理解的视觉语言模型系列。Qwen3.5-VL 包含稠密模型和混合专家（Mixture-of-Experts, MoE）变体，以提高大规模部署时的效率。

Qwen 3.5 模型采用混合架构，结合了 GDN（Gated DeltaNet）层与标准注意力层、SwiGLU 激活函数和 RMSNorm。MoE 变体使用带共享专家的 top-k 路由机制以获得更好的质量。

Qwen 3.5 模型通过 Megatron Bridge 支持，具有自动检测的配置和权重映射。

```{important}
请升级到 `transformers` >= 5.2.0 以使用 Qwen 3.5 模型。
```

## 可用模型

### 稠密模型
- **Qwen3.5 0.8B** (`Qwen/Qwen3.5-0.8B`): 0.8B 参数的视觉语言模型
  - 推荐配置：1 个节点，8 个 GPU

- **Qwen3.5 2B** (`Qwen/Qwen3.5-2B`): 2B 参数的视觉语言模型
  - 推荐配置：1 个节点，8 个 GPU

- **Qwen3.5 4B** (`Qwen/Qwen3.5-4B`): 4B 参数的视觉语言模型
  - 推荐配置：1 个节点，8 个 GPU

- **Qwen3.5 9B** (`Qwen/Qwen3.5-9B`): 9B 参数的视觉语言模型
  - 推荐配置：1 个节点，8 个 GPU

- **Qwen3.5 27B** (`Qwen/Qwen3.5-27B`): 27B 参数的视觉语言模型
  - 推荐配置：2 个节点，16 个 GPU

### 混合专家（MoE）模型
- **Qwen3.5 35B-A3B** (`Qwen/Qwen3.5-35B-A3B`): 总计 35B 参数，每个令牌激活 3B 参数
  - 推荐配置：2 个节点，16 个 GPU

- **Qwen3.5 122B-A10B** (`Qwen/Qwen3.5-122B-A10B`): 总计 122B 参数，每个令牌激活 10B 参数
  - 推荐配置：4 个节点，32 个 GPU

- **Qwen3.5 397B-A17B** (`Qwen/Qwen3.5-397B-A17B`): 总计 397B 参数，每个令牌激活 17B 参数
  - 包含 512 个专家，采用 top-10 路由和共享专家机制
  - 推荐配置：16 个节点，128 个 GPU

## 示例

关于检查点转换、推理、微调配方和分步训练指南，请参阅 [Qwen 3.5 示例](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/vlm/qwen35_vl/README.md)。

## Hugging Face 模型卡片

- Qwen3.5 0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Qwen3.5 2B: https://huggingface.co/Qwen/Qwen3.5-2B
- Qwen3.5 4B: https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3.5 9B: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen3.5 27B: https://huggingface.co/Qwen/Qwen3.5-27B
- Qwen3.5 35B-A3B (MoE): https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Qwen3.5 122B-A10B (MoE): https://huggingface.co/Qwen/Qwen3.5-122B-A10B
- Qwen3.5 397B-A17B (MoE): https://huggingface.co/Qwen/Qwen3.5-397B-A17B

## 相关文档
- 相关 VLM: [Qwen3-VL](qwen3-vl.md)
- 相关 LLM: [Qwen](../llm/qwen.md)
- 配方使用: [配方使用](../../recipe-usage.md)
- 自定义训练配方配置: [配置概述](../../training/config-container-overview.md)
- 训练入口点: [入口点](../../training/entry-points.md)