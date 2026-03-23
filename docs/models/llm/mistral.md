# Mistral

[Mistral AI](https://mistral.ai/) 致力于开发前沿的大型语言模型，重点关注效率和性能。Mistral 系列模型包括密集模型和专家混合（Mixture-of-Experts）架构，并采用了滑动窗口注意力（sliding window attention）和高效上下文处理等创新技术。

Mistral 模型通过 Bridge 系统获得支持，该系统具备自动检测配置和权重映射功能。

## 可用模型

Megatron Bridge 支持以下 Mistral 模型变体：

- **Mistral Small 3 (24B)**：240 亿参数，128K 上下文长度
- **Mistral 7B**：70 亿参数，高效的基线模型
- **Mistral 7B Instruct**：指令调优变体

其他 Mistral 模型（包括像 Mixtral 这样的 MoE 变体）可能通过标准转换流程获得支持。

## 模型架构特性

- **滑动窗口注意力（Sliding Window Attention）**：用于长序列的高效注意力机制
- **分组查询注意力（Grouped Query Attention, GQA）**：内存高效的注意力机制
- **旋转位置嵌入（Rotary Positional Embeddings, RoPE）**：相对位置编码
- **SwiGLU 激活函数**：前馈网络中的门控线性单元
- **扩展上下文**：支持长达 128K 个标记的序列（Mistral Small 3）
- **YaRN RoPE 缩放**：用于扩展上下文长度的高级 RoPE 缩放技术

## 使用 🤗 Hugging Face 进行转换

### 加载 HF → Megatron

```python
from megatron.bridge import AutoBridge

# 示例：Mistral Small 3 24B
bridge = AutoBridge.from_hf_pretrained("mistralai/Mistral-Small-24B-Base-2501")
provider = bridge.to_megatron_provider()

# 可选地在实例化模型前配置并行策略
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 从 HF 导入检查点

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model mistralai/Mistral-Small-24B-Base-2501 \
  --megatron-path /checkpoints/mistral_small_24b_megatron
```

### 导出 Megatron → HF

```python
from megatron.bridge import AutoBridge

# 从 HF 模型 ID 加载 bridge
bridge = AutoBridge.from_hf_pretrained("mistralai/Mistral-Small-24B-Base-2501")

# 将训练好的 Megatron 检查点导出为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/mistral_small_24b/checkpoints/iter_0000500",
    hf_path="/exports/mistral_small_24b_hf",
)
```

### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path mistralai/Mistral-Small-24B-Base-2501 \
  --megatron_model_path /checkpoints/mistral_small_24b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2
```

更多详情，请参阅 [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## 配方

目前尚未提供 Mistral 模型的训练配方。Bridge 支持用于推理和部署场景的检查点转换。

## Hugging Face 模型卡片与参考资料

### Hugging Face 模型卡片
- Mistral Small 3 (24B): https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501
- Mistral Small 3 (24B) Instruct: https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501
- Mistral 7B v0.1: https://huggingface.co/mistralai/Mistral-7B-v0.1
- Mistral 7B Instruct v0.2: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

### 技术论文
- Mistral 7B: https://arxiv.org/abs/2310.06825

### 其他资源
- Mistral AI 官网: https://mistral.ai/
- Mistral 文档: https://docs.mistral.ai/

## 相关文档
- 配方使用：[Recipe usage](../../recipe-usage.md)
- 自定义训练配方配置：[Configuration overview](../../training/config-container-overview.md)
- 训练入口点：[Entry points](../../training/entry-points.md)