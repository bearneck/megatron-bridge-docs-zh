# Llama Nemotron

[Llama Nemotron](https://huggingface.co/collections/nvidia/llama-nemotron) 是 NVIDIA 基于 Meta 的 Llama 架构衍生出的大语言模型系列，经过后训练以增强推理能力、人类聊天偏好以及 RAG 和工具调用等智能体任务。该系列模型采用了神经架构搜索（Neural Architecture Search, NAS）优化，以在效率和准确性之间实现更好的权衡。

Llama Nemotron 模型通过 Bridge 系统得到支持，具有自动检测的配置和权重映射。

## 可用模型

Megatron Bridge 支持以下 Llama Nemotron 模型变体：

- **Llama-3.3-Nemotron-Super-49B**: 490 亿参数（通过 NAS 从 700 亿参数优化而来）
- **Llama-3.1-Nemotron-Ultra-253B**: 2530 亿参数（大规模推理模型）
- **Llama-3.1-Nemotron-70B**: 700 亿参数（标准尺寸）
- **Llama-3.1-Nemotron-Nano-8B**: 80 亿参数（高效变体）
- **Llama-3.1-Nemotron-Nano-4B**: 40 亿参数（超紧凑变体）

所有模型均已准备好用于商业用途，并支持高达 128K 令牌的上下文长度。

## 模型架构特性

- **神经架构搜索（Neural Architecture Search, NAS）**：一种新颖的方法，可在保持准确性的同时减少内存占用
- **异构块（Heterogeneous Blocks）**：非标准且非重复的层配置以提高效率
  - 在某些块中跳过注意力机制
  - 块之间具有可变的前馈网络扩展/压缩比率
- **多阶段后训练（Multi-Phase Post-Training）**：
  - 针对数学、代码、科学和工具调用进行监督微调
  - 针对聊天的奖励感知偏好优化（Reward-aware Preference Optimization, RPO）
  - 针对推理的带可验证奖励的强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）
  - 针对工具调用的迭代直接偏好优化（Iterative Direct Preference Optimization, DPO）
- **扩展上下文（Extended Context）**：原生支持高达 128K 令牌的序列
- **商业就绪（Commercial Ready）**：完全授权用于商业部署

## 与 🤗 Hugging Face 的转换

### 加载 HF → Megatron

```python
from megatron.bridge import AutoBridge

# 示例：Llama-3.3-Nemotron-Super-49B
bridge = AutoBridge.from_hf_pretrained(
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    trust_remote_code=True
)
provider = bridge.to_megatron_provider()

# 可选地在实例化模型前配置并行策略
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

**注意**：异构的 Llama-Nemotron 模型（Super/Ultra）需要 `trust_remote_code=True`，因为它们使用自定义的 `DeciLMForCausalLM` 架构。同构模型（Nano/70B）使用标准的 Llama 架构，不需要此标志。

### 从 HF 导入检查点

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --megatron-path /checkpoints/llama_nemotron_super_49b_megatron \
  --trust-remote-code
```

### 导出 Megatron → HF

```python
from megatron.bridge import AutoBridge

# 从 HF 模型 ID 加载 bridge
bridge = AutoBridge.from_hf_pretrained(
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    trust_remote_code=True
)

# 将训练好的 Megatron 检查点导出为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/llama_nemotron_super_49b/checkpoints/iter_0000500",
    hf_path="/exports/llama_nemotron_super_49b_hf",
)
```

### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --megatron_model_path /checkpoints/llama_nemotron_super_49b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2 \
  --trust-remote-code
```

更多详情，请参阅 [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## 训练配方

Llama Nemotron 模型的训练配方目前暂不可用。

## Hugging Face 模型卡片与参考文献

### Hugging Face 模型卡片
- Llama Nemotron 合集：https://huggingface.co/collections/nvidia/llama-nemotron
- Llama-3.3-Nemotron-Super-49B-v1.5：https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
- Llama-3.1-Nemotron-Ultra-253B-v1：https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
- Llama-3.1-Nemotron-Nano-8B-v1：https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1
- Llama-3.1-Nemotron-Nano-4B-v1.1：https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1

### 技术论文
- Llama-Nemotron：高效推理模型：[arXiv:2505.00949](https://arxiv.org/abs/2505.00949)
- Puzzle：基于蒸馏的 NAS 用于推理优化的 LLM：[arXiv:2411.19146](https://arxiv.org/abs/2411.19146)
- 奖励感知偏好优化：[arXiv:2502.00203](https://arxiv.org/abs/2502.00203)

### 其他资源
- NVIDIA Build 平台：https://build.nvidia.com/
- Llama Nemotron 后训练数据集：https://huggingface.co/nvidia/Llama-Nemotron-Post-Training-Dataset

## 相关文档

- 配方使用：[配方使用](../../recipe-usage.md)
- 自定义训练配方配置：[配置概述](../../training/config-container-overview.md)
- 训练入口点：[入口点](../../training/entry-points.md)