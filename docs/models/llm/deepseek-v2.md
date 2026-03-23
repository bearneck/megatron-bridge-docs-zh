# DeepSeek V2

[DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2) 是一个混合专家（Mixture-of-Experts，MoE）语言模型，它采用创新的多头潜在注意力（Multi-head Latent Attention，MLA）机制以实现高效推理，并使用 DeepSeekMoE 架构以实现经济的训练和推理。该模型在激活参数显著减少的情况下，实现了与 GPT-4 相媲美的性能。更多信息请参阅配套论文 ["DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"](https://arxiv.org/abs/2405.04434)。

DeepSeek V2 模型通过 Bridge 系统得到支持，具有自动检测的配置和权重映射。

## 可用模型

Megatron Bridge 支持以下 DeepSeek V2 模型变体：

- **DeepSeek-V2**: 2360 亿参数（每个令牌激活 210 亿参数）
- **DeepSeek-V2-Lite**: 160 亿参数（每个令牌激活 24 亿参数）

两个模型都支持使用专家并行（expert parallelism）进行高效的 MoE 预训练。

## 模型架构特性

- **多头潜在注意力（Multi-head Latent Attention，MLA）**：新颖的注意力机制，可减少 KV 缓存需求
- **DeepSeekMoE**：高效的 MoE 架构，包含路由专家和共享专家
- **专家并行（Expert Parallelism）**：将专家分布在多个 GPU 上以实现可扩展的训练
- **RoPE 嵌入（RoPE Embeddings）**：用于位置编码的旋转位置嵌入
- **128K 上下文长度**：原生支持长序列（DeepSeek-V2）
- **预归一化（Pre-normalization）**：在每个 Transformer 子层之前使用 RMSNorm

## 与 🤗 Hugging Face 的转换

### 加载 HF → Megatron

```python
from megatron.bridge import AutoBridge

# 示例：DeepSeek-V2-Lite
bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)
provider = bridge.to_megatron_provider()

# 可选地在实例化模型前配置并行策略
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 8

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 从 HF 导入检查点

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model deepseek-ai/DeepSeek-V2-Lite \
  --megatron-path /checkpoints/deepseek_v2_lite_megatron \
  --trust-remote-code
```

### 导出 Megatron → HF

```python
from megatron.bridge import AutoBridge

# 从 HF 模型 ID 加载 bridge
bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)

# 将训练好的 Megatron 检查点导出为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/deepseek_v2_lite/checkpoints/iter_0000500",
    hf_path="/exports/deepseek_v2_lite_hf",
)
```

### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path deepseek-ai/DeepSeek-V2-Lite \
  --megatron_model_path /checkpoints/deepseek_v2_lite_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --ep 8 \
  --trust-remote-code
```

更多详情，请参阅 [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## 配方

参见：[bridge.recipes.deepseek.deepseek_v2](../../apidocs/bridge/bridge.recipes.deepseek.deepseek_v2.md)

### 可用配方

- **预训练配方**：
  - `deepseek_v2_lite_pretrain_config`: DeepSeek-V2-Lite 的预训练（160 亿参数，每个令牌激活 24 亿参数）
  - `deepseek_v2_pretrain_config`: DeepSeek-V2 的预训练（2360 亿参数，每个令牌激活 210 亿参数）

### 并行配置

| 模型 | TP | PP | EP | 总 GPU 数 | 使用场景 |
|-------|----|----|----|-----------:|----------|
| **DeepSeek-V2-Lite** | 1 | 1 | 8 | 8 | 预训练（单节点） |
| **DeepSeek-V2** | 1 | 4 | 32 | 128 | 预训练（16 节点） |

**关键特性**：
- **专家并行（Expert Parallelism）**：EP=8（V2-Lite）或 EP=32（V2），用于高效的 MoE 训练
- **选择性重计算（Selective Recomputation）**：默认启用以优化内存
- **序列长度**：默认 4096，V2 支持高达 128K 个令牌

### 预训练示例

#### DeepSeek-V2-Lite (16B)

```python
from megatron.bridge.recipes.deepseek import deepseek_v2_lite_pretrain_config

config = deepseek_v2_lite_pretrain_config(
    name="deepseek_v2_lite_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/deepseek_v2_lite",
    train_iters=500_000,
    global_batch_size=512,
    seq_length=4096,
    # 自动使用 TP=1, PP=1, EP=8 (8 GPUs)
)
```

#### DeepSeek-V2 (236B)

```python
from megatron.bridge.recipes.deepseek import deepseek_v2_pretrain_config

config = deepseek_v2_pretrain_config(
    name="deepseek_v2_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/deepseek_v2",
    train_iters=500_000,
    global_batch_size=512,
    seq_length=4096,
    # 自动使用 TP=1, PP=4, EP=32 (128 GPUs)
)
```

### 微调配方

DeepSeek V2 模型的微调配方目前暂不可用。

## Hugging Face 模型卡片与参考资料

### Hugging Face 模型卡片
- DeepSeek-V2: https://huggingface.co/deepseek-ai/DeepSeek-V2
- DeepSeek-V2-Lite: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
- DeepSeek-V2-Chat: https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat
- DeepSeek-V2-Lite-Chat: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat

### 技术论文
- DeepSeek-V2: 一个强大、经济且高效的专家混合语言模型: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

### 其他资源
- GitHub 仓库: https://github.com/deepseek-ai/DeepSeek-V2

## 相关文档
- 配方使用: [配方使用](../../recipe-usage.md)
- 自定义训练配方配置: [配置概述](../../training/config-container-overview.md)
- 训练入口点: [入口点](../../training/entry-points.md)