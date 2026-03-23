# Moonlight

[Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B) 是 **Moonshot AI** 推出的一个拥有 160 亿参数的混合专家模型（Mixture-of-Experts, MoE），它使用创新的 **Muon 优化器** 在 5.7 万亿个令牌上训练而成。虽然 Moonlight 与 DeepSeek-V3 共享相同的架构（包含多头潜在注意力（Multi-head Latent Attention）和 MoE），但它是一个独立的模型，通过使用 Muon 优化器，在性能与训练 FLOPs 的帕累托前沿上取得了进步。Muon 在计算最优训练下，其样本效率比 Adam 高出约 2 倍。

该模型包含 27 个解码器层，每层有 64 个路由专家和 8 个共享专家，每次前向传播激活 30 亿参数，总参数量为 160 亿。

Moonlight 模型通过 Bridge 系统获得支持，并针对 MoE 和 MLA 优化提供了专门的配置。

## 模型架构

- **参数**：总计 160 亿，每次前向传播激活 30 亿
- **层数**：27 个解码器层
- **注意力机制**：支持 RoPE 融合的多头潜在注意力（Multi-head Latent Attention, MLA）
- **MoE**：每层 64 个路由专家 + 8 个共享专家
- **隐藏层大小**：2048
- **中间层大小**：10944（包含 MLP 和专家门控）
- **词表大小**：151,936
- **上下文长度**：8K 令牌
- **训练**：使用 Muon 优化器在 5.7 万亿令牌上训练

## 使用 🤗 Hugging Face 进行转换

Moonlight 与 DeepSeek-V3 共享相同的架构，这使得它与 vLLM 和 SGLang 等各种推理引擎兼容。该模型可以从 HuggingFace 加载，或与 Megatron 检查点一起使用。

### 加载 HF → Megatron
```python
from megatron.bridge import AutoBridge

# 示例：Moonlight-16B-A3B
bridge = AutoBridge.from_hf_pretrained("moonshotai/Moonlight-16B-A3B")
provider = bridge.to_megatron_provider()

# 在实例化模型之前配置并行策略
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 8
provider.sequence_parallel = True

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 导出 Megatron → HF
```python
# 从 Megatron 检查点目录转换为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/moonlight_16b/checkpoints/iter_0500000",
    hf_path="./moonlight-hf-export",
)
```

## 示例

- 检查点转换：[examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)

## 配方

参见：[bridge.recipes.moonlight](../../apidocs/bridge/bridge.recipes.moonlight.md)

### 可用配方

- **预训练配方**：
  - `moonlight_16b_pretrain_config`：Moonlight-16B 的预训练（160 亿参数，每令牌激活 30 亿）

- **监督微调（SFT）配方**：
  - `moonlight_16b_sft_config`：Moonlight-16B 的完整监督微调
- **参数高效微调（PEFT）配方**（LoRA, DoRA）：
  - `moonlight_16b_peft_config`：Moonlight-16B 的参数高效微调

### 并行配置

| 模型 | 模式 | TP | PP | EP | 总计 GPU | 使用场景 |
|-------|------|----|----|----|-----------:|----------|
| **Moonlight-16B** | 预训练 | 2 | 1 | 8 | 16 | 预训练（2 个节点） |
| **Moonlight-16B** | 完整监督微调 | 2 | 1 | 8 | 16 | 完整监督微调（2 个节点） |
| **Moonlight-16B** | LoRA/DoRA | 1 | 1 | 1 | 8 | 参数高效微调（单节点！） |

**关键特性**：
- **专家并行（Expert Parallelism）**：EP=8 用于高效的 MoE 训练（64 个专家）
- **序列并行（Sequence Parallel）**：默认启用以提高内存效率
- **选择性重计算（Selective Recomputation）**：减少激活内存
- **RoPE 融合（RoPE Fusion）**：可选的 MLA 特定优化（`apply_rope_fusion=True`）
- **DeePEP**：可选的专家置换优化（`enable_deepep=True`）

**性能优化**：
- **MoE 置换融合（MoE Permute Fusion）**：融合的专家置换操作
- **RoPE 融合（RoPE Fusion）**：针对多头潜在注意力的可选融合
- **手动垃圾回收（Manual GC）**：积极的垃圾回收（间隔=5）
- **精度感知优化器（Precision-Aware Optimizer）**：BF16 梯度和优化器状态，FP32 主权重

**流水线布局**（可选）：
- **PP=1**：无流水线（默认）
- **PP=2**：14+13 层分割，包含嵌入层/损失层
- **PP=4**：8+7+7+6 层分割
- **PP=8**：5+4+4+4+4+4+4+4 层分割
- **VP**：支持 PP=2,VP=2 和 PP=4,VP=2

### 预训练示例

```python
from megatron.bridge.recipes.moonlight import moonlight_16b_pretrain_config

cfg = moonlight_16b_pretrain_config(
    name="moonlight_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/moonlight_16b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # 自动使用 TP=2, PP=1, EP=8（16 个 GPU）
)
```

### 微调示例

#### 完整微调（2 个节点）

```python
from megatron.bridge.recipes.moonlight import moonlight_16b_sft_config

cfg = moonlight_16b_sft_config(
    tokenizer_path="moonshotai/Moonlight-16B-A3B",
    name="moonlight_full_sft",
    pretrained_checkpoint="/results/moonlight_16b/checkpoints/iter_0500000",
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=5e-6,
    # 自动使用 TP=2, PP=1, EP=8（16 个 GPU）
)
```

#### LoRA 微调

```python
from megatron.bridge.recipes.moonlight import moonlight_16b_peft_config

cfg = moonlight_16b_peft_config(
    tokenizer_path="moonshotai/Moonlight-16B-A3B",
    name="moonlight_lora_finetune",
    pretrained_checkpoint="/results/moonlight_16b/checkpoints/iter_0500000",
    peft_scheme="lora",  # 或 "dora" 用于 DoRA
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # 自动使用 TP=1, PP=1, EP=1 (8 GPUs)
)
```

## Hugging Face 模型卡片

- Moonlight-16B-A3B (基础版): [moonshotai/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B)
- Moonlight-16B-A3B-Instruct: [moonshotai/Moonlight-16B-A3B-Instruct](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct)

## 技术论文

- Muon is Scalable for LLM Training: [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)

## 相关文档

- 配方使用与自定义: [配方使用](../../recipe-usage.md)
- 训练配置: [配置概述](../../training/config-container-overview.md)
- 训练入口点: [入口点](../../training/entry-points.md)