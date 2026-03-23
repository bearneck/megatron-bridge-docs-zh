# OLMoE

[OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0125) 是来自 **Allen Institute for AI (AI2)** 的一个 70 亿参数的混合专家（Mixture-of-Experts, MoE）模型，具有 64 个专家并采用 top-8 路由。该模型设计为完全开源，其训练数据、代码和模型权重均公开可用。其名称为 "OLMoE-1B-7B"，其中 1B 指激活的参数数量，7B 指总参数数量。

最新版本（OLMoE-1B-7B-0125，发布于 2025 年 1 月）是 2024 年 9 月原始版本（OLMoE-1B-7B-0924）的改进版，在 5T token 上进行了训练，在多个基准测试中性能有所提升。

该模型具有 16 个解码器层，每层有 64 个路由专家，每个 token 激活 8 个专家，每次前向传播总共激活约 13 亿个参数，总参数为 70 亿。

OLMoE 模型通过 Bridge 系统获得支持，并配有专门针对 MoE 优化的配置。

## 模型架构

- **参数**：总计 70 亿，每次前向传播激活 13 亿
- **层数**：16 个解码器层
- **注意力机制**：具有 QK LayerNorm 和 RoPE 的多查询注意力（Multi-query attention）
- **MoE**：每层 64 个路由专家，采用 top-8 路由
- **隐藏层大小**：2048
- **FFN 隐藏层大小**：1024（密集层），1024（专家层）
- **注意力头**：16 个查询头，16 个键值组
- **词表大小**：50,304
- **上下文长度**：4K token
- **激活函数**：带有门控线性单元的 SiLU
- **训练数据量**：5T token（OLMoE-1B-7B-0125）

## 主要特性

- **QK LayerNorm**：对查询和键投影应用 LayerNorm 以提高训练稳定性
- **RoPE**：基数为 10000 的旋转位置编码（Rotary Position Embeddings）
- **MoE 路由**：基于 Softmax 的路由器，带有用于负载均衡的辅助损失
- **路由器预 Softmax**：预 Softmax 路由分数
- **分组 GEMM**：针对专家计算优化的分组矩阵乘法

## 使用 🤗 Hugging Face 进行转换

### 加载 HF → Megatron
```python
from megatron.bridge import AutoBridge

# 示例：OLMoE-1B-7B-0125（最新版本）
bridge = AutoBridge.from_hf_pretrained("allenai/OLMoE-1B-7B-0125")
provider = bridge.to_megatron_provider()

# 在实例化模型之前配置并行策略
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 8
provider.sequence_parallel = False

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
# 你也可以使用旧版本：
# bridge = AutoBridge.from_hf_pretrained("allenai/OLMoE-1B-7B-0924")
```

### 导出 Megatron → HF
```python
# 从 Megatron 检查点目录转换为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/olmoe_7b/checkpoints/iter_0500000",
    hf_path="./olmoe-hf-export",
)
```

## 示例

- 检查点转换：[examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)

## 配方

参见：[bridge.recipes.olmoe](../../apidocs/bridge/bridge.recipes.olmoe.md)

### 可用配方

- **预训练配方**：
  - `olmoe_7b_pretrain_config`：OLMoE-7B 的预训练（70 亿参数，每个 token 激活 13 亿）

- **SFT 配方**：
  - `olmoe_7b_sft_config`：OLMoE-7B 的完整监督微调
- **PEFT 配方**（LoRA, DoRA）：
  - `olmoe_7b_peft_config`：OLMoE-7B 的参数高效微调

### 并行配置

| 模型 | 模式 | TP | PP | EP | 总 GPU 数 | 使用场景 |
|-------|------|----|----|----|-----------:|----------|
| **OLMoE-7B** | 预训练 | 1 | 1 | 8 | 8 | 预训练（单节点） |
| **OLMoE-7B** | 完整 SFT | 1 | 1 | 8 | 8 | 完整监督微调 |
| **OLMoE-7B** | LoRA/DoRA | 1 | 1 | 1 | 8 | PEFT 微调（单节点） |

**主要特性**：
- **专家并行**：EP=8 用于高效的 MoE 训练（64 个专家）
- **选择性重计算**：默认启用以优化内存
- **RoPE 融合**：MLA 的可选优化（`apply_rope_fusion=True`）
- **MoE 优化**：默认启用分组 GEMM 和置换融合

**性能优化**：
- **MoE 置换融合**：融合的专家置换操作
- **分组 GEMM**：优化的专家计算
- **路由器负载均衡**：用于均衡专家利用率的辅助损失
- **手动垃圾回收**：积极的垃圾回收（间隔=5）
- **精度感知优化器**：BF16 梯度和优化器状态，FP32 主权重

**流水线布局**（可选）：
- **PP=1**：无流水线（默认）
- **PP=2**：8+8 层分割，包含嵌入/损失层
- **PP=4**：4+4+4+4 层分割
- **VP**：支持 PP=2,VP=2

### 预训练示例

```python
from megatron.bridge.recipes.olmoe import olmoe_7b_pretrain_config

cfg = olmoe_7b_pretrain_config(
    name="olmoe_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/olmoe_7b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # 自动使用 TP=1, PP=1, EP=8（8 个 GPU）
)
```

### 微调示例

#### 完整微调

from megatron.bridge.recipes.olmoe import olmoe_7b_sft_config

cfg = olmoe_7b_sft_config(
    tokenizer_path="allenai/OLMoE-1B-7B-0125",
    name="olmoe_full_sft",
    pretrained_checkpoint="path/to/olmoe/checkpoint",
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=5e-6,
    # 自动使用 TP=1, PP=1, EP=8 (8个GPU)
)
```

#### LoRA 微调

```python
from megatron.bridge.recipes.olmoe import olmoe_7b_peft_config

cfg = olmoe_7b_peft_config(
    tokenizer_path="allenai/OLMoE-1B-7B-0125",
    name="olmoe_lora_finetune",
    pretrained_checkpoint="path/to/olmoe/checkpoint",
    peft_scheme="lora",  # 或使用 "dora" 表示 DoRA
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # 自动使用 TP=1, PP=1, EP=1 (8个GPU)
)
```

## Hugging Face 模型卡片

### 最新版本 (2025年1月)
- OLMoE-1B-7B-0125 (基础版): [allenai/OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
- OLMoE-1B-7B-0125-SFT: [allenai/OLMoE-1B-7B-0125-SFT](https://huggingface.co/allenai/OLMoE-1B-7B-0125-SFT)
- OLMoE-1B-7B-0125-Instruct: [allenai/OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct)

### 先前版本 (2024年9月)
- OLMoE-1B-7B-0924 (基础版): [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- OLMoE-1B-7B-0924-Instruct: [allenai/OLMoE-1B-7B-0924-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct)

## 技术资源

- OLMoE 论文: [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
- OLMoE 模型卡片 (最新): [HuggingFace 模型卡片](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
- OLMoE GitHub 仓库: [allenai/OLMoE](https://github.com/allenai/OLMoE)

## 相关文档

- 配方使用与自定义: [配方使用](../../recipe-usage.md)
- 训练配置: [配置概述](../../training/config-container-overview.md)
- 训练入口点: [入口点](../../training/entry-points.md)