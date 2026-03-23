# GLM 4.5

[GLM 4.5](https://huggingface.co/zai-org/GLM-4.5) 是由智谱 AI 开发的一系列大规模专家混合（Mixture-of-Experts，MoE）语言模型。基于 GLM（通用语言模型）架构，GLM 4.5 引入了先进特性，包括稀疏 MoE 层、共享专家机制和多令牌预测（Multi-Token Prediction，MTP），以提高训练效率和性能。

GLM 4.5 模型专为高性能文本生成和理解任务设计，通过稀疏激活提供卓越的质量和高效的推理。该系列包含两个针对不同部署场景优化的变体。

GLM 4.5 模型通过 Bridge 系统得到支持，具有自动检测的配置和权重映射。

## 可用模型

### GLM 4.5 355B-A32B
**完整模型：** `zai-org/GLM-4.5`（总参数量 355B，每令牌激活 32B）
- **架构：**
  - 92 个 Transformer 层（前 3 层为稠密层，其余 89 层为 MoE 层）
  - 隐藏层大小 5120，FFN 隐藏层大小 12288
  - 96 个注意力头，8 个查询组（GQA）
  - 每个 MoE 层 160 个专家，top-8 路由，缩放因子 2.5x
  - MoE FFN 隐藏层大小：1536
  - 共享专家中间层大小：1536
  - 包含 QK LayerNorm 以提高训练稳定性
- **上下文长度：** 131,072 个令牌
- **词表大小：** 151,552 个令牌
- **优化：**
  - 共享专家重叠以实现更好的负载均衡
  - 带专家偏置的路由器 sigmoid 评分
  - MTP（多令牌预测），1 层，缩放因子 0.3

### GLM 4.5 Air 106B-A12B
**完整模型：** `zai-org/GLM-4.5-Air`（总参数量 106B，每令牌激活 12B）
- **架构：**
  - 46 个 Transformer 层（第一层为稠密层，其余 45 层为 MoE 层）
  - 隐藏层大小 4096，FFN 隐藏层大小 10944
  - 96 个注意力头，8 个查询组（GQA）
  - 每个 MoE 层 128 个专家，top-8 路由，缩放因子 1.0x
  - MoE FFN 隐藏层大小：1408
  - 共享专家中间层大小：1408
  - 无 QK LayerNorm
- **上下文长度：** 131,072 个令牌
- **词表大小：** 151,552 个令牌
- **优化：**
  - 针对减少内存占用进行优化
  - 适用于中端 GPU 集群

两个模型均使用 RMSNorm、SiLU 激活函数、门控线性单元以及基频为 1M 的 RoPE。

## 模型架构特性

GLM 4.5 引入了多项先进的架构创新：

- **专家混合（Mixture-of-Experts，MoE）**：稀疏激活，包含 160/128 个专家，每令牌仅激活 8 个，以实现高效扩展
- **共享专家机制**：专用的共享专家，具有重叠设计，以改善负载均衡和知识迁移
- **多令牌预测（Multi-Token Prediction，MTP）**：同时预测多个未来令牌，以提高训练效率
  - 可配置的 MTP 层数（默认：1 层）
  - 可调整的损失缩放因子（默认：训练早期为 0.3，后期为 0.1）
- **扩展上下文的 RoPE**：基频为 1M 的旋转位置嵌入，用于鲁棒的长上下文建模
- **高级路由器设计**：
  - Sigmoid 评分函数，用于更好的专家选择
  - 带可配置更新率的专家偏置
  - 用于负载均衡的辅助损失
- **分组查询注意力（Grouped Query Attention，GQA）**：8 个查询组，用于高效注意力计算
- **RMSNorm**：无需均值中心化的快速层归一化

## 使用 🤗 Hugging Face 进行转换

### 加载 HF → Megatron
```python
from megatron.bridge import AutoBridge

# 示例：GLM 4.5 Air 106B
bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-4.5-Air")
provider = bridge.to_megatron_provider()

# 在实例化模型之前配置并行策略
# 对于 Air 106B：TP=1, PP=4, EP=8 (32 GPUs)
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 4
provider.expert_model_parallel_size = 8
provider.sequence_parallel = True

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 导入 HF → Megatron
```bash
# 导入 GLM 4.5 Air 模型
python examples/conversion/convert_checkpoints.py import \
--hf-model zai-org/GLM-4.5-Air \
--megatron-path /models/glm45-air-106b

# 导入 GLM 4.5 355B 模型
python examples/conversion/convert_checkpoints.py import \
--hf-model zai-org/GLM-4.5 \
--megatron-path /models/glm45-355b
```

### 导出 Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model zai-org/GLM-4.5-Air \
--megatron-path /results/glm45_air/checkpoints/iter_00001000 \
--hf-path ./glm45-air-hf-export
```

### 在转换后的检查点上运行推理
```bash
python examples/conversion/hf_to_megatron_generate_text.py \
--hf_model_path zai-org/GLM-4.5-Air \
--megatron_model_path /models/glm45-air-106b \
--prompt "Explain quantum computing in simple terms." \
--max_new_tokens 200
```

## 预训练和微调配方

### 可用配方
- 预训练配方：
  - `glm45_355b_pretrain_config`：GLM 4.5 355B 的预训练
  - `glm45_air_106b_pretrain_config`：GLM 4.5 Air 106B 的预训练
- SFT 配方：
  - `glm45_355b_sft_config`：GLM 4.5 355B 的完整 SFT
  - `glm45_air_106b_sft_config`：GLM 4.5 Air 106B 的完整 SFT

- PEFT 配方（LoRA、DoRA）：
  - `glm45_355b_peft_config`：用于 GLM 4.5 355B 的 PEFT
  - `glm45_air_106b_peft_config`：用于 GLM 4.5 Air 106B 的 PEFT

### 并行配置

#### GLM 4.5 355B-A32B
| 模式 | TP | PP | EP | 总 GPU 数 | 使用场景 |
|------|----|----|----|-----------:|----------|
| **预训练** | 2 | 8 | 16 | 256 | 完整预训练 |
| **PEFT (LoRA/DoRA)** | 2 | 4 | 4 | 32 | 参数高效微调 |
| **完整 SFT** | 2 | 8 | 16 | 256 | 完整监督微调 |

#### GLM 4.5 Air 106B-A12B
| 模式 | TP | PP | EP | 总 GPU 数 | 使用场景 |
|------|----|----|----|-----------:|----------|
| **预训练** | 1 | 4 | 8 | 32 | 完整预训练 |
| **PEFT (LoRA/DoRA)** | 1 | 2 | 4 | 8 | 参数高效微调（单节点！） |
| **完整 SFT** | 1 | 4 | 8 | 32 | 完整监督微调 |

### 预训练示例

```python
from megatron.bridge.recipes.glm import glm45_air_106b_pretrain_config

# 创建预训练配置
config = glm45_air_106b_pretrain_config(
    name="glm45_air_pretrain",
    data_paths=["path/to/data"],
    train_iters=100000,
    global_batch_size=2048,
    micro_batch_size=1,
    lr=1e-4,
    min_lr=1e-5,
    # MTP 配置
    mtp_num_layers=1,
    mtp_loss_scaling_factor=0.3,  # 前 15T token 用 0.3，之后用 0.1
    # 为内存效率启用重计算
    recompute_granularity="selective",
)
```

### 微调示例

#### 完整微调（GLM 4.5 Air）
```python
from megatron.bridge.recipes.glm import glm45_air_106b_sft_config

config = glm45_air_106b_sft_config(
    name="glm45_air_full_finetune",
    pretrained_checkpoint="/models/glm45-air-106b",
    train_iters=1000,
    global_batch_size=128,
    micro_batch_size=1,
    finetune_lr=5e-6,
)
```

#### LoRA 微调（GLM 4.5 Air）
```python
from megatron.bridge.recipes.glm import glm45_air_106b_peft_config

config = glm45_air_106b_peft_config(
    name="glm45_air_lora_finetune",
    pretrained_checkpoint="/models/glm45-air-106b",
    peft_scheme="lora",  # 或 "dora"
    train_iters=1000,
    global_batch_size=128,
    micro_batch_size=1,
    finetune_lr=1e-4,
    # 自动使用 TP=1, PP=2, EP=4 (8 GPUs)
)
```

#### DoRA 微调（GLM 4.5 355B）
```python
from megatron.bridge.recipes.glm import glm45_355b_peft_config

config = glm45_355b_peft_config(
    name="glm45_355b_dora_finetune",
    pretrained_checkpoint="/models/glm45-355b",
    peft_scheme="dora",
    train_iters=1000,
    global_batch_size=128,
    micro_batch_size=1,
    finetune_lr=1e-4,
    # 自动使用 TP=2, PP=4, EP=4 (32 GPUs)
)
```

### 命令行训练

```bash
# GLM 4.5 Air - 单节点 LoRA 微调 (8 GPUs)
torchrun --nproc-per-node=8 run/run_recipe.py \
--pretrained-checkpoint /models/glm45-air-106b \
--recipe glm45_air_106b_peft_config \
--peft_scheme lora \
train.global_batch_size=128 \
train.train_iters=1000 \
checkpoint.save=$SAVE_DIR/glm45_air_lora

# GLM 4.5 355B - 完整微调 (256 GPUs)
torchrun --nnodes=32 --nproc-per-node=8 run/run_recipe.py \
--pretrained-checkpoint /models/glm45-355b \
--recipe glm45_355b_sft_config \
train.global_batch_size=256 \
train.train_iters=1000 \
checkpoint.save=$SAVE_DIR/glm45_355b_full
```

## 高级配置

### 多令牌预测（MTP）
可以配置 MTP 以提高训练效率：

```python
config = glm45_355b_pretrain_config(
    name="glm45_with_mtp",
    mtp_num_layers=1,  # MTP 预测层数
    mtp_loss_scaling_factor=0.3,  # 早期训练 0.3，后期 0.1
    # 设置为 None 或 0 以禁用 MTP
)
```

### 激活重计算
适用于内存受限的场景：

```python
config = glm45_air_106b_pretrain_config(
    name="glm45_with_recompute",
    recompute_granularity="selective",  # 或 "full"
    recompute_method="uniform",
    recompute_num_layers=2,
)
```

### 专家并行调优
根据您的集群调整专家并行：

```python
config = glm45_air_106b_pretrain_config(
    name="glm45_custom_parallelism",
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=16,  # 根据 GPU 数量调整
    sequence_parallel=True,
)
```

## 示例
- 检查点导入/导出：[examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- 生成文本（HF→Megatron）：[examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Hugging Face 模型卡片
- GLM 4.5 355B：https://huggingface.co/zai-org/GLM-4.5
- GLM 4.5 Air 106B：https://huggingface.co/zai-org/GLM-4.5-Air

## 生产环境关键特性

1. **高效扩展**：MoE 架构支持 355B 参数，但每个令牌仅激活 32B
2. **单节点 PEFT**：GLM 4.5 Air 可以在仅 8 个 GPU 上使用 LoRA/DoRA 进行微调
3. **长上下文**：原生 131K 令牌上下文窗口，并优化了 RoPE

4. **多令牌预测（Multi-Token Prediction）**：通过 MTP 训练实现更快的收敛
5. **灵活部署**：支持多种并行策略，适用于不同的硬件配置
6. **负载均衡**：共享专家重叠和辅助损失，实现最佳的专家利用率

## 相关文档
- 配方使用：[配方使用](../../recipe-usage.md)
- 自定义训练配方配置：[配置概述](../../training/config-container-overview.md)
- 训练入口点：[入口点](../../training/entry-points.md)