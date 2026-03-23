# DeepSeek V3

[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) 是一个大规模混合专家（Mixture-of-Experts, MoE）语言模型，总参数量为 671B，每个 token 激活 37B 参数。它采用了多头潜在注意力（Multi-head Latent Attention, MLA）、创新的负载均衡策略以及多令牌预测（Multi-Token Prediction, MTP）以提高训练效率。DeepSeek-V3 在保持经济训练成本的同时，实现了最先进的性能。更多信息请参阅技术报告 ["DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437)。

DeepSeek V3 模型通过 Bridge 系统提供支持，具有自动检测的配置和权重映射。

## 可用模型

Megatron Bridge 支持以下 DeepSeek V3 模型变体：

- **DeepSeek-V3**: 671B 参数（每个 token 激活 37B）
- **DeepSeek-V3-Base**: 未经指令调优的预训练基础模型

该模型支持使用专家并行（expert parallelism）、管道并行（pipeline parallelism）以及可选的多令牌预测（Multi-Token Prediction, MTP）进行预训练。

## 模型架构特性

- **多头潜在注意力（Multi-head Latent Attention, MLA）**: 先进的注意力机制，可减少 KV 缓存并提高效率
- **DeepSeekMoE**: 增强的 MoE 架构，包含 256 个路由专家和共享专家
- **多令牌预测（Multi-Token Prediction, MTP）**: 预测多个未来 token 的辅助训练目标
- **专家并行（Expert Parallelism）**: 将 256 个专家分布在多个 GPU 上以实现可扩展训练
- **RoPE 嵌入（RoPE Embeddings）**: 带有缩放因子的旋转位置嵌入，用于位置编码
- **带专家偏置的 Sigmoid 门控（Sigmoid Gating with Expert Bias）**: 具有可学习专家偏置的新型路由机制
- **预归一化（Pre-normalization）**: 在每个 Transformer 子层之前使用 RMSNorm 以确保训练稳定性

## 使用 🤗 Hugging Face 进行转换

### 加载 HF → Megatron

```python
from megatron.bridge import AutoBridge

# 示例：DeepSeek-V3-Base
bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)
provider = bridge.to_megatron_provider()

# 可选地在实例化模型前配置并行策略
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 16
provider.expert_model_parallel_size = 64

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 从 HF 导入检查点

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model deepseek-ai/DeepSeek-V3-Base \
  --megatron-path /checkpoints/deepseek_v3_megatron \
  --trust-remote-code
```

### 导出 Megatron → HF

```python
from megatron.bridge import AutoBridge

# 从 HF 模型 ID 加载 bridge
bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)

# 将训练好的 Megatron 检查点导出为 HF 格式
bridge.export_ckpt(
    megatron_path="/results/deepseek_v3/checkpoints/iter_0000500",
    hf_path="/exports/deepseek_v3_hf",
)
```

### 在转换后的检查点上运行推理

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path deepseek-ai/DeepSeek-V3-Base \
  --megatron_model_path /checkpoints/deepseek_v3_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2 \
  --pp 16 \
  --ep 64 \
  --trust-remote-code
```

更多详情，请参阅 [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## 配方

参见：[bridge.recipes.deepseek.deepseek_v3](../../apidocs/bridge/bridge.recipes.deepseek.deepseek_v3.md)

### 可用配方

- **预训练配方**:
  - `deepseek_v3_pretrain_config`: DeepSeek-V3 的预训练（671B 参数，每个 token 激活 37B）

### 并行配置

| 模型 | TP | PP | EP | VP | 节点数 | 总 GPU 数 | 使用场景 |
|-------|----|----|-----|-----|------:|-----------:|----------|
| **DeepSeek-V3** | 2 | 16 | 64 | None | 128 | 1024 | 预训练 |

**关键特性**:
- **专家并行（Expert Parallelism）**: EP=64，用于将 256 个专家分布在 GPU 上
- **管道并行（Pipeline Parallelism）**: PP=16，采用针对嵌入层和损失层优化的非对称布局
- **选择性重计算（Selective Recomputation）**: 默认启用，用于内存优化
- **多令牌预测（Multi-Token Prediction, MTP）**: 可选的辅助训练目标（默认 1 层）
- **序列并行（Sequence Parallel）**: 默认启用，以提高内存效率

**性能优化**:
- **MoE 置换融合（MoE Permute Fusion）**: 融合的专家置换操作
- **Flex 调度器后端（Flex Dispatcher Backend）**: 可选的高性能 MoE token 调度器
- **RoPE 融合（RoPE Fusion）**: 针对多头潜在注意力的可选融合操作
- **精度感知优化器（Precision-Aware Optimizer）**: FP32 主权重，BF16 梯度和优化器状态

### 预训练示例

```python
from megatron.bridge.recipes.deepseek import deepseek_v3_pretrain_config

config = deepseek_v3_pretrain_config(
    name="deepseek_v3_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/deepseek_v3",
    train_iters=500_000,
    global_batch_size=4096,
    seq_length=4096,

# MTP 配置
mtp_num_layers=1,
mtp_loss_scaling_factor=0.1,
# 自动使用 TP=2, PP=16, EP=64 (1024 个 GPU, 128 个节点)
```

### 微调配方

DeepSeek V3 的微调配方目前暂不可用。

## Hugging Face 模型卡片与参考资料

### Hugging Face 模型卡片
- DeepSeek-V3: https://huggingface.co/deepseek-ai/DeepSeek-V3
- DeepSeek-V3-Base: https://huggingface.co/deepseek-ai/DeepSeek-V3-Base

### 技术论文
- DeepSeek-V3 技术报告: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

### 其他资源
- GitHub 仓库: https://github.com/deepseek-ai/DeepSeek-V3

## 相关文档
- 配方使用: [配方使用](../../recipe-usage.md)
- 自定义训练配方配置: [配置概述](../../training/config-container-overview.md)
- 训练入口点: [入口点](../../training/entry-points.md)