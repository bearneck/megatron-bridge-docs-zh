# NeMo 2.0 到 Megatron Bridge 迁移指南

本指南帮助您从 NeMo 2.0 的训练和配方迁移到 Megatron Bridge。Megatron Bridge 保留了 NeMo 2.0 开发的 Python 化、代码优先的 API，同时将配置简化为一个具有类型化子配置的单一 {py:class}`bridge.training.config.ConfigContainer`。来自 Megatron Core 的模型并行和性能特性仍然是一流的。

## 保持不变的部分

- **Megatron Core 基础**：Megatron Bridge 在底层使用相同的 Megatron Core 引擎。
- **模型并行**：相同的 TP/PP/CP/EP 概念，具有相同的分布式训练语义。
- **高性能特性**：混合精度、通信重叠和其他性能特性都得到原生支持。
- **保留 Python 化 API**：Megatron Bridge 保留了 NeMo 2.0 的“配置即代码”理念。

## 模型配置映射

Megatron Bridge 提供了直接映射到 NeMo 2.0 模型配置的模型提供者。

### 示例

| NeMo 2.0 | Megatron Bridge |
|----------|-----------------|
| `llm.Llama3Config8B` | {py:class}`bridge.models.Llama3ModelProvider8B` |
| `llm.Llama31Config70B` | {py:class}`bridge.models.Llama31ModelProvider70B` |
| `llm.Qwen2Config7B` | {py:class}`bridge.models.Qwen2ModelProvider7B` |
| `llm.DeepseekV2Config` | {py:class}`bridge.models.DeepseekV2ModelProvider` |

### 支持的模型系列

Megatron Bridge 通过预设提供者支持以下模型系列：
- **基础模型**：`GPTModelProvider`、`T5ModelProvider`、`MambaModelProvider`
- **Llama**：Llama2、Llama3、Llama3.1、Llama3.2、CodeLlama、Llama4
- **Qwen**：Qwen2、Qwen2.5、Qwen3、Qwen3MoE、Qwen2.5VL
- **DeepSeek**：DeepSeek、DeepSeekV2、DeepSeekV2Lite、DeepSeekV3、Moonlight
- **Nemotron**：Nemotron3、Nemotron4、NemotronH、NemotronNano
- **NVIDIA Mamba**：Mamba 变体和混合模型

有关所有模型提供者及其参数的完整列表，请参阅 {py:mod}`bridge.models`。

<!-- TODO: 创建一个包含经过测试的 HF 检查点映射的专用模型支持表 -->

---

## 快速开始：迁移示例

本节展示了常见训练场景的完整迁移示例。有关详细的配置映射，请参阅[配置迁移](#configuration-migration)。有关入口点 API 的详细信息，请参阅[入口点](#entry-points-pretrain-and-finetune)。

### 预训练迁移示例

#### 之前：NeMo 2.0
```python
from nemo import lightning as nl
from nemo.collections import llm
import nemo_run as run
from megatron.core.distributed import DistributedDataParallelConfig

# 模型配置
model = run.Config(
    llm.LlamaModel,
    config=run.Config(llm.Llama3Config8B),  # 包含所有默认值的预设配置
)

# 包含并行设置的策略
strategy = run.Config(
    nl.MegatronStrategy,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    context_parallel_size=1,
    sequence_parallel=False,
    ddp=run.Config(
        DistributedDataParallelConfig,
        grad_reduce_in_fp32=True,
    ),
)

# 训练器设置
trainer = run.Config(
    nl.Trainer,
    max_steps=1000,
    val_check_interval=100,
    limit_val_batches=50,
    log_every_n_steps=10,
    devices=8,
    num_nodes=1,
    strategy=strategy,
    plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
)

# 数据配置
data = run.Config(
    llm.PreTrainingDataModule,
    paths="/path/to/data_text_document",
    seq_length=8192,
    micro_batch_size=1,
    global_batch_size=512,
)

# 优化器配置
optim = llm.distributed_fused_adam_with_cosine_annealing(
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=100,
)

# 执行训练
llm.pretrain(model, data, trainer, optim=optim)
```

#### 现在：Megatron Bridge
```python
# Megatron Bridge 配置模式
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    TrainingConfig,
)
from megatron.bridge.models import Llama3ModelProvider8B  # Llama3Config8B 的直接等效项
from megatron.core.optimizer import OptimizerConfig
from megatron.bridge.training.config import SchedulerConfig
from megatron.bridge.training.pretrain import pretrain
# 使用提供的 GPT 前向步骤
from megatron.bridge.training.gpt_step import forward_step

def create_config():
    return ConfigContainer(
        # 内置并行性的模型 - 使用预设的 8B 配置
        model=Llama3ModelProvider8B(
            # 并行设置（从 MegatronStrategy 移入）
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
            sequence_parallel=False,
            # 如果需要，仍然可以覆盖任何模型参数
            seq_length=8192,
        ),
        # 训练循环配置
        train=TrainingConfig(
            global_batch_size=512,
            micro_batch_size=1,
            train_iters=1000,           # 原为 max_steps

```python
eval_interval=100,          # 原为 val_check_interval
            eval_iters=50,              # 原为 limit_val_batches
        ),
        # 优化与调度
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=3e-4,
            min_lr=3e-5,
            use_distributed_optimizer=True,
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="cosine",
            lr_warmup_iters=100,
            lr_decay_iters=1000,
        ),
        # 数据配置
        dataset=GPTDatasetConfig(
            blend=["/path/to/data_text_document"],
            seq_length=8192,
        ),
        # 检查点与日志记录
        checkpoint=CheckpointConfig(
            save="/path/to/checkpoints",
            save_interval=100,
            ckpt_format="torch_dist",
        ),
        logger=LoggerConfig(log_interval=10),  # 原为 log_every_n_steps
        # 混合精度
        mixed_precision="bf16_mixed",
    )

# 执行训练
cfg = create_config()

pretrain(cfg, forward_step_func=forward_step)
```

### 微调迁移示例 (SFT/PEFT)

对于微调，使用 {py:class}`bridge.training.config.FinetuningDatasetConfig` 配置数据，并将 `checkpoint.pretrained_checkpoint` 设置为基座模型。可选地添加 `peft` 配置以进行参数高效训练。

#### 之前：NeMo 2.0
```python
from nemo import lightning as nl
from nemo.collections import llm
import nemo_run as run

# 模型和训练器配置
model = run.Config(llm.LlamaModel, config=run.Config(llm.Llama3Config8B))
trainer = run.Config(
    nl.Trainer,
    max_steps=500,
    val_check_interval=100,
    devices=8,
    num_nodes=1,
)

# 数据配置
data = run.Config(
    llm.FineTuningDataModule,
    dataset_root="/path/to/sft/data",
    seq_length=2048,
    micro_batch_size=1,
    global_batch_size=128,
)

# PEFT 配置
lora = llm.peft.LoRA(
    target_modules=['linear_qkv', 'linear_proj'],
    dim=32,
    alpha=16,
)

# 使用 PEFT 执行微调
llm.finetune(
    model=model,
    data=data,
    trainer=trainer,
    peft=lora,
    tokenizer="model",
)
```

#### 现在：Megatron Bridge
```python
# Megatron Bridge 微调配置（含可选 PEFT）
from megatron.bridge.models import Llama3ModelProvider8B
from megatron.bridge.peft import LoRA

def create_finetune_config():
    return ConfigContainer(
        model=Llama3ModelProvider8B(
            # 预设配置，匹配 Llama3Config8B
        ),
        train=TrainingConfig(
            micro_batch_size=1,
            global_batch_size=128,
            train_iters=500,
        ),
        # 使用微调数据集而非预训练数据集
        dataset=FinetuningDatasetConfig(
            dataset_root="/path/to/sft/data",
            seq_length=2048,
            do_validation=True,
            do_test=True,
            # 可选：打包序列支持
            packed_sequence_specs=PackedSequenceSpecs(
                packed_sequence_size=2048,
            ),
        ),
        # 必须指定预训练检查点
        checkpoint=CheckpointConfig(
            pretrained_checkpoint="/path/to/pretrained/model",
            save="/path/to/sft/checkpoints",
            load="/path/to/sft/checkpoints",
            save_interval=50,
        ),
        # 可选：启用 PEFT
        peft=LoRA(
            target_modules=["linear_qkv", "linear_proj"],
            dim=32,
            alpha=16,
        ),
        # ... 其他配置
    )
```

---

## 配方迁移

NeMo 2.0 和 Megatron Bridge 都为流行模型提供了预构建的配方。在 NeMo 2.0 中，配方返回 `run.Partial` 配置。Megatron Bridge 配方返回 `ConfigContainer` 对象。

### 使用预构建配方

两个框架都提供了可自定义的即用型配方：

**NeMo 2.0**: 配方位于 `nemo.collections.llm.recipes/`
```python
from nemo.collections import llm

# 使用预构建配方
recipe = llm.llama3_8b.pretrain_recipe(name="my_run", num_nodes=2)
```

**Megatron Bridge**: 配方位于 `megatron.bridge.recipes/`
```python
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.training import pretrain
from megatron.bridge.training.gpt_step import forward_step

# 使用预构建配方
cfg = pretrain_config()

# 按需自定义
cfg.train.train_iters = 10000
cfg.model.tensor_model_parallel_size = 4

# 启动训练
pretrain(cfg, forward_step_func=forward_step)
```

有关使用和自定义配方的详细信息，请参阅 {doc}`recipe-usage`。

### 迁移自定义配方

如果您创建了自定义的 NeMo 2.0 配方，以下是如何将其迁移到 Megatron Bridge：

#### 之前：NeMo 2.0 配方结构

```python
# nemo/collections/llm/recipes/llama3_8b.py
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm

@run.cli.factory(name="llama3_8b")
def model() -> run.Config[pl.LightningModule]:
```

    return run.Config(llm.LlamaModel, config=run.Config(llm.Llama3Config8B))

def trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1000,
) -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
    )
    return run.Config(
        nl.Trainer,
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        max_steps=max_steps,
        strategy=strategy,
        val_check_interval=100,
        limit_val_batches=50,
    )

@run.cli.factory(target=llm.pretrain, name="llama3_8b")
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    return run.Partial(
        llm.pretrain,
        model=model(),
        trainer=trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node),
        data=run.Config(
            llm.PreTrainingDataModule,
            paths="/path/to/data_text_document",
            seq_length=8192,
            global_batch_size=512,
            micro_batch_size=1,
        ),
        log=llm.default_log(dir=dir, name=name),
        optim=llm.distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=llm.default_resume(),
    )

# 使用方式
if __name__ == "__main__":
    recipe = pretrain_recipe(name="my_run", num_nodes=2)
    # 通过 nemo-run 提交或直接执行
```

#### 现在：Megatron Bridge 配方结构

```python
# my_recipes/llama3_8b.py
from typing import Optional
from megatron.bridge.training.config import (
    ConfigContainer,
    TrainingConfig,
    GPTDatasetConfig,
    CheckpointConfig,
    SchedulerConfig,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.bridge.models import Llama3ModelProvider8B
from megatron.bridge.training import pretrain

def llama3_8b_config(
    # 模型/并行参数
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    # 训练参数
    train_iters: int = 1000,
    eval_interval: int = 100,
    eval_iters: int = 50,
    # 数据参数
    data_path: str = "/path/to/data_text_document",
    seq_length: int = 8192,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    # 检查点参数
    checkpoint_dir: Optional[str] = None,
    save_interval: int = 1000,
) -> ConfigContainer:
    """创建 Llama3 8B 预训练配置。"""
    return ConfigContainer(
        model=Llama3ModelProvider8B(
            # 来自 Llama3Config8B 的预设架构（num_layers=32, hidden_size=4096 等）
            # 只需指定并行度和覆盖项
            tensor_model_parallel_size=tensor_parallelism,
            pipeline_model_parallel_size=pipeline_parallelism,
        ),
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        dataset=GPTDatasetConfig(
            blend=[data_path],
            seq_length=seq_length,
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=3e-4,
            use_distributed_optimizer=True,
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="cosine",
            lr_warmup_iters=100,
        ),
    checkpoint=CheckpointConfig(
            save=checkpoint_dir or "/results/checkpoints",
            save_interval=save_interval,
        ),
        mixed_precision="bf16-mixed",
    )

# 使用方式
if __name__ == "__main__":
    from megatron.bridge.training.gpt_step import forward_step
    
    cfg = llama3_8b_config(
        train_iters=10000,
        checkpoint_dir="/my/checkpoints",
        tensor_parallelism=2,
    )
    pretrain(cfg, forward_step_func=forward_step)
```

**迁移步骤：**
1. 将 `run.Partial` 替换为返回 `ConfigContainer` 的函数
2. 将所有 `trainer`、`strategy` 和分布式设置移至模型提供者中
3. 将 `log`、`optim`、`resume` 整合到各自的配置对象中
4. 移除 `@run.cli.factory` 装饰器（可选：使用你自己的 CLI 框架）
5. 使用 `torchrun` 或类似的启动器启动——设备数量不再传递给训练函数

---

## 配置迁移

### 概述
过去分布在 Lightning `Trainer` 参数、回调和 `MegatronStrategy` 参数中的配置，现在集中到一组配置类中：

| 配置领域 | Megatron Bridge 配置类 |
|-------------------|-------------------|
| 训练循环设置 | {py:class}`bridge.training.config.TrainingConfig` |
| 检查点 | {py:class}`bridge.training.config.CheckpointConfig` |
| 日志记录和监控 | {py:class}`bridge.training.config.LoggerConfig` |

| 分布式训练初始化 | {py:class}`bridge.training.config.DistributedInitConfig` |
| 混合精度 | {py:class}`bridge.training.mixed_precision.MixedPrecisionConfig` |
| 性能分析 | {py:class}`bridge.training.config.ProfilingConfig` |

关于每个配置领域的详细文档，请参阅训练文档：
- {doc}`training/config-container-overview` - 配置系统概述
- {doc}`training/training-loop-settings` - 训练循环参数和验证
- {doc}`training/checkpointing` - 检查点和模型持久化
- {doc}`training/optimizer-scheduler` - 优化器和学习率调度
- {doc}`training/logging` - 日志记录、TensorBoard 和 Weights & Biases
- {doc}`training/profiling` - 使用 Nsys 和 PyTorch 进行性能分析



### 训练配置迁移
Lightning `Trainer` 参数现在通过专门的配置类进行管理。

| **设置类别** | **NeMo 2.0 位置** | **Megatron Bridge 位置** | **详情** |
|---------------------|----------------------|-------------------|-------------|
| **训练迭代次数** | `trainer.max_steps` | {py:attr}`bridge.training.config.TrainingConfig.train_iters` | 总训练迭代次数 |
| **验证频率** | `trainer.val_check_interval` | {py:attr}`bridge.training.config.TrainingConfig.eval_interval` | 验证运行之间的步数 |
| **验证迭代次数** | `trainer.limit_val_batches` | {py:attr}`bridge.training.config.TrainingConfig.eval_iters` | 每次验证运行的步数 |
| **测试迭代次数** | `trainer.limit_test_batches` | {py:attr}`bridge.training.config.TrainingConfig.eval_iters` | 测试步数（共享 eval_iters） |
| **日志记录频率** | `trainer.log_every_n_steps` | {py:attr}`bridge.training.config.LoggerConfig.log_interval` | 日志记录频率 |

#### 之前：NeMo 2.0
```python
trainer = run.Config(
    nl.Trainer,
    max_steps=1000,
    val_check_interval=100,     # 验证频率
    limit_val_batches=50,       # 每次验证运行的迭代次数
    limit_test_batches=100,     # 测试迭代次数
    log_every_n_steps=10,
)
```

#### 现在：Megatron Bridge
```python  
train_config = TrainingConfig(
    train_iters=1000,           # 原为 max_steps
    eval_interval=100,          # 原为 val_check_interval  
    eval_iters=50,              # 原为 limit_val_batches（同时用于验证和测试）
)
logger_config = LoggerConfig(log_interval=10)  # 原为 log_every_n_steps
```

### 数据配置迁移

NeMo 2.0 使用 `PreTrainingDataModule` 和 `FineTuningDataModule` 类。Megatron Bridge 使用配置对象：{py:class}`bridge.training.config.GPTDatasetConfig` 用于预训练，{py:class}`bridge.training.config.FinetuningDatasetConfig` 用于微调。

#### 预训练数据

##### 之前：NeMo 2.0 PreTrainingDataModule

```python
from nemo.collections.llm.gpt.data import PreTrainingDataModule

# 单个数据集
data = PreTrainingDataModule(
    paths="/path/to/train_data_text_document",
    seq_length=4096,
    micro_batch_size=1,
    global_batch_size=512,
    num_workers=8,
    split="949,50,1",  # 训练/验证/测试分割比例
)

# 带权重的多个数据集
data = PreTrainingDataModule(
    paths=["30", "/path/to/dataset1_text_document", 
           "70", "/path/to/dataset2_text_document"],
    seq_length=4096,
    micro_batch_size=1,
    global_batch_size=512,
    split="949,50,1",
)

# 独立的训练/验证/测试数据集
data = PreTrainingDataModule(
    paths={
        "train": ["/path/to/train_data_text_document"],
        "validation": ["/path/to/val_data_text_document"],
        "test": ["/path/to/test_data_text_document"],
    },
    seq_length=4096,
    micro_batch_size=1,
    global_batch_size=512,
)
```

##### 现在：Megatron Bridge GPTDatasetConfig

```python
from megatron.bridge.training.config import GPTDatasetConfig, TrainingConfig

# 单个数据集
dataset_config = GPTDatasetConfig(
    blend=["/path/to/train_data_text_document"],
    seq_length=4096,
    split="949,50,1",
)
train_config = TrainingConfig(
    micro_batch_size=1,
    global_batch_size=512,
)

# 带权重的多个数据集（混合）
dataset_config = GPTDatasetConfig(
    blend=[
        "/path/to/dataset1_text_document",
        "/path/to/dataset2_text_document",
    ],
    blend_weights=[0.3, 0.7],  # 显式权重（不与路径压缩在一起）
    seq_length=4096,
    split="949,50,1",
)
```

**主要区别：**
- NeMo 2.0 的 `paths` → Megatron Bridge 的 `blend`
- NeMo 2.0 的压缩列表 `["30", "path1", "70", "path2"]` → Megatron Bridge 独立的 `blend` 和 `blend_weights`
- 批次大小从数据模块移至 `TrainingConfig`
- 数据加载器选项（`num_workers`、`pin_memory` 等）在两个配置中均可用

#### 微调数据

##### 之前：NeMo 2.0 FineTuningDataModule

```python
from nemo.collections.llm.gpt.data import FineTuningDataModule

data = FineTuningDataModule(
    dataset_root="/path/to/instruction_data",
```

seq_length=2048,
    micro_batch_size=1,
    global_batch_size=128,
    num_workers=8,
)
```

##### 现在：Megatron Bridge FinetuningDatasetConfig

```python
from megatron.bridge.training.config import FinetuningDatasetConfig, TrainingConfig

dataset_config = FinetuningDatasetConfig(
    dataset_root="/path/to/instruction_data",
    seq_length=2048,
    do_validation=True,
    do_test=False,
    # Dataloader 选项（继承自 DataloaderConfig）
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
)
train_config = TrainingConfig(
    micro_batch_size=1,
    global_batch_size=128,
)
```

**主要区别：**
- 批次大小移至 `TrainingConfig`
- 通过 `do_validation` 和 `do_test` 显式控制微调验证/测试集划分
- 数据加载器选项（`num_workers`、`pin_memory` 等）可通过 `FinetuningDatasetConfig` 设置

### 分词器迁移

Megatron Bridge 使用 {py:class}`bridge.training.tokenizers.config.TokenizerConfig` 在不同模型类型间实现一致的分词器设置。

#### 之前：NeMo 2.0
```python
# 选项 1：使用 get_nmt_tokenizer 工具
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

tokenizer = get_nmt_tokenizer(
    library="megatron",
    model_name="GPT2BPETokenizer",
    vocab_file="/path/to/vocab.json",
    merges_file="/path/to/merges.txt",
)

# 选项 2：使用 run.Config 与分词器类
import nemo_run as run
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

tokenizer = run.Config(
    AutoTokenizer,
    pretrained_model_name="meta-llama/Llama-3-8B",
)
```

#### 现在：Megatron Bridge
```python
# 专用的分词器配置
from megatron.bridge.training.tokenizers.config import TokenizerConfig

# GPT2 BPE 分词器
tokenizer_config = TokenizerConfig(
    tokenizer_type="GPT2BPETokenizer",
    vocab_file="/path/to/vocab.json",
    merge_file="/path/to/merges.txt",
)

# HuggingFace 分词器
tokenizer_config = TokenizerConfig(
    tokenizer_type="HuggingFaceTokenizer",
    tokenizer_model="meta-llama/Llama-3-8B",
)
```

#### 词汇表大小优先级

在 Megatron Bridge 中，词汇表大小可以在模型提供者中指定，也可以从分词器派生。优先级顺序如下：

1. **模型提供者设置了 `vocab_size`**：使用模型的词汇表大小
   - 必须 `>= tokenizer.vocab_size`（如果更小会引发错误）
   - 设置 `should_pad_vocab=False`（无自动填充）
   - 适用于需要特定词汇表大小的情况（例如，为了检查点兼容性）

2. **模型提供者 `vocab_size` 为 None**：使用分词器的词汇表大小
   - 在构建分词器后自动从 `tokenizer.vocab_size` 派生。
   - 设置 `should_pad_vocab=True`（启用填充以实现高效并行）

```python
# 选项 1：让分词器决定词汇表大小
config = ConfigContainer(
    model=Llama3ModelProvider8B(
        # 未设置 vocab_size - 将使用分词器的词汇表大小
        vocab_size=None,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model="meta-llama/Llama-3-8B",
    ),
)

# 选项 2：在模型中显式设置词汇表大小
config = ConfigContainer(
    model=Llama3ModelProvider8B(
        vocab_size=128256,  # 显式设置（必须 >= 分词器词汇表大小）
    ),
    tokenizer=TokenizerConfig(...),
)
```

### 并行配置迁移

在 NeMo 2.0 中，并行设置是在 `MegatronStrategy` 上配置的。在 Megatron Bridge 中，这些设置直接放在模型提供者上：

| **并行类型** | **NeMo 2.0** | **Megatron Bridge** |
|---------------------|-------------|-----------|
| **张量并行** | `strategy.tensor_model_parallel_size` | `model.tensor_model_parallel_size` |
| **管道并行** | `strategy.pipeline_model_parallel_size` | `model.pipeline_model_parallel_size` |
| **虚拟管道** | `strategy.virtual_pipeline_model_parallel_size` | `model.virtual_pipeline_model_parallel_size` |
| **微批次组大小** | `strategy.microbatch_group_size_per_vp_stage` | `model.microbatch_group_size_per_vp_stage` |
| **管道层分布** | `strategy.num_layers_in_first_pipeline_stage` | `model.num_layers_in_first_pipeline_stage` |
| **管道层分布** | `strategy.num_layers_in_last_pipeline_stage` | `model.num_layers_in_last_pipeline_stage` |
| **上下文并行** | `strategy.context_parallel_size` | `model.context_parallel_size` |
| **序列并行** | `strategy.sequence_parallel` | `model.sequence_parallel` |
| **专家并行** | `strategy.expert_model_parallel_size` | `model.expert_model_parallel_size` |
| **专家张量并行** | `strategy.expert_tensor_parallel_size` | `model.expert_tensor_parallel_size` |
| **管道布局** | `strategy.pipeline_model_parallel_layout` | `model.pipeline_model_parallel_layout` |
| **管道通信后端** | `strategy.pipeline_model_parallel_comm_backend` | `model.pipeline_model_parallel_comm_backend` |

| **管道数据类型** | `strategy.pipeline_dtype` | `model.pipeline_dtype` |
| **编码器张量并行** | `strategy.encoder_tensor_model_parallel_size` | `model.encoder_tensor_model_parallel_size` |
| **编码器管道并行** | `strategy.encoder_pipeline_model_parallel_size` | `model.encoder_pipeline_model_parallel_size` |
| **管道中的嵌入层** | `strategy.account_for_embedding_in_pipeline_split` | `model.account_for_embedding_in_pipeline_split` |
| **管道中的损失层** | `strategy.account_for_loss_in_pipeline_split` | `model.account_for_loss_in_pipeline_split` |
| **TE RNG 追踪器** | `strategy.use_te_rng_tracker` | `model.use_te_rng_tracker` |

#### 之前：NeMo 2.0
```python
strategy = run.Config(
    MegatronStrategy,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=2,
    context_parallel_size=2,
    sequence_parallel=True,
)
```

#### 现在：Megatron Bridge
```python
model = GPTModelProvider(
    # 模型架构
    num_layers=32,
    hidden_size=4096,
    # 与模型共置的并行配置
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=2,
    context_parallel_size=2,
    sequence_parallel=True,
)
```

### DDP 配置迁移
部分 `MegatronStrategy` 参数移至 {py:class}`bridge.training.config.DistributedDataParallelConfig`：

| **设置项** | **NeMo 2.0** | **Megatron Bridge** |
|-------------|-------------|-----------|
| **分布式优化器实例数** | `strategy.num_distributed_optimizer_instances` | {py:attr}`bridge.training.config.DistributedDataParallelConfig.num_distributed_optimizer_instances` |

### 策略设置迁移
其他 `MegatronStrategy` 参数移至 {py:class}`bridge.training.config.DistributedInitConfig`：

| **设置项** | **NeMo 2.0** | **Megatron Bridge** |
|-------------|-------------|-----------|
| **进程组** | `strategy.use_gloo_process_groups` | {py:attr}`bridge.training.config.DistributedInitConfig.use_gloo_process_groups` |
| **SHARP** | `strategy.use_sharp` | {py:attr}`bridge.training.config.DistributedInitConfig.use_sharp` |
| **NCCL 配置** | `strategy.nccl_communicator_config_path` | {py:attr}`bridge.training.config.DistributedInitConfig.nccl_communicator_config_path` |
| **映射顺序** | `strategy.use_tp_pp_dp_mapping` | {py:attr}`bridge.training.config.DistributedInitConfig.use_tp_pp_dp_mapping` |
| **延迟初始化** | `strategy.lazy_init` | {py:attr}`bridge.training.config.DistributedInitConfig.lazy_init` |

### 混合精度迁移
在 NeMo 2.0 中，混合精度通过传递给训练器的精度插件控制。在 Megatron Bridge 中，这移至一个专用的配置类：

#### 之前：NeMo 2.0
```python
# 通过插件设置混合精度
from nemo.lightning.pytorch.plugins import MegatronMixedPrecisionPlugin

trainer = run.Config(
    nl.Trainer,
    plugins=[MegatronMixedPrecisionPlugin(precision="bf16-mixed")]
)
```

#### 现在：Megatron Bridge
```python
# 选项 1：使用预设字符串
config = ConfigContainer(
    mixed_precision="bf16_mixed",  # 简单预设
    # ... 其他配置
)

# 选项 2：详细配置
config = ConfigContainer(
    mixed_precision=MixedPrecisionConfig(
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    ),
    # ... 其他配置
)
```

```{注意}
`mixed_precision` 配置会自动同步模型、优化器和 DDP 配置中的精度设置，并覆盖任何冲突的设置。这确保了整个训练过程中精度行为的一致性。有关配置优先级和可用预设的详细信息，请参阅 {doc}`training/mixed-precision`。
```

### 检查点配置迁移
检查点配置从 `MegatronStrategy` 参数和 `ModelCheckpoint` 回调移至 {py:class}`bridge.training.config.CheckpointConfig`：

| **检查点设置** | **NeMo 2.0** | **Megatron Bridge** |
|------------------------|-------------|-----------|
| **保存目录** | `ModelCheckpoint(dirpath=...)` | {py:attr}`bridge.training.config.CheckpointConfig.save` |
| **加载目录** | `trainer.ckpt_path` | {py:attr}`bridge.training.config.CheckpointConfig.load` |
| **预训练检查点（用于微调）** | `AutoResume.import_path` 或手动加载 | {py:attr}`bridge.training.config.CheckpointConfig.pretrained_checkpoint` |
| **保存频率** | `ModelCheckpoint(every_n_train_steps=...)` | {py:attr}`bridge.training.config.CheckpointConfig.save_interval` |
| **保存 top-k** | `ModelCheckpoint(save_top_k=...)` | 无直接对应项 - Megatron Bridge 可以保留最近的检查点 |
| **最近检查点** | 无直接对应项 | {py:attr}`bridge.training.config.CheckpointConfig.most_recent_k` |
| **保存最后一个** | `ModelCheckpoint(save_last=...)` | 在 Megatron Bridge 中始终启用 |
| **检查点格式** | `strategy.save_ckpt_format` | {py:attr}`bridge.training.config.CheckpointConfig.ckpt_format` |

| **异步保存** | `strategy.ckpt_async_save` | {py:attr}`bridge.training.config.CheckpointConfig.async_save` |
| **并行保存** | `strategy.ckpt_parallel_save` | {py:attr}`bridge.training.config.CheckpointConfig.fully_parallel_save` |
| **并行加载** | `strategy.ckpt_parallel_load` | {py:attr}`bridge.training.config.CheckpointConfig.fully_parallel_load` |
| **加载优化器** | `strategy.ckpt_load_optimizer` | {py:attr}`bridge.training.config.CheckpointConfig.load_optim` |
| **保存优化器** | `strategy.ckpt_save_optimizer` | {py:attr}`bridge.training.config.CheckpointConfig.save_optim` |
| **加载主参数** | `strategy.ckpt_load_main_params` | {py:attr}`bridge.training.config.CheckpointConfig.load_main_params_from_ckpt` |
| **仅保存权重** | `ModelCheckpoint(save_weights_only=...)` | `save_optim` 的反向设置 |
| **加载严格性** | `strategy.ckpt_load_strictness` | {py:attr}`bridge.training.config.CheckpointConfig.dist_ckpt_strictness` |
| **假设结构恒定** | `strategy.ckpt_assume_constant_structure` | {py:attr}`bridge.training.config.CheckpointConfig.ckpt_assume_constant_structure` |
| **训练结束时保存优化器** | `ModelCheckpoint(save_optim_on_train_end=...)` | 由 `save_optim` 控制 |
| **从目录恢复** | `AutoResume(resume_from_directory=...)` | {py:attr}`bridge.training.config.CheckpointConfig.load` |
| **如果存在则恢复** | `AutoResume(resume_if_exists=...)` | 如果设置了 `load` 则自动启用 |
| **忽略无检查点恢复** | `AutoResume(resume_ignore_no_checkpoint=...)` | {py:attr}`bridge.training.config.CheckpointConfig.exit_on_missing_checkpoint` (反向) |

#### 之前：NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning import AutoResume, NeMoLogger

# ModelCheckpoint 回调
checkpoint_callback = ModelCheckpoint(
    dirpath="/path/to/checkpoints",
    every_n_train_steps=1000,
    save_top_k=3,           # 根据监控指标保存最佳的 3 个检查点
    save_last=True,
    save_weights_only=False,
    monitor="val_loss",     # 用于 top-k 选择的监控指标
)

# 用于检查点恢复的 AutoResume
resume = AutoResume(
    resume_if_exists=True,
    resume_ignore_no_checkpoint=True,
    resume_from_directory="/path/to/checkpoints"
)

# NeMoLogger 将所有内容整合在一起
logger = NeMoLogger(
    log_dir="/path/to/logs",
    name="my_experiment", 
    ckpt=checkpoint_callback,
)

# MegatronStrategy 参数
strategy = run.Config(
    MegatronStrategy,
    save_ckpt_format="torch_dist",
    ckpt_async_save=True,
    ckpt_parallel_save=True,
    ckpt_load_optimizer=True,
    ckpt_save_optimizer=True,
    ckpt_load_strictness=None,
)

trainer = nl.Trainer(strategy=strategy)
logger.setup(trainer, resume.resume_if_exists)
resume.setup(trainer)
```

#### 现在：Megatron Bridge
```python
checkpoint_config = CheckpointConfig(
    # 保存配置
    save="/path/to/checkpoints",
    save_interval=1000,
    most_recent_k=3,        # 保留 3 个最近的检查点（非基于指标）
    save_optim=True,
    save_rng=True,
    
    # 加载/恢复配置
    load="/path/to/checkpoints",  # 从此目录恢复（如果存在）
    load_optim=True,               # 加载优化器状态
    exit_on_missing_checkpoint=False,  # 如果未找到检查点不退出（对应之前的 resume_ignore_no_checkpoint）
    
    # 格式和性能选项
    ckpt_format="torch_dist",
    async_save=True,
    fully_parallel_save=True,
    fully_parallel_load=True,
    dist_ckpt_strictness="assume_ok_unexpected",
)
```

**主要区别：**
- **恢复行为**：设置 `load` 会在检查点存在时自动启用恢复（不再需要单独的 `AutoResume`）
- **预训练检查点**：使用 `pretrained_checkpoint` 指定用于微调的基础模型权重（在训练开始前加载）
- **Top-k**：NeMo 2.0 的 `save_top_k` 监控指标；Megatron Bridge 的 `most_recent_k` 保留最近的检查点
- **配置位置**：所有检查点设置统一在一个配置中（不再分散在回调、日志记录器和策略中）

```{Important}
所有检查点路径（`save`、`load`、`pretrained_checkpoint`）必须指向 **Megatron 格式的检查点**。不能直接使用 Hugging Face 检查点——请先使用 {py:meth}`bridge.models.conversion.auto_bridge.AutoBridge.import_ckpt` 转换它们。有关转换详情，请参阅 {doc}`bridge-guide`。
```

有关检查点格式、本地检查点、容错和高级功能的完整文档，请参阅 {doc}`training/checkpointing`。

---

### 优化器和学习率调度器迁移

优化配置从 NeMo 2.0 的 `MegatronOptimizerModule` 方法迁移到 Megatron Bridge 的直接 {py:class}`megatron.core.optimizer.OptimizerConfig` 和 {py:class}`bridge.training.config.SchedulerConfig`。

#### 之前：NeMo 2.0
```python
# NeMo 2.0 使用 MegatronOptimizerModule 的优化器配置

from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.collections.llm.recipes import distributed_fused_adam_with_cosine_annealing

# 选项 1：使用配方辅助函数
optim_config = distributed_fused_adam_with_cosine_annealing(
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=2000,
)

# 选项 2：直接使用 MegatronOptimizerModule
optim = MegatronOptimizerModule(
    config=OptimizerConfig(
        optimizer="adam",
        lr=3e-4,
        use_distributed_optimizer=True,
    ),
    lr_scheduler=CosineAnnealingScheduler(
        warmup_steps=2000,
        constant_steps=0,
        decay_steps=100000,
    )
)
```

#### 现在：Megatron Bridge
```python
# Megatron Bridge 直接配置
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing

# 选项 1：使用工具函数
optimizer_config, scheduler_config = distributed_fused_adam_with_cosine_annealing(
    max_lr=3e-4,
    min_lr=3e-5,
    lr_warmup_iters=2000,
    lr_decay_iters=100000,
)

# 选项 2：直接配置
optimizer_config = OptimizerConfig(
    optimizer="adam",
    lr=3e-4,
    min_lr=3e-5,
    weight_decay=0.1,
    use_distributed_optimizer=True,
)

scheduler_config = SchedulerConfig(
    lr_decay_style="cosine",
    lr_warmup_iters=2000,
    lr_decay_iters=100000,
)
```

### 日志配置迁移

NeMo 2.0 使用 `NeMoLogger` 进行 TensorBoard 和 Weights & Biases (W&B) 集成。Megatron Bridge 将日志配置整合在 {py:class}`bridge.training.config.LoggerConfig` 中。

#### 之前：NeMo 2.0

```python
from nemo.lightning import NeMoLogger

logger = NeMoLogger(
    log_dir="/path/to/logs",
    name="my_experiment",
    use_datetime_version=True,
    tensorboard=dict(
        log_dir="/path/to/tensorboard",
    ),
    wandb=dict(
        project="my_project",
        name="my_run",
        entity="my_team",
    ),
)
```

#### 现在：Megatron Bridge

```python
from megatron.bridge.training.config import LoggerConfig

logger_config = LoggerConfig(
    # 通用日志记录
    log_interval=10,              # 每 N 次迭代记录一次指标
    log_throughput=True,          # 记录每个 GPU 的吞吐量
    
    # TensorBoard 配置
    tensorboard_dir="/path/to/tensorboard",
    tensorboard_log_interval=1,   # 每 N 次迭代写入 TensorBoard
    log_timers_to_tensorboard=False,
    log_validation_ppl_to_tensorboard=False,
    
    # Weights & Biases 配置
    wandb_project="my_project",
    wandb_exp_name="my_run",
    wandb_entity="my_team",
    wandb_save_dir="/path/to/wandb",
)
```

**主要区别：**
- TensorBoard 和 W&B 配置统一在单个 `LoggerConfig` 中
- 对记录内容（计时器、内存、验证困惑度等）进行细粒度控制
- 无需单独的 `NeMoLogger.setup()` 调用

有关日志配置和可用选项的更多详细信息，请参阅 {doc}`training/logging`。

### 性能分析配置迁移

Megatron Bridge 将所有性能分析功能集中在 {py:class}`bridge.training.config.ProfilingConfig` 中，取代了多个 NeMo 回调函数。

#### Nsys 性能分析迁移

##### 之前：NeMo 2.0
```python
# NeMo 2.0 使用 NsysCallback
from nemo.lightning.pytorch.callbacks import NsysCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[
        NsysCallback(
            start_step=100,
            end_step=110,
            ranks=[0],
            gen_shape=True
        )
    ]
)
```

##### 现在：Megatron Bridge
```python
# Megatron Bridge 使用 ProfilingConfig  
profiling_config = ProfilingConfig(
    use_nsys_profiler=True,
    profile_step_start=100,
    profile_step_end=110,
    profile_ranks=[0],
    record_shapes=True,
)
```

#### PyTorch Profiler 迁移

##### 之前：NeMo 2.0
```python
# NeMo 2.0 使用 PytorchProfilerCallback
from nemo.lightning.pytorch.callbacks import PytorchProfilerCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[
        PytorchProfilerCallback(
            start_step=100,
            end_step=110,
            warmup_steps=1,
            active_steps=5,
            trace_dir="/path/to/traces",
        )
    ]
)
```

##### 现在：Megatron Bridge
```python
# Megatron Bridge 使用 ProfilingConfig
profiling_config = ProfilingConfig(
    use_pytorch_profiler=True,
    profile_step_start=100,
    profile_step_end=110,
    profile_ranks=[0],
    record_memory_history=True,
    memory_snapshot_path="memory_profile.pickle",
)
```

### PEFT 配置迁移

PEFT（参数高效微调）通过冻结基础模型并仅训练适配器模块，使用一小部分可训练参数实现微调。

#### 之前：NeMo 2.0

```python
from nemo.collections import llm
import nemo_run as run

# 创建 PEFT 配置
lora = llm.peft.LoRA(
    target_modules=['linear_qkv', 'linear_proj'],
    dim=32,
    alpha=16,
    dropout=0.0,
)

# 传递给 finetune()
llm.finetune(
    model=model,
    data=data,
    trainer=trainer,

peft=lora,  # PEFT 作为参数
)
```

#### 现在：Megatron Bridge

```python
from megatron.bridge.peft import LoRA
from megatron.bridge.training.config import ConfigContainer, CheckpointConfig

# 在 ConfigContainer 中包含 PEFT
config = ConfigContainer(
    model=Llama3ModelProvider8B(),
    # ... 其他配置
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/megatron/checkpoint",  # PEFT 必需
        save="/path/to/peft/checkpoints",
    ),
    peft=LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=32,
        alpha=16,
        dropout=0.0,
    ),
)
```

**主要区别：**
- PEFT 配置是 `ConfigContainer` 的一部分，而不是 `finetune()` 的单独参数
- 使用 PEFT 时必须设置 `checkpoint.pretrained_checkpoint`（在验证时强制执行）
- 目标模块名称在 NeMo 2.0 和 Megatron Bridge 之间是相同的

**支持的 PEFT 方法：**
- **LoRA**：低秩适配，通过 {py:class}`bridge.peft.lora.LoRA`
- **DoRA**：权重分解低秩适配，通过 {py:class}`bridge.peft.dora.DoRA`

有关 PEFT 的完整文档，包括适配器设计、检查点处理、通配符目标设定和最佳实践，请参阅 {doc}`training/peft`。

## 入口点：`pretrain` 和 `finetune`

NeMo 2.0 的 `llm.pretrain()` 和 `llm.finetune()` API 函数直接映射到 Megatron Bridge 的入口点函数，并具有统一的配置。

### NeMo 2.0 入口点

在 NeMo 2.0 中，您从 `nemo.collections.llm.api` 调用 `llm.pretrain()` 或 `llm.finetune()`：

```python
from nemo.collections import llm
import nemo_run as run

# 预训练
result = llm.pretrain(
    model=model_config,
    data=data_config,
    trainer=trainer_config,
    log=logger_config,
    resume=resume_config,
    optim=optimizer_config,
)

# 微调
result = llm.finetune(
    model=model_config,
    data=data_config,
    trainer=trainer_config,
    log=logger_config,
    resume=resume_config,
    optim=optimizer_config,
    peft=peft_config,   # 可选的 PEFT
    tokenizer="model",  # 或 "data"
)
```

### Megatron Bridge 入口点

在 Megatron Bridge 中，训练入口点接受一个 `ConfigContainer` 和一个 `forward_step_func`：

```python
from megatron.bridge.training import pretrain, finetune
from megatron.bridge.training.config import ConfigContainer

# 创建统一配置
cfg = ConfigContainer(
    model=model_provider,
    train=train_config,
    dataset=dataset_config,
    optimizer=optimizer_config,
    scheduler=scheduler_config,
    checkpoint=checkpoint_config,
    logger=logger_config,
    mixed_precision="bf16_mixed",
    # peft=peft_config,  # 微调时可选
)

# 预训练
from megatron.bridge.training.gpt_step import forward_step
pretrain(cfg, forward_step_func=forward_step)

# 微调（相同的函数签名）
finetune(cfg, forward_step_func=forward_step)
```

#### 理解 `forward_step_func`

`forward_step_func` 将三个职责合并到一个函数中：

1. 从数据迭代器中**获取一个批次**
2. 通过模型**运行前向传播**
3. **定义损失函数**，以根据模型输出计算损失

**签名：**
```python
def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: nn.Module,
) -> tuple[torch.Tensor, Callable]:
    """
    参数：
        state: 全局训练状态（包含配置、计时器等）
        data_iterator: 训练/验证数据的迭代器
        model: 要运行前向传播的模型
        
    返回：
        output_tensor: 模型输出（logits）
        loss_func: 从 output_tensor 计算损失的可调用对象
    """
```

对于 GPT 模型，使用提供的 {py:func}`bridge.training.gpt_step.forward_step`。对于自定义模型或专门的训练逻辑，请按照此模式实现您自己的函数。

**主要区别：**
- 所有配置都整合到单个 `ConfigContainer` 对象中
- 训练模式由数据集类型和检查点配置决定，而不是通过单独的函数调用
- 必须提供处理批次获取、前向传播和损失计算的 `forward_step_func`
- 没有单独的 `resume`、`log`、`optim` 参数——所有配置都是 `ConfigContainer` 的一部分

### `pretrain`

使用 `pretrain()` 和 `GPTDatasetConfig` 从头开始训练模型：

```python
from megatron.bridge.training import pretrain
from megatron.bridge.training.gpt_step import forward_step

config = ConfigContainer(
    model=Llama3ModelProvider8B(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
    ),
    train=TrainingConfig(
        train_iters=100000,
        eval_interval=1000,
        micro_batch_size=1,
        global_batch_size=512,
    ),
    dataset=GPTDatasetConfig(
        blend=["/path/to/train_data_text_document"],
        seq_length=4096,
        split="949,50,1",
    ),
    optimizer=OptimizerConfig(optimizer="adam", lr=3e-4),

checkpoint=CheckpointConfig(save="/path/to/checkpoints", save_interval=1000),
    mixed_precision="bf16_mixed",
)

pretrain(config, forward_step_func=forward_step)
```

### `finetune`

使用 `finetune()` 和 `FinetuningDatasetConfig` 进行全参数微调（SFT）和参数高效微调（PEFT）：

#### 监督微调（SFT）

不使用 PEFT 的全参数微调 - 更新所有模型参数：

```python
from megatron.bridge.training import finetune
from megatron.bridge.training.gpt_step import forward_step

config = ConfigContainer(
    model=Llama3ModelProvider8B(),
    train=TrainingConfig(
        train_iters=1000,
        eval_interval=100,
        micro_batch_size=1,
        global_batch_size=128,
    ),
    dataset=FinetuningDatasetConfig(
        dataset_root="/path/to/instruction_data",
        seq_length=4096,
        do_validation=True,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/megatron/checkpoint",  # 必须是 Megatron 格式
        save="/path/to/sft_checkpoints",
    ),
    optimizer=OptimizerConfig(optimizer="adam", lr=1e-5),
    mixed_precision="bf16_mixed",
)

finetune(config, forward_step_func=forward_step)
```

#### 参数高效微调（PEFT）

添加 `peft` 配置以启用参数高效训练：

```python
from megatron.bridge.peft import LoRA

config = ConfigContainer(
    model=Llama3ModelProvider8B(),
    train=TrainingConfig(
        train_iters=1000,
        eval_interval=100,
        micro_batch_size=1,
        global_batch_size=128,
    ),
    dataset=FinetuningDatasetConfig(
        dataset_root="/path/to/instruction_data",
        seq_length=4096,
        do_validation=True,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/megatron/checkpoint",
        save="/path/to/peft_checkpoints",
    ),
    peft=LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=32,
        alpha=16,
    ),
    optimizer=OptimizerConfig(optimizer="adam", lr=1e-4),
    mixed_precision="bf16_mixed",
)

finetune(config, forward_step_func=forward_step)
```

**转换 Hugging Face 检查点**：如果你有一个 Hugging Face 模型，请先将其转换为 Megatron 检查点格式：

```python
from megatron.bridge import AutoBridge

# 将 HuggingFace 转换为 Megatron 格式
AutoBridge.import_ckpt(
    "meta-llama/Meta-Llama-3-8B",
    "/path/to/megatron/checkpoint"
)
```

有关模型转换的更多详细信息，请参阅 {doc}`bridge-guide`。

### 高级：自定义前向步骤和损失归约

有关入口点、前向步骤函数和自定义模式的完整文档，请参阅 {doc}`training/entry-points`。

#### 前向步骤自定义

在 NeMo 2.0 中，自定义的 `forward_step` 和 `data_step` 函数可以附加到模型配置上。在 Megatron Bridge 中，前向步骤函数直接作为参数传递给 `pretrain()` 或 `finetune()`。

##### NeMo 2.0：自定义步骤附加到配置

```python
# NeMo 2.0: 定义自定义函数并附加到模型配置
import torch

def custom_forward_step(model, batch) -> torch.Tensor:
    """用于专门损失计算的自定义前向步骤。"""
    output = model(batch['tokens'], batch['attention_mask'])
    loss = compute_custom_loss(output, batch['labels'])
    return loss

# 在 NeMo 2.0 中附加到配置
model_config = llm.Llama3Config8B()
model_config.forward_step_fn = custom_forward_step  # 覆盖默认前向步骤

model = run.Config(llm.LlamaModel, config=model_config)
```

##### Megatron Bridge：传递自定义前向步骤

```python
# Megatron Bridge: 定义并传递前向步骤函数
import torch
from typing import Iterable
from functools import partial
from megatron.bridge.training.state import GlobalState

def custom_forward_step(
    state: GlobalState,
    data_iterator: Iterable, 
    model: torch.nn.Module,
) -> tuple[torch.Tensor, partial]:
    """用于专门损失计算的自定义前向步骤。"""
    # 从迭代器获取批次
    batch = next(data_iterator)
    tokens = batch['tokens'].cuda()
    labels = batch['labels'].cuda()
    loss_mask = batch['loss_mask'].cuda()
    
    # 自定义前向逻辑
    output = model(tokens, attention_mask=batch.get('attention_mask'))
    
    # 定义自定义损失函数
    def loss_func(output_tensor):
        return compute_custom_loss(output_tensor, labels, loss_mask)
    
    return output, loss_func

# 传递给训练函数
pretrain(cfg, forward_step_func=custom_forward_step)
```

#### 损失归约模式

NeMo 2.0 使用 `MegatronLossReduction` 进行跨微批次的自定义损失计算和归约。Megatron Bridge 通过 `forward_step` 返回的损失函数实现相同的功能。

##### NeMo 2.0：MegatronLossReduction

```python
from nemo.lightning.megatron_parallel import MegatronLossReduction

class CustomLossReduction(MegatronLossReduction):
    def forward(self, batch, forward_out):

"""从前向输出计算损失。"""
        loss = compute_loss(forward_out, batch['labels'])
        return loss, {"custom_metric": some_metric}
    
    def reduce(self, losses_reduced_per_micro_batch):
        """跨微批次减少损失。"""
        losses = [x["custom_metric"] for x in losses_reduced_per_micro_batch]
        return torch.stack(losses).mean()

# 附加到模型
model._training_loss_reduction = CustomLossReduction()
```

##### Megatron Bridge：损失函数模式

```python
def custom_forward_step(state, data_iterator, model):
    """带有自定义损失归约的前向步骤。"""
    batch = next(data_iterator)
    tokens = batch['tokens'].cuda()
    labels = batch['labels'].cuda()
    loss_mask = batch['loss_mask'].cuda()
    
    output = model(tokens)
    
    def loss_func(output_tensor):
        """计算并以归约友好的格式返回损失。
        
        返回格式：
        - 单个值：损失（仅在微批次上平均）
        - 元组：(loss, num_tokens) - 在微批次和令牌上平均
        - 字典：{"loss": loss, "custom_metric": value, ...} - 用于日志记录
        """
        loss = compute_loss(output_tensor, labels, loss_mask)
        num_tokens = loss_mask.sum()
        
        # 返回 (loss, num_tokens) 以进行正确的平均
        # 训练循环会自动跨微批次和数据并行秩进行归约
        return {
            "loss": torch.cat([loss.view(1), num_tokens.view(1)]),
            "custom_metric": torch.cat([some_metric.view(1), num_tokens.view(1)]),
        }
    
    return output, loss_func

# 传递给训练 - 归约自动处理
pretrain(cfg, forward_step_func=custom_forward_step)
```

**关键区别：**
- **NeMo 2.0**：独立的 `MegatronLossReduction` 类，包含 `forward()` 和 `reduce()` 方法
- **Megatron Bridge**：损失函数返回格式为 `{key: [value, count]}` 的字典以进行自动归约
- **归约逻辑**：Megatron Bridge 自动跨微批次和数据并行秩平均 `value/count`

Megatron Bridge 中的训练循环（参见 {py:func}`bridge.training.train.train_step`）自动执行以下操作：
1. 为每个微批次调用损失函数
2. 跨微批次聚合结果
3. 执行数据并行全归约
4. 计算最终的平均值

#### 何时需要自定义

在以下情况下使用自定义前向步骤：
- 需要超出标准语言建模的自定义损失函数
- 具有多个损失组件的多任务学习
- 训练期间计算额外指标
- 专门的批次预处理

有关入口点签名、损失计算模式、状态访问以及更高级自定义选项的完整文档，请参阅 {doc}`training/entry-points`。

---

## 回调迁移

Megatron Bridge 将大多数 NeMo 2.0 回调转换为显式的配置字段或实用函数。

### DDP 奇偶校验器

验证模型权重在数据并行副本之间是否同步。

#### 之前：NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import DDPParityChecker

trainer = run.Config(
    nl.Trainer,
    callbacks=[DDPParityChecker(check_interval=100)]
)
```

#### 现在：Megatron Bridge
```python
# 内置到 TrainingConfig 中
train_config = TrainingConfig(
    check_weight_hash_across_dp_replicas_interval=100,
)
```

### 垃圾回收

在训练期间手动进行垃圾回收以释放内存。

#### 之前：NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import GarbageCollectionCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[
        GarbageCollectionCallback(
            gc_interval_train=100,
            gc_interval_val=100,
        )
    ]
)
```

#### 现在：Megatron Bridge
```python
# 内置到 TrainingConfig 中
train_config = TrainingConfig(
    manual_gc=True,              # 启用手动垃圾回收
    manual_gc_interval=100,      # 训练期间的 GC 间隔（原 gc_interval_train）
    manual_gc_eval=True,         # 在评估开始/结束时启用 GC（原 gc_interval_val）
)
```

### 通信重叠

启用张量/管道并行通信与计算的重叠。

#### 之前：NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import MegatronCommOverlapCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[
        MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            ...
        )
    ]
)
```

#### 现在：Megatron Bridge
```python
from megatron.bridge.training.comm_overlap import CommOverlapConfig

config = ConfigContainer(
    comm_overlap=CommOverlapConfig(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=...,  # 详细的 TP 重叠设置
    ),
)
```

有关通信重叠策略（TP、PP、DP、CP、MoE）、硬件要求和性能调优的全面文档，请参阅 {doc}`training/communication-overlap`。

### 抢占处理

优雅地处理 SLURM/集群抢占信号。

#### 之前：NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import PreemptionCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[PreemptionCallback()]
)
```

#### 现在：Megatron Bridge
```python
# 内置在 TrainingConfig 中
train_config = TrainingConfig(
    exit_signal_handler=True,  # 启用抢占信号处理
)
```

有关抢占处理和容错性的更多详细信息，请参阅 {doc}`training/resiliency`。

### 实验性功能

启用 Megatron Core 实验性功能。

#### 之前：NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import MegatronEnableExperimentalCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[MegatronEnableExperimentalCallback()]
)
```

#### 现在：Megatron Bridge
```python
from megatron.bridge.training.config import DistributedInitConfig

dist_config = DistributedInitConfig(
    enable_megatron_core_experimental=True,
)
```

### MoE Token Drop

为 MoE 模型配置专家容量和令牌填充。

#### 之前：NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import MegatronTokenDropCallback

callbacks = [
    MegatronTokenDropCallback(
        moe_expert_capacity_factor=1.0,
        moe_pad_expert_input_to_capacity=True
    )
]
```

#### 现在：Megatron Bridge
```python
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop

model = GPTModelProvider(
    # MoE 架构
    num_moe_experts=8,
    moe_router_topk=2,
    moe_token_dispatcher_type="alltoall",
)

# 应用令牌丢弃优化
apply_moe_token_drop(
    model,
    moe_expert_capacity_factor=1.0,
    moe_pad_expert_input_to_capacity=True
)
```

### MoE 的 DeepEP

在支持的硬件（Ampere/Hopper GPU）上为 MoE 模型启用 DeepEP 优化。

#### 之前：NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import DeepEPCallback

callbacks = [DeepEPCallback()]  # 如果硬件支持则自动应用
```

#### 现在：Megatron Bridge
```python
from megatron.bridge.training.deepep import apply_deepep

model = GPTModelProvider(
    num_moe_experts=8,
    # ... 其他 MoE 设置
)

# 应用 DeepEP 优化（仅在 Ampere/Hopper GPU 上）
# 硬件验证会在训练期间自动执行
apply_deepep(model)
```

---

## NeMo-Run、插件和启动

Megatron Bridge 支持直接 Python 执行和 NeMo-Run 编排。虽然 NeMo 2.0 严重依赖 NeMo-Run 的配方系统，但 Megatron Bridge 提供了更大的灵活性。

有关启动训练作业、配置覆盖和 NeMo-Run 集成的完整详细信息，请参阅 {doc}`recipe-usage`。

### NeMo-Run 集成

#### 直接 Python 执行
Megatron Bridge 支持标准的 PyTorch 分布式执行模式：

```bash
# 使用 torchrun 直接执行脚本
python -m torch.distributed.run --nproc_per_node=8 my_training_script.py

# 多节点执行
torchrun --nnodes=4 --nproc_per_node=8 \
    --master_addr="node0" --master_port=12345 \
    my_training_script.py
```

#### 使用插件的 NeMo-Run（推荐脚本模式）
如果使用 NeMo-Run，**强烈建议使用 `run.Script` 模式**以获得更好的依赖管理。Megatron Bridge 插件设计为与此方法良好配合：

```python
# Megatron Bridge NeMo-Run 集成
import nemo_run as run
from megatron.bridge.recipes.run_plugins import (
    NsysPlugin, WandbPlugin, PreemptionPlugin, FaultTolerancePlugin
)

# 创建任务
task = run.Script("my_training_script.py", args=[])

# 使用插件配置执行器
executor = run.SlurmExecutor(nodes=2, nproc_per_node=8)
executor.plugins = [
    NsysPlugin(profile_step_start=100, profile_step_end=110),
    WandbPlugin(project="my_project", entity="my_team"),
    PreemptionPlugin(preempt_time=120),
]

# 提交作业
run.run(task, executor=executor)
```

### 插件迁移对比

| **插件** | **NeMo 2.0** | **Megatron Bridge** |
|------------|-------------|-----------|
| **Nsys** | `nemo.lightning.run.plugins.NsysPlugin` | {py:class}`bridge.recipes.run_plugins.NsysPlugin` |
| **Wandb** | `nemo.lightning.run.plugins.WandbPlugin` | {py:class}`bridge.recipes.run_plugins.WandbPlugin` |
| **Preemption** | `nemo.lightning.run.plugins.PreemptionPlugin` | {py:class}`bridge.recipes.run_plugins.PreemptionPlugin` |
| **Fault Tolerance** | `nemo.lightning.run.plugins.FaultTolerancePlugin` | {py:class}`bridge.recipes.run_plugins.FaultTolerancePlugin` |