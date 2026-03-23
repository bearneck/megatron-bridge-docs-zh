# Megatron-LM 到 Megatron Bridge 指南

Megatron Bridge 是 Python 优先的：通过类型化的 Python API 配置模型、数据和训练。所有配置都存在于结构化的 `ConfigContainer` 中（参见[配置概述](training/config-container-overview.md)）。在示例训练脚本中，可以使用 Hydra/OmegaConf 语法从命令行覆盖任何字段。

## 自动化配置转换脚本

`scripts/translate_mlm_to_bridge.py` 可以在 Megatron-LM `pretrain_gpt.py` 的 CLI 参数和 Megatron Bridge `run_recipe.py` 的 Hydra 覆盖配置之间进行双向转换。这对于在两个框架之间运行损失相关性实验以及迁移现有的 MLM 配置非常有用。

### MLM → Bridge（默认方向）

```bash
# 从 YAML 配置文件（MODEL_ARGS 部分）
python scripts/translate_mlm_to_bridge.py --yaml model_configs/DeepSeek-V3.yaml

# 从内联 CLI 参数
python scripts/translate_mlm_to_bridge.py \
    --args "--num-layers 32 --hidden-size 4096 --num-attention-heads 32 --bf16 --swiglu"

# 生成一个独立的 Bridge 配方 Python 文件（输出到 stdout；使用 -o 写入文件）
python scripts/translate_mlm_to_bridge.py \
    --yaml DeepSeek-V3.yaml --emit recipe --recipe-name deepseek_v3

# 将输出写入文件而不是 stdout
python scripts/translate_mlm_to_bridge.py \
    --yaml DeepSeek-V3.yaml -o bridge_overrides.txt
```

### Bridge → MLM（反向转换）

```bash
# 从 Bridge 配方名称（默认导出为 MLM 参数）
python scripts/translate_mlm_to_bridge.py --reverse \
    --recipe llama32_1b_pretrain_config

# 从配方加上内联覆盖配置
python scripts/translate_mlm_to_bridge.py --reverse \
    --recipe llama32_1b_pretrain_config \
    --args "train.train_iters=1000 model.tensor_model_parallel_size=2"

# 仅从 Bridge 覆盖配置（无配方）
python scripts/translate_mlm_to_bridge.py --reverse \
    --args "model.num_layers=32 model.activation_func=silu model.gated_linear_unit=true"

# 从 Bridge YAML/OmegaConf 配置文件（例如导出的 ConfigContainer）
python scripts/translate_mlm_to_bridge.py --reverse \
    --yaml bridge_config.yaml
```

### 关键映射关系

| MLM 标志 | Bridge 覆盖配置 | 备注 |
|---|---|---|
| `--num-layers N` | `model.num_layers=N` | |
| `--hidden-size N` | `model.hidden_size=N` | |
| `--ffn-hidden-size N` | `model.ffn_hidden_size=N` | |
| `--num-attention-heads N` | `model.num_attention_heads=N` | |
| `--num-query-groups N` | `model.num_query_groups=N` | |
| `--seq-length N` | `model.seq_length=N dataset.sequence_length=N` | 双重映射 |
| `--swiglu` | `model.gated_linear_unit=true model.activation_func=silu` | 扩展为两个键 |
| `--squared-relu` | `model.activation_func=squared_relu` | |
| `--data-path PATH [W PATH...]` | `dataset.data_path=PATH` | 空格分隔的路径（和可选权重） |
| `--bf16` | `mixed_precision=bf16_mixed` | |
| `--fp16` | `mixed_precision=16-mixed` | |
| `--disable-bias-linear` | `model.add_bias_linear=false` | 反转的标志 |
| `--untie-embeddings-and-output-weights` | `model.share_embeddings_and_output_weights=false` | 反转的标志 |
| `--sequence-parallel` | `model.sequence_parallel=true` | |
| `--tensor-model-parallel-size N` | `model.tensor_model_parallel_size=N` | |
| `--pipeline-model-parallel-size N` | `model.pipeline_model_parallel_size=N` | |
| `--context-parallel-size N` | `model.context_parallel_size=N` | |
| `--micro-batch-size N` | `train.micro_batch_size=N` | |
| `--global-batch-size N` | `train.global_batch_size=N` | |
| `--train-iters N` | `train.train_iters=N` | |
| `--lr LR` | `optimizer.lr=LR` | |
| `--min-lr LR` | `optimizer.min_lr=LR` | |
| `--weight-decay WD` | `optimizer.weight_decay=WD` | |
| `--clip-grad CG` | `optimizer.clip_grad=CG` | |
| `--lr-decay-style S` | `scheduler.lr_decay_style=S` | |
| `--lr-warmup-iters N` | `scheduler.lr_warmup_iters=N` | |
| `--seed S` | `rng.seed=S` | |
| `--save PATH` | `checkpoint.save=PATH` | |
| `--load PATH` | `checkpoint.load=PATH` | |
| `--save-interval N` | `checkpoint.save_interval=N` | |
| `--tokenizer-type T` | `tokenizer.tokenizer_type=T` | |
| `--tokenizer-model M` | `tokenizer.tokenizer_model=M` | |
| `--normalization N` | `model.normalization=N` | |
| `--position-embedding-type T` | `model.position_embedding_type=T` | |
| `--rotary-base N` | `model.rotary_base=N` | |
| `--num-experts N` | `model.num_moe_experts=N` | |
| `--moe-router-topk K` | `model.moe_router_topk=K` | |
| `--mock-data` | `dataset.mock=true` | 使用合成数据（无需文件） |

Bridge 中不存在的标志（例如 `--use-mcore-models`、`--use-flash-attn`）会被静默跳过并附带注释。`--mock-data` 转换为 `dataset.mock=true`。未知的标志会列在一个单独的部分中，以便您手动处理。

> **激活函数 CLI 覆盖**：`model.activation_func` 现在可以通过 Hydra CLI 字符串覆盖来设置（例如 `model.activation_func=silu`、`model.activation_func=gelu`）。该字符串在 `TransformerConfig.finalize()` 中被解析为可调用对象。这使得 `--swiglu` → `model.gated_linear_unit=true model.activation_func=silu` 可以通过 CLI 实现往返。

## 快速开始

运行你的示例训练入口点并直接覆盖配置键：

```bash
python examples/models/llama/pretrain_llama3_8b.py \
  train.micro_batch_size=2 \
  train.global_batch_size=128 \
  model.num_layers=32 model.hidden_size=4096 model.num_attention_heads=32 \
  model.max_position_embeddings=4096 \
  dataset.sequence_length=4096 \
  checkpoint.save=/workspace/ckpts checkpoint.save_interval=1000 \
  logger.wandb_project=my_proj logger.wandb_exp_name=exp1
```

注意：
- 配置组是嵌套的：`rng`、`train`、`model`、`optimizer`、`ddp`、`scheduler`、`dataset`、`logger`、`tokenizer`、`checkpoint`、`dist`、`profiling`、`peft`、`comm_overlap`、`mixed_precision`、`inprocess_restart`。
- 应用覆盖后，运行时验证会计算任何依赖字段（例如，数据并行大小、调度器步数）并检查一致性。

## Megatron-LM 参数到 Megatron Bridge 配置的映射

以下是常见 `megatron-lm/megatron/training/arguments.py` 参数到新数据类字段的简明映射。如果某个字段未在此列出（例如，高度模型特定的旋钮），它通常位于 `model.*`、`optimizer.*`、`dataset.*` 或 `tokenizer.*` 下，且名称相似。

### 模型拓扑和并行性

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--tensor-model-parallel-size` | `model.tensor_model_parallel_size` | 张量并行（TP）度。 |
| `--pipeline-model-parallel-size` | `model.pipeline_model_parallel_size` | 管道并行（PP）度。 |
| `--context-parallel-size` | `model.context_parallel_size` | 上下文并行（CP）度。 |
| `--expert-model-parallel-size` | `model.expert_model_parallel_size` | 专家并行（EP）度。 |
| `--expert-tensor-parallel-size` | `model.expert_tensor_parallel_size` | 专家张量并行度。 |
| `--sequence-parallel` | `model.sequence_parallel` | 启用序列并行。 |
| `--account-for-embedding-in-pipeline-split` | `model.account_for_embedding_in_pipeline_split` | 非对称 PP：嵌入层。 |
| `--account-for-loss-in-pipeline-split` | `model.account_for_loss_in_pipeline_split` | 非对称 PP：损失层。 |

### 模型架构旋钮

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--untie-embeddings-and-output-weights` | `model.share_embeddings_and_output_weights=false` | 解绑嵌入/输出权重。 |
| `--position-embedding-type` | `model.position_embedding_type` | `learned_absolute` 或 `rope`。 |
| `--rotary-percent` | `model.rotary_percent` | 旋转维度比例。 |
| `--rotary-base` | `model.rotary_base` | RoPE 基数。 |
| `--rotary-seq-len-interpolation-factor` | `model.seq_len_interpolation_factor` | RoPE 序列长度插值因子。 |
| `--normalization` | `model.normalization` | LayerNorm/RMSNorm 等。 |
| `--swiglu` | `model.gated_linear_unit=true` | 启用 SwiGLU MLP。 |
| `--norm-epsilon` | `model.layernorm_epsilon` | 归一化层的 epsilon。 |
| `--num-layers` | `model.num_layers` | Transformer 层数。 |
| `--hidden-size` | `model.hidden_size` | 模型隐藏层大小。 |
| `--ffn-hidden-size` | `model.ffn_hidden_size` | MLP 扩展大小。 |
| `--num-attention-heads` | `model.num_attention_heads` | 注意力头数。 |
| `--kv-channels` | `model.kv_channels` | 每个注意力头的键/值通道数。 |
| `--group-query-attention` | `model.num_query_groups` | 设置分组（启用 GQA）。 |
| `--num-query-groups` | `model.num_query_groups` | 查询组数。 |
| `--qk-layernorm` | `model.qk_layernorm` | 启用 QK LayerNorm。 |
| `--seq-length` | `model.seq_length` | 模型最大序列长度。 |
| `--max-position-embeddings` | `model.seq_length` | HF 转换使用的别名。 |
| `--make-vocab-size-divisible-by` | `model.make_vocab_size_divisible_by` | TP 填充倍数。 |
| `--disable-bias-linear` | `model.add_bias_linear=false` | 禁用线性层偏置。 |
| `--use-flash-attn` | `model.attention_backend=flash` | 使用 FlashAttention 后端。 |
| `--init-method-std` | `model.init_method_std` | 权重初始化标准差。 |
| `--attention-dropout` | `model.attention_dropout` | 注意力丢弃率。 |
| `--hidden-dropout` | `model.hidden_dropout` | 隐藏层丢弃率。 |

### MoE

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--num-experts` | `model.num_moe_experts` | 每个 MoE 层的专家数。 |
| `--moe-ffn-hidden-size` | `model.moe_ffn_hidden_size` | 专家 MLP 隐藏层大小。 |
| `--moe-router-load-balancing-type` | `model.moe_router_load_balancing_type` | 例如，aux_loss 或 seq_aux_loss。 |
| `--moe-router-topk` | `model.moe_router_topk` | 每个令牌的 Top-k 专家数。 |

| `--moe-router-pre-softmax` | `model.moe_router_pre_softmax` | 预 softmax 路由。 |
| `--moe-grouped-gemm` | `model.moe_grouped_gemm` | 用于 MoE 的分组 GEMM。 |
| `--moe-aux-loss-coeff` | `model.moe_aux_loss_coeff` | 辅助损失系数。 |
| `--moe-token-dispatcher-type` | `model.moe_token_dispatcher_type` | Token 分发器类型：alltoall 或 flex。 |
| `--moe-flex-dispatcher-backend` | `model.moe_flex_dispatcher_backend` | MoE token 分发器后端：deepep 或 hybridep |
| `--moe-permute-fusion` | `model.moe_permute_fusion` | 启用 MoE 置换融合。 |
| `--moe-router-fusion` | `model.moe_router_fusion` | 启用 MoE 路由器融合。 |
| `--moe-router-dtype` | `model.moe_router_dtype` | 路由器数据类型（例如，fp32）。 |

### 混合精度

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--bf16` | `mixed_precision` 预设（例如，"bf16_mixed"） | 选择混合精度方案；设置 `model.bf16`/`optimizer.bf16`。 |

混合精度通过 `mixed_precision` 配置键选择（例如，预设名称如 `bf16_mixed`、`bf16` 或 `fp16`，具体取决于您的代码库），并在 `runtime_config_update` 期间应用于 `model`、`optimizer` 和 `ddp`。

### 训练

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--micro-batch-size` | `train.micro_batch_size` | 梯度累积前的每 rank 批次大小。 |
| `--global-batch-size` | `train.global_batch_size` | 跨 DP 和 micro-batch 的总批次大小。 |
| `--train-samples` | `train.train_samples` | 总训练样本数（基于样本的模式）。 |
| `--rampup-batch-size` | `train.rampup_batch_size` | 线性批次大小递增的起始大小、增量和样本计数。 |
| `--decrease-batch-size-if-needed` | `train.decrease_batch_size_if_needed` | 当 DP 变化时调整 GBS 以保持可整除性。 |
| `--empty-unused-memory-level` | `train.empty_unused_memory_level` | PyTorch CUDA empty_cache 调用频率（0、1 或 2）。 |
| `--check-weight-hash-across-dp-replicas-interval` | `train.check_weight_hash_across_dp_replicas_interval` | 验证 DP 权重一致性的间隔。 |
| `--train-iters` | `train.train_iters` | 训练迭代次数。 |
| `--exit-interval` | `train.exit_interval` | 当迭代次数 % 间隔 == 0 时退出。 |
| `--exit-duration-in-mins` | `train.exit_duration_in_mins` | N 分钟后退出。 |
| `--exit-signal-handler` | `train.exit_signal_handler` | 收到 SIGTERM 时保存并关闭。 |
| `--manual-gc` | `train.manual_gc` | 启用手动 Python GC 调度。 |
| `--manual-gc-interval` | `train.manual_gc_interval` | 手动 GC 运行之间的步数。 |
| `--no-manual-gc-eval` | `train.manual_gc_eval=false` | 在评估边界禁用 GC。 |
| `--eval-iters` | `train.eval_iters` | 每次验证运行的评估迭代次数。 |
| `--eval-interval` | `train.eval_interval` | 验证之间的步数间隔。 |
| `--skip-train` | `train.skip_train` | 跳过训练循环（仅评估）。 |

### 调度器 / 正则化

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--lr-decay-style` | `scheduler.lr_decay_style` | 学习率调度：constant/linear/cosine/ISR/WSD。 |
| `--lr-decay-iters` | `scheduler.lr_decay_iters` | 学习率衰减所覆盖的迭代次数。 |
| `--lr-wsd-decay-style` | `scheduler.lr_wsd_decay_style` | WSD 退火风格。 |
| `--lr-wsd-decay-iters` | `scheduler.lr_wsd_decay_iters` | WSD 退火阶段的迭代次数。 |
| `--lr-warmup-fraction` | `scheduler.lr_warmup_fraction` | 预热阶段占衰减区间的比例。 |
| `--lr-warmup-iters` | `scheduler.lr_warmup_iters` | 预热迭代次数（绝对值）。 |
| `--lr-warmup-init` | `scheduler.lr_warmup_init` | 预热开始时的初始学习率。 |
| `--lr-decay-samples` | `scheduler.lr_decay_samples` | 学习率衰减所覆盖的样本数（基于样本的训练）。 |
| `--lr-warmup-samples` | `scheduler.lr_warmup_samples` | 预热样本数（基于样本的训练）。 |
| `--lr` | `optimizer.lr` | 基础学习率。 |
| `--min-lr` | `optimizer.min_lr` | 最小学习率。 |
| `--clip-grad` | `optimizer.clip_grad` | 梯度裁剪值。 |
| `--weight-decay` | `optimizer.weight_decay` | 权重衰减。 |
| `--adam-beta1` | `optimizer.adam_beta1` | Adam beta1。 |
| `--adam-beta2` | `optimizer.adam_beta2` | Adam beta2。 |
| `--override-opt_param-scheduler` | `scheduler.override_opt_param_scheduler` | 忽略检查点中的调度器并使用配置。 |
| `--use-checkpoint-opt-param-scheduler` | `scheduler.use_checkpoint_opt_param_scheduler` | 从检查点加载调度器。 |
| `--start-weight-decay` | `scheduler.start_weight_decay` | 起始权重衰减（非恒定模式）。 |
| `--end-weight-decay` | `scheduler.end_weight_decay` | 结束权重衰减（非恒定模式）。 |
| `--weight-decay-incr-style` | `scheduler.weight_decay_incr_style` | 权重衰减调度：constant/linear/cosine。 |

### 检查点

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--save` | `checkpoint.save` | 写入检查点的目录。 |

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--save-interval` | `checkpoint.save_interval` | 持久化保存之间的迭代次数。 |
| `--no-save-optim` | `checkpoint.save_optim=false` | 不保存优化器状态。 |
| `--no-save-rng` | `checkpoint.save_rng=false` | 不保存 RNG 状态。 |
| `--load` | `checkpoint.load` | 加载检查点的目录。 |
| `--no-load-optim` | `checkpoint.load_optim=false` | 不加载优化器状态。 |
| `--load-main-params-from-ckpt` | `checkpoint.load_main_params_from_ckpt` | 直接加载 FP32 主参数。 |
| `--no-load-rng` | `checkpoint.load_rng=false` | 不加载 RNG 状态。 |
| `--non-persistent-save-interval` | `checkpoint.non_persistent_save_interval` | 临时保存的频率。 |
| `--non-persistent-ckpt-type` | `checkpoint.non_persistent_ckpt_type` | 临时检查点类型（global/local/memory）。 |
| `--non-persistent-global-ckpt-dir` | `checkpoint.non_persistent_global_ckpt_dir` | 全局临时保存目录。 |
| `--non-persistent-local-ckpt-dir` | `checkpoint.non_persistent_local_ckpt_dir` | 每个 rank 本地临时保存目录。 |
| `--non-persistent-local-ckpt-algo` | `checkpoint.non_persistent_local_ckpt_algo` | 本地保存算法选择。 |
| `--finetune` | `checkpoint.finetune` | 加载权重，重置迭代次数，不加载优化器/RNG。 |
| `--pretrained-checkpoint` | `checkpoint.pretrained_checkpoint` | 用于微调/SFT 的预训练权重路径。 |
| `--ckpt-step` | `checkpoint.ckpt_step` | 要加载的明确步骤。 |
| `--use-checkpoint-args` | `checkpoint.use_checkpoint_args` | 使用检查点元数据中的参数覆盖模型参数。 |
| `--exit-on-missing-checkpoint` | `checkpoint.exit_on_missing_checkpoint` | 如果未找到 `load` 路径则退出。 |
| `--ckpt-format` | `checkpoint.ckpt_format` | 格式：torch_dist/zarr/fsdp_dtensor。 |
| `--ckpt-convert-format` | `checkpoint.ckpt_convert_format` | 转换目标格式。 |
| `--ckpt-convert-save` | `checkpoint.ckpt_convert_save` | 转换后检查点的输出目录。 |
| `--no-ckpt-fully-parallel-save` | `checkpoint.fully_parallel_save=false` | 禁用数据并行（DP）并行保存。 |
| `--async-save` | `checkpoint.async_save` | 启用异步保存（仅限 torch_dist 格式）。 |
| `--use-persistent-ckpt-worker` | `checkpoint.use_persistent_ckpt_worker` | 用于异步保存的后台工作进程。 |
| `--ckpt-fully-parallel-load` | `checkpoint.fully_parallel_load` | 启用数据并行（DP）并行加载。 |
| `--ckpt-assume-constant-structure` | `checkpoint.ckpt_assume_constant_structure` | 为固定结构优化。 |
| `--dist-ckpt-strictness` | `checkpoint.dist_ckpt_strictness` | 加载时对键不匹配的处理方式。 |
| `--auto-detect-ckpt-format` | `checkpoint.auto_detect_ckpt_format` | 加载时自动检测检查点格式。 |
| `--replication` | `checkpoint.replication` | 启用本地检查点的复制。 |
| `--replication-jump` | `checkpoint.replication_jump` | 副本 rank 之间的间隔。 |
| `--replication-factor` | `checkpoint.replication_factor` | 副本数量。 |
| `--no-strict-fsdp-dtensor-load` | `checkpoint.strict_fsdp_dtensor_load=false` | 放宽 FSDP-DTensor 的严格加载要求。 |

### 日志记录

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--log-interval` | `logger.log_interval` | 控制台日志之间的步数间隔。 |
| `--log-params-norm` | `logger.log_params_norm` | 计算并记录参数的 L2 范数。 |
| `--log-throughput` | `logger.log_throughput` | 记录每个 GPU 的 tokens/秒。 |
| `--log-progress` | `logger.log_progress` | 将 tokens 和 FLOPs 写入 progress.txt。 |
| `--timing-log-level` | `logger.timing_log_level` | 0=最少；1=粗略操作；2=许多操作。 |
| `--timing-log-option` | `logger.timing_log_option` | 跨 rank 的 max/minmax/all 计时选项。 |
| `--tensorboard-dir` | `logger.tensorboard_dir` | TensorBoard 日志目录。 |
| `--tensorboard-log-interval` | `logger.tensorboard_log_interval` | TensorBoard 事件之间的步数间隔。 |
| `--tensorboard-queue-size` | `logger.tensorboard_queue_size` | 待处理的 TensorBoard 事件队列大小。 |
| `--log-timers-to-tensorboard` | `logger.log_timers_to_tensorboard` | 将计时器写入 TensorBoard。 |
| `--no-log-loss-scale-to-tensorboard` | `logger.log_loss_scale_to_tensorboard=false` | 禁用损失缩放（loss-scale）的 TensorBoard 日志。 |
| `--log-validation-ppl-to-tensorboard` | `logger.log_validation_ppl_to_tensorboard` | 将验证困惑度（ppl）写入 TensorBoard。 |
| `--log-memory-to-tensorboard` | `logger.log_memory_to_tensorboard` | 在 TensorBoard 中启用内存统计。 |
| `--log-world-size-to-tensorboard` | `logger.log_world_size_to_tensorboard` | 在 TensorBoard 中记录 world size。 |
| `--wandb-project` | `logger.wandb_project` | Weights & Biases 项目。 |
| `--wandb-entity` | `logger.wandb_entity` | Weights & Biases 实体/团队。 |
| `--wandb-exp-name` | `logger.wandb_exp_name` | W&B 中的运行名称。 |
| `--wandb-save-dir` | `logger.wandb_save_dir` | 用于 W&B 产物的本地目录。 |
| `--logging-level` | `logger.logging_level` | Python 日志记录级别（例如，20=INFO）。 |
| `--log-energy` | `logger.log_energy` | 记录能量消耗（焦耳）（如果可用）。 |

### RNG / 初始化

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--seed` | `rng.seed` | 全局随机种子。 |
| `--data-parallel-random-init` | `rng.data_parallel_random_init` | 启用每个数据并行（DP）等级的随机初始化。 |
| `--te-rng-tracker` | `rng.te_rng_tracker` | 使用 TE RNG（CUDA 图所需）。 |
| `--inference-rng-tracker` | `rng.inference_rng_tracker` | 为推理稳定性调整的 RNG。 |

### 分布式初始化与拓扑

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--distributed-backend` | `dist.distributed_backend` | 进程组后端（nccl/gloo）。 |
| `--distributed-timeout-minutes` | `dist.distributed_timeout_minutes` | 进程组初始化和集合操作超时时间。 |
| `--no-align-grad-reduce` | `dist.align_grad_reduce=false` | 在每个管道并行（PP）阶段独立启动 DP 规约。 |
| `--disable-gloo-process-groups` | `dist.use_gloo_process_groups=false` | 禁用辅助 Gloo 进程组的创建。 |
| `--use-sharp` | `dist.use_sharp` | 为 DP 进程组启用 SHARP 集合通信。 |
| `--sharp-enabled-group` | `dist.sharp_enabled_group` | 哪个 DP 组启用 SHARP。 |
| `--high-priority-stream-groups` | `dist.high_priority_stream_groups` | 为指定组使用高优先级通信流。 |
| `--use-tp-pp-dp-mapping` | `dist.use_tp_pp_dp_mapping` | 在初始化时使用 TP-PP-DP 等级排序。 |

额外的分布式/优化器重叠设置：

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--use-distributed-optimizer` | `ddp.use_distributed_optimizer` 和 `optimizer.use_distributed_optimizer` | 启用分布式优化器；设置是同步的。 |
| `--overlap-grad-reduce` | `ddp.overlap_grad_reduce` | 重叠 DP 梯度规约-分散（reduce-scatter）。 |
| `--overlap-param-gather` | `ddp.overlap_param_gather` | 重叠参数全收集（all-gather）与前向传播（fprop）。 |

### 性能分析

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--profile` | `profiling.use_nsys_profiler` | 启用 nsys 性能分析（捕获通过外部 CLI 控制）。 |
| `--use-pytorch-profiler` | `profiling.use_pytorch_profiler` | 启用 PyTorch 性能分析器（兼容 TensorBoard）。 |
| `--profile-step-start` | `profiling.profile_step_start` | 开始性能分析的全局步数。 |
| `--profile-step-end` | `profiling.profile_step_end` | 停止性能分析的全局步数。 |
| `--profile-ranks` | `profiling.profile_ranks` | 要进行性能分析的全局等级。 |
| `--record-memory-history` | `profiling.record_memory_history` | 跟踪内存历史。 |
| `--memory-snapshot-path` | `profiling.memory_snapshot_path` | 内存快照的输出路径。 |
| (shapes) | `profiling.record_shapes` | 记录张量形状（有开销）。 |

### 进程内重启

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--inprocess-restart` | `inprocess_restart.enabled` | 启用 nvrx 进程内重启。 |
| `--inprocess-max-iterations` | `inprocess_restart.max_iterations` | 最大重启尝试次数。 |
| `--inprocess-monitor-thread-interval` | `inprocess_restart.monitor_thread_interval` | 监控线程轮询间隔。 |
| `--inprocess-monitor-process-interval` | `inprocess_restart.monitor_process_interval` | 监控进程轮询间隔。 |
| `--inprocess-progress-watchdog-interval` | `inprocess_restart.progress_watchdog_interval` | 自动更新进度时间戳的频率。 |
| `--inprocess-heartbeat-interval` | `inprocess_restart.heartbeat_interval` | 无响应等级的心跳检测频率。 |
| `--inprocess-soft-timeout` | `inprocess_restart.soft_timeout` | 软进度超时时间。 |
| `--inprocess-hard-timeout` | `inprocess_restart.hard_timeout` | 强制终止前的硬超时时间。 |
| `--inprocess-heartbeat-timeout` | `inprocess_restart.heartbeat_timeout` | 心跳丢失超时时间。 |
| `--inprocess-barrier-timeout` | `inprocess_restart.barrier_timeout` | 内部屏障（barrier）超时时间。 |
| `--inprocess-completion-timeout` | `inprocess_restart.completion_timeout` | 完成屏障超时时间。 |
| `--inprocess-last-call-wait` | `inprocess_restart.last_call_wait` | 收集最终故障前的延迟时间。 |
| `--inprocess-termination-grace-time` | `inprocess_restart.termination_grace_time` | SIGTERM 到 SIGKILL 的宽限期。 |
| `--inprocess-granularity` | `inprocess_restart.granularity` | 重启粒度（节点/等级）。 |
| `--inprocess-active-world-size` | `inprocess_restart.active_world_size` | 活跃等级数量；其余为备用等级。 |
| `--inprocess-empty-cuda-cache` | `inprocess_restart.empty_cuda_cache` | 在重启完成时清空 CUDA 缓存。 |

### 掉队检测

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--log-straggler` | `straggler.log_straggler` | 跟踪并记录掉队 GPU。 |
| `--disable-straggler-on-startup` | `straggler.disable_straggler_on_startup` | 启动时禁用掉队检测器。 |
| `--straggler-ctrlr-port` | `straggler.straggler_ctrlr_port` | 用于开关控制的控制器端口。 |

| `--straggler-minmax-count` | `straggler.straggler_minmax_count` | 报告最小/最大吞吐量的排名数量。 |

### 重运行状态机

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--error-injection-rate` | `rerun_state_machine.error_injection_rate` | 注入验证扰动的频率。 |
| `--error-injection-type` | `rerun_state_machine.error_injection_type` | 注入类型（正确/瞬时/持久）。 |
| `--rerun-mode` | `rerun_state_machine.rerun_mode` | 禁用/验证结果/报告确定性统计。 |

### 数据 / 分词器参数

| megatron-lm 参数 | Megatron Bridge 配置 | 描述 |
| --- | --- | --- |
| `--tokenizer-type` | `tokenizer.tokenizer_type` | 分词器实现（例如，HuggingFaceTokenizer）。 |
| `--tokenizer-model` | `tokenizer.tokenizer_model` | 分词器的模型名称/路径。 |
| `--num-workers` | `dataset.num_workers` | DataLoader 工作进程数。 |
| `--no-create-attention-mask-in-dataloader` | `dataset.skip_getting_attention_mask_from_dataset=true` | 使用后端生成的掩码。 |