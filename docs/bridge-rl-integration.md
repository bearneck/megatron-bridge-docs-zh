# 在强化学习框架中适配 Megatron Bridge

Megatron Bridge 提供了一条清晰、支持并行化的路径，用于将 🤗 Hugging Face 模型与 Megatron-Core 训练结合使用，并可在推理时转换回来。本指南展示了如何将 Megatron Bridge 适配到新的 RL 框架中，以实现：

- 将 Hugging Face (HF) 检查点转换为 Megatron 格式，以进行可扩展训练
- 使用 Megatron-Core 进行训练，支持 TP/PP/CP/MoE 并行、检查点和高效数据路径
- 将训练后的权重重新适配回 HF 格式，以便与推理引擎（例如 vLLM）一起部署，包括零拷贝/IPC 流程

这些示例反映了 NeMo-RL 如何集成 Megatron Bridge：

- [nemo_rl/models/megatron/community_import.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/megatron/community_import.py)
- [nemo_rl/models/policy/megatron_policy_worker.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/policy/megatron_policy_worker.py)

- 本仓库中的本地示例脚本：[examples/rl/rlhf_with_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/rl/rlhf_with_bridge.py)


## 先决条件

- 可用的 PyTorch + NCCL GPU 环境
- 已安装 Megatron-LM (MCore) 和 Megatron-Bridge
- 用于多 GPU 设置的分布式启动器（例如 `torchrun`、`srun`）
- 如果需要，可以访问受限制的 HF 仓库（导出 `HF_TOKEN`）

```bash
export HF_TOKEN=<your_hf_token_if_needed>
```


## 1) 一次性 HF → Megatron 检查点转换

使用 `AutoBridge` 将 HF 模型导入为 Megatron 格式。这将写入一个包含 `run_config.yaml` 的 Megatron 检查点目录，你将在训练期间重复使用它。

```python
from megatron.bridge import AutoBridge

# 将模型导入为 Megatron 检查点格式（一次调用）
AutoBridge.import_ckpt(
    hf_model_id="meta-llama/Llama-3.2-1B",
    megatron_path="/path/to/megatron_ckpt/llama32_1b",
)
```

或者，使用显式的提供程序和并行化设置（类似于 [nemo_rl/models/megatron/community_import.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/megatron/community_import.py)）：

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=True)

# 配置导入期间使用的分布式并行化
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 1
provider.expert_tensor_parallel_size = 1
provider.num_layers_in_first_pipeline_stage = 0
provider.num_layers_in_last_pipeline_stage = 0
provider.finalize()

# 创建分布式模型并保存为 Megatron 检查点
megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)
bridge.save_megatron_model(megatron_model, "/path/to/megatron_ckpt")
```

你也可以查看并尝试我们的多 GPU 转换示例脚本：[examples/conversion/hf_megatron_roundtrip_multi_gpu.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_megatron_roundtrip_multi_gpu.py)


注意事项：
- 导入时的并行化仅用于加载/转换。保存的配置会恢复为规范值，以避免训练时出现验证问题。
- 如果你在框架内运行，请确保在导入前后根据需要销毁或初始化进程组，以清理任何现有的分布式状态。如果尚未设置分布式环境，`provide_distributed_model` 方法将初始化一个新的分布式环境。


## 2) 构建训练配置并初始化 Megatron-Core

将你的 RL 框架配置转换为 Megatron Bridge 的 `ConfigContainer`，用于模型、优化器、调度器、DDP、分词器和检查点。

```python
import torch
from megatron.bridge.training.config import (
    ConfigContainer,
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    DistributedDataParallelConfig,
    CheckpointConfig,
    TokenizerConfig,
)
from nemo_rl.models.policy import PolicyConfig  # 或你自己的策略配置类型

# 示例：将你的 RL 配置映射到 Megatron 配置
def build_megatron_config(rl_cfg: PolicyConfig, pretrained_ckpt_dir: str) -> ConfigContainer:
    model_cfg = rl_cfg["megatron_cfg"].copy()
    # 精度
    dtype = rl_cfg["precision"]
    model_cfg["bf16"] = dtype == "bfloat16"
    model_cfg["fp16"] = dtype == "float16"

    checkpoint = CheckpointConfig(
        save_interval=100,
        save=rl_cfg["train_ckpt_dir"],
        load=rl_cfg["train_ckpt_dir"],
        pretrained_checkpoint=pretrained_ckpt_dir,
        async_save=False,
        fully_parallel_save=True,
        fully_parallel_load=True,
        load_rng=False,
    )

    ddp = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["grad_reduce_in_fp32"],
        overlap_grad_reduce=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["overlap_grad_reduce"],
```

overlap_param_gather=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["overlap_param_gather"],
        average_in_collective=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["average_in_collective"],
        use_distributed_optimizer=rl_cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"],
        data_parallel_sharding_strategy=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["data_parallel_sharding_strategy"],
    )

    opt = OptimizerConfig(**rl_cfg["megatron_cfg"]["optimizer"])  # lr, wd, etc.
    sch = SchedulerConfig(**rl_cfg["megatron_cfg"]["scheduler"])  # warmup, decay, etc.

    train = TrainingConfig(
        micro_batch_size=rl_cfg["train_micro_batch_size"],
        global_batch_size=rl_cfg["train_global_batch_size"],
        train_iters=rl_cfg["megatron_cfg"]["train_iters"],
    )

    tokenizer = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=rl_cfg["model_name"],
    )

    return ConfigContainer(
        model=model_cfg,
        checkpoint=checkpoint,
        logger=None,
        train=train,
        optimizer=opt,
        ddp=ddp,
        scheduler=sch,
        dataset=None,
        tokenizer=tokenizer,
    )
```

使用一个类似于 NeMo-RL 中 `setup_megatron_model` 的辅助函数来初始化 Megatron-Core：

```python
from megatron.bridge.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.checkpointing import init_checkpointing_context, load_checkpoint
from megatron.bridge.training.state import GlobalState

# 最小化引导
state = GlobalState()
state.cfg = megatron_cfg
initialize_megatron(cfg=megatron_cfg)
set_jit_fusion_options(megatron_cfg.model, megatron_cfg.train.micro_batch_size)

ckpt_ctx = init_checkpointing_context(megatron_cfg.checkpoint)
model_list = get_model(
    megatron_cfg.model,
    megatron_cfg.ddp,
    use_torch_fsdp2=megatron_cfg.dist.use_torch_fsdp2,
    overlap_param_gather_with_optimizer_step=megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
    data_parallel_random_init=megatron_cfg.rng.data_parallel_random_init,
)
optimizer, scheduler = setup_optimizer(
    optimizer_config=megatron_cfg.optimizer,
    scheduler_config=megatron_cfg.scheduler,
    model=model_list,
    use_gloo_process_groups=megatron_cfg.dist.use_gloo_process_groups,
)

# 可选：加载预训练检查点
load_checkpoint(
    state,
    model_list,
    optimizer,
    scheduler,
    checkpointing_context=ckpt_ctx,
    skip_load_to_model_and_opt=False,
)

model = model_list[0]
```

需要在你的 RL 配置中处理的关键映射：
- 并行性：`tensor_model_parallel_size`、`pipeline_model_parallel_size`、`context_parallel_size`（需要序列打包）以及 MoE（`expert_*`）。
- 精度：`bf16`/`fp16` 加上 `pipeline_dtype`。
- 激活重计算：用于节省内存的重计算设置。
- FP8（高级）：如果启用，请注意对齐/填充要求。


## 3) 训练循环集成（前向/反向/微批次）

Megatron-Core 暴露了 `get_forward_backward_func()` 来运行微批次循环。插入你的 RL 损失函数。

```python
from functools import partial
from megatron.core.pipeline_parallel import get_forward_backward_func

model.train()
forward_backward = get_forward_backward_func()

# 你的损失函数应返回 (loss_tensor, metrics_dict)
def rl_loss_fn(outputs, batch):
    # 为你的 RL 目标（例如 PPO、DPO）计算 logits → 损失
    loss = outputs.sum() * 0.0  # 占位符
    return loss, {"loss": loss.detach()}

# 前向步骤：准备输入；返回输出和一个产生损失的收集器

def forward_step_fn(data_iterator, model):
    batch = next(data_iterator).to("cuda")
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        packed_seq_params=batch.get("packed_seq_params"),  # 如果使用序列打包
        # 多模态特征可以作为 kwargs 传递
    )
    return outputs, (lambda _out: rl_loss_fn(outputs, batch))

losses_reduced = forward_backward(
    forward_step_func=forward_step_fn,
    data_iterator=your_microbatch_iterator,
    model=model,
    num_microbatches=num_microbatches,
    seq_length=sequence_length,
    micro_batch_size=micro_batch_size,
    decoder_seq_length=sequence_length,
    forward_only=False,
    do_not_average_loss=True,
)

# 优化器/调度器步骤
update_successful, grad_norm, _ = optimizer.step()
scheduler.step(increment=global_batch_size)
```

序列打包和上下文并行性：
- 如果 `context_parallel_size > 1`，启用序列打包，并在调用模型之前为每个微批次构建 `packed_seq_params` 和 `cu_seqlens`。
- 使用 FP8 时，确保序列填充符合硬件友好的倍数（例如，lcm(16, 2 × TP × CP)）。


## 4) 用于 RL 目标的 Token 对数概率（优势函数、DPO 等）

为了评估词元对数概率，运行仅前向传播并将 TP 分片的逻辑值归约为每个词元的对数概率。

```python
import torch
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_rank

@torch.no_grad()
def get_token_logprobs(model, batch):
    model.eval()
    input_ids = batch["input_ids"].to("cuda")
    outputs = model(input_ids=input_ids)

    # 将 TP 逻辑值归约 → 针对目标的本地对数概率
    tp_group = get_tensor_model_parallel_group()
    tp_rank = get_tensor_model_parallel_rank()

    # 使用类似于 NeMo-RL 中 `from_parallel_logits_to_logprobs` 的归约器
    token_logprobs = your_reduce_parallel_logits_to_logprobs(
        outputs,
        target=input_ids,
        vocab_start_index=tp_rank * outputs.shape[-1],
        vocab_end_index=(tp_rank + 1) * outputs.shape[-1],
        tp_group=tp_group,
        inference_only=True,
    )

    # 在前面补零以保持与输入相同的序列长度
    token_logprobs = torch.cat([torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1)
    return token_logprobs.cpu()
```

如果使用序列打包（sequence packing）和上下文并行（context parallelism），请切换到使用 `packed_seq_params` 和 `cu_seqlens` 进行正确对齐的打包变体。

## 5) 检查点（保存/加载）

使用 Megatron-Bridge 的检查点辅助工具。在保存期间，如果需要，可以临时禁用重叠的参数收集钩子。

```python
from megatron.bridge.training.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    init_checkpointing_context,
)

ckpt_ctx = init_checkpointing_context(megatron_cfg.checkpoint)
save_checkpoint(
    state=state,
    model=[model],
    optimizer=optimizer,
    opt_param_scheduler=scheduler,
    num_floating_point_operations_so_far=state.train_state.floating_point_operations_so_far,
    checkpointing_context=ckpt_ctx,
)
```

提示：
- 在大规模情况下，优先使用完全并行保存/加载（`fully_parallel_save=True`, `fully_parallel_load=True`）。

## 6) 适配：Megatron → HF 用于推理（vLLM, Triton 等）

两种常见途径：

### A) 导出完整的 HF 检查点（最简单）

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
# 从你的训练检查点加载 Megatron 模型
megatron_model = bridge.load_megatron_model("/path/to/train_ckpt")

# 逐个参数地遍历 HF 权重
for name, weight in bridge.export_hf_weights(megatron_model, cpu=True, show_progress=False):
    # process_or_save(name, weight)
    pass
```

将你的推理引擎（例如 vLLM）指向 `"/path/to/hf_export"`。

### B) 通过 ZMQ 进行零拷贝流式传输（快速适配，共置）

将张量从训练端流式传输到你的推理运行时，无需写入磁盘。传输使用 ZMQ 点对点，带有异步发送/接收和乒乓缓冲区以实现重叠；Ray 仅用于轻量级协调。这取代了早期临时的每个张量 IPC 句柄传递，并与 [NVIDIA-NeMo/RL#1267](https://github.com/NVIDIA-NeMo/RL/pull/1267) 中的重构保持一致。

**概念（计划和分块如何工作）：**
- **传输和重叠：** ZMQ P2P 流式传输结合异步发送/接收和乒乓缓冲区，使得权重收集和应用之间能够重叠。
- **转换任务（规划）：** `bridge.get_conversion_tasks([model])` 返回一个有序的每个参数转换任务列表，这些任务编码了如何将分片的 Megatron 权重（TP/PP/MoE/CP）转换回 HF 张量。工作器将其存储在 `self.refit_conversion_tasks` 中，并在流式传输分块时推进游标（`self.refit_conversion_tasks_current_index`）。参见 `nemo_rl/models/policy/megatron_policy_worker.py` 中的方法 `prepare_refit_info()`、`_calculate_refit_param_info()` 和 `get_weights_ipc_handles()`。
- **跨 PP 等级的大小估计：** 参数仅在其所属的 PP 等级上具体化。工作器计算每个参数的字节大小，然后将这些大小广播到所有 PP 等级，以便整个流水线能就分块边界达成一致。参见 `megatron_policy_worker.py` 中的 `broadcast_object_across_pp_ranks()` 和 `_calculate_refit_param_info()`。
- **内存感知分块：** 使用你的空闲 GPU 内存预算（例如 `NRL_REFIT_BUFFER_MEMORY_RATIO`）来决定下一个分块中包含多少参数（`keys` 集合）。工作器暴露 `prepare_weights_for_ipc()` 方法，该方法返回 `(param_info, total_available_bytes)` 并重置转换游标；然后控制器重复选择累积字节大小 ≤ 预算的 `keys`，并通过 ZMQ 将它们流式传输给消费者。
- **设备路由：** 句柄在 `device_uuid` 键（CUDA 设备的 NVML UUID）下返回。推理端应映射同一设备上的句柄（或通过你的通信器进行协调）。对于集体更新，工作器也可以直接广播张量（`broadcast_weights_for_collective`）。

- **并行性细节：** 使用张量并行（TP）/专家并行（EP）时，导出的 HuggingFace 张量是从分片中重新组装的；使用检查点并行（CP）/序列打包时，形状/数据类型在导出时已经一致。FP8 或混合精度会影响大小估计；工作器在估计字节数时会考虑数据类型缩放。

```python
import os
import torch
from collections import defaultdict
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)

# 1) 规划：检查名称/形状/数据类型并估计内存
refit_param_info_hf = {}
for name, tensor in bridge.export_hf_weights([model], show_progress=False):
    refit_param_info_hf[name] = (tuple(tensor.shape), tensor.dtype)

# 2) 为暂存缓冲区做预算（可选）
from nemo_rl.utils.nvml import get_free_memory_bytes  # 或你自己的 NVML 包装器
free_bytes = get_free_memory_bytes(torch.cuda.current_device())
ratio = float(os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.2"))
allowed_bytes = int(free_bytes * ratio)

# 3) 通过 ZMQ 流式传输数据块
from nemo_rl.utils.nvml import get_device_uuid

# 一次性构建转换任务，并在流式传输时推进索引
refit_conversion_tasks = bridge.get_conversion_tasks([model])
refit_tasks_current_index = 0

def stream_next_chunk(keys: list[str]):
    """为此数据块生成 ZMQ 多部分帧。
    帧通常包括：(metadata_json_bytes, payload_bytes)。
    """
    global refit_tasks_current_index
    conversion_tasks = refit_conversion_tasks[
        refit_tasks_current_index : refit_tasks_current_index + len(keys)
    ]
    refit_tasks_current_index += len(keys)

    device_uuid = get_device_uuid(torch.cuda.current_device())

    # 工作器暴露一个流式生成器，重叠收集和发送操作
    for frames in worker.stream_refit_chunks(
        conversion_tasks=conversion_tasks, device_uuid=device_uuid
    ):
        yield frames  # 通过 zmq_socket.send_multipart(frames) 发送

# 示例用法（生产者）
for frames in stream_next_chunk(list(refit_param_info_hf.keys())):
    zmq_socket.send_multipart(frames)
```

**实践中的数据分块（控制器端选择键）：**

```python
# param_info 类似 [(name, size_bytes), ...]，来自 prepare_refit_info 或 prepare_weights_for_ipc
param_info, budget_bytes = worker.prepare_weights_for_ipc()

cursor = 0
while cursor < len(param_info):
    batch_keys = []
    used = 0
    # 贪婪地将参数打包到此数据块中，直到超出预算
    while cursor < len(param_info):
        name, size_bytes = param_info[cursor]
        # size_bytes 已经广播到所有流水线并行（PP）等级；可以是 int 类型
        if used + int(size_bytes) > budget_bytes and len(batch_keys) > 0:
            break
        batch_keys.append(name)
        used += int(size_bytes)
        cursor += 1

    # 流式传输此数据块并在推理端消费
    for frames in worker.stream_refit_chunks(keys=batch_keys):
        zmq_socket.send_multipart(frames)
```

环境变量控制：
- `NRL_REFIT_BUFFER_MEMORY_RATIO`（默认值 `0.2`）—— 用于规划暂存区的空闲 GPU 内存比例

## 7) 最小适配器骨架

使用此骨架将 Megatron Bridge 嵌入到你的强化学习代码库中。填写配置映射、微批处理和损失逻辑。

```python
import torch
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.checkpointing import init_checkpointing_context, save_checkpoint
from megatron.core.pipeline_parallel import get_forward_backward_func

class MegatronBridgeAdapter:
    def __init__(self, rl_cfg, pretrained_ckpt_dir: str):
        self.rl_cfg = rl_cfg
        self.megatron_cfg = build_megatron_config(rl_cfg, pretrained_ckpt_dir)
        self.state = GlobalState(); self.state.cfg = self.megatron_cfg
        self.ckpt_ctx = init_checkpointing_context(self.megatron_cfg.checkpoint)
        self._init_model()

    def _init_model(self):
        from megatron.bridge.training.initialize import initialize_megatron, set_jit_fusion_options
        from megatron.bridge.models.model_provider import get_model
        from megatron.bridge.training.optim import setup_optimizer
        initialize_megatron(cfg=self.megatron_cfg)
        set_jit_fusion_options(self.megatron_cfg.model, self.megatron_cfg.train.micro_batch_size)
        self.model_list = get_model(self.megatron_cfg.model, self.megatron_cfg.ddp,
                                    use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                                    overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step)
        self.model = self.model_list[0]
        self.optimizer, self.scheduler = setup_optimizer(self.megatron_cfg.optimizer, self.megatron_cfg.scheduler, self.model_list,
                                                         use_gloo_process_groups=self.megatron_cfg.dist.use_gloo_process_groups)

    @torch.no_grad()
    def get_logprobs(self, batch):
```

```python
        self.model.eval()
        # 实现从并行 logits 到 token logprobs 的归约
        ...

    def train_step(self, mb_iter, num_microbatches, seq_len, mbs, loss_fn):
        self.model.train()
        fb = get_forward_backward_func()
        def fwd(data_it, model):
            batch = next(data_it).to("cuda")
            out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
            return out, (lambda _o: loss_fn(out, batch))
        fb(forward_step_func=fwd, data_iterator=mb_iter, model=self.model, num_microbatches=num_microbatches,
           seq_length=seq_len, micro_batch_size=mbs, decoder_seq_length=seq_len, forward_only=False, do_not_average_loss=True)
        ok, _, _ = self.optimizer.step(); self.scheduler.step(increment=self.rl_cfg["train_global_batch_size"])
        return ok

    def save_ckpt(self, path: str):
        save_checkpoint(self.state, [self.model], self.optimizer, self.scheduler,
                        num_floating_point_operations_so_far=self.state.train_state.floating_point_operations_so_far,
                        checkpointing_context=self.ckpt_ctx)

    def export_hf(self, out_dir: str, trust_remote_code: bool = False):
        from megatron.bridge import AutoBridge
        bridge = AutoBridge.from_hf_pretrained(self.rl_cfg["model_name"], trust_remote_code=trust_remote_code)
        # 使用 AutoBridge.export_hf_weights 直接流式传输权重；根据需要消费（保存/IPC）
        for name, tensor in bridge.export_hf_weights([self.model], show_progress=False):
            # process_or_save(name, tensor, out_dir)  # 实现你的消费者（例如，safetensors 或 IPC）
            pass
        # 可选地，持久化 safetensors 分片而不包含配置/分词器：
        # bridge.save_hf_weights([self.model], out_dir, show_progress=False)
```


## 8) 最佳实践与常见陷阱

- 并行性
  - 如果 `context_parallel_size > 1`，启用序列打包。使用打包的 logprob 归约器。
  - 使用 FP8 时，填充到硬件友好的倍数（例如，lcm(16, 2 × TP × CP)）。
- 卸载
  - 待补充
- MoE 路由器稳定性
  - 考虑冻结路由器权重并禁用路由器负载均衡，以减少训练不稳定性（参见 `freeze_moe_router`, `moe_router_bias_update_rate=0.0`）。
- 钩子
  - 在交换权重或保存时，暂时禁用 DDP 前向钩子，以避免与重叠的参数收集发生冲突。
- 检查点
  - 在大规模时使用完全并行保存/加载。除非在你的环境中已验证，否则避免异步保存。


## 另请参阅

- [与 🤗 Hugging Face 的桥接](./bridge-guide.md) 了解 HF↔Megatron 转换概述
- [nemo_rl/models/megatron/community_import.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/megatron/community_import.py) 获取导入/导出辅助工具
- [nemo_rl/models/policy/megatron_policy_worker.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/policy/megatron_policy_worker.py) 获取端到端 RL 集成（训练、logprobs、生成、重新拟合）