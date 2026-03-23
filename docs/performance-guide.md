# 性能调优指南

Megatron-Bridge 提供了广泛的功能，用于在 GPU 上进行高性能且内存高效的大语言模型训练，并预置了最优设置。然而，模型架构、超参数、GPU 数量和 GPU 类型等因素会影响可用选项，可能需要进行额外的调优才能达到最佳性能。本文档探讨了影响训练性能的因素，重点介绍了常见问题，并概述了可提高 MFU（模型浮点运算利用率）和 TCO（总拥有成本）的性能调优技术。

```{注意}
本指南引用了多个配置设置。这些设置将相对于包含它们的配置类进行引用，例如 `OptimizerConfig.lr`。有关配置设置的更多详细信息，请参阅 <project:apidocs/index.rst>。
```

```{注意}
本指南引用了 `TransformerConfig` 中的多个配置设置。请将这些设置应用于您模型的相应 ModelProvider，例如 `GPTModelProvider`，因为 `ConfigContainer` 不接受原始的 `TransformerConfig`。
```

## 低精度训练

1.  FP8 训练相比 BF16 训练的预期加速

    > 1.  默认的低精度 LLM 训练方案仅将 FP8 计算应用于 Transformer 块内的线性层，通常可实现 1.2–1.5 倍的加速。
    > 2.  然而，实际加速取决于训练时间中花费在这些线性层上的比例。例如，隐藏大小有限的小型 LLM 表现出较低的 FP8 加速，因为线性层的复杂度按 O(序列长度 × 隐藏大小²) 缩放，而其他逐元素计算层（例如，层归一化、Dropout、RoPE 和简单数学函数）按 O(序列长度 × 隐藏大小) 缩放，点积注意力按 O(序列长度² × 隐藏大小) 缩放。因此，在此类模型中，线性层对总训练时间的贡献较小。
    > 3.  不同的 FP8 方案使用不同的量化块大小，这会影响性能。较小的量化块通常在量化和 GEMM 执行中都会产生更高的开销。例如，使用 1×32 量化块的 MXFP8 比全张量级 FP8 缩放的效率低。

2.  FP8 训练加速较低的常见问题

    > 1.  当 LLM 使用小型 GPU 内核时，主机性能受限（参见[降低主机开销和抖动](#降低主机开销和抖动)）。
    > 2.  训练步骤时间中使用 FP8 计算的线性层比例较低。

## 并行映射策略

1.  使用分布式优化器的数据并行

    > 1.  您应该从数据并行（DP）映射开始。只要模型和激活内存能容纳在 GPU 内，数据并行通常能提供最佳性能，最大限度地减少通信开销，并最大化每个 GPU 的张量大小（与按张量分片相比）。
    >
    > 2.  Megatron-Bridge 使用分布式优化器作为数据并行训练的默认方法。它在数据并行秩之间分片主参数和优化器状态，与传统的数据并行训练相比，减少了模型状态内存使用量，且不增加通信开销。
    >
    >    > 1.  `OptimizerConfig.use_distributed_optimizer=true`

2.  按张量分片（张量并行或上下文并行映射）

    > 1.  当模型在数据并行映射下超出 GPU 内存容量时，张量并行（TP）是主要的推荐方案。然而，由于它涉及更高的通信开销，张量并行大小最好限制在高带宽的节点内网络（NVLink 域）中。
    >
    >    > 1.  `TransformerConfig.tensor_model_parallel_size=<int>`
    >
    > 2.  当训练运行中的序列长度显著大于隐藏大小时，激活内存可能会溢出。在这种情况下，上下文并行（CP）通过沿序列维度分片张量来提供帮助，使工作负载能够适应有限的 GPU 内存并提高性能。与张量并行（TP）类似，CP 需要 GPU 间通信来传递激活。然而，对于相同的张量大小，CP 通常会导致更低的通信量。

尽管如此，CP 的有效性取决于序列长度和隐藏大小的相对大小。当序列长度小于隐藏大小时，CP 会在每个 GPU 上产生窄（或"瘦"）的张量分片。这会减少数据重用并可能降低性能。

此外，由于 CP 分片了激活，它也会在分布式训练中分割优化器状态。因此，优化器状态分区同时跨越数据并行（DP）和上下文并行（CP）维度。

> > 1.  `TransformerConfig.context_parallel_size=<int>`
>
> 1.  性能提示：

>    > 1. 除非隐藏层大小或序列长度足够大，能够维持足够的每GPU并行度并避免过度的通信开销，否则不建议使用较大的张量并行（Tensor Parallelism）或上下文并行（Context Parallelism）规模。例如，对于LLAMA 3 70B模型，使用张量并行规模为8可能导致GPU利用率低下，并使训练受限于主机性能。
>    > 2. 你可以结合使用TP和CP，通过平衡通信开销来优化性能。例如，当序列大小大于隐藏层大小时，使用TP=2配合CP=2可以获得比TP=4更好的性能。
>    > 3. 更多技巧，请参阅[长序列训练](#long-sequence-train)。

1. 管道并行

   > 1. 当模型无法在仅使用张量并行的情况下放入GPU内存时，管道并行（Pipeline Parallelism, PP）是必要的。此外，应结合使用虚拟管道并行（Virtual Pipeline Parallelism, VPP）来减少由管道预热和冲刷气泡引起的开销。
   >
   >    > 1. `TransformerConfig.pipeline_model_parallel_size=<int>`
   >    > 2. `TransformerConfig.virtual_pipeline_model_parallel_size=<int>`
   >
   > 2. PP和VPP规模设定的性能提示：
   >
   >    > 1. PP也可以与逐张量分片方法结合使用，以减轻分片效率低下和管道气泡的影响。例如，当两种映射都能放入内存时，TP4 + PP2可能优于TP8，因为使用较大的TP会减少每GPU的张量大小，但会增加通信成本，从而增加暴露的通信。
   >    > 2. VPP会增加阶段间的通信开销。当全局批次包含许多微批次时，使用较小的VPP规模可以提高性能，因为暴露的通信成本超过了管道气泡的减少。
   >
   > 3. 跨管道阶段的非对称Transformer层分配
   >
   >    > 1. 具有大词汇量的LLM具有计算量大的嵌入查找和投影操作，这会导致跨管道阶段的负载不平衡。为了解决这个问题，Megatron-Bridge提供了一个选项，可以在处理嵌入查找和投影的第一个和最后一个管道阶段中少分配一个Transformer层，以更好地平衡工作负载。
   >    >
   >    >    > 1. `GPTProvider.account_for_embedding_in_pipeline_split=true`
   >    >    > 2. `GPTProvider.account_for_loss_in_pipeline_split=true`

2. 专家并行

   > 1. 专家并行（Expert Parallelism, EP）专为专家混合（Mixture-of-Experts, MoE）模型设计，用于在多个芯片间高效地分布稀疏MLP权重。它可以与其他并行策略结合使用，例如张量并行（TP）、上下文并行（CP）、管道并行（PP）、数据并行（DP）和全分片数据并行（FSDP）。在当前设计中，稠密注意力部分和稀疏MLP部分在它们的TP、CP和DP并行配置方面是完全解耦的。引入了专家张量并行（Expert Tensor Parallelism, ETP）来专门控制稀疏MLP部分的张量并行。ETP在分配给稀疏层EP的秩上，对稠密层使用TP。另一方面，基线是DEP，它在稀疏层的EP中，对稠密层折叠使用DP。
   >
   >    > 1. `TransformerConfig.expert_model_parallel_size=<int>`
   >    > 2. `TransformerConfig.expert_tensor_parallel_size=<int>`
   >
   > 2. 混合折叠选项和EP规模设定的性能提示：
   >
   >    > 1. 通常，EP保持在节点内高带宽网络（NVLink域）内，以最小化其可能引入的通信开销。然而，结合使用通信重叠技术（如管道重叠或1F1B重叠）与PP（例如DualPipe），可能使得将EP扩展到节点间网络成为可能。
   >    >
   >    > 2. 在稀疏MLP块内部，DP取代了CP，因为基于每个EP秩上分派的令牌，DP对计算模式没有影响。
   >    >
   >    > 3. 通常，ETP设置为1，以避免将TP应用于MLP GEMMs所带来的显著通信开销。
   >    >
   >    > 4. 在应用专家并行后，当多个专家被放置在单个芯片上时，启用分组GEMM可以显著提高计算效率。
   >    >
   >    >    > 1. `TransformerConfig.moe_grouped_gemm=True`

3. 全分片数据并行

   > 1. Megatron-Bridge支持PyTorch原生的FSDP。FSDP可以与逐张量分片方法结合使用。
   >
   >    > 1. 要使用PyTorch FSDP2：
   >    >
   >    >    > 1. `DistributedInitConfig.use_torch_fsdp2=True`
   >
   > 2. 在以下场景中，FSDP可能优于TP+PP+DP映射：
   >
   >    > 1. 对于具有大序列的小模型，参数AllGather和梯度ReduceScatter可以有效地隐藏在计算之下，并且短暂的通信重叠对重叠下的计算造成的干扰较小。

>    > 2. 在 FSDP 训练中，激活存储仍然是主要的内存瓶颈，因为 FSDP 仅对模型状态内存进行分片，并且需要较大的每 GPU 激活量来隐藏昂贵的 FSDP 通信。在 GB200 GPU 上，Megatron-Bridge 提供了一个选项，可以通过高速芯片间互连将激活卸载到主机内存。
>    > 3. 基线训练受主机性能限制，但 FSDP 通过消除 TP 或启用更大的微批次大小，允许更大的每 GPU 张量大小。

<!-- TODO: support megatron custom fsdp -->
<!-- > 1. Megatron-Bridge 支持两种全分片数据并行（FSDP）实现：PyTorch 原生 FSDP 和 Megatron Core 内构建的自定义 Megatron FSDP。虽然两者遵循相同的分片原则，但自定义实现针对性能进行了进一步优化。自定义 FSDP 的性能提升主要来自于最小化到通信张量的数据移动和重用通信缓冲区。两种 FSDP 方法都可以与每张量分片方法结合使用。 -->
<!-- > -->
<!-- >    > 1. 要使用 PyTorch FSDP2： -->
<!-- >    > -->
<!-- >    >    > 1. `DistributedInitConfig.use_torch_fsdp2=True` -->
<!-- >    > -->
<!-- >    > 2. 要使用自定义 Megatron FSDP： -->
<!-- >    > -->
<!-- >    >    > 1. `recipe.trainer.strategy.fsdp="megatron"` -->
<!-- >    >    > 2. `recipe.trainer.strategy.ddp.data_parallel_sharding_strategy="optim_grads_params"` -->
<!-- > -->
<!-- > 2. 在以下场景中，FSDP 可能优于 TP+PP+DP 映射： -->
<!-- > -->
<!-- >    > 1. 具有大序列的小模型，因此参数 AllGather 和梯度 ReduceScatter 可以有效地隐藏在计算之下，并且较短的通信重叠对重叠下的计算造成的干扰较小。 -->
<!-- >    > 2. 在 FSDP 训练中，激活存储仍然是主要的内存瓶颈，因为 FSDP 仅对模型状态内存进行分片，并且需要较大的每 GPU 激活量来隐藏昂贵的 FSDP 通信。在 GB200 GPU 上，Megatron-Bridge 提供了一个选项，可以通过高速芯片间互连将激活卸载到主机内存。 -->
<!-- >    > 3. 基线训练受主机性能限制，但 FSDP 通过消除 TP 或启用更大的微批次大小，允许更大的每 GPU 张量大小。 -->

4. 异构编码器并行

   > 1. 编码器管道并行
   >
   >    > 1. 使用 `T5ModelProvider.encoder_pipeline_model_parallel_size`。
   >    > 2. 在编码器-解码器架构中，如多模态模型（NeVA 等 VLM），编码器管道并行可用于向编码器添加管道并行。
   >    > 3. 管道并行控制解码器部分的流水线数量。
   >    > 4. 目前编码器管道并行限制为 1，即编码器最多只能占据 1 个 PP 阶段。
   >    > 5. 默认情况下，编码器管道并行为 0，解码器管道并行为 1。
   >    > 6. 当编码器管道并行大小为 0 时，它与解码器的第一个 PP 阶段共享。
   >
   > 2. 编码器张量并行
   >
   >    > 1. 使用 `T5ModelProvider.encoder_tensor_model_parallel_size`。
   >    > 2. 由于编码器往往比解码器小得多，我们还提供了为编码器设置与解码器不同数量的张量并行的能力。
   >    > 3. 默认情况下，编码器张量并行设置为 0，即编码器中的张量并行数量等于解码器中的张量并行数量。
   >    > 4. 要使用此选项，编码器管道并行必须大于 0，因为我们需要编码器位于其自己的管道阶段上。
   >    > 5. 编码器张量并行大小限制为小于或等于张量并行大小。
   >
   > 3. 使用这些功能时所需的 GPU 总数为：
   >
   >    > 1. 数据并行大小 * 上下文并行大小 * ((编码器 TP * 编码器 PP) + (解码器 TP * 解码器 PP))
   >
   > 4. 这些功能是实验性的，可能仍存在错误。关键的 bug 修复将在未来的版本中进行。

5. 使用 NVL72 的并行映射策略

   > 1. 仅使用数据并行或 FSDP 进行训练可以直截了当地充分利用 NVL72 系统的带宽。然而，当结合多种并行策略时，确保高通信量的通信器保持在每个 NVL72 域内非常重要。例如，当 TP=4、DP=16、PP=4 时，DP1/PP1 的第一个 TP 组中的 GPU 跨越了 NVLink 和网络域，导致通信性能受限于较慢的网络链路。为了避免这种情况，您可以选择 TP 和 DP 的大小，使得 TP × DP 的乘积能整除 NVL72 配置。如果模型并行大小不能自然对齐，则可能需要填充以支持不可整除的组大小。

> 2. 为了避免这种分区复杂性，您可以只使用 72 个 GPU 中的 64 个。

## 通信重叠与调优

1. 分布式优化器的数据并行通信

   > 1. 分布式优化器将参数的 AllGather 与第一个微批次的前向计算重叠，并将梯度的 ReduceScatter 与最后一个微批次的反向计算重叠。
   >
   >    > 1. `DistributedDataParallelConfig.overlap_param_gather=true`
   >    > 2. `DistributedDataParallelConfig.overlap_grad_reduce=true`
   >
   > 2. 当使用分布式优化器配合管道并行（PP）+ 虚拟管道并行（VPP）时，DP 通信与多个微批次重叠，增加了有效重叠的机会。此外，Megatron-Bridge 会跨管道并行 rank 对齐 DP 通信的执行时机，以同步因重叠导致的计算内核减速。
   >
   >    > 1. `DistributedDataParallelConfig.align_param_gather=true`
   >
   > 3. 大规模训练中缓慢的 DP 通信：
   >
   >    > 1. 在部分 DP 域内分布优化器状态，可以减少高延迟以太网上的通信开销。模型状态在分布式域外保持复制。在最后一个微批次的反向传播期间，梯度 ReduceScatter 在分布式域内进行，随后在非分布式域内进行 AllReduce。参数 AllGather 仅在分布式域内执行。
   >    >
   >    >    > 1. `DistributedDataParallelConfig.num_distributed_optimizer_instances= <int>`
   >    >
   >    > 2. 建议使用较大的 DP 通信消息大小，以最大化网络带宽利用率。您可以通过增加通信桶大小来实现这一点。
   >    >
   >    >    > 1. `DistributedDataParallelConfig.bucket_size=<number_of_elements: int>`
   >
   > 4. DP 通信重叠失败的常见原因：
   >
   >    > 1. Transformer Engine 中的持久化层归一化（LN）内核会对 GPU 中的所有 SM 使用自旋等待，导致 LN 内核及后续计算内核只有在 DP 通信完成后才会被调度。为防止这种情况，应使用以下环境变量配置适当的 SM 余量。
   >    >
   >    >    > 1. `NVTE_FWD_LAYERNORM_SM_MARGIN=<#SM for DP collectives = 16>`
   >    >    > 2. `NVTE_BWD_LAYERNORM_SM_MARGIN=<#SM for DP collectives = 16>`

<!-- 2. 自定义 Megatron FSDP -->

<!--    > 1. 除非您指定通信桶大小，否则 MCORE FSDP 使用固定的通信重叠，将每个 Transformer 层的参数 AllGather 和梯度 ReduceScatter 与其相关的前向和反向计算重叠。 -->

3. 张量并行（TP）通信（配合序列并行）

   > 1. Megatron-Bridge 目前使用 Transformer Engine 中的 userbuffer 后端来实现 TP 通信重叠。这提供了 TP 通信与依赖计算的流水线式重叠。
   >
   >    > 1. `CommOverlapConfig.tp_comm_overlap`
   >
   > 2. TP 通信重叠的方法、资源和精度是可配置的，默认情况下，性能最佳的配置已在 Megatron-Bridge 训练配方中设置。此外，您可以通过以下接口，按照 TransformerLayerTPOverlapCfg 类的结构设置自定义的 TP 通信重叠配置。
   >
   >    > 1. `CommOverlapConfig.tp_comm_overlap_cfg=<TransformerLayerTPOverlapCfg>`
   >
   > 3. TP 通信重叠设置技巧
   >
   >    > 1. 平衡通信与 GEMM 之间的 SM 数量
   >    >
   >    >    > 1. 对于 AllGather/ReduceScatter 批量重叠和 ReduceScatter 流水线重叠，您可以调整 SM 数量以平衡通信和 GEMM 的执行。为通信分配过多的 SM 可能会降低 GEMM 性能，而分配过少则可能暴露通信开销。通信的默认 SM 分配是 16，但您可以根据性能分析结果进行微调。
   >    >    > 2. `TPOverlapCfg.num_sm=<int>`
   >    >
   >    > 2. 设置 CGA 大小以提高 SM 利用率
   >    >
   >    >    > 1. CGA 大小可以设置在 1 到 4 之间，但不应超过为通信分配的 SM 数量。我们建议使用 CGA ≤ 2，以防止可能影响 GEMM 性能的潜在 SM 光栅化。
   >    >    > 2. `TPOverlapCfg.cga_size=<int≤4>`
   >    >
   >    > 3. 使用 4× 分割进行 ReduceScatter 和 GEMM 重叠，以优化 GEMM 效率与通信暴露之间的平衡。
   >    >
   >    >    > 1. 在 GEMM-then-ReduceScatter 流水线重叠中，一个 1× 的 ReduceScatter 块仍然会暴露。较小的分割大小会增加通信暴露，而较大的分割大小可能会因聚合的 GEMM 波量化而降低性能。我们发现 num_splits = 4 通常能提供最佳性能。
   >    >    > 2. `TPOverlapCfg.num_split=<int>`

> 4. Hopper 架构上 TP 通信重叠失败的常见原因
>
>    > 1. 在 H100 GPU 上，应设置环境变量 `CUDA_DEVICE_MAX_CONNECTIONS=1`。否则，TP 通信内核可能会被调度到 GEMM 操作的末尾，从而无法实现重叠。
>    > 2. 流水线式 TP 通信重叠使用在模型初始化时注册的静态用户缓冲区。因此，它不支持在训练步之间或 Transformer 层之间动态变化的激活张量。

4. 上下文并行（CP）通信

   > 1. CP 通信可通过 "cp_comm_type" 配置，可选值有 "p2p"、"all_gather"、"a2a" 或 "a2a+p2p"。"p2p" 通信被实现为环形交换发送/接收操作，并且被硬编码为与序列块的注意力计算重叠。更多细节请参见[长序列训练](#long-sequence-train)。

5. 专家并行通信

   > 1. 为了隐藏 EP 引入的 A2A/AG 通信，可以结合流水线并行（Pipeline Parallelism）使用流水线分割重叠或 1F1B 重叠。此功能将在未来版本的 Megatron-Bridge 中添加。

6. 流水线并行（PP）发送/接收通信

   > 1. 在稳定的 1F1B 状态下，PP 发送/接收默认设置为与计算重叠。
   > 2. 预热（warmup）和清空（flush）阶段的 PP 发送/接收默认是暴露的（即不重叠）。

(comm-data-types)=
## 通信数据类型

1. 分布式优化器和 FSDP 中的 FP8 数据并行参数 AllGather

   > 1. Megatron-Bridge 支持针对每张量 FP8 缩放方案的 FP8 参数 AllGather。此操作是无损的，可在减少内存使用的同时提升性能。
   >
   >    > 1. `MixedPrecisionConfig.fp8_param=true`

2. 分布式优化器和 FSDP 中的 BF16（替代 FP32）数据并行规约

   > 1. 我们已在大量模型训练运行中验证，BF16 规约在数值上是安全的。然而，在较大的数据并行规模下（例如 DP ≥ 128）使用 BF16 规约，特别是采用顺序累加副本的环形规约算法时，可能会影响数值稳定性。当在 NVIDIA InfiniBand 上使用 SHARP 时，BF16 规约会更加鲁棒，因为它以更高精度对中间部分规约结果执行二进制加法。
   >
   >    > 1. `DistributedDataParallelConfig.grad_reduce_in_fp32=false`

3. FP8 张量并行 ReduceScatter

   > 1. 当通信延迟超过 GEMM 执行时间时，使用 FP8 输入的 ReduceScatter 可以更好地隐藏通信开销。这种方法对数值影响较小，因为 GEMM 输出必须先转换为 FP8，然后在规约过程中再转换回高精度。
   >
   >    > 1. `TPOverlapCfg.fp8_buf=true`

4. 专家并行通信的 FP8 A2A 分发

   > 1. Megatron-Bridge 正在致力于支持 FP8 A2A 分发（在专家 FC1 之前），但仍将保持 BF16 A2A 合并（在专家 FC2 之后）。

## 大规模性能

1. 扩展训练作业通常通过增加数据并行域的规模来实现。在大规模训练中，这通常会导致每个全局批次只有少量微批次，甚至只有一个微批次，从而使大部分计算与数据并行通信重叠。为了在此类场景下保持高性能，您应专注于最小化数据并行通信的开销并减少主机驱动的 GPU 间抖动。

2. 您可以通过以下方式降低数据并行通信的开销：(1) 降低通信精度，例如使用 BF16 进行梯度规约和 FP8 进行参数收集；(2) 通过增加数据并行通信消息大小或使用分层数据并行规约来提高通信效率；或 (3) 在 InfiniBand 网络情况下使用 SHARP 进行多播和交换机规约。

   > 1. 使用 BF16 梯度规约和 FP8 参数收集在[通信数据类型](#comm-data-types)中有所描述。
   >
   > 2. 对于非流水线并行训练，可以使用下面的旋钮调整数据并行通信的桶大小。然而，在流水线并行训练中，桶大小是固定的，由分配给每个虚拟流水线等级的参数量决定。
   >
   >    > 1. `DistributedDataParallelConfig.bucket_size=<int: bytes>`
   >
   > 3. 设置下面的旋钮可以将分布式优化器的数据并行域拆分为分片域和复制域。然后梯度规约分两个阶段进行——每个域内一次——避免使用单个大的扁平环进行高延迟的集合操作。
   >
   >    > 1. `DistributedDataParallelConfig.num_distributed_optimizer_instances=<int: ≤dp_size>`

3. 减少主机驱动的 GPU 间抖动的思路在[降低主机开销和抖动](#lowering-overhead-jitter)中讨论。

(lowering-overhead-jitter)=
## 降低主机开销和抖动

1. 与主机开销相关的常见现象

   > 1. GPU FLOPS 显著偏低。
   > 2. 低精度（FP8）训练带来的性能提升很小。

> 3. 隐藏维度小或序列长度小的 LLM，或未使用序列打包的微调场景
> 4. 多 GPU 间通信延迟差异大的情况

2. 增大微批次大小并减少张量分片

   > 1. 在 GPU 内存允许的情况下，增大每个 GPU 上张量大小的最常见方法是增加微批次大小，或尽量减少不必要的张量分片（例如，张量并行或序列并行）。

3. 手动垃圾回收以对齐跨 GPU 的主机中断

   > 1. 与基线自动垃圾回收相比，Megatron-Bridge 手动对齐了跨 GPU 的垃圾回收时机，显著减轻了主机开销。
   >
   >    > 1. `TrainingConfig.manual_gc_interval=<int>`

4. 使用 CUDA 图消除重复的静态主机代码执行

   > 1. Megatron-Bridge 支持图捕获，可显著减少主机开销。CUDA 图仅适用于训练步骤间张量形状保持静态的 LLM。例如，它支持固定大小的打包序列，但无法处理每一步序列长度都变化的情况。此外，具有无令牌丢弃传播的 MoE 模型对 CUDA 图的支持有限，仅限于稠密模块。
   > 2. CUDA 图需要额外的内存用于静态缓冲区管理，通常会增加几 GB 的静态缓冲区开销，而当流水线并行规模大于 1 时，可能消耗超过 10GB 的内存。我们正在积极努力减少这部分内存开销。
   > 3. 配置详情（`cuda_graph_impl`, `cuda_graph_scope`）请参阅 [CUDA 图](training/cuda-graphs.md)。

5. 为 GPU 进程绑定 CPU 内存

   > 1. 将 CPU 核心绑定到 GPU 进程有助于缓解长延迟问题，并确保跨 GPU 的 GPU 队列延迟差异最小。当通信域规模较大时，此优化效果尤为显著。
   > 2. 基于 X86 的 GPU 系统示例命令行：`numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) <运行脚本>`
   > 3. 基于 Grace 的 GPU 系统示例命令行：`numactl --cpunodebind=$((SLURM_LOCALID/2)) --membind=$((SLURM_LOCALID/2)) <运行脚本>`

(reducing-memory-overflow)=
## 减少内存以避免内存溢出并提升训练效率的技术

1. 激活重计算

   > 1. Megatron-Bridge LLM 默认使用 Flash Attention 进行仅点积注意力的重计算，能以最小的计算开销高效地重新生成注意力操作产生的大型中间激活。
   >
   > 2. Megatron-Bridge 还支持重计算整个 Transformer 块的中间激活，这能显著减少激活内存使用，代价是增加约 30% 的计算量。可通过可配置设置调整要重计算的 Transformer 块数量。
   >
   >    > 1. `TransformerConfig.recompute_granuality=full`
   >    > 2. `TransformerConfig.recompute_method=block`
   >    > 3. `TransformerConfig.recompute_num_layers=<int:≤模型中的层数>`

2. 激活卸载到主机内存

   > 1. Megatron-Bridge 支持将激活内存卸载到主机内存，这对于受激活内存限制的训练任务至关重要。这在以下场景中特别有用：(1) FSDP，其中模型状态内存通过分片最小化，但激活内存仍然很高；(2) LoRA，其参数被冻结但激活内存需求很大；(3) 使用长序列进行训练。激活卸载的效率取决于 GPU 与主机之间的互连带宽以及主机内存带宽。从这个角度看，像 GB200 这样的基于 Grace 的系统通过优化这些带宽提升了卸载性能。
   >
   > 2. 应配置以下旋钮以启用卸载，并指定要卸载到主机内存的 Transformer 层数。可卸载的最大层数取决于主机内存容量，当多个 GPU 共享 CPU 时，此容量可能会降低。
   >
   >    > 1. `TransformerConfig.cpu_offloading=True`
   >    > 2. `TransformerConfig.cpu_offloading_weights=False`
   >    > 3. `TransformerConfig.cpu_offloading_num_layers= <int:≤激活卸载层数>`
   >
   > 3. 避免 CPU 内存卸载与网络通信之间资源冲突的环境变量设置
   >
   >    > 1. `NCCL_NET_GDR_LEVEL=PHB # NCCL <=2.25`
   >    > 2. `NCCL_NET_GDR_C2C=1     # NCCL >=2.26`
   >
   > 4. 优化建议
   >
   >    > 1. 考虑到激活数据量与计算操作之间的比率，简单地卸载所有层的激活可能会成为性能瓶颈。优化性能需要调整卸载的层数，同时与重计算进行平衡。

3. 权重内存优化的 BF16 训练

> 1. 在 BF16 训练中，Megatron-Bridge 通过仅存储主权重副本的 BF16 余数用于下一次优化器更新来优化内存使用。这是可能的，因为 BF16 数据可以使用 FP32 位的一个子集来表示，这使得 Megatron-Bridge 能够避免冗余存储用于 BF16 表示的 FP32 部分。当在 Megatron Core 中使用精度感知优化器时，此功能默认启用。
>
>    > 1. `OptimizerConfig.use_precision_aware_optimizer=True`

4. 环境变量设置导致的常见内存使用激增

   > 1. 以下环境变量将 (1) 避免为 NCCL 通信保留缓冲区，以及 (2) 在不使用时禁用 NVLSharp。这两个选项都会降低 GPU 内存使用量。
   >
   >    > 1. `TORCH_NCCL_AVOID_RECORD_STREAMS=1`
   >    > 2. `NCCL_NVLS_ENABLE=0`
   >
   > 2. 虽然默认未启用，但您可以通过设置如下所示的环境变量来进一步减少由分段惩罚引起的内存使用。
   >
   >    > 1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

5. 在 FP8 训练中保持参数为 FP8

   > 1. 在 FP8 训练中，执行优化器步骤后，我们可以将参数保持在 FP8 格式。与将中间权重值保持在 BF16 的基线相比，FP8 参数降低了内存使用量并提高了通信性能。以下开关启用将参数保持在 FP8 的功能。
   >
   >    > 1. `MixedPrecisionConfig.fp8_param_gather=True`

## 算子融合

1. 您可以使用以下配置开关控制特定的融合行为：

   > 1. `TransformerConfig.masked_softmax_fusion=true`
   > 2. `GPTProvider.cross_entropy_loss_fusion=true`
   > 3. `GPTProvider.gradient_accumulation_fusion=true`
   > 4. `TransformerConfig.bias_activation_fusion=true`
   > 5. `TransformerConfig.bias_dropout_fusion=true`
   > 6. `TransformerConfig.apply_rope_fusion=true`

2. Megatron-Bridge 提供不同的 Flash Attention 选项，可以通过模型配置进行选择：

   > 1. 让 Transformer Engine 决定（默认）：`TransformerConfig.attention_backend=AttnBackend.auto`
   > 2. FlashAttention2：`TransformerConfig.attention_backend=AttnBackend.flash`
   > 3. cuDNN 融合注意力：`TransformerConfig.attention_backend=AttnBackend.fused`

(long-sequence-train)=
## 长序列训练

1. 长序列训练的问题

   > 1. 使用长序列长度进行训练可能由于激活的巨大内存成本而导致内存溢出。这个问题可以通过在反向传播中重新计算激活来解决，但这可能会在每个训练步骤中带来高达约 30% 的开销。上下文并行（Context Parallelism）是一个更好的解决方案，它将序列维度在多个 GPU 上进行划分，使得每个 GPU 只计算和保存一个序列块的激活。通过这种方式，可以在不引入任何冗余计算的情况下解决内存溢出问题。

2. 使用 CP 对激活进行分片（开关）

   > 1. `TransformerConfig.context_parallel_size=<int>`
   >
   >    > 1. TP 和 CP 都可以减少激活内存开销。偏向其中任何一种都是不明智的。TP 和 CP 的通信分别与 GEMM 和 Attention 重叠。盲目增大它们的规模可能会使某些通信难以重叠。建议尝试 TP+CP 配置的组合。最优配置应能充分利用所有相关计算并实现最佳重叠，从而达到最佳的端到端性能。
   >
   > 2. `TransformerConfig.cp_comm_type=<str> 或 <list of str>`
   >
   >    > 1. Megatron-Core 提供了 CP 的多种实现变体，并允许您通过配置 "cp_comm_type" 根据具体用例进行选择。配置值可以是 `p2p`、`all_gather`、`a2a` 或 `a2a+p2p`。这些通信类型彼此兼容，因此可以在 Transformer 层之间灵活地交错使用。您只需要提供一个列表，其中每个元素对应一个层。
   >    > 2. `p2p`：以环形拓扑交换 KV 序列块。P2P 通信可以完全重叠。
   >    > 3. `all_gather`：在注意力计算之前插入一个 all-gather 操作以获取完整的 KV 序列。all-gather 操作是暴露的，但如果使用 GQA/MQA，则不应带来大的开销，因为它们只有很少的 KV 头。
   >    > 4. `a2a`：是 DeepSpeed Ulysses 的一种实现。在注意力模块前后添加 A2A 通信，以收集完整序列长度并在 CP 域内进一步分散头。A2A 无法重叠。
   >    > 5. `a2a+p2p`：是 `a2a` 和 `p2p` 之间的折中方案。这对于 CP 规模较大的情况很有用，因为每个序列块太短而无法重叠 P2P 通信。它首先在部分 CP 组中进行 A2A 以收集相对较长的序列块，然后对收集到的块应用 P2P 实现。它也有助于分层 CP 通信，例如 A2A 和 P2P 分别发生在 NVLink 和 IBLink 域中。

> 6. 对于中小型 CP 大小，`p2p` 是推荐的配置，因为通信可以完全重叠；"all_gather" 对于 GQA/MQA 也应该能正常工作。至于在大 CP 大小下对序列长度进行强扩展，短分块长度几乎无法重叠 `p2p` 通信，因此 `a2a+p2p` 应该是首选。在某些情况下，`a2a` 因其简单性而被采用。然而，使用 "a2a" 可能会限制 CP 大小，因为它要求注意力头的数量能被 CP 大小整除。受限的 CP 大小最终会限制可运行的序列长度。

3. 激活重计算（在[减少内存以避免内存溢出和提高训练效率的技术](#reducing-memory-overflow)中）

4. 激活卸载到主机内存（在[减少内存以避免内存溢出和提高训练效率的技术](#reducing-memory-overflow)中）

## 用于高效微调的序列打包

1. 数据集准备

   > 1. 可以将具有可变长度的较短序列的微调数据集打包成更长的序列，直至设定的最大长度，以获得最佳效率。

2. 要使用此功能，必须将微批次大小设置为 1。代替增加微批次大小，可以增加最大序列长度，这将有效地增加每个打包序列中的独立序列数量。

3. 通过以下方式启用：

   > 1. `FinetuningDatasetConfig.packed_sequence_specs.packed_sequence_size=<最大序列长度>`
   > 2. `TrainingConfig.micro_batch_size=1`

4. 性能优势还包括：

   > 1. 微调数据集中序列之间的长度不一致会降低计算效率。当微批次大小大于 1 时，所有序列都必须用空标记填充到该微批次中最长序列的长度。类似地，像 CUDA 图这样的优化要求微批次之间的序列长度一致。打包序列的排列使得每个打包序列的总标记数尽可能接近最大长度，从而使大多数处理的标记都是有用的。
   > 2. 同样，当使用数据并行时，处理不同批次所需时间的差异可能导致所有批次都需要等待最长的批次完成——而使用打包序列可以减少这种差异。

## GPU 核心时钟优化

1. 提高 GPU 核心相对于片外内存系统的时钟比率

   > 1. NVIDIA GPU 支持 CPU 核心时钟加速模式，该模式通过降低片外内存时钟速率来提高核心时钟速率。这对于通常受计算吞吐量限制的大语言模型尤其有益。
   >
   >    > 1. `sudo nvidia-smi boost-slider --vboost 1 <运行命令行>`

## 用于基于分析的性能调优的性能分析选项

1. Nsight 系统性能分析

   > 1. Megatron-Bridge 提供了一个接口来启用 NVIDIA Nsight Systems 性能分析器，该分析器显示所有 CUDA 流的 GPU 执行轨迹。您可以检查通信内核是否与计算内核重叠，并调整资源分配以平衡通信和计算。可以使用 ProfilingConfig 启用 Nsight Systems 性能分析，如下所示。
   > 2. `ProfilingConfig(use_nsys_profiler=True, profile_start_step=<int>, profile_end_step=<int>, profile_ranks=<[0,...]>)`

2. 内存快照

   > 1. Megatron-Bridge 提供了一个接口来提取内存快照，该快照显示内存分配字节数、分配生命周期和函数调用堆栈。可以通过 ProfilingConfig 启用提取内存快照，如下所示。
   > 2. `ProfilingConfig(record_memory_history=True, memory_snapshot_path=</path/to/store/the/output/file, profile_ranks=<[0,...]>)`

## DeepEP：常见问题与解决方案

DeepEP 是一个为混合专家（Mixture-of-Experts, MoE）全对全操作优化的通信库。当使用 DeepEP 进行跨节点的专家并行（Expert Parallelism, EP）时，有几个与网络传输和 GPU-NIC 亲和性相关的常见问题会显著影响性能。

> 注意：DeepEP 针对 NVL8 系统（如 DGX-B200 NVL8 或 DGX-H200 NVL8）进行了最佳优化。对于 GB200 NVL72 机架级系统，其中 72 个 GPU 在同一 NVLINK 域内互连，我们建议使用 [HybridEP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep) 而不是 DeepEP。HybridEP 由 NVIDIA 维护，专门针对 NVL72 机架级系统进行了优化。它也作为 `flex` 令牌分发器下的替代后端集成到 Megatron-core 的[融合全对全模块](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.fused_a2a.html)中。
>
> 了解更多关于 GB200 MoE 训练最佳实践的信息，请访问[此处](https://github.com/NVIDIA/Megatron-LM/blob/dev/docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-reproduce-guide.md)。

### 1. 为什么我的 DeepEP 不工作

1. 什么是 IBGDA 以及为什么它是一个问题

DeepEP 使用 InfiniBand GPU Direct Async（IBGDA）实现最优的跨节点通信性能，该功能在 InfiniBand 和 RoCEv2 模式下均受 ConnectX 网卡支持。然而，IBGDA 并非默认启用——通常需要集群管理员主动配置系统，并在 InfiniBand（或 RoCEv2）网络中启用 GPU Direct RDMA 支持。如果此配置步骤被跳过或在集群环境中不受支持，IBGDA 可能不可用，这将导致 DeepEP 的跨节点 EP 功能无法工作。

1. 网络传输：IBGDA 与 IBRC

   > 1. IBGDA（InfiniBand GPU Direct Async）要求集群管理员启用 GPU Direct RDMA 并配置 InfiniBand 子系统。许多集群默认并未启用 IBGDA。
   > 2. 官方 DeepEP 主分支已移除对 IBRC（InfiniBand Reliable Connection）的支持，该机制此前作为备用方案。使用 IBRC 时，一个 CPU 代理线程将协助处理 EP 通信，与 IBGDA 相比可能存在性能下降，但我们发现这种性能下降并不会掩盖在生产训练中启用 wideEP 所带来的收益。

2. 解决方案：支持自动传输回退的 NVSHMEM 3.5

   > 1. NVSHMEM 3.5 针对各种网络配置下的跨节点通信引入了改进的自动回退支持。它可以根据集群能力自动选择最佳的可用传输机制（IBGDA、IBRC 或其他支持的机制）。
   > 2. 要在 DeepEP 中受益于 NVSHMEM 的自动回退功能：
   >    - 下载 [官方 NVSHMEM 3.5.19-1 版本](https://github.com/NVIDIA/nvshmem/releases/tag/v3.5.19-1)。您也可以选择在容器环境中从源代码编译；本指南后续会提供相关示例。
   >    - 切换到 [集成了原生 NVSHMEM API 的 DeepEP 分支](https://github.com/seth-howell/DeepEP/tree/nvshmem_native_apis)。此分支能够自动使用 NVSHMEM 的回退机制，无需任何手动代码修改。

### 2. GPU-NIC 亲和性与带宽争用

DeepEP 性能不佳的一个常见原因是 GPU 到 NIC（网络接口卡）的亲和性配置不当，导致多个 GPU 在单个 NIC 上争抢带宽。如 [DeepEP PR #466](https://github.com/deepseek-ai/DeepEP/pull/466) 所述，在某些集群中，由于特定的 GPU-NIC 亲和性，如果多个 GPU 使用同一个 NIC，跨节点 EP 性能可能会下降。该 PR 提供了一个解决方案，通过支持环境变量 `DEEP_EP_DEVICE_TO_HCA_MAPPING` 来指定 GPU 到 NIC 的映射关系，从而自动将每个 GPU 绑定到最优的 NIC，以实现最大的 DeepEP 吞吐量。

使用此 PR 的解决方案，我们需要以下环境变量来正确映射 GPU 到 NIC。首先，您需要通过运行 `ibstat` 来找出 NIC 的名称。在我们的示例中，针对一个 RoCEv2 DGX-B200 集群，我们发现了以下信息：
```
> ibstat | grep ^CA
CA 'rocep145s0'
CA 'rocep146s0'
CA 'rocep152s0'
CA 'rocep153s0'
CA 'rocep198s0'
CA 'rocep199s0'
CA 'rocep205s0'
CA 'rocep206s0'
```

使用以下环境变量将 GPU 映射到 NIC。请注意，`0:rocep145s0:1` 的格式为 `<CUDA_device_id>:<NIC_name>:<port>`，以便每个 GPU 仅映射到一个专用的 NIC。
```bash
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export DEEP_EP_DEVICE_TO_HCA_MAPPING="0:rocep145s0:1,1:rocep146s0:1,2:rocep152s0:1,3:rocep153s0:1,4:rocep198s0:1,5:rocep199s0:1,6:rocep205s0:1,7:rocep206s0:1"
```

### 3. 构建 DeepEP

在本节中，我们提供一个参考 Dockerfile，展示如何将 NVSHMEM 3.5 和定制的 DeepEP 构建到您的容器环境中。

请注意，以下示例是为 DGX-B200 NVL8 系统提供的，但类似的方法也适用于 Hopper 架构——只需相应地修改 Dockerfile。例如，您只需要更改 SM90 的编译目标。

关键点：

- NVSHMEM 源代码：https://github.com/NVIDIA/nvshmem/tree/v3.5.19-1
- 我们精选了包含上述所有修复的 DeepEP 分支：https://github.com/zhongbozhu/DeepEP/tree/nvshmem_deepep_gcp
- DGX-B200 的训练容器模板示例：https://github.com/yanring/Megatron-MoE-ModelZoo/blob/main/dockers/B200.Dockerfile

**Dockerfile**
```bash
FROM nvcr.io/nvidia/pytorch:25.11-py3 as base

# 您可能需要的其他依赖
...

# IBGDA 的依赖
RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

# 克隆定制的 DeepEP 版本
WORKDIR /home/dpsk_a2a
RUN git clone https://github.com/zhongbozhu/DeepEP.git ./deepep
RUN cd ./deepep && git checkout nvshmem_deepep_gcp && cd /home/dpsk_a2a

# 克隆 NVSHMEM 3.5 https://github.com/NVIDIA/nvshmem
RUN git clone --branch v3.5.19-1 https://github.com/NVIDIA/nvshmem.git ./deepep-nvshmem
RUN cd ./deepep-nvshmem && git checkout v3.5.19-1 && cd /home/dpsk_a2a

# 从源代码构建 nvshmem
# 您也可以下载预编译的二进制文件，并跳过以下

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        clang \
        llvm-dev \
        libclang-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/dpsk_a2a/deepep-nvshmem
RUN mkdir -p build && mkdir -p install && \
    cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=/home/dpsk_a2a/deepep-nvshmem/install \
    -DCUDA_HOME=/usr/local/cuda \
    -DMPI_HOME=/opt/hpcx/ompi \
    -DMPI_C_COMPILER=/opt/hpcx/ompi/bin/mpicc \
    -DMPI_CXX_COMPILER=/opt/hpcx/ompi/bin/mpicxx \
    -DNVSHMEM_MPI_SUPPORT=OFF \
    -DNVSHMEM_IBRC_SUPPORT=ON \
    -DNVSHMEM_IBGDA_SUPPORT=ON \
    -DNVSHMEM_IBDEVX_SUPPORT=OFF \
    -DNVSHMEM_UCX_SUPPORT=OFF \
    -DNVSHMEM_SHMEM_SUPPORT=OFF \
    -DNVSHMEM_PMIX_SUPPORT=OFF \
    -DNVSHMEM_USE_NCCL=OFF \
    -DNVSHMEM_USE_GDRCOPY=ON \
    -DGDRCOPY_HOME=/usr \
    -DNVSHMEM_USE_MLX5DV=ON \
    -DNVSHMEM_BUILD_TESTS=ON \
    -DNVSHMEM_BUILD_EXAMPLES=ON \
    -DNVSHMEM_BUILD_PYTHON_LIB=OFF \
    -DNVSHMEM_BUILD_BITCODE_LIBRARY=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="100" && \
    cmake --build build -j && \
    cmake --install build

ENV NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install
ENV LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH
ENV PATH=${NVSHMEM_DIR}/bin:$PATH

## 构建 DeepEP
WORKDIR /home/dpsk_a2a/deepep
ENV TORCH_CUDA_ARCH_LIST="10.0"
ENV PIP_NO_BUILD_ISOLATION=1
ENV CPATH=${CUDA_HOME}/include/cccl:$CPATH
RUN pip install --no-build-isolation .

```

DeepEP 提供了 `test_internode.py` 来测试和基准测试跨节点的 EP 通信。在我们的实验中，当使用 4 个 DGX-B200 节点（即 EP32）时，使用 IBRC 实现的跨 EP 吞吐量约为 50 GB/s。我们在下面提供了一个示例 SLURM 脚本，用于使用 DeepEP 运行此类测试。

在同一集群的另一个实验中，在集群管理员启用 IBGDA 的情况下，我们观察到节点间性能提高了大约 10%——大约 55 GB/s。要启用 IBGDA，您需要设置环境变量 `export NVSHMEM_IB_ENABLE_IBGDA=true`；无需更改软件版本或容器，因为使用上面提供的软件，两种模式都将正常工作。

```bash
srun --account=<your_account> -N 4 -p batch --time 30 \
     --ntasks-per-node=1 --gpus-per-node=8 \
     --no-container-mount-home --container-mounts "/lustre:/lustre" \
     --container-image <your_container_path> \
     --mpi=none --export=ALL \
     bash -lc '
set -eo pipefail 

# GPU-NIC 映射的环境变量
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export DEEP_EP_DEVICE_TO_HCA_MAPPING="0:rocep145s0:1,1:rocep146s0:1,2:rocep152s0:1,3:rocep153s0:1,4:rocep198s0:1,5:rocep199s0:1,6:rocep205s0:1,7:rocep206s0:1"


# 1) 展开 SLURM_JOB_NODELIST 并获取第一个主机名
headnode=$(python - <<PY
import os, re
nl = os.environ.get("SLURM_JOB_NODELIST", "") or os.environ.get("SLURM_NODELIST", "")
if not nl:
    print(""); raise SystemExit(0)
m = re.match(r"^([^-\\[]+)-(\\[(.+)\\]|(\\d+))$", nl)
if not m:
    # 没有括号/范围，直接按原样打印
    print(nl); raise SystemExit(0)
prefix = m.group(1)
br_or_num = m.group(3) or m.group(4)
candidates = []
for part in br_or_num.split(","):
    part = part.strip()
    if "-" in part:
        a,b = part.split("-",1)
        # 保留前导零
        width = max(len(a), len(b))
        start, end = int(a), int(b)
        candidates.append(f"{prefix}-{start:0{width}d}")
    else:
        candidates.append(f"{prefix}-{part}")
print(sorted(candidates)[0])
PY
)

if [[ -z "$headnode" ]]; then
  echo "Could not determine master host from SLURM_JOB_NODELIST"; exit 1
fi

# 2) 解析为两个节点都能访问的 IP（回退到主机名）
if command -v getent >/dev/null 2>&1; then
  master_ip=$(getent ahostsv4 "$headnode" | awk "{print \$1; exit}")
else
  master_ip=""
fi
MASTER_ADDR="${master_ip:-$headnode}"

# 3) 导出与 test_internode.py 期望匹配的 rendezvous 环境变量
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=${SLURM_NNODES:-2}   # 节点数
export RANK=${SLURM_NODEID:-0}         # 每个节点的 0..N-1

export OMP_NUM_THREADS=1
python -u /home/dpsk_a2a/deepep/tests/test_internode.py
'

```










## 索引 - 调优旋钮列表

- `CommOverlapConfig.tp_comm_overlap`
- `CommOverlapConfig.tp_comm_overlap_cfg`
- `CUDA_DEVICE_MAX_CONNECTIONS`
- `TrainingConfig.manual_gc_interval`
- `MixedPrecisionConfig.fp8_param`
- `ProfilingConfig`
- `NCCL_NET_GDR_C2C`
- `NCCL_NET_GDR_LEVEL`
- `NCCL_NVLS_ENABLE`
- `NVTE_BWD_LAYERNORM_SM_MARGIN=<#SM for DP collectives`
- `TransformerConfig.attention_backend`
- `AttnBackend`
- `NVTE_FWD_LAYERNORM_SM_MARGIN=<#SM for DP collectives`
- `PYTORCH_CUDA_ALLOC_CONF`
- `TrainingConfig.micro_batch_size`
- `FinetuningDatasetConfig.packed_sequence_specs.packed_sequence_size`
- `TransformerConfig.apply_rope_fusion`
- `TransformerConfig.bias_activation_fusion`
- `TransformerConfig.bias_dropout_fusion`
- `TransformerConfig.cp_comm_type`
- `TransformerConfig.cpu_offloading`

- `TransformerConfig.cpu_offloading_num_layers`
- `TransformerConfig.cpu_offloading_weights`
- `GPTProvider.cross_entropy_loss_fusion`
- `TransformerConfig.cuda_graph_impl` / `cuda_graph_scope` (参见 [CUDA 图](training/cuda-graphs.md))
- `MixedPrecisionConfig.fp8_param_gather`
- `GPTProvider.gradient_accumulation_fusion`
- `TransformerConfig.masked_softmax_fusion`
- `TransformerConfig.recompute_granuality`
- `TransformerConfig.recompute_method`
- `TransformerConfig.recompute_num_layers`
- `OptimizerConfig.use_precision_aware_optimizer`
- `GPTProvider.account_for_embedding_in_pipeline_split`
- `GPTProvider.account_for_loss_in_pipeline_split`
- `TransformerConfig.context_parallel_size`
- `DistributedDataParallelConfig.align_param_gather`
- `DistributedDataParallelConfig.bucket_size`
- `DistributedDataParallelConfig.bucket_size`
- `DistributedDataParallelConfig.data_parallel_sharding_strategy`
- `DistributedDataParallelConfig.grad_reduce_in_fp32`
- `DistributedDataParallelConfig.num_distributed_optimizer_instances`
- `DistributedDataParallelConfig.overlap_grad_reduce`
- `DistributedDataParallelConfig.overlap_param_gather`
- `T5ModelProvider.encoder_pipeline_model_parallel_size`
- `T5ModelProvider.encoder_tensor_model_parallel_size`
- `TransformerConfig.expert_model_parallel_size=<int>`
- `TransformerConfig.expert_tensor_parallel_size=<int>`
- `TransformerConfig.moe_grouped_gemm`
- `DistributedInitConfig.use_torch_fsdp2`
- `TransformerConfig.pipeline_model_parallel_size`
- `TransformerConfig.tensor_model_parallel_size`
- `TransformerConfig.virtual_pipeline_model_parallel_size`
- `OptimizerConfig.use_distributed_optimizer`
- `TORCH_NCCL_AVOID_RECORD_STREAMS`
- `TPOverlapCfg.cga_size`
- `TPOverlapCfg.fp8_buf`
- `TPOverlapCfg.num_sm`
- `TPOverlapCfg.num_split`
<!-- - `garbageCollectionCallback.gc_interval_val` -->
<!-- - `NsysPlugin` -->