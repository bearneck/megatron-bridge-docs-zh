# 混合/分层上下文并行

本文档涵盖分层上下文并行（hierarchical context parallelism）在 Megatron Bridge 中的稳定含义，特别是 `a2a+p2p` 传输路径和 `hierarchical_context_parallel_sizes` 参数。

关于操作设置、代码锚点和验证命令，请参阅 [skills/perf-techniques/hybrid-context-parallel/SKILL.md](../skills/perf-techniques/hybrid-context-parallel/SKILL.md)。

## 概念解析

上下文并行（Context Parallelism, CP）将输入序列拆分到多个 GPU 上，使得每个计算单元（rank）处理一个序列块。在注意力计算期间，GPU 之间必须通信键值（KV）数据。目前有几种 CP 通信后端：

| `cp_comm_type` | 机制 | 异步/重叠 | 约束条件 |
|---|---|---|---|
| `"p2p"` | KV 块的环形交换 | 是 | 无 |
| `"all_gather"` | 注意力计算前全收集完整 KV | 否 | 无 |
| `"a2a"` | 全对全：分散注意力头，收集完整序列（Ulysses 风格） | 不适用 | **CP <= num_kv_heads** |
| `"a2a+p2p"` | 分层：组内使用 a2a，组间使用 p2p | 部分（p2p 部分） | 需要 `hierarchical_context_parallel_sizes` |

**分层上下文并行（HCP，即 `a2a+p2p`）** 的存在是为了突破 KV 头数的限制，实现更大规模的 CP。它结合了节点内链路上的 a2a（快速，头并行）和节点间链路上的 p2p（异步，序列并行）。

需要将其与上游的布尔参数 `hybrid_context_parallel` 区分开来，后者是用于平衡打包或变长工作负载的不同功能。这两个概念不应被视为可互换的。

### 为什么 a2a 受限于 KV 头数

a2a 会转置并行维度：每个计算单元用自己的序列块交换一部分注意力头。在全对全操作之后，每个计算单元都拥有**完整的序列**，但只拥有 `heads / CP` 个头。这意味着：

- `heads / CP` 必须是一个正整数。
- 瓶颈在于 KV 头数（而非 Q 头数），因为在分组查询注意力（GQA）中，KV 头是不可分割的单位。
- 如果模型有 8 个 KV 头，纯 a2a 最多支持 CP=8。

HCP 通过仅在足够小的子组内应用 a2a（该子组大小需在 KV 头数限制内）来打破这个限制。

## 使用时机

**当以下条件全部满足时，使用 HCP：**

1.  您需要的 CP 规模大于 `num_kv_heads / TP`（纯 a2a 无法满足）。
2.  您不能（或不想）通过增加 TP 来减小 CP。
3.  您的集群具有明显的带宽层次结构（例如，节点内 NVLink >> 节点间 IB）。

**在以下情况优先使用纯 `a2a`：**

-   您可以调整 TP，使得 `CP <= num_kv_heads / TP`。这更简单，避免了 p2p 的开销，并且通常能提供相同的吞吐量，同时内存余量更大。

**在以下情况优先使用纯 `p2p`：**

-   您的 KV 头数非常少，或者希望获得最大的 CP 灵活性。
-   您的工作负载可以将 p2p 延迟隐藏在计算之后（长序列有助于此）。

### 决策示例

模型：8 个 KV 头。集群：4 节点 x 8 GPU。目标：训练 128K 序列。

| 选项 | TP | CP | `cp_comm_type` | 说明 |
|---|---|---|---|---|
| A | 1 | 16 | 使用 `[8,2]` 的 `a2a+p2p` | 节点内 a2a（8 GPU），跨 2 个节点组 p2p |
| B | 2 | 4 | `a2a` | CP=4 <= 8 KV 头。更简单。通常吞吐量相同。 |
| C | 1 | 16 | `p2p` | 可行，但无法获得节点内 a2a 的带宽优势 |

在实践中，**通常首选选项 B** —— 基准测试显示其与选项 A 吞吐量相同，且内存余量更大。

应将其视为高级功能，而非默认推荐。

## Megatron Bridge 的稳定限制

最重要的 Bridge 特定限制是，分层上下文并行目前仅在 MPU 初始化路径上受支持。

在实践中，这意味着：

-   `dist.use_decentralized_pg=False` 是受支持的 Bridge 路径
-   不应假设去中心化进程组（decentralized process-group）路径能实现 HCP 分组

## 稳定约束

持久的约束条件是：

-   `hierarchical_context_parallel_sizes` 必须与 `context_parallel_size` 在乘积上匹配
-   通常的 CP 序列长度整除规则仍然适用
-   Transformer Engine 版本对 `a2a+p2p` 的支持很重要

## 推荐等级

仅在您确实需要该传输路径，并准备好验证执行路径细节时，才在 Bridge 中使用分层上下文并行。它目前还不是那种可以宣称在所有 Bridge 初始化模式下都普遍安全的功能。

## 相关文档

-   [docs/performance-guide.md](../performance-guide.md)
-   [docs/training/communication-overlap.md](communication-overlap.md)
-   [skills/perf-techniques/hybrid-context-parallel/SKILL.md](../skills/perf-techniques/hybrid-context-parallel/SKILL.md)
-   [skills/perf-techniques/hybrid-context-parallel/card.yaml](../skills/perf-techniques/hybrid-context-parallel/card.yaml)