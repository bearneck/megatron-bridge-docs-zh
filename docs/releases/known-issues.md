# 已知问题

本文档列出了当前版本中的已知问题和限制。

## 26.02

- **仅限 AWS EKS**：由于 AWS-OFI-NCCL v1.17.0 存在内存泄漏问题，长时间运行的作业会随着时间推移出现性能下降。可以通过升级到 [v1.17.3](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.3) 来缓解此问题。
- 在 r0.3.0 版本中，Qwen 3 VL 模型尚不支持结合序列打包（sequence packing）的上下文并行（Context Parallelism）。如需在 Qwen 3 VL 上使用此功能，请使用主分支（main branch）。
- 当前的 NeMo 框架 26.02 容器（nvcr.io/nvidia/nemo:26.02）不支持 DeepEP，这导致在 H100 机器上，与 NeMo 框架 25.09 容器（nvcr.io/nvidia/nemo:25.09）相比，DSv3 性能有所下降。为了获得最佳的 H100 性能，我们建议使用 NeMo 框架 25.09 容器。

## 25.11

- 在 H100 上使用 DeepEP 运行 Deepseek V3 时存在问题，会因 `RuntimeError: DeepEP error: timeout (dispatch CPU)` 而失败。
- 对于所有混合模型（Hybrid models），例如 Nemotron-H 56B，`MODEL_TFLOP/s/GPU` 在标准输出中打印为 0。

## 25.09

- **以子通道 FP8（subchannel FP8）精度预训练 DeepSeek 无法正常工作。** 使用当前缩放 FP8（scaling FP8）预训练 DeepSeek 是一种变通方法，但 MTP 损失（MTP loss）无法收敛。