# 模型优化

本目录包含使用 NVIDIA ModelOpt 通过 Megatron Bridge 优化模型的综合文档。学习如何应用各种优化技术，在保持模型质量的同时提高推理效率。

## 概述

NVIDIA ModelOpt 提供了一套用于提升推理性能的模型优化技术：

- **量化（Quantization）** - 将模型从高精度（FP32/BF16）转换为低精度格式（FP8, INT8, INT4），以实现高效部署
- **蒸馏（Distillation）** - 将知识从预训练的教师模型迁移到更小、更快的学生模型
- **剪枝（Pruning）** - 通过移除层（深度）或减少维度（宽度）（例如注意力头和隐藏层大小）来减小模型尺寸

## 快速导航

### 我想要

**🔧 量化一个预训练模型**
→ 查看[训练后量化部分](quantization.md#post-training-quantization-ptq)，了解完整的 PTQ 工作流程（量化、恢复和生成、导出）

**🏋️ 使用量化进行训练**
→ 查看[量化感知训练部分](quantization.md#quantization-aware-training-qat)，了解 QAT 工作流程

## 参考

- [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)