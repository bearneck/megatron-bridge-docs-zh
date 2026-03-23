# 知识蒸馏

Megatron Bridge 为知识蒸馏（Knowledge Distillation，KD）训练提供了一个简化的设置，使其易于启用并集成到您的工作流程中。本节将解释如何有效使用此功能。

知识蒸馏是一种技术，其中预训练模型（“教师”）将其学到的知识转移给第二个模型（“学生”），后者通常更小、更快。通过模仿教师模型的行为，这个过程帮助学生模型更高效地学习。与传统训练相比，知识蒸馏具有两个关键优势：更快的收敛速度和更高的最终准确率。

在 Megatron Bridge 中，知识蒸馏功能由 NVIDIA Model Optimizer（ModelOpt）启用——这是一个用于优化深度学习模型以在 GPU 上进行推理的库。

## 知识蒸馏过程

知识蒸馏过程包括以下步骤：

1.  **加载检查点**：加载学生模型和教师模型的检查点。
2.  **替换损失函数**：用输出 logits 之间的 KL 散度（以及可能存在的中间模型状态对之间的额外损失）替换标准损失函数。
3.  **训练模型**：在两个模型上运行前向传播，但仅对学生模型执行反向传播。
4.  **保存检查点**：仅保存学生模型的检查点，使其可以像以前一样在后续使用。

## 限制

*   目前仅支持基于 GPT 的检查点。
*   学生模型和教师模型必须支持相同的并行策略。
*   如果启用了管道并行（Pipeline Parallelism），则基于中间状态的知识蒸馏损失仅支持在最终的管道阶段计算。

## 配置

### 知识蒸馏配置

您可以通过 `ModelOptDistillConfig` 类或 YAML 文件来配置知识蒸馏过程。配置包括：

*   `logit_layers`：学生模型和教师模型 logit 层的层名称。这些名称对应于 Megatron Core 模型的 PyTorch 子模块属性。（对于基于 GPT 的模型，这是 `"output_layer"`）。默认值：`["output_layer", "output_layer"]`
*   `intermediate_layer_pairs`：中间层名称对的列表。默认情况下，这些对之间会计算余弦相似度损失。如果启用了张量并行（Tensor Parallelism），这些层必须具有序列并行输出（例如 LayerNorm），因为余弦损失不能有分割的隐藏维度。默认值：`[["decoder.final_layernorm", "decoder.final_layernorm"]]`
*   `skip_lm_loss`：是否跳过默认的语言建模（LM）损失。如果为 `false`，它将被添加到蒸馏损失中。（注意这会消耗更多内存）。默认值：`true`
*   `kd_loss_scale`：蒸馏损失的相对缩放因子。累积的 logits 和中间损失将被缩放到 LM 损失幅度的 `kd_loss_scale` 倍。如果 `skip_lm_loss` 为 `true` 则不使用。默认值：`1.0`
*   `logit_kl_temperature`：用于 KL 散度损失计算的温度变量。默认值：`1.0`

YAML 配置示例：

```yaml
logit_layers: ["output_layer", "output_layer"]
intermediate_layer_pairs:
  - ["decoder.final_layernorm", "decoder.final_layernorm"]
logit_kl_temperature: 2.0
```

## 使用方法

### 使用默认配置的基本用法

运行知识蒸馏最简单的方法是使用或修改提供的配方脚本之一。以下是将 Llama3.2-3B 蒸馏到 Llama3.2-1B 的示例：

```bash
torchrun --nproc_per_node=1 examples/distillation/llama/distill_llama32_3b-1b.py
```

### 使用自定义 YAML 配置文件

您可以提供一个自定义的 YAML 配置文件来覆盖默认设置：

```bash
torchrun --nproc_per_node=1 examples/distillation/llama/distill_llama32_3b-1b.py \
    --config-file my_custom_config.yaml
```

### 使用命令行覆盖

Megatron Bridge 支持 Hydra 风格的命令行覆盖，以实现灵活的配置：

```bash
torchrun --nproc_per_node=2 examples/distillation/llama/distill_llama32_3b-1b.py \
    model.tensor_model_parallel_size=2 \
    model.teacher.tensor_model_parallel_size=2
```

### 结合 YAML 和命令行覆盖

命令行覆盖的优先级高于 YAML 配置：

```bash
torchrun --nproc_per_node=2 examples/distillation/llama/distill_llama32_3b-1b.py \
    --config-file conf/my_config.yaml \
    train.global_batch_size=512
```

## 模型支持

目前，知识蒸馏支持基于 GPT 和 Mamba 的模型。

要为模型启用知识蒸馏：

1.  将 `teacher` 属性设置为教师模型配置。
2.  使用所需的蒸馏设置配置 `kd_config`（否则使用默认值）。
3.  使用 `convert_to_distillation_provider()` 转换您现有的模型提供程序。

## 检查点

在蒸馏训练期间：

*   仅保存**学生模型**的检查点。
*   教师模型保持冻结状态且不会被修改。
*   检查点可以像任何标准检查点一样用于推理或进一步训练。

## 最佳实践

1.  **匹配并行策略**：确保学生模型和教师模型使用兼容的并行配置。

2. **监控损失**：同时跟踪蒸馏损失和（如果启用）语言建模损失
3. **批大小**：使用更大的批大小以获得蒸馏过程中更好的稳定性
4. **学习率**：从比预训练更小的学习率开始
5. **数据质量**：使用高质量、多样化的训练数据以获得最佳蒸馏结果

## 故障排除

### 内存不足错误

* 减小 `train.micro_batch_size`
* 增加并行度大小
* 设置 `model.kd_config.skip_lm_loss = True` 以节省内存

## 参考资料

有关底层实现的更多信息，请参阅：
* [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)