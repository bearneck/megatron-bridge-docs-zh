# 剪枝

剪枝通过移除冗余参数（例如，缩小隐藏维度或层数）来减小模型大小，同时保持准确性。在 Megatron Bridge 中，剪枝功能由 [NVIDIA 模型优化器 (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer) 提供，使用 Minitron 算法对从 HuggingFace 加载的 GPT 和基于 Mamba 的模型进行剪枝。

## 前提条件

运行剪枝示例需要 Megatron-Bridge 和 Model-Optimizer 依赖项。我们建议使用 NeMo 容器（例如 `nvcr.io/nvidia/nemo:26.02`）。要使用最新的 ModelOpt 脚本，请将你的 Model-Optimizer 仓库挂载到容器中。

```bash
export MODELOPT_DIR=${PWD}/Model-Optimizer # 或者设置为你的本地 Model-Optimizer 仓库路径（如果你已经克隆了它）
if [ ! -d "${MODELOPT_DIR}" ]; then
  git clone https://github.com/NVIDIA/Model-Optimizer.git ${MODELOPT_DIR}
fi

export DOCKER_IMAGE=nvcr.io/nvidia/nemo:26.02
docker run \
  --gpus all \
  --shm-size=20g \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v ${MODELOPT_DIR}:/opt/Model-Optimizer \
  -v ${MODELOPT_DIR}/modelopt:/opt/venv/lib/python3.12/site-packages/modelopt \
  -w /opt/Model-Optimizer/examples/megatron_bridge \
  ${DOCKER_IMAGE} bash
```

进入容器后，你需要使用你的 HuggingFace 令牌登录以下载受限制的数据集/模型。
请注意，剪枝的默认数据集是 [`nemotron-post-training-dataset-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)，该数据集是受限制的。

```bash
huggingface-cli login --token <你的令牌>
```

## 使用方法

### 剪枝至目标参数量（使用神经架构搜索）

示例：在 2 个 GPU 上（管道并行度 = 2）将 Qwen3-8B 剪枝至 6B，跳过 `num_attention_heads` 的剪枝。默认值：使用 [nemotron-post-training-dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) 中的 1024 个样本进行校准，每个可剪枝超参数（`hidden_size`、`ffn_hidden_size`、...）的深度（`num_layers`）最多减少 20%，宽度最多减少 40%，评估前 10 个候选模型在 MMLU（5% 采样数据）上的表现以选择最佳模型。

```bash
torchrun --nproc_per_node 2 prune_minitron.py \
    --pp_size 2 \
    --hf_model_name_or_path Qwen/Qwen3-8B \
    --prune_target_params 6e9 \
    --hparams_to_skip num_attention_heads \
    --output_hf_path /tmp/Qwen3-8B-Pruned-6B
```

### 剪枝至特定架构（使用手动配置）

示例：将 Qwen3-8B 剪枝至固定架构。默认值：使用 [nemotron-post-training-dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) 中的 1024 个样本进行校准。

```bash
torchrun --nproc_per_node 2 prune_minitron.py \
    --pp_size 2 \
    --hf_model_name_or_path Qwen/Qwen3-8B \
    --prune_export_config '{"hidden_size": 3584, "ffn_hidden_size": 9216}' \
    --output_hf_path /tmp/Qwen3-8B-Pruned-6B-manual
```

要查看高级配置的完整选项列表，请运行：

```bash
torchrun --nproc_per_node 1 prune_minitron.py --help
```

### 非均匀管道并行

如果层数不能被 GPU 数量（管道并行大小）整除，请设置 `--num_layers_in_first_pipeline_stage` 和 `--num_layers_in_last_pipeline_stage`。例如，Qwen3-8B 有 36 层，在 8 个 GPU 上运行：将两者都设置为 3，得到每个 GPU 的层数分布为 3-5-5-5-5-5-5-3。

## 更多信息

更多详细信息，请参阅 [ModelOpt 剪枝 README](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge#readme)。

## 后续步骤：知识蒸馏

需要知识蒸馏来恢复剪枝模型的性能。更多详细信息，请参阅 [知识蒸馏](distillation.md) 指南。