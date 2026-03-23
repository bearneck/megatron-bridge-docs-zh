# 为 Megatron Bridge 贡献新模型

本指南解释了如何为 Megatron Bridge 添加对新的 🤗 Hugging Face 模型（或系列）的支持，以便在 HF ↔ Megatron-Core 格式之间进行转换并参与训练流程。

请使用此清单式流程：搭建脚手架 → 提供者映射 → 参数映射 → 测试 → 验证。

## 前提条件

- 熟悉 Megatron Bridge 仓库结构。
- 一个安装了 Megatron Bridge 的 Python 3.10+ 工作环境（参见[安装说明](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/README.md#installation)），推荐使用容器。
- 熟悉 Megatron-Core GPT 风格的模块和 🤗 Transformers 配置对象。
- 拥有一个用于本地测试的小型 HF 检查点。
- 首先阅读：
  - [Bridge 用户指南](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-guide.md)
  - [技术细节](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)
  - [模型桥接器概述](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/README.md)

## 1) 确定集成策略

大多数 GPT 风格的模型（例如 Qwen 和 Llama 系列）可以通过映射其配置来重用 Megatron-Core GPT 模型。如果模型需要自定义构建块（例如，注意力变体、RoPE 变体或 VLM 模块），请添加一个轻量级的特化实现，类似于 🤗 HuggingFace 实现 `modeling_xxx.py` 的方式。

- **标准 GPT 风格模型**：实现一个 `Provider` 和一个 `Bridge`。例如，参见 [Llama 提供者](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_provider.py) 和 [Llama 桥接器](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_bridge.py)。
- **具有自定义组件的模型**：如果你的模型具有自定义操作或块（例如，独特的注意力机制），请在相同目录中添加一个最小的建模模块，并从 `Provider` 中引用它（示例即将提供）。

## 2) 搭建模型文件夹脚手架

在 `src/megatron/bridge/models/<your_model>/` 下创建一个文件夹并添加：

- `<your_model>_provider.py`：构建一个与 `TransformerConfig` 兼容的提供者（或现有提供者的子类），并暴露 `.provide_distributed_model()` 方法。例如：[Llama 提供者](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_provider.py)、[Qwen 提供者](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen_provider.py) 或 [Qwen2 提供者](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen2_provider.py)。
- `<your_model>_bridge.py`：特定于架构的桥接器，用于映射 HF 配置 → 提供者并定义参数映射。例如：[Llama 桥接器](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_bridge.py)、[Qwen3 桥接器](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen3_bridge.py) 或 [Qwen2 桥接器](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen2_bridge.py)。
- 可选：包含任何模型特殊说明的 `README.md`。例如：[Llama README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/README.md)。

## 3) 实现提供者

你的提供者将 Hugging Face 配置映射到 Megatron-Core 转换器配置字段，并惰性地构建分布式模型。从通用的 GPT 提供者 (`src/megatron/bridge/models/gpt_provider.py`) 开始，并特化必要的字段和标志：

- 并行性：`tensor_model_parallel_size`、`pipeline_model_parallel_size`，可选的 VPP/EP 设置。
- 数值：`fp16`、`bf16`、`params_dtype`、激活重计算。
- 架构特性：RoPE 基数/缩放、QK 层归一化、绑定嵌入、KV 分组、最大序列长度等。
- 可选的自定义模块：如果需要，使用层规范指向自定义注意力/MLP 实现。

暴露：
```python
provider = YourModelProvider(...)
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 建议的 Cursor 提示（提供者）[实验性]
```text
你正在 Megatron Bridge 仓库中工作。创建 `src/megatron/bridge/models/<your_model>/<your_model>_provider.py`。

目标：实现 `YourModelProvider`，将 HF 配置映射到 Megatron-Core 转换器配置，并暴露 `.provide_distributed_model()`。

要求：
- 从 `src/megatron/bridge/models/gpt_provider.py` 开始并适配。
- 映射核心字段：层数、隐藏层大小、FFN 大小、注意力头数、KV 分组数、最大序列长度、RoPE 基数/缩放、绑定嵌入。
- 配置并行性：`tensor_model_parallel_size`、`pipeline_model_parallel_size`（VPP/EP 可选）。
- 配置数值：`fp16`/`bf16`、`params_dtype`、激活重计算。
- 如果需要，通过层规范指向自定义注意力/MLP。
```

- 在 `.provide_distributed_model()` 中返回一个延迟构建的分布式模型。

参考提供者：
- Llama：`src/megatron/bridge/models/llama/llama_provider.py`
- Qwen：`src/megatron/bridge/models/qwen/qwen_provider.py`

验收标准：
- 无 linter 错误。
- 最小化冒烟测试能通过桥接器构建模型并加载一个微小的 HuggingFace 检查点。

## 4) 定义配置和参数映射

使用 `provider_bridge` 方法将 Hugging Face 配置映射到 Megatron 模型提供者，并使用 `MegatronMappingRegistry` 将 Megatron 参数名映射到 Hugging Face 参数名。从基本部分开始（嵌入层、最终归一化层、QKV、MLP），然后添加额外部分（偏置、旋转嵌入、专家和视觉块）。

- `provider_bridge`：参见 [model_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/model_bridge.py)
- `MegatronMappingRegistry`：参见 [mapping_registry.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/mapping_registry.py)
- 映射实现：参见 [param_mapping.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/param_mapping.py)
- 背景知识：参见 [桥接器技术细节](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)

注册示例框架：

```python
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import LlamaForCausalLM  # 替换为你的 HF 类
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping, GatedMLPMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from .<your_model>_provider import YourModelProvider

@MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
class YourModelBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> YourModelProvider:
        cfg = hf_pretrained.config
        return YourModelProvider(
            num_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            ffn_hidden_size=getattr(cfg, "intermediate_size", 4 * cfg.hidden_size),
            num_attention_heads=cfg.num_attention_heads,
            num_query_groups=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            # 如果需要，通过辅助函数设置 dtype 标志
            params_dtype=self.dtype_from_hf(cfg),
            ...
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(
            AutoMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            AutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            AutoMapping(
                megatron_param="decoder.final_layernorm.weight",
                hf_param="model.norm.weight",
            ),
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            ...
        )
```

注意：
- 对每层模式使用 `*` 通配符；`megatron_param` 和 HF 模式之间的通配符数量必须匹配。
- `*` 通常捕获层索引；`**` 可以跨点匹配。例如，要同时映射 `.weight` 和 `.bias`：
  ```python
  AutoMapping(
      megatron_param="output_layer.**",
      hf_param="lm_head.**",
  ),
  ```
- 在某些情况下，同一个模块可能根据你使用的是 Transformer Engine 后端还是 PyTorch 后端而具有不同的 Megatron 参数名。在这种情况下，列出两个映射，例如 `[AutoMapping(megatron_param="te_backend_name", hf_param="hf_name"), AutoMapping(megatron_param="pytorch_backend_name", hf_param="hf_name")]`。多个 Megatron 参数可以映射到同一个 Hugging Face 参数，因为在转换期间，注册表只查询当前模型的模块名。
- 当 Megatron 层类型自动隐含了张量并行（TP）切分时，优先使用 `AutoMapping`。
- 对融合的 QKV 使用 `QKVMapping`，对门控/上投影拼接使用 `GatedMLPMapping`。

### 建议的 Cursor 提示（桥接器）[实验性]
```text
你正在 Megatron Bridge 仓库中工作。创建 `src/megatron/bridge/models/<your_model>/<your_model>_bridge.py`。
```

目标：实现一个桥接类，使用 `MegatronModelBridge` 将 HF 模型类连接到 Megatron 模型。

任务：
- 添加 `@MegatronModelBridge.register_bridge(source=<HFClass>, target=GPTModel)`。
- 实现 `provider_bridge(self, hf_pretrained)` 以读取 `hf_pretrained.config` 并返回 `YourModelProvider(...)`，其中包含映射的字段（层数、隐藏层大小、FFN、注意力头数、分组数、RoPE、通过 `self.dtype_from_hf(cfg)` 获取的数据类型）。
- 实现 `mapping_registry(self)` 返回 `MegatronMappingRegistry(...)`，包含：
  - 用于嵌入层、最终归一化层、输出层、一对一映射权重的 `AutoMapping`。
  - 如果适用，用于融合 QKV 的 `QKVMapping`。
  - 如果适用，用于门控/向上投影的 `GatedMLPMapping`。
- 在 Megatron 和 HF 模式之间一致地使用 `*` 通配符。
- 将模型组织添加到 `megatron.bridge.models.hf_pretrained.utils` 的 `SAFE_REPOS` 列表中。

参考：
- `src/megatron/bridge/models/conversion/model_bridge.py`
- `src/megatron/bridge/models/conversion/mapping_registry.py`
- `src/megatron/bridge/models/conversion/param_mapping.py`
- `src/megatron/bridge/models/qwen/qwen2_bridge.py`

验收标准：
- HF → Megatron 加载完成，无缺失参数（针对一个小型模型）。
- Megatron → HF 导出为多个键返回预期形状/数据类型的张量。

## 5) 最小化冒烟测试（本地）

一个最小化的双向端到端检查：
```python
from megatron.bridge import AutoBridge

# HF → Megatron
bridge = AutoBridge.from_hf_pretrained("<org>/<model-id>", trust_remote_code=True)
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
bridge.load_hf_weights(model)

# Megatron → HF（流式传输少量张量）
for i, (name, tensor) in enumerate(bridge.export_hf_weights(model, cpu=True)):
    print(name, tuple(tensor.shape))
    if i > 10:
        break
```

## 6) 使用示例进行验证

使用 `examples/conversion/` 中的示例来验证双向转换以及更复杂的模型并行设置下的基本生成。

- 直接使用桥接器从 HF 生成
- 来回转换检查点
- 多 GPU HF 加载到 Megatron

```sh
python examples/conversion/hf_to_megatron_generate_text.py --hf_model_path <org>/<model-id> --prompt "Hello"
python examples/conversion/convert_checkpoints.py import --hf-model <org>/<model-id> --megatron-path ./checkpoints/<model-dir>
```

## 7) 添加测试

在 `tests/functional_tests/models/<your_model>/` 和 `tests/unit_tests/models/` 下添加或扩展测试：

测试按模型特定的子目录组织，这些子目录镜像了 `src/megatron/bridge/models/` 中的源结构。

- 转换覆盖：
  - HF → Megatron 加载成功，无缺失参数
  - Megatron → HF 导出往返形状和数据类型一致
- 提供者覆盖：
  - 提供者字段与 HF 配置对齐（注意力头数、分组数、FFN 大小、RoPE）
- 可选的数值检查：
  - 在少量 token 上比较 HF 与 Megatron 输出的前向传播一致性

参考示例：
- `tests/functional_tests/models/qwen/test_qwen3_provider.py`
- `tests/functional_tests/models/qwen/test_qwen3_conversion.py`

本地运行快速测试：
```sh
uv run pytest -q tests/functional_tests/models/<your_model>/test_<your_model>_provider.py -k your_model | cat
uv run pytest -q tests/functional_tests/models/<your_model>/test_<your_model>_conversion.py -k your_model | cat
```

完整测试套件（较慢）：
```sh
uv run pytest -q tests | cat
```

### 7.1) CI 缓存中未找到模型

Megatron Bridge 功能测试在 `HF_HUB_OFFLINE=1` 环境下运行。这意味着，如果贡献包含一个新的桥接器以及针对一个未缓存在我们 CI 的 `$HF_HOME` 目录中的 HuggingFace 模型的测试，将会失败并出现类似以下错误：

```
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled.
```

如果在 CI 中遇到此类错误，请请求仓库维护者为你的 PR 中添加支持的模型启动“缓存 HuggingFace 模型”工作流。

### 建议的 Cursor 提示（测试）[实验性]
```text
你正在 Megatron Bridge 仓库中工作。为新的模型 `<your_model>` 添加测试。

创建一个子目录 `tests/functional_tests/models/<your_model>/`，包含一个 `__init__.py` 文件和两个测试模块：
1) `test_<your_model>_provider.py`
   - 构建一个微小的 HF 模型/配置（或者使用 `<org>/<tiny-model-id>`，如果可用）。
   - 使用桥接器派生一个提供者，并使用 TP=PP=1 构建模型。
   - 断言提供者字段与 HF 配置匹配（注意力头数、分组数、隐藏层大小、FFN、RoPE、词汇表大小、最大位置）。

2) `test_<your_model>_conversion.py`
   - HF → Megatron：通过桥接器将 HF 权重加载到 Megatron 模型中；断言没有缺失/多余的参数。
   - Megatron → HF：导出一个张量子集；断言与 HF 的形状/数据类型一致。
   - 可选地在 CPU 上运行一个简短的生成过程，并在容差范围内数值比较 logits。
```

以 `tests/functional_tests/models/qwen/test_qwen3_provider.py` 和 `test_qwen3_conversion.py` 为模板。

提供 `-k your_model` 选择器，如果外部权重不可用，则使用 `pytest.skip` 来保护长时间运行的测试。

## 8) 故障排除

- **形状不匹配**：仔细检查张量并行（TP）/管道并行（PP）切分和模型配置。
- **权重缺失**：确保每个 Megatron 参数都有映射；打印未解析的名称。
- **数据类型问题**：必要时在映射内部将 HuggingFace 权重转换为目标数据类型。
- **专家并行（EP）/混合专家（MoE）层**：请参阅 `param_mapping.py` 中针对 EP 的 gather/scatter 辅助函数。

启用详细日志：
```python
import logging
logging.getLogger("megatron.bridge").setLevel(logging.DEBUG)
```

## 9) PR 检查清单

- 在 PR 描述中提供详细信息
- 提供程序映射所有必需的配置字段
- 所有参数都被映射覆盖
- 从 HuggingFace 转换到 Megatron 后的生成结果与 Megatron 匹配，包括多 GPU 运行
- 已添加单元/功能测试并通过
- 如果适用，请将您的模型添加到仓库 `README.md` 的“支持的模型”表中

## 10) 有用的链接

- 用户指南：[docs/bridge-guide.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-guide.md)
- 技术深度解析：[docs/bridge-tech-details.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)
- 代码示例：[examples/conversion/](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/conversion)
- 提供程序和桥接器：[src/megatron/bridge/models/](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models)
- GitHub 源码树：[Megatron Bridge src/megatron/bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge)