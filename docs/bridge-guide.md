# 开始使用 🤗 Hugging Face 转换

Megatron Bridge 提供了 🤗 Hugging Face Transformers 与 Megatron 模型定义之间的无缝双向转换。本指南涵盖了加载模型、检查兼容性以及在格式之间转换的主要 API。

## 设计与目标

- **单一高级入口点**：`AutoBridge` 检测 HF 模型架构并分派到正确的桥接器。
- **双向转换**：导入 HF → Megatron 用于训练；导出 Megatron → HF 用于部署。
- **并行性感知**：在转换过程中处理 TP/PP/VPP/CP/EP/ETP 分布。
- **流式处理和内存效率**：使用 safetensors 进行逐参数流式处理。
- **提供者模式**：在通过 `to_megatron_provider()` 实例化之前，配置与 Megatron-Core `TransformerConfig` 兼容的属性。
- **便捷工作流**：`import_ckpt` 和 `export_ckpt` 提供一键式 HF↔Megatron 检查点流程。

有关安装、支持的模型和项目亮点，请参阅仓库的 `README.md`。

## 将 🤗 Hugging Face 模型加载到 Megatron 中

加载 🤗 Hugging Face 模型最简单的方法是使用 `AutoBridge.from_hf_pretrained()`，它会自动检测模型架构并选择适当的桥接器进行转换。然后，您可以使用 `AutoBridge.to_megatron_model()` 从 🤗 Hugging Face 配置初始化 Megatron 模型，并同时加载 🤗 Hugging Face 权重。

### 访问受限制的 🤗 Hugging Face 模型

Megatron Bridge 中的某些模型需要访问 Hugging Face 上的受限制仓库。这些模型需要模型作者的明确许可才能下载或使用。

如果您在使用模型时遇到如下错误：

```
OSError: You are trying to access a gated repo.
Make sure to have access to it at <URL>
```

请按照以下步骤解决问题：

1.  **请求访问权限**：访问错误消息中提供的 URL 并请求访问受限制的模型。
2.  **生成令牌**：按照[本教程](https://huggingface.co/docs/hub/en/security-tokens#how-to-manage-user-access-tokens)创建 Hugging Face 访问令牌。
3.  **设置环境变量**：在您的环境中导出您的令牌：

```bash
export HF_TOKEN=<您的访问令牌>
```

### 基本用法

```python
from megatron.bridge import AutoBridge

# 自动加载支持的模型
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")

# 创建一个提供者，在实例化前进行配置，然后构建模型
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### 高级加载选项

您还可以使用特定设置加载模型，例如精度、设备放置或通过信任远程代码：

```python
import torch
from megatron.bridge import AutoBridge

# 使用特定设置加载
bridge = AutoBridge.from_hf_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# 从本地路径加载
bridge = AutoBridge.from_hf_pretrained("/path/to/local/hf_model")
```

### 使用模型提供者

为了更精细地控制模型配置，请使用提供者模式。提供者允许您配置任何 `TransformerConfig` 属性：

```python
from megatron.bridge import AutoBridge

# 自动加载支持的模型
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")

# 获取模型提供者（延迟加载）
provider = bridge.to_megatron_provider()

# 配置并行性（多 GPU 需要 torchrun 或 srun）
provider.tensor_model_parallel_size = 8
provider.pipeline_model_parallel_size = 2

# 配置融合
provider.bias_activation_fusion = True
provider.bias_dropout_fusion = True

# 最终确定提供者以运行验证检查并完成初始化
provider.finalize()

# 创建应用了所有配置的模型
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

提供者模式在以下情况下特别有用：
- 覆盖默认模型参数
- 配置高级功能，如 MoE、激活重计算或混合精度
- 设置分布式训练参数

## 检查支持的模型

在加载模型之前，您可以检查它是否受 Megatron Bridge 支持。

您可以按如下方式列出所有支持的 🤗 Hugging Face 模型架构：

```python
from megatron.bridge import AutoBridge

# 获取所有支持的模型架构列表
supported_models = AutoBridge.list_supported_models()

print(f"找到 {len(supported_models)} 个支持的模型：")
for i, model in enumerate(supported_models, 1):
    print(f"  {i:2d}. {model}")
```

或者，检查特定模型是否受支持：

```python
from megatron.bridge import AutoBridge

if AutoBridge.can_handle("meta-llama/Llama-3.2-1B"):
    print("✅ 模型受支持！")
```

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
else:
    print("❌ 模型需要自定义桥接实现")
```

## 转换回 🤗 Hugging Face 格式

在训练或修改 Megatron 模型后，您可以将其转换回 🤗 Hugging Face 格式以进行部署或共享。根据您的需求，桥接器提供了几种转换方法。

要保存包含配置、分词器和权重的完整模型：

```python
# 保存完整模型（配置、分词器、权重）
bridge.save_hf_pretrained(megatron_model, "./my-fine-tuned-llama")

# 保存的模型可以用 🤗 Hugging Face 加载
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained("./my-fine-tuned-llama")
```

您可以保存模型权重（safetensors 格式）：

```python
# 仅保存模型权重（更快、更小）
bridge.save_hf_weights(megatron_model, "./model_weights")

# 保存时不显示进度条（在脚本中有用）
bridge.save_hf_weights(megatron_model, "./weights", show_progress=False)
```

您也可以在转换过程中流式传输权重而不保存到磁盘，以便在 RL 框架等场景中即时使用，例如：

```python
# 在转换过程中流式传输权重（内存高效）
for name, weight in bridge.export_hf_weights(megatron_model):
    print(f"正在导出 {name}: {weight.shape}")

for name, weight in bridge.export_hf_weights(megatron_model, cpu=True):
    print(f"已导出 {name}: {tuple(weight.shape)}")
```

## 常用模式和最佳实践
在使用 Megatron Bridge 时，有几种模式可以帮助您有效地使用 API 并避免常见陷阱。

### 1. 始终使用高级 API
始终优先使用像 `AutoBridge` 这样的高级 API 进行自动模型检测。除非您知道所需的特定类型，否则避免直接使用桥接器：

```python
# ✅ 推荐：使用 AutoBridge 进行自动检测
bridge = AutoBridge.from_hf_pretrained("any-supported-model")

# ❌ 避免：除非您知道特定类型，否则避免直接使用桥接器
```

### 2. 在创建模型前进行配置
使用提供者模式时，务必在创建模型之前配置并行性和其他设置。先创建模型会使用可能并非最优的默认设置：

```python
# ✅ 正确：在创建模型前配置提供者
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 8
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)

# ❌ 避免：在配置并行性之前创建模型
model = bridge.to_megatron_model()  # 使用默认设置
```

### 3. 利用参数流式传输 API
您可以在不保存到磁盘的情况下，将转换后的权重从 Megatron 流式传输到 HF：

```python
# ✅ 对大模型使用流式传输
for name, weight in bridge.export_hf_weights(model, cpu=True):
    process_weight(name, weight)
```

### 4. 在导出工作流中使用 `from_hf_pretrained`

当将 Megatron 检查点导出回 🤗 Hugging Face 格式时，始终使用 `from_hf_pretrained()` 而不是 `from_hf_config()`。`from_hf_config()` 方法不会加载保存完整 🤗 Hugging Face 检查点所需的分词器和其他构件：

```python
from megatron.bridge import AutoBridge

# ✅ 正确：在导出工作流中使用 from_hf_pretrained
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
bridge.export_ckpt("./megatron_checkpoints/llama32_1b", "./hf_exports/llama32_1b")

# ❌ 避免：from_hf_config 缺少保存所需的构件
# config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
# bridge = AutoBridge.from_hf_config(config)  # 缺少分词器等
# bridge.export_ckpt(...)  # 将会失败！
```

`from_hf_config()` 方法仅适用于架构探索和内省（例如，检查 `transformer_config`），不适用于检查点转换工作流。

更多示例和高级使用模式，请参阅代码仓库中的 `examples/conversion/` 目录。

## 便捷工作流（命令）

这些示例可以直接作为 shell 命令运行。

### HF → Megatron 检查点导入（单次调用）

```bash
huggingface-cli login --token <your token>
python -c "from megatron.bridge import AutoBridge; AutoBridge.import_ckpt('meta-llama/Llama-3.2-1B','./megatron_checkpoints/llama32_1b')"
```

### Megatron → HF 导出（单次调用）

```bash
python -c "from megatron.bridge import AutoBridge; b=AutoBridge.from_hf_pretrained('meta-llama/Llama-3.2-1B'); b.export_ckpt('./megatron_checkpoints/llama32_1b','./hf_exports/llama32_1b')"
```

### 创建 Megatron 模型并在本地运行

```bash
python - << 'PY'
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained('meta-llama/Llama-3.2-1B')
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)

# 导出到 HF 文件夹

bridge.save_hf_pretrained(model, './hf_exports/llama32_1b')
PY
```

### 使用多个 GPU 启动（示例）

```bash
torchrun --nproc-per-node=2 -m examples.conversion.generate_from_hf
```

## AutoBridge API 参考

最新的公共 API 和签名（参见 {doc}`apidocs/bridge/bridge.models.conversion.auto_bridge`）：

```python
from megatron.bridge import AutoBridge

# 创建和能力
AutoBridge.from_hf_pretrained(path: str | Path, **kwargs) -> AutoBridge
AutoBridge.from_hf_config(config: PretrainedConfig) -> AutoBridge
AutoBridge.can_handle(path: str | Path, trust_remote_code: bool = False) -> bool
AutoBridge.list_supported_models() -> list[str]
AutoBridge.supports(config: Any) -> bool

# 提供者/模型构建
AutoBridge.to_megatron_provider(load_weights: bool = True, hf_path: str | Path | None = None) -> GPTModelProvider
AutoBridge.to_megatron_model(load_weights: bool = True, hf_path: str | Path | None = None, **kwargs) -> list[MegatronModule]

# HF → Megatron 权重
AutoBridge.load_hf_weights(model: list[MegatronModule], hf_path: str | Path | None = None) -> None

# Megatron → HF 转换
AutoBridge.export_hf_weights(model: list[MegatronModule], cpu: bool = False, show_progress: bool = True, conversion_tasks: Optional[list[WeightConversionTask]] = None) -> Iterable[HFWeightTuple]
AutoBridge.save_hf_pretrained(model: list[MegatronModule], path: str | Path, show_progress: bool = True) -> None
AutoBridge.save_hf_weights(model: list[MegatronModule], path: str | Path, show_progress: bool = True) -> None

# Megatron 原生检查点
AutoBridge.save_megatron_model(model: list[MegatronModule], path: str | Path) -> None
AutoBridge.load_megatron_model(path: str | Path, **kwargs) -> list[MegatronModule]

# 单次调用工作流
AutoBridge.import_ckpt(hf_model_id: str | Path, megatron_path: str | Path, **kwargs) -> None  # HF → Megatron 检查点
AutoBridge.export_ckpt(megatron_path: str | Path, hf_path: str | Path, show_progress: bool = True) -> None  # Megatron → HF

# 配置提取
AutoBridge.transformer_config -> TransformerConfig
AutoBridge.mla_transformer_config -> MLATransformerConfig

# 内省 / 规划
AutoBridge.get_conversion_tasks(megatron_model: MegatronModule | list[MegatronModule], hf_path: str | Path | None = None) -> list[WeightConversionTask]
```