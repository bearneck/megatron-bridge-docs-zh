# 使用配方

Megatron Bridge 为多个流行模型提供了生产就绪的训练配方。您可以在此处找到支持的配方和 🤗 HuggingFace 桥接器的概述。
本指南将介绍使用训练配方的后续步骤，包括如何[覆盖配置](#overriding-configuration)以及如何[启动作业](#launch-methods)。

## 概述

- **覆盖范围**：我们为选定的模型系列和尺寸提供配方，包括 Llama、Qwen、DeepSeek 和 Nemotron-H（基于 Mamba）。
- **默认值**：每个配方都设置了旨在实现收敛和性能的默认值，涵盖并行策略、精度数据类型以及优化器和调度器选择。这些配方可以作为高质量的起点。
- **集成**：配方返回一个单一的 `ConfigContainer`，可直接插入我们的训练[入口点](training/entry-points.md)（另请参阅已发布的文档：https://docs.nvidia.com/nemo/megatron-bridge/latest/training/entry-points.html）。
- **自定义**：您可以覆盖配方的任何部分（Python、YAML、CLI），以适应您的数据、规模和目标。

## 覆盖配置

配方通过 {py:class}`~bridge.training.config.ConfigContainer` 对象提供。这是一个数据类，保存了训练所需的所有配置对象。您可以在此处找到关于 `ConfigContainer` 的更详细概述。
通过 Python 结构提供完整配方的好处在于，它对用户可能偏好的任何配置方法（无论是 YAML、`argparse` 还是其他方法）都是不可知的。换句话说，用户可以按照他们认为合适的方式覆盖配方。

以下部分详细介绍了覆盖配置配方的几种不同方法。有关完整的训练脚本，请参阅[此示例](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/llama/pretrain_llama3_8b.py)。

### Python

如果您更喜欢在 Python 中管理配置，可以直接修改 `ConfigContainer` 的属性：

```python
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config

# 从配方中获取基础 ConfigContainer
cfg: ConfigContainer = pretrain_config()

# 应用覆盖。注意层次结构
cfg.train.train_iters = 20
cfg.train.global_batch_size = 8
cfg.train.micro_batch_size = 1
cfg.logger.log_interval = 1
```

您还可以替换 `ConfigContainer` 的整个子配置：

```python
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.models.llama import Llama3ModelProvider

cfg: ConfigContainer = pretrain_config()

small_llama = Llama3ModelProvider(
    num_layers=2,
    hidden_size=768,
    ffn_hidden_size=2688,
    num_attention_heads=16,
)
cfg.model = small_llama
```

### YAML

可以使用 OmegaConf 工具通过 YAML 文件覆盖配置配方：

```python
from omegaconf import OmegaConf
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
)

cfg: ConfigContainer = pretrain_config()
yaml_filepath = "conf/llama3-8b-benchmark-cfg.yaml"

# 将初始的 Python 数据类转换为 OmegaConf DictConfig 以进行合并
# excluded_fields 保存了一些无法序列化为 DictConfig 的配置
merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

# 加载并合并 YAML 覆盖
yaml_overrides_omega = OmegaConf.load(yaml_filepath)
merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)

# 应用覆盖，同时保留 excluded_fields
final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
apply_overrides(cfg, final_overrides_as_dict, excluded_fields)
```

上述代码片段将使用 `llama3-8b-benchmark-cfg.yaml` 中的所有覆盖来更新 `cfg`。

### Hydra 风格

Megatron Bridge 提供了一些实用工具，用于使用 Hydra 风格的 CLI 覆盖来更新 ConfigContainer：

```python
import sys
from omegaconf import OmegaConf
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)

cfg: ConfigContainer = pretrain_config()
cli_overrides = sys.argv[1:]

# 将初始的 Python 数据类转换为 OmegaConf DictConfig 以进行合并
# excluded_fields 保存了一些无法序列化为 DictConfig 的配置
merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

# 解析并合并 CLI 覆盖
merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

# 应用覆盖，同时保留 excluded_fields
final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
apply_overrides(cfg, final_overrides_as_dict, excluded_fields)
```

在上述代码片段之后，`cfg` 将更新所有通过命令行界面提供的覆盖参数。
包含上述代码的脚本可以这样调用：

```sh
torchrun <torchrun 参数> pretrain_cli_overrides.py model.tensor_model_parallel_size=4 train.train_iters=100000 ...
```

## 启动方法

Megatron Bridge 支持使用 `torchrun` 和 [NeMo-Run](https://github.com/NVIDIA-NeMo/Run) 启动脚本。
当您的脚本准备好启动时，请参考以下章节之一。

### Torchrun
Megatron Bridge 训练脚本可以使用大多数 PyTorch 用户熟悉的 `torchrun` 命令启动。
只需使用 `--nproc-per-node` 指定要使用的 GPU 数量，并使用 `--nnodes` 指定节点数量。例如，在单个节点上：

```sh
torchrun --nnodes 1 --nproc-per-node 8 /path/to/train/script.py <训练脚本参数>
```

对于多节点训练，建议使用像 SLURM 这样的集群编排系统。
`torchrun` 命令应按照集群编排系统的要求进行包装。
例如，使用 Slurm 时，将 `torchrun` 命令包装在 `srun` 内：

```sh
# launch.sub

srun --nodes 2 --gpus-per-node 8 \
    --container-image <镜像标签> --container-mounts <挂载点> \
    bash -c "
        torchrun --nnodes $SLURM_NNODES --nproc-per-node $SLURM_GPUS_PER_NODE /path/to/train/script.py <训练脚本参数>
    "
```

以及任何其他必需的标志。还建议将 NeMo Framework 容器与 Slurm 一起使用。您可以在 [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags) 上找到容器标签列表。

### NeMo-Run

Megatron Bridge 也支持使用 [NeMo-Run](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/index.html) 启动训练。NeMo-Run 是一个 Python 包，支持在多个平台上配置和执行实验。
对于多节点训练，NeMo-Run 将生成一个包含适当命令的脚本，类似于上面描述的 `srun` 命令。

使用 NeMo-Run 启动 Megatron Bridge 脚本的推荐方法是通过 `run.Script` API。
您可以在新文件中根据需求修改以下 3 个步骤：

```python
import nemo_run as run

if __name__ == "__main__":
    # 1) 配置 `run.Script` 对象
    train_script = run.Script(path="/path/to/train/script.py", entrypoint="python")

    # 2) 为目标平台定义一个执行器
    executor = run.LocalExecutor(ntasks_per_node=8, launcher="torchrun")

    # 3) 执行
    run.run(train_script, executor=executor)
```

NeMo-Run 支持在多个不同的平台上启动，包括 [SLURM 集群](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/guides/execution.html#slurmexecutor)。
更多详情，请参阅 NeMo-Run [文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/guides/execution.html#)，了解支持的平台列表、对应的执行器以及配置说明。

您还可以将参数从 NeMo-Run 启动脚本转发到目标脚本：

```python
import nemo_run as run
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ...
    known_args, args_to_fwd = parser.parse_known_args()
    train_script = run.Script(..., args=args_to_fwd)
```

有关 `run.Script` API 的完整示例，包括参数转发，请参阅 [此脚本](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/llama/pretrain_llama3_8b_nemo_run_script.py)。

#### 插件

Megatron Bridge 提供了几个 NeMo-Run 插件来简化某些功能的使用。
这些插件可以简单地添加到 `run.run()` 调用中：

```python
import nemo_run as run
from megatron.bridge.recipes.run_plugins import NsysPlugin

if __name__ == "__main__":
    train_script = run.Script(path="/path/to/train/script.py", entrypoint="python")
    executor = run.LocalExecutor(ntasks_per_node=8, launcher="torchrun")

    plugins = [] # plugins 参数期望一个列表
    nsys = NsysPlugin(profile_step_start=10, profile_step_end=15, ...)
    plugins.append(nsys)
    run.run(train_script, plugins=plugins, executor=executor)
```

##### 自定义参数转换器

默认情况下，插件在与 `run.Script` 任务一起使用时，会将其配置转换为 Hydra 风格的 CLI 参数。如果您的训练脚本使用不同的参数格式（例如，argparse），您可以通过 `script_args_converter_fn` 参数提供自定义转换函数。

```python
import nemo_run as run
from typing import List
from megatron.bridge.recipes.run_plugins import (
    PreemptionPlugin,
    PreemptionPluginScriptArgs,
)

# 为 argparse 风格的参数定义自定义转换器
def argparse_preemption_converter(args: PreemptionPluginScriptArgs) -> List[str]:
    result = []
    if args.enable_exit_handler:
        result.append("--enable-exit-handler")
    if args.enable_exit_handler_for_data_loader:
```

```python
    result.append("--enable-exit-handler-dataloader")
    return result

if __name__ == "__main__":
    train_script = run.Script(path="/path/to/train/script.py", entrypoint="python")
    executor = run.LocalExecutor(ntasks_per_node=8, launcher="torchrun")

    # 使用插件及其自定义转换器
    plugin = PreemptionPlugin(
        preempt_time=120,
        enable_exit_handler=True,
        script_args_converter_fn=argparse_preemption_converter,
    )
    run.run(train_script, plugins=[plugin], executor=executor)
```

每个插件都提供了其对应的数据类（例如 `PreemptionPluginScriptArgs`、`NsysPluginScriptArgs`），用于定义可供转换的参数。

有关可用的 NeMo-Run 插件列表，请参阅 [API 参考](#bridge.recipes.run_plugins)。

### 避免程序挂起

在使用 Megatron Bridge 中的任何脚本时，请确保将您的代码包装在 `if __name__ == "__main__":` 代码块中。否则，您的代码可能会意外挂起。

原因是 Megatron Bridge 在运行多 GPU 作业时，后端使用了 Python 的 `multiprocessing` 模块。该模块会创建新的 Python 进程，这些进程会导入当前模块（您的脚本）。如果您没有添加 `__name__== "__main__"`，那么您的模块将生成新的进程，这些进程导入模块后又会各自生成新的进程。这将导致进程生成的无限循环。

## 资源

- [OmegaConf 文档](https://omegaconf.readthedocs.io/en/2.3_branch/)
- [torchrun 文档](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [PyTorch 多节点训练文档](https://docs.pytorch.org/tutorials/intermediate/ddp_series_multinode.html)
- [NeMo-Run 文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/index.html#)