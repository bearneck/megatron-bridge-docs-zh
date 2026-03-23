# 性能分析

Megatron Bridge 内置支持使用一系列性能分析工具来分析训练任务。这些工具包括用于工作流优化的 NVIDIA Nsight Systems (Nsys)，以及基于 PyTorch 的性能分析器和内存跟踪器，用于监控训练期间的性能和内存使用模式。

## ProfilingConfig 概览

{py:class}`bridge.training.config.ProfilingConfig` 是一个数据类，封装了与训练相关的性能分析设置。它位于整体的 {py:class}`bridge.training.config.ConfigContainer` 内部，后者代表一次训练运行的完整配置。

### 性能分析选项

配置支持两种互斥的性能分析选项：

- **NSys 性能分析** (`use_nsys_profiler`)
- **PyTorch 性能分析** (`use_pytorch_profiler`)

您可以启用其中一种，但不能同时启用两者。

### 步骤范围和目标 Rank

所有性能分析模式都允许您配置：

- **步骤范围**：`profile_step_start` 和 `profile_step_end`
- **目标 Rank**：`profile_ranks`

默认情况下，性能分析针对 rank 0。您可以指定多个 rank 来分析分布式训练设置的不同部分。

### 高级性能分析功能

配置包括记录张量形状 (`record_shapes`) 和启用内存性能分析 (`record_memory_history`) 的选项，并带有可自定义的输出路径 (`memory_snapshot_path`)。这些功能提供了对训练期间模型内存消耗和张量级操作的更深入可见性。

## NSys 性能分析

NVIDIA Nsys 是一个系统级的性能分析工具，旨在帮助您调整和优化 CUDA 应用程序。Megatron Bridge 与 Nsys 集成，支持对训练任务的特定步骤进行性能分析，从而无需手动插桩即可轻松收集详细的性能数据。

```{note}
由于实现冲突，NSys 性能分析不能与 `FaultTolerancePlugin` 一起使用。如果两者都启用，框架将自动禁用 NSys 性能分析并发出警告。
```

### 配置 NSys 性能分析

通过在 `ProfilingConfig` 中设置 `use_nsys_profiler=True` 来启用 NSys 性能分析。关键的配置选项包括：

```python
from megatron.bridge.training.config import ProfilingConfig

# 在您的 ConfigContainer 设置中，cfg 是一个 ConfigContainer 实例
cfg.profiling = ProfilingConfig(
    use_nsys_profiler=True,
    profile_step_start=10,
    profile_step_end=15,
    profile_ranks=[0, 1],  # 分析前两个 rank
    record_shapes=False,   # 可选：记录张量形状
)
```

### 使用 NSys 启动

使用 NSys 性能分析时，使用 NSys 命令包装器启动您的训练脚本：

```bash
nsys profile -s none -o <profile_filepath> -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python <path_to_script>
```

将 `<profile_filepath>` 替换为您期望的输出路径，将 `<path_to_script>` 替换为您的训练脚本。`--capture-range=cudaProfilerApi` 选项确保性能分析由框架的步骤范围配置控制。

### 使用 NeMo Run NSys 插件配置性能分析

配方（Recipe）用户可以利用 {py:class}`bridge.recipes.run_plugins.NsysPlugin` 通过 NeMo Run 执行器来配置 NSys 性能分析。该插件提供了一个便捷的接口来设置性能分析，而无需手动配置底层的 NSys 命令。

```python
import nemo_run as run
from megatron.bridge.recipes.run_plugins import NsysPlugin

# 创建您的配方和执行器
recipe = your_recipe_function()
executor = run.SlurmExecutor(...)

# 通过插件配置 NSys 性能分析
plugins = [
    NsysPlugin(
        profile_step_start=10,
        profile_step_end=15,
        profile_ranks=[0, 1],
        nsys_trace=["nvtx", "cuda"],  # 可选：指定跟踪事件
        record_shapes=False,
        nsys_gpu_metrics=False,
    )
]

# 启用性能分析运行
with run.Experiment("nsys_profiling_experiment") as exp:
    exp.add(recipe, executor=executor, plugins=plugins)
    exp.run()
```

该插件会自动配置 NSys 命令行选项，并在您的训练任务中设置性能分析配置。

### 分析结果

性能分析运行完成后，将生成 NSys 性能分析文件 (`.nsys-rep`)。要分析它们，请从 NVIDIA 开发者网站安装 [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)，在 NSys GUI 中打开文件，并使用时间线视图来探索训练任务的性能特征。

## PyTorch 性能分析器

Megatron Bridge 支持内置的 PyTorch 性能分析器，这对于在 TensorBoard 中查看性能分析结果和理解 PyTorch 级别的性能特征非常有用。

### 配置 PyTorch 性能分析器

通过在 `ProfilingConfig` 中设置 `use_pytorch_profiler=True` 来启用 PyTorch 性能分析：

```python
from megatron.bridge.training.config import ProfilingConfig

cfg.profiling = ProfilingConfig(

use_pytorch_profiler=True,
    profile_step_start=10,
    profile_step_end=15,
    profile_ranks=[0],
    record_shapes=True,    # 记录张量形状以进行详细分析
)
```

### 使用 PyTorch Profiler 插件配置性能分析

与 NSys 类似，配方用户可以使用 {py:class}`bridge.recipes.run_plugins.PyTorchProfilerPlugin` 进行便捷配置：

```python
from megatron.bridge.recipes.run_plugins import PyTorchProfilerPlugin

plugins = [
    PyTorchProfilerPlugin(
        profile_step_start=10,
        profile_step_end=15,
        profile_ranks=[0],
        record_memory_history=True,
        memory_snapshot_path="memory_snapshot.pickle",
        record_shapes=True,
    )
]
```

## 内存性能分析

Megatron Bridge 提供对 CUDA 内存性能分析的内置支持，以跟踪和分析训练期间的内存使用模式，包括 GPU 内存分配和消耗跟踪。

有关生成的内存分析文件的更多信息，请参阅[此处](https://pytorch.org/blog/understanding-gpu-memory-1/)。

### 配置内存性能分析

通过在您的 `ProfilingConfig` 中设置 `record_memory_history=True` 来启用内存性能分析。这可以与任一性能分析模式一起使用：

```python
from megatron.bridge.training.config import ProfilingConfig

cfg.profiling = ProfilingConfig(
    use_pytorch_profiler=True,  # 或 use_nsys_profiler=True
    profile_step_start=10,
    profile_step_end=15,
    profile_ranks=[0],
    record_memory_history=True,
    memory_snapshot_path="memory_trace.pickle",  # 自定义输出路径
)
```

### 分析内存使用情况

运行完成后，每个指定 rank 的内存快照将保存到指定路径。使用 PyTorch Memory Viz 工具加载这些跟踪文件，以绘制随时间变化的内存使用情况图，并检测训练管道中的瓶颈或内存泄漏。

## 优化性能分析准确性

性能分析会给您的训练任务带来开销，因此测量的时间可能略高于正常操作。为了获得准确的性能分析结果，请在分析的步骤范围内禁用其他密集型操作，例如频繁的检查点保存。请仔细选择您的性能分析步骤范围，以捕获具有代表性的训练行为，同时最小化对整个任务性能的影响。