# 回调函数

Megatron Bridge 提供了一个轻量级的回调系统，用于在不修改框架代码的情况下，将自定义逻辑注入训练和评估循环。这非常适合专有集成或自定义日志记录和指标跟踪。

## 快速开始

### 基于类的回调函数

继承 {py:class}`bridge.training.callbacks.Callback` 类并重写事件方法：

```python
import time

from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.recipes.qwen import qwen25_500m_pretrain_config

class MyCallback(Callback):
    def on_train_start(self, context):
        context.user_state['start_time'] = time.time()
        print(f"Training started at step {context.state.train_state.step}")

    def on_train_step_end(self, context):
        if context.loss_dict:
            print(f"Step {context.state.train_state.step}: loss={context.loss_dict}")

    def on_train_end(self, context):
        elapsed = time.time() - context.user_state['start_time']
        print(f"Training completed in {elapsed:.2f}s")

# 创建一个适合单个 GPU 的配置
config = qwen25_500m_pretrain_config()

# 将回调函数传递给 pretrain
pretrain(config, forward_step, callbacks=[MyCallback()])
```

### 函数式回调函数

直接向 {py:class}`bridge.training.callbacks.CallbackManager` 注册函数：

```python
from megatron.bridge.training.callbacks import CallbackManager
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.recipes.qwen import qwen25_500m_pretrain_config

def log_step(context):
    step = context.state.train_state.step
    if context.loss_dict:
        print(f"Step {step}: {context.loss_dict}")

callback_manager = CallbackManager()
callback_manager.register("on_train_step_end", log_step)

# 创建一个适合单个 GPU 的配置
config = qwen25_500m_pretrain_config()

pretrain(config, forward_step, callbacks=callback_manager)
```

### 混合两种模式

两种注册模式可以结合使用：

```python
from megatron.bridge.training.callbacks import CallbackManager
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.recipes.qwen import qwen25_500m_pretrain_config

manager = CallbackManager()
manager.add(MyCallback())
manager.add([TimingCallback(), MetricsCallback()])
manager.register("on_eval_end", lambda ctx: print("Evaluation complete!"))

# 创建一个适合单个 GPU 的配置
config = qwen25_500m_pretrain_config()

pretrain(config, forward_step, callbacks=manager)
```

## 可用事件

### 训练事件

| 事件 | 触发时机 | 可用的上下文字段 |
|-------|------------|-------------------------|
| `on_train_start` | 在 `model.train()` 之后，训练循环开始之前 | `state`, `model`, `user_state`, `optimizer`, `scheduler` |
| `on_train_step_start` | 每个训练步骤开始之前 | `state`, `model`, `user_state`, `optimizer`, `scheduler` |
| `on_train_step_end` | 每个训练步骤结束之后 | `state`, `model`, `user_state`, `optimizer`, `scheduler`, `loss_dict`, `grad_norm`, `skipped_iter` |
| `on_train_end` | 训练循环完成之后 | `state`, `model`, `user_state`, `optimizer`, `scheduler` |

### 验证事件

| 事件 | 触发时机 | 可用的上下文字段 |
|-------|------------|-------------------------|
| `on_eval_start` | 在 `model.eval()` 之后，验证循环开始之前 | `state`, `model`, `user_state` |
| `on_eval_step_start` | 每个验证步骤开始之前 | `state`, `model`, `user_state` |
| `on_eval_step_end` | 每个验证步骤结束之后 | `state`, `model`, `user_state` |
| `on_eval_end` | 验证完成之后 | `state`, `model`, `user_state`, `total_loss_dict` |

### 测试事件

| 事件 | 触发时机 | 可用的上下文字段 |
|-------|------------|-------------------------|
| `on_test_start` | 在 `model.eval()` 之后，测试循环开始之前 | `state`, `model`, `user_state` |
| `on_test_step_start` | 每个测试步骤开始之前 | `state`, `model`, `user_state` |
| `on_test_step_end` | 每个测试步骤结束之后 | `state`, `model`, `user_state` |
| `on_test_end` | 测试完成之后 | `state`, `model`, `user_state`, `total_loss_dict` |

## CallbackContext

{py:class}`bridge.training.callbacks.CallbackContext` 提供了对框架状态的访问：

### 始终可用

- **`state`**: {py:class}`bridge.training.state.GlobalState` - 包含配置、训练状态、计时器和日志记录器
- **`model`**: 模型块的列表
- **`user_state`**: 用于在回调函数调用之间存储数据的可变字典

### 仅训练事件可用

- **`optimizer`**: 优化器实例
- **`scheduler`**: 学习率调度器

### 事件特定字段

- **`loss_dict`** (`on_train_step_end`): 从训练步骤中归约的损失字典

- **`grad_norm`** (`on_train_step_end`): 梯度范数（如果已计算）
- **`skipped_iter`** (`on_train_step_end`): 该迭代是否被跳过
- **`total_loss_dict`** (`on_eval_end`, `on_test_end`): 聚合的评估/测试损失

## 用户状态

`CallbackManager` 拥有一个 `user_state` 字典，该字典在一次训练运行期间的所有回调调用中持续存在。使用它在回调之间共享数据或累积指标：

```python
class StepCounterCallback(Callback):
    def on_train_start(self, context):
        context.user_state['callback_step_count'] = 0

    def on_train_step_end(self, context):
        context.user_state['callback_step_count'] += 1

    def on_train_end(self, context):
        print(f"Callback saw {context.user_state['callback_step_count']} steps")
```

## 分布式训练

回调在**所有进程（rank）**上触发，没有框架级别的同步。如果你的回调应该只在特定进程上运行，请添加防护：

```python
import torch.distributed as dist

class RankZeroCallback(Callback):
    def on_train_step_end(self, context):
        if dist.get_rank() == 0:
            print(f"Step {context.state.train_state.step} complete")
```

## 异常处理

来自回调的异常会传播给调用者。框架不会捕获或处理回调异常。如果你的回调可能失败，请将其包装在 try-except 中：

```python
def safe_callback(context):
    try:
        # 你的逻辑写在这里
        external_service.log(context.loss_dict)
    except Exception as e:
        print(f"Callback failed: {e}")
        # 不要重新抛出异常，以避免停止训练
```

## 执行顺序

回调按注册顺序触发：

1. 通过 `add()` 添加的回调按其添加顺序触发
2. 通过 `register()` 注册的回调按其注册顺序触发
3. 如果两种方法都使用了，顺序取决于每种方法被调用的时间

## 内省

查询已注册的回调：

```python
manager = CallbackManager()
manager.register("on_train_start", my_fn)

# 检查某个事件是否有任何回调存在
if manager.has_callbacks("on_train_start"):
    print("Callbacks registered for on_train_start")

# 列出某个事件的所有回调
callbacks = manager.list_callbacks("on_train_start")
print(f"Found {len(callbacks)} callbacks")

# 获取所有有效的事件名称
print(manager.events)  # 有效事件名称的 frozenset
```

## 设计原则

回调系统遵循以下原则：

1.  **第一方隔离**：框架代码从不使用回调来实现其自身逻辑。回调严格用于第三方扩展。
2.  **零开销**：当没有注册任何回调时，性能开销为零。
3.  **安全性**：回调接收框架状态，但修改它由用户自行承担风险。框架不保证修改的效果。

## 示例

### 专有指标

```python
class ProprietaryMetricsCallback(Callback):
    """发送指标到内部监控系统。"""

    def __init__(self, endpoint: str):
        self.client = InternalMetricsClient(endpoint)

    def on_train_step_end(self, context):
        if context.loss_dict:
            self.client.send({
                "step": context.state.train_state.step,
                "loss": context.loss_dict.get("lm loss"),
                "grad_norm": context.grad_norm,
                "cluster_id": os.environ.get("CLUSTER_ID"),
            })
```

## API 参考

- {py:class}`bridge.training.callbacks.Callback`
- {py:class}`bridge.training.callbacks.CallbackContext`
- {py:class}`bridge.training.callbacks.CallbackManager`
- {py:func}`bridge.training.callbacks.normalize_callbacks`
- {py:func}`bridge.training.callbacks.should_fire`