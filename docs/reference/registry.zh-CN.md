# Plugin & Registry Reference

> **受众**：想快速扩展框架功能的算法研究员。本手册只说"输入什么、输出什么"。

---

## 1. 注册语法

### Import

```python
from lighttrain.registry import (
    register,           # 注册装饰器 / 函数
    get,                # 按名称取回
    list_entries,       # 列出某类别的所有名称
    categories,         # 列出所有类别
    register_category,  # 新增自定义类别
    contains,           # 是否已注册
    unregister,         # 删除注册
)
```

**源文件**：[lighttrain/registry/_core.py](../../lighttrain/registry/_core.py)

---

### 三种使用方式

**① 装饰器（最常用）**

```python
from lighttrain.registry import register

@register("optimizer", "my_optim")
class MyOptimWrapper:
    ...
```

**② 多名称注册（同一个类绑定多个 key）**

```python
@register("loss", "cross_entropy")
@register("loss", "ce")            # 别名
class CrossEntropyLoss:
    ...
```

**③ 强制覆盖（用于插件替换内置实现）**

```python
@register("model", "hf_causal", force=True)
class MyPatchedCausalLM:
    ...
```

**④ 函数调用形式**

```python
register("model", "my_model", MyModelClass)
```

---

### 查询 API

```python
cls = get("optimizer", "adamw")         # 取回类/对象
names = list_entries("optimizer")       # → ['adamw', 'lion']
exists = contains("model", "hf_causal") # → True
```

---

### 新增自定义类别

预声明 33 个 `KNOWN_CATEGORIES`（见第 2 节）以外的类别需先注册：

```python
register_category("my_plugin_category")

@register("my_plugin_category", "my_impl")
class MyImpl:
    ...
```

---

## 2. 注册类别清单

共 **35 个** `KNOWN_CATEGORIES`（`metric` 暂无内置实现；`architecture` 已注册 `transformer` / `rwkv`）。

### Core 8

| 类别 | 作用 | YAML 挂载节点 |
|------|------|--------------|
| `model` | 模型前向 | `model:` |
| `loss` | 损失函数 | `loss:` |
| `optimizer` | 优化器包装器 | `optimizer:` |
| `scheduler` | 学习率调度 | `scheduler:` |
| `dataset` | 训练数据集（Map-style / Iterable） | `data.dataset:` |
| `processor` | 多模态预处理（图像/音频/文本） | `data.processor:` |
| `collator` | batch 组装（padding / stacking） | `data.collator:` |
| `sampler` | 索引采样顺序 | `data.sampler:` |

### Training Orchestration

| 类别 | 作用 | YAML 挂载节点 |
|------|------|--------------|
| `trainer` | 训练主循环 | `trainer:` |
| `engine` | 单步前向+反向 dispatch | `engine:` |
| `update_rule` | 梯度更新规则（clip / 累积 / SAM...） | `update_rule:` |
| `callback` | 训练生命周期钩子 | `callbacks:` (列表) |
| `metric` | 评估指标（accumulate → compute） | `metrics:` |
| `logger` | 日志后端 | `logger:` 或 `loggers:` (列表) |
| `objective` | 目标函数封装（batch 变换 + loss） | `objective:` |
| `architecture` | 架构元信息（block 迭代器、head 获取...） | `architecture:` |

### Frontier 6

| 类别 | 作用 | YAML 挂载节点 |
|------|------|--------------|
| `generation_strategy` | 可控生成策略（beam / MCTS / best-of-N...） | `generation_strategy:` |
| `judge` | 响应评估（验证器 / 成对 LLM 打分） | `judge:` |
| `environment` | RL 环境（reset / step） | `environment:` |
| `retriever` | 检索器（index + query） | `retriever:` |
| `chunker` | 文档分块 | `chunker:` |
| `probe` | 表示探针（attach + compute） | `probe:` |

### Artifact & Data Pipeline

| 类别 | 作用 | YAML 挂载节点 |
|------|------|--------------|
| `artifact_producer` | 运行模型前向、收集中间张量 | `artifact_producer:` |
| `artifact_store` | 张量数据存储后端（safetensors / memmap / parquet） | `artifact_store:` |
| `prep_node` | PrepGraph DAG 节点（数据预处理流水线） | `prep.nodes:` (列表) |
| `data_module` | 完整数据模块（封装 dataset + collator + loader） | `data:` |
| `tokenizer` | 分词器 | `data.tokenizer:` |

### Failure-First & RL

| 类别 | 作用 | YAML 挂载节点 |
|------|------|--------------|
| `invariant` | 运行时断言（返回 `bool`，失败则阻断步骤） | `invariants:` (列表) |
| `rl_backend` | RL rollout 生成后端（ppo/grpo 经注册表解析） | `trainer.rollout_backend:` |
| `value_head` | 价值/奖励头（PPO critic / RM 打分头） | `value_head:` |
| `reward_adapter` | judge → RL `reward_fn` 适配器 | `reward_adapter:` |

### Distributed Strategies

| 类别 | 已注册名称 | 对应 Protocol | YAML 挂载节点 |
|------|-----------|--------------|--------------|
| `grad_sync_strategy` | `noop` · `ddp` · `fsdp` · `deepspeed` | `GradSyncStrategy` | `parallel.grad_sync.name:` |

> 仅支持数据并行；张量 / 流水线 / 专家 / 序列并行（`model_parallel_strategy` · `pipeline_schedule`）已于 v0.4.5 移除。

### Sweep

| 类别 | 已注册名称 | YAML 挂载节点 |
|------|-----------|--------------|
| `sweep_backend` | `optuna`（plugin，需 `pip install -e '.[sweep]'`） | `sweep --strategy optuna` |

---

## 3. 数据载体 Dataclass

这三个类型贯穿所有协议，需提前了解。

```python
# lighttrain/protocols.py

@dataclass
class ModelOutput:
    outputs: dict[str, torch.Tensor]       # 主输出张量（logits / eps / recon...）
    loss: torch.Tensor | None              # 模型内置 loss（可选）
    hidden_states: tuple[torch.Tensor, ...] | None
    attentions: tuple[torch.Tensor, ...] | None
    extras: dict[str, torch.Tensor]        # ExtraOutputSpec 捕获的额外张量
    state: Any | None                      # 有状态架构（RWKV / Mamba）的 state

@dataclass
class LossContext:
    step: int                              # 当前全局步数
    epoch: int
    metrics: dict[str, float]
    loss_family: str | None               # "next_token" / "mlm" / "rl" / ...
    extras: dict[str, Any]

@dataclass
class StepOutput:
    loss: Any | None                       # 主优化目标（标量 tensor 或 float）
    metrics: dict[str, Any]               # 包含 "loss" 键的完整 metrics
    logs: dict[str, Any]
    extras: dict[str, Any]
```

---

## 4. 协议要求与内置注册项

每个小节格式：**协议方法签名 → 基类（如有）→ 极简真实范例 → 内置注册项清单**。

---

### 4.1 `model`

**Protocol**：`ModelProtocol`（[lighttrain/protocols.py:86](../../lighttrain/protocols.py#L86)）

```python
def forward(self, **batch: Any) -> ModelOutput: ...
```

**扩展 Protocol**（用于 RL rollout）：`GenerativeModelProtocol`

```python
def forward(self, **batch: Any) -> ModelOutput: ...
def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...
```

**要求**：
- 继承 `torch.nn.Module`（非强制，但推荐）
- `forward` 接受 `**batch`（即 collator 返回的 dict 展开），返回 `ModelOutput`
- `outputs` dict 中必须有下游 loss/objective 需要的 key（通常为 `"logits"`）

**极简范例**：[lighttrain/builtin_plugins/models/text/tiny_lm.py:113](../../lighttrain/builtin_plugins/models/text/tiny_lm.py#L113)

```python
@register("model", "tiny_lm")
class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=260, d_model=512, n_layers=6, n_heads=8,
                 max_seq_len=512, dropout=0.0, tie_weights=True, init_std=0.02,
                 output_hidden_states=False, output_attentions=False) -> None: ...

    def forward(self, input_ids, attention_mask=None, labels=None,
                *, output_hidden_states=None, output_attentions=None, **_) -> ModelOutput:
        ...
        return ModelOutput(outputs={"logits": logits})
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `tiny_lm` | 轻量 Pre-norm Transformer，支持 tied weights | [builtin_plugins/models/text/tiny_lm.py](../../lighttrain/builtin_plugins/models/text/tiny_lm.py) |
| `hf_causal` | HuggingFace CausalLM 适配器（支持任意 pretrained） | [builtin_plugins/models/text/hf_causal.py](../../lighttrain/builtin_plugins/models/text/hf_causal.py) |
| `tiny_cnn` *(plugin)* | 卷积图像分类器（conv→自适应池化→线性头，分辨率无关） | [builtin_plugins/models/vision/tiny_cnn.py](../../lighttrain/builtin_plugins/models/vision/tiny_cnn.py) |
| `lora` | LoRA PEFT 适配器（包装 base model） | [builtin_plugins/models/peft/_lora.py](../../lighttrain/builtin_plugins/models/peft/_lora.py) |
| `ia3` | IA³ PEFT 适配器 | [builtin_plugins/models/peft/_ia3.py](../../lighttrain/builtin_plugins/models/peft/_ia3.py) |
| `adalora` | AdaLoRA 自适应秩 PEFT | [builtin_plugins/models/peft/_adalora.py](../../lighttrain/builtin_plugins/models/peft/_adalora.py) |
| `jepa` *(plugin)* | JEPA 架构（图像 / 语言 JEPA 训练） | [lighttrain/builtin_plugins/optim/architectures/jepa.py](../../lighttrain/builtin_plugins/optim/architectures/jepa.py) |
| `qlora` *(plugin)* | QLoRA（4-bit 量化 base + LoRA）PEFT | [lighttrain/builtin_plugins/quant/_qlora.py](../../lighttrain/builtin_plugins/quant/_qlora.py) |
| `tiny_rwkv` *(plugin)* | RWKV 时间混合架构 | [lighttrain/builtin_plugins/optim/architectures/rwkv/](../../lighttrain/builtin_plugins/optim/architectures/rwkv/__init__.py) |
| `tiny_mamba` *(plugin)* | Mamba / SSM 架构 | [lighttrain/builtin_plugins/optim/architectures/mamba/](../../lighttrain/builtin_plugins/optim/architectures/mamba/__init__.py) |
| `tiny_unet` *(plugin)* | Diffusion U-Net | [lighttrain/builtin_plugins/optim/architectures/diffusion_unet/](../../lighttrain/builtin_plugins/optim/architectures/diffusion_unet/__init__.py) |

---

### 4.2 `loss`

**Protocol**：`LossFnProtocol`（[lighttrain/protocols.py:102](../../lighttrain/protocols.py#L102)）

```python
def __call__(
    self,
    model_output: ModelOutput,
    batch: Mapping[str, Any],
    ctx: LossContext,
) -> dict[str, Any]: ...
```

**返回值要求**：dict 中必须包含 `"loss"` key（`torch.Tensor`，标量，requires_grad=True）。

**极简范例**：[lighttrain/builtin_plugins/losses/core.py:37](../../lighttrain/builtin_plugins/losses/core.py#L37)

```python
@register("loss", "cross_entropy")
@register("loss", "ce")
class CrossEntropyLoss:
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0) -> None: ...

    def __call__(self, model_output, batch, ctx) -> dict[str, Any]:
        logits = model_output.outputs["logits"]   # (B, T, V)
        labels = batch["labels"]                  # (B, T)
        loss = F.cross_entropy(shift_logits, shift_labels, ...)
        return {"loss": loss}
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `cross_entropy` / `ce` | 标准 Causal-LM next-token CE | [builtin_plugins/losses/core.py](../../lighttrain/builtin_plugins/losses/core.py) |
| `classification` | 监督分类 CE + top-1 准确率（rank-2 logits (B,C)，`loss_family="classification"`） | [builtin_plugins/losses/classification.py](../../lighttrain/builtin_plugins/losses/classification.py) |
| `mlm` | Masked-LM CE | [builtin_plugins/losses/core.py](../../lighttrain/builtin_plugins/losses/core.py) |
| `z_loss` | Z-loss 正则（防 logit 爆炸） | [builtin_plugins/losses/core.py](../../lighttrain/builtin_plugins/losses/core.py) |
| `composite` | 多 loss 加权组合 | [builtin_plugins/losses/core.py](../../lighttrain/builtin_plugins/losses/core.py) |
| `dpo` | Direct Preference Optimization | [builtin_plugins/losses/preference.py](../../lighttrain/builtin_plugins/losses/preference.py) |
| `bradley_terry` / `bt` | Bradley-Terry 偏好 loss | [builtin_plugins/losses/preference.py](../../lighttrain/builtin_plugins/losses/preference.py) |
| `ipo` | Identity Preference Optimization | [builtin_plugins/losses/preference.py](../../lighttrain/builtin_plugins/losses/preference.py) |
| `simpo` | SimPO | [builtin_plugins/losses/preference.py](../../lighttrain/builtin_plugins/losses/preference.py) |
| `orpo` | ORPO | [builtin_plugins/losses/preference.py](../../lighttrain/builtin_plugins/losses/preference.py) |
| `kto` | KTO | [builtin_plugins/losses/preference.py](../../lighttrain/builtin_plugins/losses/preference.py) |
| `ppo_surrogate` | PPO clip 代理目标 | [builtin_plugins/losses/rl.py](../../lighttrain/builtin_plugins/losses/rl.py) |
| `grpo` | GRPO loss | [builtin_plugins/losses/rl.py](../../lighttrain/builtin_plugins/losses/rl.py) |
| `info_nce` | InfoNCE / 对比学习 | [builtin_plugins/losses/aux.py](../../lighttrain/builtin_plugins/losses/aux.py) |
| `moe_balance` | MoE 负载均衡辅助 loss | [builtin_plugins/losses/aux.py](../../lighttrain/builtin_plugins/losses/aux.py) |
| `kl_topk` | Top-k KL 蒸馏 | [builtin_plugins/losses/distill.py](../../lighttrain/builtin_plugins/losses/distill.py) |
| `hidden_mse` | 隐层 MSE 蒸馏 | [builtin_plugins/losses/distill.py](../../lighttrain/builtin_plugins/losses/distill.py) |
| `hidden_cosine` | 隐层余弦相似度蒸馏 | [builtin_plugins/losses/distill.py](../../lighttrain/builtin_plugins/losses/distill.py) |
| `attention_transfer` | 注意力图迁移蒸馏 | [builtin_plugins/losses/distill.py](../../lighttrain/builtin_plugins/losses/distill.py) |

---

### 4.3 `optimizer`

**Protocol**：`OptimizerWrapperProtocol`（[lighttrain/protocols.py:112](../../lighttrain/protocols.py#L112)）

**完整契约**（更新规则与 checkpoint 管理器都调用在 **wrapper** 上，而非内部
optimizer——务必全部实现，或继承 `OptimizerWrapperBase`）：

```python
optimizer: torch.optim.Optimizer          # build() 后必须设置；须暴露 .param_groups
                                          # （LR 日志读 optimizer.optimizer.param_groups[0]["lr"]）
def build(self, model: Any) -> torch.optim.Optimizer: ...
def step(self, *a, **k) -> Any: ...        # 每步调用
def zero_grad(self, set_to_none=True): ...
def state_dict(self) -> dict: ...          # checkpoint 管理器调用
def load_state_dict(self, sd) -> None: ...

# 可选：estimate() 在场时调用，否则回退 2×params
def optim_state_bytes(self, model) -> int: ...
```

**调用时序**（[builtin_plugins/engine/update_rules/standard.py:280-307](../../lighttrain/builtin_plugins/engine/update_rules/standard.py#L280-L307)）：每步
`clip_grad_norm_(...)` → `optimizer.step()` → `zero_grad()`。**`step()` 时梯度为全秩、未被框架
改动**（无 closure、无预先 grad mutation）——梯度操纵型优化器（如 GaLore）可在 `.step()` 内
安全读取并投影 `.grad`。

**自定义状态 checkpoint**：optimizer state 经
`torch.save(..., weights_only=False)`（[checkpoint/manager.py:136](../../lighttrain/engine/checkpoint/manager.py#L136)、
[:177](../../lighttrain/engine/checkpoint/manager.py#L177)）整体 round-trip。放进 `optimizer.state[p]` 的任意
对象（如 GaLore 的 `GaLoreProjector`）能存活，前提是 **(a) 可 pickle、(b) 加载时其类可
import**。若按引用 pickle 自定义类，checkpoint 即**不自包含**（加载进程缺该包会
`ModuleNotFoundError`）。要可移植，请在 `state_dict()` 里把自定义状态序列化为**纯张量**
（无类引用），`load_state_dict()` 再重建对象。

> ⚠️ **Aliasing 陷阱**：`torch.optim.Optimizer.state_dict()` 返回的内层
> `state[param]` dict 与活动优化器**同引用**。若在 `state_dict()` 覆写里**原地**改写
> 自定义状态（例如把 projector 对象换成纯张量），会**污染正在运行的优化器**——下一个
> `.step()` 会拿到你序列化后的形式而非活动对象，直接崩。**务必先 copy 再改写**。
> `OptimizerWrapperBase._safe_state_dict(convert)` 已替你做好这件事（内层 dict 先 copy，再对每个
> 条目应用 `convert(key, value) -> value`）：
>
> ```python
> def state_dict(self):
>     def conv(k, v):
>         return v.as_plain_tensors() if k == "projector" else v
>     return self._safe_state_dict(conv)   # 安全：不会 alias 活动状态
> ```

**`optim_state_bytes(model)`（可选）**：返回优化器真实的每步状态字节数。`lighttrain estimate`
在场时调用它，否则回退 `2 × trainable_param_bytes`（全秩 Adam 假设）。内存高效优化器
（GaLore / 8-bit Adam / Adam-mini）覆写它，`estimate` 才能**看见**其节省（issue #4）。

**基类**：`OptimizerWrapperBase`（[lighttrain/optim/base.py](../../lighttrain/optim/base.py)）

提供 `step / zero_grad / state_dict / load_state_dict` 与默认 `optim_state_bytes`（Adam 类
`2×`、Lion `1×`），子类只需实现 `build()`。

**`__init__` 约定**：所有超参数通过 `**kwargs` 传入（由 config resolver 注入），支持
`param_groups` 列表进行分组配置；每条支持名字正则 `pattern` + 可选谓词 `min_ndim` /
`module_type`（详见下方 YAML）。

**极简范例**：[lighttrain/builtin_plugins/optim/wrappers.py:210](../../lighttrain/builtin_plugins/optim/wrappers.py#L210)

```python
@register("optimizer", "adamw")
class AdamWWrapper(OptimizerWrapperBase):
    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        self._check_unbuilt()
        groups = _split_param_groups(model, self.param_groups, self._kwargs)
        self.optimizer = torch.optim.AdamW(groups)
        self._built = True
        return self.optimizer
```

**YAML 示例**

```yaml
optimizer:
  name: adamw
  lr: 3e-4
  weight_decay: 0.1
  param_groups:
    - pattern: ".*bias|.*norm.*"
      weight_decay: 0.0
    # 可选谓词（名字正则之后追加过滤；默认不设=旧行为）：
    - pattern: "attn|mlp"
      min_ndim: 2          # 仅 ndim>=2 的权重矩阵（排除 1-D bias/norm）
      module_type: Linear  # 仅 nn.Linear 拥有的参数（按 type(module).__name__ 匹配）
      weight_decay: 0.1
```

> `min_ndim` / `module_type` 让**内置** param-group DSL 直接表达
> 「Linear 权重、ndim≥2」这类选层（如 GaLore），无需自定义 `build()`（issue #3）。

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `adamw` | `torch.optim.AdamW` 包装 | [builtin_plugins/optim/wrappers.py](../../lighttrain/builtin_plugins/optim/wrappers.py) |
| `lion` | Lion 优化器（纯 PyTorch 参考实现） | [builtin_plugins/optim/wrappers.py](../../lighttrain/builtin_plugins/optim/wrappers.py) |
| `cpu_offload` *(plugin)* | 优化器状态 CPU offload 包装 | [lighttrain/builtin_plugins/layer_offload/_optim_offload.py](../../lighttrain/builtin_plugins/layer_offload/_optim_offload.py) |

---

### 4.4 `scheduler`

**Protocol**：`SchedulerProtocol`（[lighttrain/protocols.py:158](../../lighttrain/protocols.py#L158)）

```python
step_per_batch: bool   # 总为 True：每个 optimizer step 调用一次

def step(self, *args, **kwargs) -> None: ...
def state_dict(self) -> dict[str, Any]: ...
def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...
```

**基类**：`SchedulerBase`（[lighttrain/optim/base.py](../../lighttrain/optim/base.py)）

子类只需实现 `_factor(step: int) -> float`（返回 lr 缩放因子）。

**极简范例**：[lighttrain/builtin_plugins/optim/schedulers.py:95](../../lighttrain/builtin_plugins/optim/schedulers.py#L95)

```python
@register("scheduler", "warmup_cosine")
class WarmupCosineScheduler(SchedulerBase):
    def __init__(self, optimizer=None, *, warmup_steps=100,
                 total_steps=1000, min_lr_ratio=0.1) -> None: ...

    def _factor(self, step: int) -> float:
        # linear warmup → cosine decay
        ...
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
```

**内置注册项**

| name | 行为 | 关键参数 |
|------|------|---------|
| `constant` | 恒定 lr | — |
| `linear` | 线性衰减 | `total_steps`, `end_factor`, `warmup_steps` |
| `warmup_cosine` | 线性预热 + 余弦衰减 | `warmup_steps`, `total_steps`, `min_lr_ratio` |
| `wsd` | Warmup→Stable→Decay 三阶段 | `warmup_steps`, `stable_steps`, `decay_steps`, `min_lr_ratio` |

---

### 4.5 `dataset`

**无正式 Protocol**，duck-typing：需实现 `__len__` 和 `__getitem__(idx: int)`（Map-style）。

**极简范例**：[lighttrain/builtin_plugins/data/core/datasets.py:16](../../lighttrain/builtin_plugins/data/core/datasets.py#L16)

```python
@register("dataset", "line_file_text")
class LineFileTextDataset:
    def __init__(self, path: str | Path, *, tokenizer: Any,
                 max_len: int = 256, encoding: str = "utf-8") -> None: ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict:
        # 返回 {"input_ids": [...], "attention_mask": [...], "labels": [...]}
        ...
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `line_file_text` | 每行一条文本，byte tokenizer | [builtin_plugins/data/core/datasets.py](../../lighttrain/builtin_plugins/data/core/datasets.py) |
| `preference_jsonl` | JSONL 偏好对（chosen/rejected input_ids + labels） | [builtin_plugins/data/core/datasets.py](../../lighttrain/builtin_plugins/data/core/datasets.py) |
| `artifact_joined` | 从 ArtifactStore 读取预计算张量 | [builtin_plugins/data/artifacts/joined_dataset.py](../../lighttrain/builtin_plugins/data/artifacts/joined_dataset.py) |

---

### 4.6 `processor`

**Protocol**：`ProcessorProtocol`（[lighttrain/protocols.py:195](../../lighttrain/protocols.py#L195)）

```python
modality: str   # "image" / "audio" / "text" / "video"

def __call__(self, inputs: Any, **kwargs: Any) -> Mapping[str, Any]: ...
```

**返回值要求**：dict，至少含 `"modality"` 键和 modality 对应的张量（如 `"pixel_values"` / `"input_features"`）。

**极简范例**：[lighttrain/builtin_plugins/data/processors/image.py:55](../../lighttrain/builtin_plugins/data/processors/image.py#L55)

```python
@register("processor", "simple_image")
class SimpleImageProcessor:
    modality = "image"

    def __init__(self, *, size=(224, 224), mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5)) -> None: ...

    def __call__(self, images, **_) -> dict:
        return {"pixel_values": np.stack(...), "modality": "image"}
```

**内置注册项**

| name | modality | 依赖 | 文件 |
|------|----------|------|------|
| `simple_image` | image | Pillow / numpy | [builtin_plugins/data/processors/image.py](../../lighttrain/builtin_plugins/data/processors/image.py) |
| `hf_image` | image | transformers | [builtin_plugins/data/processors/image.py](../../lighttrain/builtin_plugins/data/processors/image.py) |
| `mel_spectrogram` | audio | librosa / numpy | [builtin_plugins/data/processors/audio.py](../../lighttrain/builtin_plugins/data/processors/audio.py) |
| `hf_audio` | audio | transformers | [builtin_plugins/data/processors/audio.py](../../lighttrain/builtin_plugins/data/processors/audio.py) |
| `chat_template` | text | transformers tokenizer | [builtin_plugins/data/processors/text.py](../../lighttrain/builtin_plugins/data/processors/text.py) |
| `hf_text` | text | transformers | [builtin_plugins/data/processors/text.py](../../lighttrain/builtin_plugins/data/processors/text.py) |
| `frame_folder` | video | Pillow / numpy | [builtin_plugins/data/processors/video.py](../../lighttrain/builtin_plugins/data/processors/video.py) |
| `decord_video` | video | decord | [builtin_plugins/data/processors/video.py](../../lighttrain/builtin_plugins/data/processors/video.py) |

---

### 4.7 `collator`

**Protocol**：`CollatorProtocol`（[lighttrain/protocols.py:182](../../lighttrain/protocols.py#L182)）

```python
def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, Any]: ...
```

**返回值要求**：dict，值为 `torch.Tensor`，含 `input_ids` / `attention_mask` / `labels`（具体 key 由 model 决定）。

**极简范例**：[lighttrain/builtin_plugins/data/collators/text.py:18](../../lighttrain/builtin_plugins/data/collators/text.py#L18)

```python
@register("collator", "causal_lm")
class CausalLMCollator:
    def __init__(self, pad_id: int, max_len: int = 1024,
                 label_ignore: int = -100) -> None: ...

    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        # right-pad to longest in batch
        return {"input_ids": ..., "attention_mask": ..., "labels": ...}
```

**内置注册项**

| name | 说明 |
|------|------|
| `causal_lm` | 右填充至批内最长，`labels` 右移 | [builtin_plugins/data/collators/text.py](../../lighttrain/builtin_plugins/data/collators/text.py) |
| `preference` | 同时填充 chosen / rejected 序列对 | [builtin_plugins/data/collators/text.py](../../lighttrain/builtin_plugins/data/collators/text.py) |
| `multimodal` | 多模态字段合并（文本 + 图像/音频等） | [builtin_plugins/data/collators/multimodal.py](../../lighttrain/builtin_plugins/data/collators/multimodal.py) |
| `image` | 堆叠 pixel_values (B,C,H,W) + 整数 labels (B,)（监督视觉） | [builtin_plugins/data/collators/image.py](../../lighttrain/builtin_plugins/data/collators/image.py) |

---

### 4.8 `sampler`

**Protocol**：`SamplerProtocol`（[lighttrain/protocols.py:187](../../lighttrain/protocols.py#L187)）

```python
def __iter__(self) -> Iterable[int]: ...
def __len__(self) -> int: ...
def state_dict(self) -> dict[str, Any]: ...
def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...
```

**极简范例**：[lighttrain/builtin_plugins/data/core/samplers.py:31](../../lighttrain/builtin_plugins/data/core/samplers.py#L31)

```python
@register("sampler", "shuffle")
class ShuffleSampler:
    def __init__(self, dataset: Sized, *, seed: int = 0) -> None: ...
    def __iter__(self): ...   # 每个 epoch 固定种子随机排列
    def __len__(self) -> int: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...
```

**内置注册项**

| name | 说明 |
|------|------|
| `sequential` | 顺序采样，支持 state_dict | [builtin_plugins/data/core/samplers.py](../../lighttrain/builtin_plugins/data/core/samplers.py) |
| `shuffle` | 每 epoch 确定性随机打乱（seed + epoch） | [builtin_plugins/data/core/samplers.py](../../lighttrain/builtin_plugins/data/core/samplers.py) |
| `length_grouped` | 按长度分桶以减少 padding | [builtin_plugins/data/samplers/length_grouped.py](../../lighttrain/builtin_plugins/data/samplers/length_grouped.py) |
| `curriculum` | 课程学习采样（按 step 调难度带） | [builtin_plugins/data/samplers/curriculum.py](../../lighttrain/builtin_plugins/data/samplers/curriculum.py) |
| `stateful_resumable` | 可精确恢复的有状态采样 | [builtin_plugins/data/samplers/stateful_resumable.py](../../lighttrain/builtin_plugins/data/samplers/stateful_resumable.py) |

---

### 4.9 `tokenizer`

**Protocol**：`TokenizerProtocol`（[lighttrain/protocols.py:176](../../lighttrain/protocols.py#L176)）

```python
def encode(self, text: str, **kwargs: Any) -> list[int]: ...
def decode(self, ids: list[int], **kwargs: Any) -> str: ...
```

**极简范例**：[lighttrain/builtin_plugins/data/core/tokenizers.py:22](../../lighttrain/builtin_plugins/data/core/tokenizers.py#L22)

```python
@register("tokenizer", "byte")
class ByteTokenizer:
    pad_id = 256;  bos_id = 257;  eos_id = 258;  unk_id = 259;  vocab_size = 260

    def encode(self, text: str, **_) -> list[int]: ...  # UTF-8 bytes
    def decode(self, ids: list[int], **_) -> str: ...
```

**内置注册项**

| name | 说明 |
|------|------|
| `byte` | UTF-8 字节级分词，vocab_size=260 | [builtin_plugins/data/core/tokenizers.py](../../lighttrain/builtin_plugins/data/core/tokenizers.py) |

---

### 4.10 `data_module`

**Protocol**：`DataModuleProtocol`（[lighttrain/protocols.py:167](../../lighttrain/protocols.py#L167)）

```python
def train_loader(self) -> Iterable[Any]: ...
def val_loader(self) -> Iterable[Any] | None: ...
def predict_loader(self) -> Iterable[Any] | None: ...
def state_dict(self) -> dict[str, Any]: ...
def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `simple` | 单数据集模块，封装 dataset + collator + sampler + DataLoader | [builtin_plugins/data/core/_module.py](../../lighttrain/builtin_plugins/data/core/_module.py) |
| `prep_graph` | 从 PrepGraph DAG 输出中加载数据 | [builtin_plugins/data/core/_prep_module.py](../../lighttrain/builtin_plugins/data/core/_prep_module.py) |

---

### 4.11 `trainer`

**Protocol**：`TrainerProtocol`（[lighttrain/protocols.py:319](../../lighttrain/protocols.py#L319)）

**扁平基类**：`Trainer`（[lighttrain/trainers/base.py](../../lighttrain/trainers/base.py)）— 拥有共享状态
（engine/optimizer/scheduler/logger/ckpt_manager/callbacks/`bus`/`ctx`、`models`/`optimizers`
集、BUG-1 resume guard），并有**具体** `fit()`。90% 场景（causal-LM 预训练/SFT）无需子类体——
`pretrain` 就是裸 `Trainer`。

```python
class Trainer:
    def fit(self, *, steps=None):           # 具体：run_train_loop 组合 produce_batch/forward_loss
        ...
    def produce_batch(self, raw):           # 默认 move-to-device；RL/OPD 重写为 rollout
        ...
    def forward_loss(self, batch):          # 默认 None → 走 engine.step（pretrain no-op）；
        ...                                 #   自定义范式返回 (loss, metrics) → 走 apply_update
    def before_step(self, batch): ...       # 可选：GAE / 组优势预计算
```

**公共原语**（可不经 `Trainer` 直接调，re-entrant）：
- `run_train_loop(trainer, *, target_steps)`（[trainers/_primitives.py](../../lighttrain/trainers/_primitives.py)）—— epoch rollover + 信号 + log/ckpt/eval + crash bundle。
- `apply_update(*, loss, model, optimizer, ctx, micro_state, ...)`（[update_rules/_primitives.py](../../lighttrain/engine/update_rules/_primitives.py)）—— backward/clip/step/sched 半边。
- `forward_with_activations(model, batch, *, layers=None)`（[trainers/_primitives.py](../../lighttrain/trainers/_primitives.py)）—— 层粒度激活捕获。

**写一个新范式**：重写 `produce_batch` / `forward_loss`（多模型经 `self.models[...]`、多优化器经
`self.optimizers[...]`），或写一个调上述原语的短 `fit()`（如逐层蒸馏的「loop over training loops」）。
`loss:` 经 `ctx.loss_fn` 到达；base 永不覆盖 recipe 提供的 loss。

**内置 `_step` / `train_step`**：`train_step` 是公共入口，默认委派到具体 `_step` → `_run_step`
（`forward_loss` 决定走 engine 还是 apply_update）。RL/preference/reward_model 仍重写 `_step`。

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `pretrain` | 标准 Causal-LM 预训练 | [builtin_plugins/trainers/pretrain.py](../../lighttrain/builtin_plugins/trainers/pretrain.py) |
| `preference` | 离线偏好训练（算法由 `loss:` 选：dpo/ipo/simpo/orpo/kto） | [builtin_plugins/trainers/_preference_base.py](../../lighttrain/builtin_plugins/trainers/_preference_base.py) |
| `grpo` | GRPO（在线 RL） | [builtin_plugins/trainers/grpo.py](../../lighttrain/builtin_plugins/trainers/grpo.py) |
| `ppo` | PPO（在线 RL） | [builtin_plugins/trainers/ppo.py](../../lighttrain/builtin_plugins/trainers/ppo.py) |
| `reward_model` | Reward Model 训练（Bradley-Terry） | [builtin_plugins/trainers/rm.py](../../lighttrain/builtin_plugins/trainers/rm.py) |

> 偏好算法是 `loss:` seam，不是单独的 trainer：`trainer: {name: preference}` +
> `loss: {name: dpo, beta: 0.1}`（dpo/ipo/simpo/orpo/kto）。扁平 `Trainer` 的 `fit()`
> 由公共原语 `run_train_loop` / `apply_update` / `forward_with_activations` 组合；
> 新范式重写 `produce_batch` / `forward_loss`（可选 `before_step`）两个 seam，或写一个
> 调这些原语的短 `fit()`。多模型/多优化器见 `models:` / `optimizers:`。

---

### 4.12 `engine`

**Protocol**：`EngineProtocol`（[lighttrain/protocols.py:240](../../lighttrain/protocols.py#L240)）

```python
def step(self, batch: Mapping[str, Any], ctx: Any) -> dict[str, Any]: ...
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `standard` | 委托给 `update_rule.step(model, batch, ctx)` | [builtin_plugins/engine/standard.py](../../lighttrain/builtin_plugins/engine/standard.py) |

---

### 4.13 `update_rule`

**Protocol**：`UpdateRuleProtocol`（[lighttrain/protocols.py:245](../../lighttrain/protocols.py#L245)）

```python
def setup(self, model: Any, sample: Any) -> None: ...
def step(self, model: Any, batch: Mapping[str, Any], ctx: Any) -> dict[str, Any]: ...
def state_dict(self) -> dict[str, Any]: ...
def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...
```

**step 返回值要求**：dict，必须含 `"loss"` key，可选 `"grad_norm"` 等。

**极简范例**：[lighttrain/builtin_plugins/engine/update_rules/standard.py:45](../../lighttrain/builtin_plugins/engine/update_rules/standard.py#L45)

```python
@register("update_rule", "standard")
class StandardUpdateRule:
    def __init__(self, *, grad_clip=1.0, accumulate_grad_batches=1,
                 max_retries=3) -> None: ...

    def setup(self, model, sample) -> None: ...
    def step(self, model, batch, ctx) -> dict[str, Any]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, sd) -> None: ...
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `standard` | 前向+反向+clip grad+optimizer step，支持梯度累积 | [builtin_plugins/engine/update_rules/standard.py](../../lighttrain/builtin_plugins/engine/update_rules/standard.py) |
| `sam` | Sharpness-Aware Minimization（两次前向） | [builtin_plugins/engine/update_rules/sam.py](../../lighttrain/builtin_plugins/engine/update_rules/sam.py) |
| `mezo` | Memory-Efficient Zeroth-Order Optimization | [builtin_plugins/engine/update_rules/mezo.py](../../lighttrain/builtin_plugins/engine/update_rules/mezo.py) |
| `rl` | RL 更新规则（PPO/GRPO 内部用） | [builtin_plugins/engine/update_rules/rl.py](../../lighttrain/builtin_plugins/engine/update_rules/rl.py) |
| `forward_forward` *(plugin)* | Forward-Forward 算法 | [builtin_plugins/engine/update_rules/forward_forward/](../../lighttrain/builtin_plugins/engine/update_rules/forward_forward/__init__.py) |
| `pcn` *(plugin)* | 预测编码网络 | [builtin_plugins/engine/update_rules/pcn/](../../lighttrain/builtin_plugins/engine/update_rules/pcn/__init__.py) |
| `dfa` *(plugin)* | Direct Feedback Alignment | [builtin_plugins/engine/update_rules/dfa/](../../lighttrain/builtin_plugins/engine/update_rules/dfa/__init__.py) |

---

### 4.14 `callback`

**Protocol**：`CallbackProtocol`（[lighttrain/protocols.py:311](../../lighttrain/protocols.py#L311)）

**所有事件方法均为可选**（`EventBus` 通过 `getattr` 检查）。完整事件列表见 `CALLBACK_EVENTS`（46 个）。

**常用事件**

```python
def on_train_start(self, **kwargs) -> None: ...
def on_step_begin(self, **kwargs) -> None: ...
def on_step_end(self, *, batch, metrics, **kwargs) -> None: ...
def on_optimizer_step_post(self, *, model, **kwargs) -> None: ...
def on_eval_begin(self, *, model, **kwargs) -> None: ...
def on_eval_end(self, *, model, **kwargs) -> None: ...
def on_train_end(self, **kwargs) -> None: ...
```

**极简范例**：[lighttrain/builtin_plugins/callbacks/builtins/ema.py:10](../../lighttrain/builtin_plugins/callbacks/builtins/ema.py#L10)

```python
@register("callback", "ema")
class EMACallback:
    def __init__(self, decay: float = 0.999) -> None: ...

    def on_optimizer_step_post(self, *, model=None, **_) -> None:
        # 更新 shadow 参数
        ...

    def on_eval_begin(self, *, model=None, **_) -> None:
        # 将 EMA 权重换入
        ...

    def on_eval_end(self, *, model=None, **_) -> None:
        # 还原原始权重
        ...
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `ema` | 指数移动平均影子权重 | [builtin_plugins/callbacks/builtins/ema.py](../../lighttrain/builtin_plugins/callbacks/builtins/ema.py) |
| `target_ema` | 各优化器步后调用模型自身 `update_ema()`（JEPA 等内部 EMA 目标；无该方法的模型为 no-op） | [builtin_plugins/callbacks/builtins/target_ema.py](../../lighttrain/builtin_plugins/callbacks/builtins/target_ema.py) |
| `throughput` | 滚动窗口 tokens/sec + samples/sec 统计 | [builtin_plugins/callbacks/builtins/throughput.py](../../lighttrain/builtin_plugins/callbacks/builtins/throughput.py) |
| `best_ckpt` | 按 metric 保留最优 checkpoint | [builtin_plugins/callbacks/builtins/best_ckpt.py](../../lighttrain/builtin_plugins/callbacks/builtins/best_ckpt.py) |
| `early_stop` | 早停（patience + metric monitor） | [builtin_plugins/callbacks/builtins/early_stop.py](../../lighttrain/builtin_plugins/callbacks/builtins/early_stop.py) |
| `nan_skip` | 检测到 NaN loss 跳过该 step | [builtin_plugins/callbacks/builtins/nan_skip.py](../../lighttrain/builtin_plugins/callbacks/builtins/nan_skip.py) |
| `frozen_step` | 前 N 步冻结指定模块参数 | [builtin_plugins/callbacks/builtins/frozen_step.py](../../lighttrain/builtin_plugins/callbacks/builtins/frozen_step.py) |
| `lineage_recorder` | 记录训练 lineage 元信息 | [builtin_plugins/callbacks/builtins/lineage_recorder.py](../../lighttrain/builtin_plugins/callbacks/builtins/lineage_recorder.py) |
| `invariants` | 在每步运行一组 invariant 检查 | [builtin_plugins/callbacks/invariants/callback.py](../../lighttrain/builtin_plugins/callbacks/invariants/callback.py) |
| `dynamic_artifact` | 训练中动态收集模型输出张量 | [builtin_plugins/data/artifacts/dynamic_producer.py](../../lighttrain/builtin_plugins/data/artifacts/dynamic_producer.py) |
| `dead_neuron` | 检测死亡神经元比例 | [builtin_plugins/observability/diagnostics/dead_neuron.py](../../lighttrain/builtin_plugins/observability/diagnostics/dead_neuron.py) |
| `grad_flow` | 梯度流可视化（各层 grad norm） | [builtin_plugins/observability/diagnostics/grad_flow.py](../../lighttrain/builtin_plugins/observability/diagnostics/grad_flow.py) |
| `loss_attribution` | loss 逐层归因分析 | [builtin_plugins/observability/diagnostics/loss_attribution.py](../../lighttrain/builtin_plugins/observability/diagnostics/loss_attribution.py) |
| `nan_hunter` | NaN 溯源 hook（定位到具体层） | [builtin_plugins/observability/diagnostics/nan_hunter.py](../../lighttrain/builtin_plugins/observability/diagnostics/nan_hunter.py) |
| `sample_preview` | 训练中采样输出预览 | [builtin_plugins/observability/diagnostics/sample_preview.py](../../lighttrain/builtin_plugins/observability/diagnostics/sample_preview.py) |
| `file_signals` | 文件信号控制训练（动态调 lr / 暂停） | [builtin_plugins/callbacks/realtime_control/file_signals.py](../../lighttrain/builtin_plugins/callbacks/realtime_control/file_signals.py) |

---

### 4.15 `logger`

**Protocol**：`LoggerProtocol`（[lighttrain/protocols.py:211](../../lighttrain/protocols.py#L211)）

```python
def log_scalars(self, scalars: Mapping[str, float], step: int) -> None: ...
def log_histograms(self, hists: Mapping[str, Any], step: int) -> None: ...
def log_text(self, text: str, step: int) -> None: ...
def log_artifact(self, path: str, name: str | None = None) -> None: ...
def flush(self) -> None: ...
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `console` | Rich 单行滚动输出 | [builtin_plugins/callbacks/logging/console.py](../../lighttrain/builtin_plugins/callbacks/logging/console.py) |
| `jsonl` | JSON Lines 日志文件 | [builtin_plugins/callbacks/logging/jsonl.py](../../lighttrain/builtin_plugins/callbacks/logging/jsonl.py) |
| `tensorboard` / `tb` | TensorBoard SummaryWriter | [builtin_plugins/callbacks/logging/tb.py](../../lighttrain/builtin_plugins/callbacks/logging/tb.py) |

---

### 4.16 `objective`

**Protocol**：（非 `ObjectiveProtocol`，见下方说明）

实际上，已注册的 objective 类遵循 **ObjectiveProfile** 接口（在 [lighttrain/optim/architectures/profile.py](../../lighttrain/optim/architectures/profile.py) 中定义）：

```python
loss_family: str   # "next_token" / "mlm" / "diffusion" / "flow_matching" / "jepa"

def prepare_batch(self, batch: dict, *, step: int, device: Any) -> dict: ...
def __call__(self, outputs: ModelOutput, batch: dict, ctx: LossContext) -> dict: ...
```

**极简范例**：[lighttrain/builtin_plugins/objectives/next_token.py:16](../../lighttrain/builtin_plugins/objectives/next_token.py#L16)

```python
@register("objective", "next_token")
class NextTokenObjective:
    loss_family: str = "next_token"

    def prepare_batch(self, batch, *, step, device) -> dict:
        return batch   # next-token 无需额外变换

    def __call__(self, outputs: ModelOutput, batch: dict,
                 ctx: LossContext) -> dict:
        ctx.loss_family = self.loss_family
        return self._loss_fn(outputs, batch, ctx)   # → {"loss": tensor}
```

**内置注册项**

| name | loss_family | 文件 |
|------|-------------|------|
| `next_token` *(plugin)* | `next_token` | [lighttrain/builtin_plugins/objectives/next_token.py](../../lighttrain/builtin_plugins/objectives/next_token.py) |
| `masked_denoising` *(plugin)* | `masked_denoising` | [lighttrain/builtin_plugins/objectives/masked_denoising.py](../../lighttrain/builtin_plugins/objectives/masked_denoising.py) |
| `diffusion` *(plugin)* | `denoising` | [lighttrain/builtin_plugins/objectives/diffusion.py](../../lighttrain/builtin_plugins/objectives/diffusion.py) |
| `flow_matching` *(plugin)* | `flow_matching` | [lighttrain/builtin_plugins/objectives/flow_matching.py](../../lighttrain/builtin_plugins/objectives/flow_matching.py) |
| `jepa` *(plugin)* | `jepa` | [lighttrain/builtin_plugins/objectives/jepa.py](../../lighttrain/builtin_plugins/objectives/jepa.py) |

---

### 4.16b `architecture`

把 `trainer.arch_profile` 字符串解析为 `ArchitectureProfile`（block / embedding / head 缝 + 有状态 reset）。

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `transformer` | 标准 Transformer 架构 profile（core） | [builtin_plugins/optim/architectures/transformer.py](../../lighttrain/builtin_plugins/optim/architectures/transformer.py) |
| `rwkv` *(plugin)* | RWKV 架构 profile | [builtin_plugins/optim/architectures/rwkv/](../../lighttrain/builtin_plugins/optim/architectures/rwkv/__init__.py) |

---

### 4.17 `judge`

**Protocol**：`JudgeProtocol`（[lighttrain/protocols.py:343](../../lighttrain/protocols.py#L343)）

```python
def score(self, items: Iterable[Any], ctx: Any | None = None) -> list[Any]: ...
```

**内置注册项**

| name | 说明 | `reward_kind` | 文件 |
|------|------|---------------|------|
| `verifier` *(plugin)* | 规则验证器（格式 / 数学正确性等） | `pointwise` | [lighttrain/builtin_plugins/judges/judge.py](../../lighttrain/builtin_plugins/judges/judge.py) |
| `pairwise_llm` *(plugin)* | 基于 LLM 的成对打分 | `pairwise` | [lighttrain/builtin_plugins/judges/judge.py](../../lighttrain/builtin_plugins/judges/judge.py) |

作为 RL reward 用时，judge 的 `reward_kind` 决定用哪个 `reward_adapter`（§4.27c）。`pointwise`
有内置适配器；`pairwise` 需自行注册一个 `pairwise` 适配器（把成对胜负折成 pointwise reward）。

---

### 4.18 `generation_strategy`

**Protocol**：`GenerationStrategyProtocol`（[lighttrain/protocols.py:331](../../lighttrain/protocols.py#L331)）

```python
def generate(self, model: Any, prompts: Any,
             sampling: Mapping[str, Any],
             scorer: Any | None = None,
             ctx: Any | None = None) -> Any: ...
```

当前无内置注册项。

---

### 4.19 `environment`

**Protocol**：`EnvironmentProtocol`（[lighttrain/protocols.py:348](../../lighttrain/protocols.py#L348)）

```python
def reset(self, ctx: Any | None = None) -> Any: ...
def step(self, action: Any) -> Any: ...
```

当前无内置注册项。

---

### 4.20 `retriever`

**Protocol**：`RetrieverProtocol`（[lighttrain/protocols.py:354](../../lighttrain/protocols.py#L354)）

```python
def index(self, corpus: Any, ctx: Any | None = None) -> Any: ...
def query(self, queries: Any, k: int, ctx: Any | None = None) -> Any: ...
```

当前无内置注册项。

---

### 4.21 `chunker`

**Protocol**：`ChunkerProtocol`（[lighttrain/protocols.py:360](../../lighttrain/protocols.py#L360)）

```python
def chunk(self, rows: Iterable[Any], ctx: Any | None = None) -> Iterable[Any]: ...
```

当前无内置注册项。

---

### 4.22 `probe`

**Protocol**：`ProbeProtocol`（[lighttrain/protocols.py:365](../../lighttrain/protocols.py#L365)）

```python
def attach(self, model: Any, layers: Iterable[str], ctx: Any | None = None) -> Any: ...
def compute(self, activations: Any) -> Any: ...
```

当前无内置注册项。

---

### 4.23 `artifact_producer`

**Protocol**：`ArtifactProducerProtocol`（[lighttrain/protocols.py:401](../../lighttrain/protocols.py#L401)）

```python
def prepare(self, cfg: Mapping[str, Any] | None = None) -> None: ...
def produce(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]: ...
def finalize(self) -> Path: ...
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `model_forward` | eval 模式前向，捕获指定层的输出张量 | [builtin_plugins/data/artifacts/producer.py](../../lighttrain/builtin_plugins/data/artifacts/producer.py) |

---

### 4.24 `artifact_store`

**Protocol**：`ArtifactStoreProtocol`（[lighttrain/protocols.py:408](../../lighttrain/protocols.py#L408)）

```python
def put(self, sample_id: str, tensors_dict: Mapping[str, torch.Tensor]) -> None: ...
def get(self, sample_id: str) -> dict[str, torch.Tensor]: ...
def contains(self, sample_id: str) -> bool: ...
def iter_keys(self) -> Iterable[str]: ...
```

**内置注册项**

| name | 存储格式 | 文件 |
|------|---------|------|
| `safetensors-shards` | safetensors 分片，变长张量 | [builtin_plugins/data/artifacts/store.py](../../lighttrain/builtin_plugins/data/artifacts/store.py) |
| `memmap-fixed` | numpy memmap + header.json，定长张量 | [builtin_plugins/data/artifacts/store.py](../../lighttrain/builtin_plugins/data/artifacts/store.py) |
| `parquet-rows` | PyArrow / Parquet 行存储 | [builtin_plugins/data/artifacts/store.py](../../lighttrain/builtin_plugins/data/artifacts/store.py) |

---

### 4.25 `prep_node`

**Protocol**：`PrepNodeProtocol`（[lighttrain/protocols.py:416](../../lighttrain/protocols.py#L416)）

**基类**：`PrepNode`（[lighttrain/data/prepgraph/node.py:67](../../lighttrain/data/prepgraph/node.py#L67)）— **推荐继承**

```python
class PrepNode:
    kind: str          # 子类设置为类属性，与注册名一致
    schema_kind: str   # 输出的数据 schema（"rows" / "tokenized_rows" / ...）

    def __init__(self, *, name: str, inputs: list[str] | None = None,
                 config: Mapping[str, Any] | None = None,
                 device_hint: str = "any") -> None: ...

    # 必须实现：
    def run(self, ctx: RunContext) -> NodeResult: ...

    # 可选重写：
    def estimate(self, ctx: RunContext) -> NodeEstimate: ...
```

**`RunContext` / `NodeResult`** 定义在 [lighttrain/data/prepgraph/node.py](../../lighttrain/data/prepgraph/node.py)。

**内置注册项**

| name | 作用 | 文件 |
|------|------|------|
| `load` | 从磁盘加载原始文本/JSON 行 | [builtin_plugins/data/prepgraph/nodes/load.py](../../lighttrain/builtin_plugins/data/prepgraph/nodes/load.py) |
| `tokenize` | 对行数据分词 | [builtin_plugins/data/prepgraph/nodes/tokenize.py](../../lighttrain/builtin_plugins/data/prepgraph/nodes/tokenize.py) |
| `chunk` | 固定长度分块（含 stride） | [builtin_plugins/data/prepgraph/nodes/chunk.py](../../lighttrain/builtin_plugins/data/prepgraph/nodes/chunk.py) |
| `pack` | 多文档打包至固定上下文窗口，`strategy: concat_chunk`（默认，padding-free 基线）/ `next_fit`（greedy-pad-flush）/ `best_fit`（BFD，opt-in），各自吐 `truncation_rate`/`token_utilization` 等指标 | [builtin_plugins/data/prepgraph/nodes/pack.py](../../lighttrain/builtin_plugins/data/prepgraph/nodes/pack.py) |
| `mix` | 多 upstream 流按比例混合 | [builtin_plugins/data/prepgraph/nodes/mix.py](../../lighttrain/builtin_plugins/data/prepgraph/nodes/mix.py) |
| `join` | 拼接多个 upstream 数据集 | [builtin_plugins/data/prepgraph/nodes/join.py](../../lighttrain/builtin_plugins/data/prepgraph/nodes/join.py) |
| `index` | 为随机访问建索引 | [builtin_plugins/data/prepgraph/nodes/index.py](../../lighttrain/builtin_plugins/data/prepgraph/nodes/index.py) |
| `validate` | 模式验证（schema 检查） | [builtin_plugins/data/prepgraph/nodes/validate.py](../../lighttrain/builtin_plugins/data/prepgraph/nodes/validate.py) |
| `materialize` | 将流式数据写入磁盘分片 | [builtin_plugins/data/prepgraph/nodes/materialize.py](../../lighttrain/builtin_plugins/data/prepgraph/nodes/materialize.py) |

---

### 4.26 `invariant`

**无 Protocol**，直接注册一个 **callable**，签名：

```python
def my_invariant(*, loss=None, batch=None, metrics=None, model=None,
                 step=None, **kwargs) -> bool:
    """返回 True 表示通过，False 表示违反。"""
    ...
```

**极简范例**：[lighttrain/builtin_plugins/callbacks/invariants/builtins.py:29](../../lighttrain/builtin_plugins/callbacks/invariants/builtins.py#L29)

```python
@register("invariant", "loss_finite")
def loss_finite(*, loss: Any = None, **_) -> bool:
    if loss is None:
        return True
    if isinstance(loss, torch.Tensor):
        return bool(torch.isfinite(loss).all().item())
    return bool(loss == loss) and abs(float(loss)) != float("inf")
```

**内置注册项**

| name | 检查内容 | 文件 |
|------|---------|------|
| `loss_finite` | loss 无 NaN / Inf | [builtin_plugins/callbacks/invariants/builtins.py](../../lighttrain/builtin_plugins/callbacks/invariants/builtins.py) |
| `grad_norm_bounded` | `metrics["grad_norm"] < max`（默认 1000） | [builtin_plugins/callbacks/invariants/builtins.py](../../lighttrain/builtin_plugins/callbacks/invariants/builtins.py) |
| `lr_nonneg` | 当前 lr ≥ 0 | [builtin_plugins/callbacks/invariants/builtins.py](../../lighttrain/builtin_plugins/callbacks/invariants/builtins.py) |
| `label_mask_nonzero` | 至少一个 label 位置非 ignore_index | [builtin_plugins/callbacks/invariants/builtins.py](../../lighttrain/builtin_plugins/callbacks/invariants/builtins.py) |
| `param_count_stable` | 参数量未在步间变化 | [builtin_plugins/callbacks/invariants/builtins.py](../../lighttrain/builtin_plugins/callbacks/invariants/builtins.py) |
| `dtype_stable` | 参数 dtype 未发生意外转换 | [builtin_plugins/callbacks/invariants/builtins.py](../../lighttrain/builtin_plugins/callbacks/invariants/builtins.py) |
| `batch_nonempty` | batch 非空 | [builtin_plugins/callbacks/invariants/builtins.py](../../lighttrain/builtin_plugins/callbacks/invariants/builtins.py) |
| `regression_gate` | 当前 loss ≤ 历史基准（防退化） | [builtin_plugins/callbacks/invariants/regression_gate.py](../../lighttrain/builtin_plugins/callbacks/invariants/regression_gate.py) |

---

### 4.27 `rl_backend`

**无 Protocol**，需实现 `generate` 方法：

```python
def generate(self, model: Any, input_ids: torch.Tensor, **kwargs) -> torch.Tensor: ...
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `hf_generate` | 使用 HF `model.generate()` 采集 rollout（暴露 `temperature`/`top_p`/`do_sample`/`max_new_tokens`/`num_return_sequences`） | [builtin_plugins/rl/rollout.py](../../lighttrain/builtin_plugins/rl/rollout.py) |
| `vllm` *(plugin)* | vLLM 高吞吐 rollout 后端（opt-in） | [lighttrain/builtin_plugins/rl/backends/vllm/](../../lighttrain/builtin_plugins/rl/backends/vllm/__init__.py) |

ppo/grpo 经 `rollout_backend:`（默认 `hf_generate`）从注册表解析后端并转发采样 knob，
不再内联构造。

### 4.27b `value_head`

价值/奖励头。单个参数化 `LinearValueHead(hidden_size, *, bias, zero_init, reduction)`
覆盖两种语义：PPO critic（`reduction="per_token"`，零初始化，逐 token V(s)）与 RM 打分头
（`reduction="last"`，默认初始化，读末 token → 标量）。ppo/rm 默认各用自己的配置；recipe
可经 `value_head: {name: linear, ...}` 覆盖。

| name | 说明 | 文件 |
|------|------|------|
| `linear` | 线性价值/奖励头（per-token 或 last-token） | [builtin_plugins/rl/value_heads.py](../../lighttrain/builtin_plugins/rl/value_heads.py) |

### 4.27c `reward_adapter`

把 judge 包成 RL `reward_fn(prompt_ids, response_ids) -> list[float]`。judge 声明
`reward_kind`（默认 `"pointwise"`），runtime 据此解析适配器（recipe `reward_adapter:` 可覆盖）；
任何 pointwise judge 都能背 RL reward。

| name | 说明 | 文件 |
|------|------|------|
| `pointwise` | 逐 (prompt, response) 打分（verifier 类） | [builtin_plugins/rl/reward_adapters.py](../../lighttrain/builtin_plugins/rl/reward_adapters.py) |

### 4.28 `grad_sync_strategy`

**源文件**：[lighttrain/distributed/_protocols.py](../../lighttrain/distributed/_protocols.py)

```python
class GradSyncStrategy(Protocol):
    def prepare(
        self,
        model: nn.Module,
        optimizer_factory: Callable[[nn.Module], Any],
        loader: Any,
        parallel_ctx: "ParallelContext",
        *,
        device: torch.device,
    ) -> tuple[nn.Module, Any, Any]: ...
    # Returns (wrapped_model, optimizer, loader)

    def accumulate(self, model: nn.Module) -> ContextManager: ...
    # Context manager suppressing gradient sync (no_sync) during accumulation steps

    def backward(self, loss: torch.Tensor, model: nn.Module) -> None: ...

    def clip_grad_norm(
        self, model: nn.Module, max_norm: float, parallel_ctx: "ParallelContext"
    ) -> float: ...

    def optimizer_step(self, optimizer: Any, model: nn.Module) -> None: ...

    def unwrap_model(self, model: nn.Module) -> nn.Module: ...
    # Returns the underlying unwrapped model (for checkpoint saving)

    def save_checkpoint(self, model: nn.Module, path: Path, parallel_ctx: "ParallelContext") -> None: ...
    def load_checkpoint(self, model: nn.Module, path: Path, parallel_ctx: "ParallelContext") -> None: ...
```

**内置注册项**

| name | 说明 | 文件 |
|------|------|------|
| `noop` | 单卡直通，无分布式开销 | [lighttrain/builtin_plugins/distributed/_noop.py](../../lighttrain/builtin_plugins/distributed/_noop.py) |
| `ddp` | `torch.nn.parallel.DistributedDataParallel` | [lighttrain/builtin_plugins/distributed/strategies/ddp.py](../../lighttrain/builtin_plugins/distributed/strategies/ddp.py) |
| `fsdp` | `torch.distributed.fsdp.FullyShardedDataParallel` | [lighttrain/builtin_plugins/distributed/strategies/fsdp.py](../../lighttrain/builtin_plugins/distributed/strategies/fsdp.py) |
| `deepspeed` | DeepSpeed ZeRO-1/2/3 engine | [lighttrain/builtin_plugins/distributed/strategies/zero.py](../../lighttrain/builtin_plugins/distributed/strategies/zero.py) |

---

## 5. 异常说明

```python
from lighttrain.registry import RegistryError, RegistryConflictError, \
    UnknownCategoryError, NotRegisteredError
```

**源文件**：[lighttrain/registry/_exceptions.py](../../lighttrain/registry/_exceptions.py)

| 异常 | 触发场景 |
|------|---------|
| `RegistryConflictError` | 向同一 `(category, name)` 二次注册，且未传 `force=True` |
| `UnknownCategoryError` | 使用了 `KNOWN_CATEGORIES` 之外的 category，且未先调 `register_category()` |
| `NotRegisteredError` | `get(category, name)` 时该 name 不存在 |
| `RegistryError` | 上述三类的公共基类，可用于统一捕获 |
