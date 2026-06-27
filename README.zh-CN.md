# lighttrain

> [English](README.md) · 中文

面向快速研究迭代的 PyTorch 语言模型训练框架：预训练、SFT、偏好学习、在线 RL、蒸馏。
核心——Registry、Config、Engine、UpdateRule、Trainer、EventBus、PrepGraph——小到可以
通读；研究级扩展（PEFT、其他架构、扫描、分布式）按需启用。

设计目标：**registry-first**、**failure-first**、**plugin-clean**、
**lab-friendly**、**audit-ready**。

> 状态：测试阶段。分布式**仅支持数据并行**（DDP / FSDP / DeepSpeed ZeRO）；
> DDP、FSDP、DeepSpeed ZeRO-2 已在真实单机多卡（NCCL）验证，但**未**在多机 GPU
> 集群验证——生产环境自行评估风险。（张量 / 流水线 / 专家 / 序列并行已移除。）
> 测试套件约 82K 行 / 4400+ 测试，含经变异测试验证的对抗性回归测试。

## 安装

```bash
git clone <this-repo> lighttrain && cd lighttrain
pip install -e .
pip install -e ".[peft]"          # 可选：LoRA / IA³ / AdaLoRA
pip install -e ".[peft,quant]"    # 可选：+ bitsandbytes 4-bit（Linux+CUDA）
```

## 快速开始

```bash
lighttrain init my_project        # 脚手架：带注释、可直接跑的 recipe
cd my_project
lighttrain dry-run -c cfg.yaml    # 解析并打印配置（不训练）
lighttrain train   -c cfg.yaml ++trainer.max_steps=50   # 50 步冒烟
```

生成的 `cfg.yaml` 放一个 `corpus.txt`（每行一条样本）后即可跑，并带大量教程式注释——
取消注释可选块（`models:`、`parallel:`、`prep_graph:`、PEFT 等）即可扩展。→ [快速开始](docs/guide/getting-started.zh-CN.md)

## 架构

1. **Registry** —— 在固定类别集上做短名 → 类解析。
2. **Config** —— OmegaConf + Pydantic v2；模型是配置组（`model_profiles:` +
   `model: <名字>`）。
3. **Engine + UpdateRule** —— engine 拥有 accelerator，把每步数学
   （前向/反向/clip/step）下放给可替换的 `UpdateRule`，于是改训练数学无需动循环。
   扁平 `Trainer` 组合公共原语（`run_train_loop`、`apply_update`、
   `forward_with_activations`）。
4. **EventBus** —— 46 个生命周期事件；单 callback 异常隔离；结果聚合为 `Signal`
   （`STOP_TRAINING > RETRY_STEP > SKIP_STEP > CONTINUE`）。
5. **PrepGraph** —— 内容寻址的数据预处理 DAG；按 config + 代码 + schema + 上游的
   fingerprint 缓存。

→ [架构](docs/concepts/architecture.zh-CN.md)

## 示例：训练，然后分叉与恢复

```bash
lighttrain train -c examples/references/recipes/pretrain_causal.yaml
lighttrain fork  --from runs/<...>/checkpoints/step_500 -c examples/references/recipes/finetune.yaml
lighttrain resume --run runs/<...>
```

## 示例：一个 recipe 表达新范式

通常写新的`loss:` ，而非新 trainer。在线 RL：

```yaml
trainer: { name: ppo, rollout_steps: 32, rollout_backend: hf_generate }
loss:    { name: ppo_surrogate, clip_eps: 0.2 }
judge:   { name: verifier, verify_pattern: "\\d+" }   # → reward_fn
```

多模型（冻结 teacher + 可训练 student）是命名模型集；自定义 trainer 读
`self.models["teacher"]`。可跑的端到端模板：
[examples/online_distill.py](examples/online_distill.py)
（`lighttrain train -c examples/references/recipes/online_distill_demo.yaml`）。
→ [训练范式](docs/concepts/training.zh-CN.md)

## 训练产出

`runs/<exp>/<时间戳>-<slug>-<hash>/` 下的自包含运行胶囊：配置快照 + 解析后配置、
`env.json`、`logs/metrics.jsonl`、`checkpoints/`（`manifest.json` 最后写 = 完整性标记）。
→ [快速开始](docs/guide/getting-started.zh-CN.md#训练产出什么)

## 内置组件（速览）

| 类别 | 名称 |
| ---- | ---- |
| 模型 | `tiny_lm`、`hf_causal`、`tiny_rwkv`、`tiny_mamba`、`jepa`，+ PEFT `lora`/`ia3`/`adalora` |
| Trainer | `pretrain`、`preference`、`reward_model`、`ppo`、`grpo` |
| Loss | `cross_entropy`、`dpo`/`ipo`/`simpo`/`orpo`/`kto`、`ppo_surrogate`、`grpo`、`kl_topk` 等 |
| 优化器 | `adamw`、`lion` · 调度器 `constant`/`linear`/`warmup_cosine`/`wsd` |
| 数据 | dataset、collator、sampler、byte 分词器、PrepGraph 节点 |
| 诊断 | invariant、nan_hunter、frozen_step、loss_attribution、`doctor` |

以上均为具体 `@register` 实现，统一位于 `lighttrain.builtin_plugins`（核心层只保留协议与框架），按短名解析、与代码位置无关。完整表：[注册表与协议](docs/reference/registry.zh-CN.md)。

## 文档

全部在 [`docs/`](docs/README.md)（中英双语，按主题拆分）：

- [快速开始](docs/guide/getting-started.zh-CN.md) · [CLI](docs/guide/cli.zh-CN.md) ·
  [配置](docs/guide/configuration.zh-CN.md)
- [架构](docs/concepts/architecture.zh-CN.md) · [训练](docs/concepts/training.zh-CN.md) ·
  [数据与 PrepGraph](docs/concepts/data-prepgraph.zh-CN.md)
- [分布式](docs/operations/distributed.zh-CN.md) · [诊断](docs/operations/diagnostics.zh-CN.md)
- [其他架构](docs/extending/architectures.zh-CN.md) · [扩展](docs/extending/extending.zh-CN.md) ·
  [配方](docs/extending/recipes.zh-CN.md) · [常见问题](docs/extending/troubleshooting.zh-CN.md)
- 参考：[注册表与协议](docs/reference/registry.zh-CN.md)

## 许可

MIT。借助 Claude Code 实现；架构、测试设计与质量门由人类主导。
