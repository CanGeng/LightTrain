# 架构

> [English](architecture.md) · [文档索引](../README.md)

lighttrain 定义五个清晰的缝，其余部分保持直白。

## 五个缝

1. **Registry** ([lighttrain/registry/_core.py](../../lighttrain/registry/_core.py)) ——
   在预声明的类别集（`model`、`loss`、`optimizer`、`dataset`、`trainer`、`engine`、
   `update_rule`、`judge`、`rl_backend`、`prep_node` 等）上做短名 → 类解析。见
   [扩展](../extending/extending.zh-CN.md)。

2. **Config** ([lighttrain/config/](../../lighttrain/config)) —— OmegaConf 加载 +
   Pydantic v2 schema。模型经配置组选择（`model_profiles:` + `model: <名字>`）。见
   [配置](../guide/configuration.zh-CN.md)。

3. **Engine + UpdateRule** ([engine/](../../lighttrain/engine)、
   [update_rules/](../../lighttrain/engine/update_rules)) —— engine 拥有 accelerator
   （混合精度、设备），把每步数学（前向/反向/clip/step/scheduler）下放给可替换的
   `UpdateRule`，于是研究代码改训练数学无需动 engine。

   **Trainer 原语** ([trainers/](../../lighttrain/trainers)) —— 扁平 `Trainer` 有
   具体 `fit()`，由公共可重入原语组合：`run_train_loop`、`apply_update`、
   `forward_with_activations`。90% 场景纯 YAML；新范式重写 `produce_batch` /
   `forward_loss`（可选 `before_step`），或写一个短的注册 `fit()`。见
   [训练](training.zh-CN.md)、[扩展](../extending/extending.zh-CN.md)。

4. **EventBus** ([callbacks/base.py](../../lighttrain/callbacks/base.py)) —— 46 个
   生命周期事件经 `getattr` 分发；单 callback 异常隔离；结果聚合为 `Signal`
   （`STOP_TRAINING > RETRY_STEP > SKIP_STEP > CONTINUE`）。见
   [诊断](../operations/diagnostics.zh-CN.md)。

5. **PrepGraph** ([prepgraph/](../../lighttrain/data/prepgraph)) —— 内容寻址的数据预处理
   DAG；fingerprint = `sha256(config + code_version + schema_version + 排序后的
   upstream_fps)`；结果原子落盘，`MANIFEST_COMPLETE.json` 最后写。见
   [数据与 PrepGraph](data-prepgraph.zh-CN.md)。

## 初始化顺序（`train`）

`setup_run_from_config` 然后 `trainer.fit()`：

1. **配置加载** —— YAML → defaults → overrides → 插值 → Pydantic。
2. **准备** —— 导入 `user_modules`（装饰器注册）、`seed_everything`；拓扑（3-A）
   *先*初始化，从而 **rank 0 拥有带时间戳的 run 目录并把路径广播**给所有 rank
   （否则各 rank 各自 `datetime.now()`，跨整秒的 launch 会把各 rank 劈进相邻目录）；
   随后写 `config.snapshot.yaml` + `env.json`。
3. **组件（严格顺序）**
   - **A — 拓扑**：`parallel_ctx`（单卡退化或进程组 + 1 维 `dp` DeviceMesh）；`device` 由它得出。
   - **B — 模型**：按 spec 构建主模型（仅数据并行——grad-sync 包装前无模型手术）。
   - **C — grad-sync 包装**（`noop`/`ddp`/`fsdp`/`deepspeed`；FSDP 经
     `optimizer_factory` 在包装*之后*建优化器）。
   - **公共**：data_module → scheduler → loss → callbacks → logger → ckpt。
4. **引擎组装** —— update_rule + accelerator + loss 装入 `StandardEngine`。
5. **Trainer 组装** —— runtime 总是把 `model` / `models` / `optimizers` 传给
   trainer；lab 模式诊断 callback 自动挂载。
6. **训练循环** —— `trainer.fit()` 跑 epoch/信号/log/ckpt 循环。

关键不变量：拓扑先于一切（rank 0 拥有并广播 run 目录）；FSDP 优化器后置；rank-0 门控
checkpoint 与日志 IO（run 元数据初始化与 PrepGraph 尚未完全按 rank-0 协调）；
`loss_fn` 是独立组件（经 `loss:` 替换）；`manifest.json` 最后写。

## 相关

- [扩展](../extending/extending.zh-CN.md) —— 针对这些缝实现
- [训练范式](training.zh-CN.md) —— Engine/Trainer 缝的实战
- [分布式](../operations/distributed.zh-CN.md) —— A/B/C/D 详解
