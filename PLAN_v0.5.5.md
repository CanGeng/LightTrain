# lighttrain v0.5.5 — 执行计划

> Phase 0 收尾性迭代。目标：清空 `KNOWN_ISSUES.md` 开放账，把 pause 收成「界定的停」。
> 在个人/实验室内部工具定位下：不引入社区负债、不扩研究范式；新增 `prune-tokenizer` / `check-tokenizer`
> 两个 CLI 子命令属「内置工具体现」（端口并改进已实测的 voca-prune），而非新研究能力。

## 决议快查（grilling Q1–Q22 整合）

| 决议点 | 选择 |
|---|---|
| Q1 CLI I/O 契约 | 1:1 端口 voca-prune，typer 外壳 |
| Q2 Qwen 内置 | Vendored Qwen3-0.6B tokenizer + LICENSE |
| Q3 裁后入口 | 复用 `hf_auto`，不新增 `hf_pruned` 短名 |
| Q4 裁剪算法 | 仅 lossless，**bool 集合计数**（不存频次） |
| Q4b 递归策略 | Fixpoint worklist（按字节长度无依赖） |
| Q5 可选路径 | 保留 `--support-lang` + `--inherit-ids` |
| Q6a chatml 拼接 | 去掉，所有 key 直接 `encode` |
| Q6b .txt 支持 | 加，一行一例 |
| Q7a token_mapping 落盘 | JSON（含 tokenizer_fingerprint + new_vocab_size + new2old 列表） |
| Q7b check 子命令 | 端口为 `lighttrain check-tokenizer` |
| Q8 activation offload 技术 | 自定义 `torch.autograd.Function`：save input → pin_memory，backward recompute forward |
| Q8b offload 测试边界 | 仅 `gpu` marker，真 GPU 跑；CPU 上 fail-loud |
| Q9 vLLM 处理 | Docstring "stub"→"available when installed" + `vllm = ["vllm>=0.6"]` extras |
| Q10 offload extras | 删 `offload = []` 整行 + 注释块 |
| Q11 changelog 版本 | v0.5.5 |
| Q12 输出布局 | 分层：单 tokenizer 与 +remap-embed 输出解耦 |
| Q13 check-tokenizer 失败行为 | failure-first：非零 exit + mismatch 报告 |
| Q14 子命令文件布局 | `cli/commands/tokenizer.py` 一个文件两函数 |
| Q15 langdetect extras | 新 extras 组 `prune=["langdetect>=1.0"]` |
| Q16 safetensors extras | 已在 core deps，无需新增 |
| Q17 算法代码位置 | `lighttrain/builtin_plugins/data/prune/` 子包 |
| Q18 测试 boundary | langdetect 用 `pytest.skip`，activation offload 用 `gpu` marker，changelog 全列 T1-T7 |
| Q19 多机守卫位置 | `ParallelContext.from_env` 内 NNODES 检测 |
| Q20 README 文案 | "Multi-node is not supported" + 守卫 wording 对齐 |
| Q21 vendored tokenizer | Qwen/Qwen3-0.6B 的 tokenizer 文件组 |
| Q22 saved_tensors 策略 | save input → `.pin_memory()`，`.cuda(non_blocking=True)` + recompute forward |

参考的裁剪工具原作：`/mnt/c/Users/admin/Desktop/llm_trian/voca-prune/`（main.py / vocab_count.py / vocab_save.py / model_save.py / check.py / utils.py）。

---

## 任务总览

```
Block A  分布式诚实化  (P0.1 README 双语 + P0.2 ParallelContext NNODES 守卫)
Block B  hf_auto 上提  (P0.4-P0.7: core/tokenizers.py + MiniMind 清理 + vendored Qwen3 baseline + LICENSE)
Block C  prune/check CLI (P0.8-P0.10: builtin_plugins/data/prune/ 子包 + cli/commands/tokenizer.py + _app 注册)
Block D  activation offload 真实现  (P0.11: 自定义 torch.autograd.Function)
Block E  vLLM docstring / extras  (P0.12)
Block F  pyproject 清理  (P0.13 删 offload=[] + P0.14 新增 prune=["langdetect>=1.0"])
Block G  KNOWN_ISSUES 与 changelog  (P0.15 E3 tombstone + P0.16 v0.5.5.md)
Block H  测试撰写  (P0.17: T1-T7 + 分布式 + vLLM docstring)
Block I  质量门禁  (P0.18: ruff/mypy/pytest 全绿)
```

执行顺序：A → B → C → D → E+F → G → I。每步独立验证通过再下一步
（按 `experience.md` #3 教训：全量绿可能掩盖隔离 bug，每步跑相关子集验证）。

---

## 详细任务分解

### Block A — 分布式诚实化

**P0.1 README 双语文案**

改 `README.md:14-19`、`README.zh-CN.md` 对应行。

英文版从：

```
> Status: Development pauses. Distributed is **data-parallel only** (DDP / FSDP /
> DeepSpeed ZeRO); DDP, FSDP, and DeepSpeed ZeRO-2 are validated on a real
> single-node multi-GPU box (NCCL), but **not** on multi-node GPU clusters — use
> at your own risk for production.
```

改为：

```
> Status: Development pauses. Distributed is data-parallel only (DDP / FSDP /
> DeepSpeed ZeRO), validated on a single-node multi-GPU box (NCCL). Multi-node
> is not supported — ParallelContext.from_env raises on NNODES>1.
```

中文版同步：`状态：暂停开发。分布式仅支持数据并行（DDP / FSDP / DeepSpeed ZeRO），
已在真实单机多卡（NCCL）验证。多机不支持——ParallelContext.from_env 在 NNODES>1 时 raise。`

**P0.2 ParallelContext 多机守卫**

文件：`lighttrain/distributed/_context.py:ParallelContext.from_env`（line 54-100）。

在 `import torch.distributed as dist` 之后、`dist.init_process_group` 之前插入
NNODES 检测（必须在 init 之前，否则多机 init 会 hang 在等待 rendezvous）：

```python
_nnodes = int(os.environ.get("NNODES", "1"))
if _nnodes > 1:
    raise RuntimeError(
        f"Multi-node training is not supported (NNODES={_nnodes}). "
        "lighttrain validates only single-node multi-GPU (DDP / FSDP / ZeRO). "
        "Run torchrun with --nnodes=1 or use ParallelContext.single_gpu()."
    )
```

在 `dist.init_process_group` 之后、`dist.get_world_size()` 之后插入双信号防御
（`LOCAL_WORLD_SIZE < WORLD_SIZE` 表示 torchrun 把 world 拆到了多机）：

```python
_local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
if _local_world_size < world_size:
    raise RuntimeError(
        f"Multi-node detected (LOCAL_WORLD_SIZE={_local_world_size} < "
        f"WORLD_SIZE={world_size}); multi-node is not supported."
    )
```

**验证**：
- 新增 `tests/distributed/test_context_multinode.py::test_multi_node_raises`：
  设 `os.environ["NNODES"]="2"` 调用 `from_env`，断言 `RuntimeError`。
  测试结束 `finally` 还原 `NNODES` 避免污染其它测试（按 `experience.md` #3 全局状态教训）。
- 现有分布式测试保持通过。
- `mypy` + `ruff` 双绿。

**Block A 改动文件**：
```
M  README.md
M  README.zh-CN.md
M  lighttrain/distributed/_context.py
A  tests/distributed/test_context_multinode.py
```

---

### Block B — hf_auto tokenizer 上提

**P0.4 在 `lighttrain/builtin_plugins/data/core/tokenizers.py`（当前只含 ByteTokenizer）追加 `HFAutoTokenizer`**

接口契约（显式满足 `TokenizerProtocol`, `protocols.py:174-176`）：

```python
@register("tokenizer", "hf_auto")
class HFAutoTokenizer:
    def __init__(self, path: str, *, use_fast: bool = True, **kwargs: Any) -> None:
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(path, use_fast=use_fast, **kwargs)
        # 缓存常用属性（避免每访问一次都走 HF 内部逻辑）
        self._vocab_size = int(self._tok.vocab_size)
        self._pad_id = int(self._tok.pad_token_id) if self._tok.pad_token_id is not None else 0
        self._bos_id = self._tok.bos_token_id
        self._eos_id = self._tok.eos_token_id

    @property
    def vocab_size(self) -> int: return self._vocab_size
    @property
    def pad_id(self) -> int: return self._pad_id
    @property
    def bos_id(self) -> int | None: return self._bos_id
    @property
    def eos_id(self) -> int | None: return self._eos_id

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return self._tok.encode(text, **kwargs)

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return self._tok.decode(ids, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any: return self._tok(*args, **kwargs)
```

**关键判定**：
- 不再使用 `__getattr__` 魔法（原 `examples/MiniMind/model/model_adapter.py:101-105` 那段）——
  runtime_checkable Protocol 在 `__getattr__` 兜底下的"通过"是 silent duck-type，应显式实现。
- `__call__` 保留：corpus reader / chainer 仍可能用 `tok(batch)` 形态，但 `encode`/`decode` 现在显式满足协议。

**P0.6 删除 `examples/MiniMind/model/model_adapter.py:87-108` 的 `HFAutoTokenizer` 块**

仅删 `@register("tokenizer","hf_auto")` 那一块（连同 `__all__` 里的 `HFAutoTokenizer` 条目），
保留 `MiniMindLightTrain` 模型注册（`model_adapter.py:40-84`）和其它 user_modules 注册项。

**Registry 冲突核查**：注册去重按 `_core.py:170-172` 的 `_same_source`——
两个不同源的对象注册同一 `(category, name)` 时 raise `RegistryConflictError`。
删除 examples 副本是必须的，不能"两份共存"。

**MiniMind 的 configs 不需要改**：`user_modules: [examples.MiniMind.model.model_adapter]`
仍然导入 `model_adapter`（现在只剩 MiniMindLightTrain 注册项），而 `hf_auto` 由
`import_all_components()`（`config/_components.py:80-94` walk_packages 递归）走 builtin_plugins
自动注册。MiniMind recipe 写 `tokenizer: {name: hf_auto, path: examples/MiniMind/model}`
仍能解析——只是实现体现在来自 builtin_plugins。

**P0.7 vendored Qwen3-0.6B tokenizer**

新建目录 `lighttrain/builtin_plugins/data/_q3_tok_baseline/`，复制以下文件（从 HuggingFace
`Qwen/Qwen3-0.6B` repo 下载后人手或脚本放置——**本文件下载动作不在 plan 内执行，仅规划位置**）：

```
tokenizer.json
tokenizer_config.json
special_tokens_map.json
added_tokens.json
vocab.json (如该模型有)
merges.txt (如该模型有)
LICENSE                        # Qwen3 LICENSE 文件原文复制附
```

为 `HFAutoTokenizer` 与 CLI 默认提供路径常量：

```python
# lighttrain/builtin_plugins/data/core/tokenizers.py 末尾
from pathlib import Path
QWEN3_BASELINE_DIR: Path = (
    Path(__file__).resolve().parents[1] / "_q3_tok_baseline"
)
```

注意：**CLI `--tokenizer` 默认值**指向此目录；**生产使用时**用户也可显式给
`--tokenizer <本地下载的 Qwen3>`。

**`.gitignore` 与体积**：Qwen3-0.6B tokenizer 文件组合计约几 MB，需 commit 进仓库（vendor 化策略），
确认 `.gitignore` 不误排除 `tokenizer.json` / `*.txt` 等通用名。

**Block B 改动文件**：
```
M  lighttrain/builtin_plugins/data/core/tokenizers.py
M  examples/MiniMind/model/model_adapter.py
A  lighttrain/builtin_plugins/data/_q3_tok_baseline/   (tokenizer 文件组 + LICENSE)
A  tests/data/test_tokenizers.py                       (T1, T2)
```

---

### Block C — prune-tokenizer / check-tokenizer CLI 子命令

**P0.8 算法位置：`lighttrain/builtin_plugins/data/prune/` 子包**

```
lighttrain/builtin_plugins/data/prune/
├── __init__.py     # 公开 API: prune_tokenizer(...), check_tokenizer(...)
├── corpus.py       # corpus_reader: 扫目录 .json/.jsonl/.txt, 统一 key 集合
├── algorithm.py    # fixpoint worklist 递归闭包 + bool seen set
├── save.py         # 写新 tokenizer 文件组 + token_mapping.json + seen_ids.json
├── remap.py        # --remap-embed: 切 safetensors + config.json + generation_config.json
└── langfilter.py   # --support-lang: langdetect per-token 语言白名单
```

**算法骨架实现** (`algorithm.py`)：

```python
def compute_seen_set(
    *,
    vocab_size: int,
    old_bytes_list: list[bytes],
    corpus_ids: Iterable[set[int]] | None = None,
    support_lang: list[str] | None = None,
    inherit_ids: Path | None = None,
) -> set[int]:
    seen: set[int] = set()
    # 1. corpus 扫描结果
    if corpus_ids is not None:
        for ids in corpus_ids:
            seen |= ids
    # 2. support-lang 路径
    if support_lang:
        _import_langdetect_or_raise()  # fail-loud If langdetect not installed
        seen |= _lang_filter(old_bytes_list, support_lang)
    # 3. inherit-ids 跨批 union
    if inherit_ids:
        seen |= _load_seen_ids_json(inherit_ids, expected_vocab_size=vocab_size)

    # 4. fixpoint worklist 递归闭包：把被保留的长 token 所有 sub-fragment 也加入 seen
    bytes_to_index: dict[bytes, int] = {b: i for i, b in enumerate(old_bytes_list)}
    work: list[int] = list(seen)
    while work:
        i = work.pop()
        b = old_bytes_list[i]
        if len(b) <= 1:
            continue
        n = len(b)
        for start in range(0, n):
            for end in range(start + 1, n + 1):
                j = bytes_to_index.get(b[start:end])
                if j is not None and j not in seen:
                    seen.add(j)
                    work.append(j)

    # 5. 强制保留所有 special tokens（索引 >= len(old_bytes_list)）
    for i in range(len(old_bytes_list), vocab_size):
        seen.add(i)
    return seen


def make_mapping(seen: set[int], vocab_size: int) -> list[int]:
    """返回 new2old list: 新 id -> 旧 id（按 old_id 升序保持 BPE id 紧凑）."""
    return sorted(seen)
```

**关键决策**：
- `mapping_new2old` 按 **old_id 升序**排列（不按 seen 集合发现顺序）。这是 BPE 词表保持
  id 紧凑、检查可复现的关键。
- special token 强制保留（与 voca-prune `main.py:198-200` 一致）。
- 涉及 `bytes_to_index.get` 性能：150k vocab 的 dict，每次切片查询 O(1)，整轮 fixpoint =
  O(N * avg_len^2)，~150k × 8^2 ≈ 10M 查询，秒级。

**P0.8 corpus_reader 设计** (`corpus.py`)：

```python
CORPUS_KEYS: tuple[str, ...] = (
    "text", "prompt", "query", "response", "instruction", "input", "output",
)

def iter_corpus_texts(corpus_dir: Path) -> Iterator[str]:
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".txt":
            for line in path.read_text("utf-8").splitlines():
                if line.strip():
                    yield line
        elif path.suffix == ".json":
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, list):
                for item in data:
                    yield from _extract_keys(item)
            elif isinstance(data, dict):
                yield from _extract_keys(data)
        elif path.suffix == ".jsonl":
            for line in path.read_text("utf-8").splitlines():
                if line.strip():
                    yield from _extract_keys(json.loads(line))


def _extract_keys(obj: dict) -> Iterator[str]:
    for k in CORPUS_KEYS:
        v = obj.get(k)
        if isinstance(v, str):
            yield v
        elif isinstance(v, list):
            yield from (x for x in v if isinstance(x, str))
```

**langfilter.py**（端口自 `vocab_count.py:121-144`，bool 化）：

```python
def _import_langdetect_or_raise() -> None:
    try:
        import langdetect  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "--support-lang requires 'langdetect'. "
            "Install it with: pip install 'lighttrain[prune]'"
        ) from exc


def _is_special_token(s: str) -> bool:
    return (
        (s.startswith("<") and s.endswith(">") and len(s) > 2)
        or (s.startswith("[") and s.endswith("]") and len(s) > 2)
    )


def lang_filter(old_bytes_list: list[bytes], support_lang: list[str]) -> set[int]:
    """对每个原 vocab token 跑 langdetect; 命中白名单或 special token → 加进 seen."""
    from langdetect import detect as langdetect_detect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0  # 复现稳定

    seen: set[int] = set()
    for i, b in enumerate(old_bytes_list):
        try:
            token_str = b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if _is_special_token(token_str):
            seen.add(i)
            continue
        try:
            if langdetect_detect(token_str) in support_lang:
                seen.add(i)
        except Exception:  # noqa: BLE001 langdetect 对随机字节抛 LangDetectException
            pass
    return seen
```

**save.py** — 写新 tokenizer 文件组（端口自 `main.py:20-102` `save_tokenizer_json_all`，删除

```python
def save_pruned_tokenizer(
    old_tokenizer, new_bytes_list: list[bytes], new_merges: list,
    output_path: Path,
) -> None:
    """写 tokenizer.json / tokenizer_config.json / special_tokens_map.json /
    added_tokens.json(若原 in)。."""
    # 完端口 main.py save_tokenizer_json_all 的四步逻辑:
    #   1. id2token_new / token2id_new → 替换 tokenizer.json model 部分 (vocab + merges)
    #   2. tokenizer_config.json: 复用旧 init_kwargs, 更新 vocab_size, 删 _name_or_path
    #   3. special_tokens_map.json: 直接复制
    #   4. added_tokens.json (若原始目录有): 直接复制
    ...


def save_mapping_and_seen(
    output_path: Path, mapping_new2old: list[int],
    tokenizer_fingerprint: str, seen: set[int],
) -> None:
    """写 token_mapping.json + seen_ids.json (后者供 --inherit-ids 跨批 union)."""
    mapping_payload = {
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "new_vocab_size": len(mapping_new2old),
        "new2old": mapping_new2old,
    }
    (output_path / "token_mapping.json").write_text(
        json.dumps(mapping_payload, ensure_ascii=False), encoding="utf-8"
    )
    seen_payload = {
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "ids": sorted(seen),
    }
    (output_path / "seen_ids.json").write_text(
        json.dumps(seen_payload, ensure_ascii=False), encoding="utf-8"
    )


def _load_seen_ids_json(path: Path, *, expected_vocab_size: int) -> set[int]:
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("tokenizer_fingerprint") != expected_vocab_size:  # 简化: 用 vocab_size 作 fingerprint
        raise ValueError(
            f"inherit-ids file '{path}' tokenizer fingerprint mismatch: "
            f"expected vocab_size={expected_vocab_size}, got "
            f"{payload.get('tokenizer_fingerprint')}"
        )
    return set(payload["ids"])
```

**校验 fingerprint 的设计**：简化用 `vocab_size` 作 fingerprint（tokenizer_fingerprint 字段存 vocab_size）。
两层用例都 cover：跨批裁同一个 base tokenizer 时 vocab_size 相同，安全；换了 base tokenizer
（不同 vocab_size）直接 raise，防用户拿错 inherit 文件污染结果。

**remap.py** — 端口自 `model_save.py:99-142`：

```python
def remap_embed_and_lm_heads(
    model_dir: Path, new_model_dir: Path,
    mapping_new2old: list[int], old_tokenizer,
) -> None:
    """遍历 model_dir 所有 .safetensors, 切 embed_tokens.weight / lm_head.weight,
    原 tensor 保留与位置复制。索引文件 (.safetensors.index.json) 整盘复制。."""
    new_model_dir.mkdir(parents=True, exist_ok=True)
    index_filename = "model.safetensors.index.json"
    src_index = model_dir / index_filename
    if src_index.exists():
        shutil.copy2(src_index, new_model_dir / index_filename)

    for f in os.listdir(model_dir):
        if not f.endswith(".safetensors"):
            continue
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(model_dir / f, framework="pt") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k)
                if "embed_tokens.weight" in k or "lm_head.weight" in k:
                    t = t[mapping_new2old]
                tensors[k] = t
        save_file(tensors, new_model_dir / f)


def remap_config_and_generation(
    old_model_dir: Path, new_model_dir: Path, mapping_new2old: list[int],
) -> None:
    """config.json: vocab_size 改新值; 所有 *_token_id 字段按 new2old 重映射.
    generation_config.json: 同上, 支持 eos 为 int 或 list. 若该文件不存在则跳过."""
    old2new = {old_id: new_id for new_id, old_id in enumerate(mapping_new2old)}

    # config.json
    new_config = AutoConfig.from_pretrained(old_model_dir, trust_remote_code=True)
    new_config.vocab_size = len(mapping_new2old)
    for key, old_id in new_config.to_dict().items():
        if "token_id" in key and isinstance(old_id, int):
            if old_id in old2new:
                setattr(new_config, key, old2new[old_id])
            else:
                # 必须 special token 被裁了——按 voca-prune 原行为告警并置 None
                import warnings
                warnings.warn(
                    f"config key '{key}' (token_id={old_id}) was pruned; set to None."
                )
                setattr(new_config, key, None)
    new_config.save_pretrained(new_model_dir)

    # generation_config.json (optional)
    gen_path = old_model_dir / "generation_config.json"
    if gen_path.exists():
        gen = json.loads(gen_path.read_text("utf-8"))
        for key, v in list(gen.items()):
            if "token_id" not in key:
                continue
            if isinstance(v, int) and v in old2new:
                gen[key] = old2new[v]
            elif isinstance(v, list):
                gen[key] = [old2new[i] for i in v if i in old2new]
        (new_model_dir / "generation_config.json").write_text(
            json.dumps(gen, ensure_ascii=False, indent=2), encoding="utf-8"
        )
```

**__init__.py** — 编排：

```python
def prune_tokenizer(
    *, tokenizer_path: Path, out: Path,
    corpus: Path | None = None, support_lang: list[str] | None = None,
    inherit_ids: Path | None = None, remap_embed: Path | None = None,
) -> None:
    # 1. 加载基础 tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    vocab_size = len(old_tokenizer)
    old_bytes_list = [tok.encode("utf-8") for tok, _ in sorted(
        old_tokenizer.get_vocab().items(), key=lambda x: x[1]
    )]

    # 2. 扫 corpus (若给)
    corpus_ids_iter = None
    if corpus is not None:
        def gen():
            for text in iter_corpus_texts(corpus):
                yield set(old_tokenizer.encode(text))
        corpus_ids_iter = gen()

    # 3. 算 seen 集合
    seen = compute_seen_set(
        vocab_size=vocab_size, old_bytes_list=old_bytes_list,
        corpus_ids=corpus_ids_iter, support_lang=support_lang,
        inherit_ids=inherit_ids,
    )

    # 4. 生成 mapping + new_bytes_list
    mapping_new2old = make_mapping(seen, vocab_size)
    new_bytes_list = [old_bytes_list[i] for i in mapping_new2old]

    # 5. 裁 BPE merges (端口自 main.py:211-227)
    new_merges = _prune_merges(old_tokenizer, old_bytes_list, seen)

    # 6. 写文件
    out.mkdir(parents=True, exist_ok=True)
    save_pruned_tokenizer(old_tokenizer, new_bytes_list, new_merges, out)
    save_mapping_and_seen(out, mapping_new2old, str(vocab_size), seen)

    # 7. (可选) remap-embed
    if remap_embed is not None:
        remap_embed_and_lm_heads(remap_embed, out, mapping_new2old, old_tokenizer)
        remap_config_and_generation(remap_embed, out, mapping_new2old)


def check_tokenizer(
    *, old_tokenizer_path: Path, new_tokenizer_path: Path, corpus: Path,
) -> int:
    """Fail-loud 等价性验证. mismatch > 0 返回非 0, CLI 层据此 raise Exit(1)."""
    old_tok = AutoTokenizer.from_pretrained(str(old_tokenizer_path), trust_remote_code=True)
    new_tok = AutoTokenizer.from_pretrained(str(new_tokenizer_path), trust_remote_code=True)
    mapping = json.loads((new_tokenizer_path / "token_mapping.json").read_text("utf-8"))
    new2old = mapping["new2old"]

    mismatch_count = 0
    n_samples = 0
    for text in iter_corpus_texts(corpus):
        n_samples += 1
        old_ids = old_tok.encode(text)
        new_ids = new_tok.encode(text)
        if len(old_ids) != len(new_ids):
            mismatch_count += 1
            _print_mismatch(text, old_ids, new_ids, new2old)
            continue
        for old_id, new_id in zip(old_ids, new_ids):
            if old_id != new2old[new_id]:
                mismatch_count += 1
                _print_mismatch(text, old_ids, new_ids, new2old)
                break
    return mismatch_count


def _print_mismatch(text: str, old_ids: list[int], new_ids: list[int], new2old: list[int]) -> None:
    snippet = text[:80].replace("\n", " ")
    mapped = [new2old[i] if i < len(new2old) else -1 for i in new_ids]
    first_diff = next(
        (p for p, (a, b) in enumerate(zip(old_ids, mapped)) if a != b), None
    )
    print(
        f"  MISMATCH: text={snippet!r}\n"
        f"    old_ids (len={len(old_ids)}): {old_ids[:10]}...\n"
        f"    new_ids (len={len(new_ids)}): {new_ids[:10]}...\n"
        f"    new2old mapped: {mapped[:10]}...\n"
        f"    first divergence at position {first_div}",
        file=sys.stderr,
    )
```

**P0.9 CLI 文件**：`lighttrain/cli/commands/tokenizer.py`

```python
"""lighttrain prune-tokenizer / check-tokenizer CLI.

Lossless (bool 集合) 词表裁剪工具，端口并改进 voca-prune 工具设计。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from lighttrain.builtin_plugins.data.core.tokenizers import QWEN3_BASELINE_DIR
from lighttrain.builtin_plugins.data.prune import prune_tokenizer, check_tokenizer


def prune_tokenizer_cmd(
    tokenizer: Annotated[Path, typer.Option(
        "--tokenizer", help="基础 tokenizer 目录路径 (默认指向内置 Qwen3-0.6B)",
    )] = QWEN3_BASELINE_DIR,
    corpus: Annotated[Path | None, typer.Option(
        "--corpus", help="语料目录 (.json/.jsonl/.txt 递归扫)",
    )] = None,
    support_lang: Annotated[list[str] | None, typer.Option(
        "--support-lang", help="保留语言集 (e.g. zh en), 不给 corpus 时必给此项",
    )] = None,
    inherit_ids: Annotated[Path | None, typer.Option(
        "--inherit-ids", help="上一轮 seen_ids.json, 跨批 set union",
    )] = None,
    remap_embed: Annotated[Path | None, typer.Option(
        "--remap-embed", help="基础模型目录 (.safetensors), 给定则同时裁权重",
    )] = None,
    out: Annotated[Path, typer.Option(
        "--out", "-o", help="输出目录",
    )] = Path("./pruned_tok"),
) -> None:
    if corpus is None and not support_lang:
        print(
            "ERROR: 必须提供 --corpus 或 --support-lang 之一", file=sys.stderr,
        )
        raise typer.Exit(1)
    prune_tokenizer(
        tokenizer_path=tokenizer, out=out,
        corpus=corpus, support_lang=support_lang or None,
        inherit_ids=inherit_ids, remap_embed=remap_embed,
    )
    print(f"==> Pruned tokenizer saved to {out}")


def check_tokenizer_cmd(
    old: Annotated[Path, typer.Option("--old", help="原 tokenizer 目录")],
    new: Annotated[Path, typer.Option("--new", help="新 tokenizer 目录")],
    corpus: Annotated[Path, typer.Option("--corpus", help="等价性验证语料")],
) -> None:
    mismatch = check_tokenizer(
        old_tokenizer_path=old, new_tokenizer_path=new, corpus=corpus,
    )
    if mismatch > 0:
        print(f"VERIFICATION FAILED: {mismatch} mismatches", file=sys.stderr)
        raise typer.Exit(1)
    print(f"VERIFIED: all samples encode-equivalent")
```

**P0.10 `_app.py` 注册**（在 `cli/_app.py:104` 之后追加两行）：

```python
from .commands import tokenizer  # noqa: E402
app.command("prune-tokenizer")(tokenizer.prune_tokenizer_cmd)
app.command("check-tokenizer")(tokenizer.check_tokenizer_cmd)
```

注：现有 `app.command()` 装饰器风格是 `app.command("name")(func)`（line 80-104），
保持一致；导入放在 `cli/_app.py` 顶部 import 块。

**分层输出布局**（Q12 决议）：

```
--out pruned_tok/
└── 仅裁 tokenizer 时:
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── added_tokens.json (若原 in)
    ├── token_mapping.json     # {"tokenizer_fingerprint", "new_vocab_size", "new2old": [...]}
    └── seen_ids.json          # {"tokenizer_fingerprint", "ids": [...]}, 跨批 union 用

└── 若 --remap-embed 给了基础模型目录: 上面所有 + 追加
    ├── config.json             # vocab_size + 所有 *_token_id 重映射
    ├── generation_config.json  # 重映射 eos/pad
    └── model.safetensors (+ index.json)
```

**Block C 改动文件**：
```
A  lighttrain/builtin_plugins/data/prune/__init__.py
A  lighttrain/builtin_plugins/data/prune/corpus.py
A  lighttrain/builtin_plugins/data/prune/algorithm.py
A  lighttrain/builtin_plugins/data/prune/save.py
A  lighttrain/builtin_plugins/data/prune/remap.py
A  lighttrain/builtin_plugins/data/prune/langfilter.py
A  lighttrain/cli/commands/tokenizer.py
M  lighttrain/cli/_app.py
A  tests/cli/test_prune_tokenizer.py (T3, T4, T5)
```

---

### Block D — activation offload 真实现

**P0.11 `lighttrain/builtin_plugins/layer_offload/_activation.py` 的 mode='offload' 块**

替换当前 warn-and-fallback（`_activation.py:52-59`）为自定义 `torch.autograd.Function`：

```python
class _ActivationOffloadFunction(torch.autograd.Function):
    """Save layer input only; offload to pinned host after forward.
    Backward pre-fetches to GPU and recomputes the layer forward to
    reconstruct intermediates via PyTorch's autograd, then backprops.

    CPU 输入选 mode='offload' 直接 raise（fail-loud, 守 GPU 边界）。
    """

    @staticmethod
    def forward(ctx, run_layer, x, *args, **kwargs):
        ctx.run_layer = run_layer
        ctx.args_for_recompute = args
        ctx.kwargs_for_recompute = kwargs
        ctx.x_device = x.device
        ctx.x_requires_grad = x.requires_grad
        y = run_layer(x, *args, **kwargs)

        if x.requires_grad:
            if not x.is_cuda:
                raise RuntimeError(
                    f"activation mode=offload requires CUDA tensors to offload; "
                    f"got device={x.device}. Use mode='recompute' on CPU."
                )
            ctx.x_pinned = x.detach().cpu().pin_memory()
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_pinned = ctx.x_pinned
        x_pre = x_pinned.to(ctx.x_device, non_blocking=True)

        # Recompute forward on original input to obtain autograd intermediates.
        with torch.enable_grad():
            x = x_pre.requires_grad_(ctx.x_requires_grad)
            y = ctx.run_layer(x, *ctx.args_for_recompute, **ctx.kwargs_for_recompute)
            grad_outputs = grad_y
            grad_input = torch.autograd.grad(
                outputs=y,
                inputs=(x,) if ctx.x_requires_grad else (),
                grad_outputs=grad_outputs,
                allow_unused=True,
                retain_graph=False,
            )
        # 返回: 对应 forward 签名 (run_layer, x, *args) 的梯度.
        # run_layer 不需要梯度, 返 None. 后续 args 暂不切 (本工具仅放 forward 的第一输入).
        grad_x = grad_input[0] if ctx.x_requires_grad else None
        return (None, grad_x, *([None] * len(ctx.args_for_recompute)))


class _OffloadWrap(torch.nn.Module):
    """Apply layer with activation offload: forward 计算完把输入搬到 pinned host,
    backward 预取并重算 forward 以拿到 intermediates."""

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x, *args, **kwargs):
        return _ActivationOffloadFunction.apply(self.layer, x, *args, **kwargs)
```

`ActivationManager.wrap` 改 (`_activation.py:44-60`)：

```python
def wrap(self, layer_module: torch.nn.Module) -> torch.nn.Module:
    if self.mode in ("recompute", "recompute_or_offload"):
        return _CheckpointWrap(layer_module)
    if self.mode == "offload":
        # Q8b 决议: 仅 CUDA 启用; CPU 立即 fail-loud.
        if self.device is None or self.device.type == "cpu":
            raise RuntimeError(
                "activation mode=offload requires a CUDA device. "
                "Use mode='recompute' or 'recompute_or_offload' on CPU."
            )
        return _OffloadWrap(layer_module)
    return layer_module
```

`recompute_or_offload` 模式仍回落到 `_CheckpointWrap`（保持现状，与 voca-prune 移植无关，
是 layer_offload 现有的探测策略——本 plan 不动 recompute_or_offload 行为）。

**关键决策**：
- CPU 输入选 mode='offload' 时立即 raise（不在 warn-and-fallback），符合 fail-loud 哲学。
- `non_blocking=True` 需要 `x_pinned` 是 pinned memory，否则是阻塞 copy（但是 functional 正确的）。
- `allow_unused=True` 处理 layer 无 args 的 case，避免 grad 严格断言崩溃。
- 删除原 `_activation.py:53-58` 的 `warnings.warn` 与 fallback 块。

**Block D 改动文件**：
```
M  lighttrain/builtin_plugins/layer_offload/_activation.py
A  tests/layer_offload/test_activation_offload.py (T6, T7)
```

---

### Block E — vLLM docstring / extras 处理

**P0.12 `lighttrain/builtin_plugins/rl/backends/vllm/__init__.py`**

- 第 1 行 module docstring："VLLMBackend — vLLM rollout backend stub." →
  "VLLMBackend — vLLM rollout backend (available when vllm is installed)."
- 第 42 行 class docstring："This is a stub that raises `ImportError` unless `vllm` is installed." →
  "Constructing this backend without `vllm` installed raises `ImportError`."
- 第 6 行 docstring 行 "Importing this module without vLLM installed will raise `ImportError` at
  backend construction time (not at import time)" 保留（事实正确，符合 lazy import 设计）。
- 自动注册入口保留不变（已在 `_components.py:80-94` 的 walk_packages 路径覆盖，
  未安装 vllm 也能完成 `@register("rl_backend","vllm")`，构造时才 raise `ImportError`）。

**自动注册核查**：完成 grilling 时已确认 `import_all_components()` 走 walk_packages 递归遍历
`lighttrain.builtin_plugins.*`，能到达 `rl/backends/vllm/__init__.py:38`，模块顶层
只 import `torch`/`register`/`typing`——无顶层 `import vllm`，所以未安装也注册成功。
PPO (`ppo.py:128-187`) 与 GRPO (`grpo.py:101-153`) 配 `rollout_backend: {name: vllm}` 被
registry 接受，构造时直白报错要装 vllm——与 plugin-clean 哲学一致。

**Block E 改动文件**：
```
M  lighttrain/builtin_plugins/rl/backends/vllm/__init__.py
A  tests/rl/test_vllm_docstring.py
```

---

### Block F — pyproject 清理

**P0.13 删 `offload = []` extras**

`pyproject.toml:43-48` 当前段：

```toml
# Reserved extras (no hard dep today): the CPU/layer-offload path is pure-torch,
# and the vLLM rl_backend lazy-imports ``vllm`` only at use time. Kept for API
# symmetry — ``pip install '.[offload]'`` / ``'.[vllm]'`` are accepted no-ops;
# populate when a backend grows a required dependency.
offload = []
vllm = []
```

改为：

```toml
# Optional rl_backend (lazy-imported at runtime). Install ``vllm`` to enable
# high-throughput PPO/GRPO rollouts with the vLLM backend.
vllm = ["vllm>=0.6"]
```

删除整个 comment block 与 `offload = []` 行。

**P0.14 新增 `prune` extras**

在 extras 组合适位置插入（按字母序或就近 quant 之后）：

```toml
prune = ["langdetect>=1.0"]
```

完整 extras 块改后形态：

```toml
[project.optional-dependencies]
dev = [...]
peft = ["peft>=0.7"]
prune = ["langdetect>=1.0"]
quant = ["bitsandbytes>=0.43; platform_system == 'Linux'"]
vllm = ["vllm>=0.6"]
sweep = ["optuna>=3.0"]
```

（保持按字母序 peft / prune / quant / vllm / sweep 排列。）

**验证**：`pip install -e .` 验证 extras 列表与 changelog 对齐。

**Block F 改动文件**：
```
M  pyproject.toml
```

---

### Block G — KNOWN_ISSUES 与 changelog

**P0.15 `docs/changelog/KNOWN_ISSUES.md`**

把「E3」条目改为 tombstone 风格（与已有 B1/A1/B2 row 一致格式）：

在 `## 已解决 / 已勾销（Resolved / Dismissed）` 段下追加（按 E 序号位置插入）：

```markdown
### E3 — 无内置 `hf_auto` tokenizer；HF 分词器需由用户在 `user_modules` 中自注册
✅ 已解决 → v0.5.5（见 [v0.5.5](v0/v0.5/v0.5.5.md)）：在
`lighttrain/builtin_plugins/data/core/tokenizers.py` 内置显式满足
`TokenizerProtocol` 的 `HFAutoTokenizer`（接受 `path:` 参数），删除
`examples/MiniMind/model/model_adapter.py` 中的 `hf_auto` 重复注册。

---

**Phase 0 收尾后 KNOWN_ISSUES 开放账完全清空。**
```

并把原 `## 开放（Open）` 段下的 E3 条目整段删除或改为「见 Resolved」一行 link（按项目
现有 B1/A1 等 tombstone 风格——保留标题、改成一句话指向）。

**P0.16 `docs/changelog/v0/v0.5/v0.5.5.md`**

完整撰写，结构对齐 v0.5.4：

```markdown
# v0.5.5

> **Phase 0 收尾迭代**：清空 KNOWN_ISSUES 开放账，多机分布式从
> "use at your own risk" 收为 "not supported + fail-loud"；新增
> `lighttrain prune-tokenizer` / `check-tokenizer` CLI 子命令（lossless
> bool 词表裁剪）；上提 `hf_auto` 为内置 tokenizer；activation offload
> 真模式接入；vLLM rl_backend 文案与 extras 接通。零运行时行为回退；
> 注册表 33 category 不变（新增 entry: tokenizer/hf_auto 复用 builtin；
> CLI 新增 2 子命令）。Xia你的中文摘要。

## 多机 fail-loud 守卫

### ParallelContext.from_env NNODES 检测
- 现状: README 称 DDP/FSDP/ZeRO "use at your own risk for production", 但
  代码未阻止多机启动, 二者口径互相别扭.
- 改动: 在 from_env 内 dist.init_process_group 之前检测 NNODES env var,
  >1 即 raise RuntimeError; init 后再查 LOCAL_WORLD_SIZE < WORLD_SIZE 双信号防御.
- README 双语同步: "Multi-node is not supported — ParallelContext.from_env
  raises on NNODES>1."

## hf_auto tokenizer 上提（E3 闭项）

- 现 E3 开放: 无内置 hf_auto, HF tokenizer 须由 user_modules 自注册;
  examples/MiniMind/model/model_adapter.py 提供的 hf_auto 是绕过.
- 改动:
  - lighttrain/builtin_plugins/data/core/tokenizers.py 追加 HFAutoTokenizer,
    显式实现 encode/decode/vocab_size/pad_id/bos_id/eos_id, 满足 TokenizerProtocol.
    不再依赖 __getattr__ magic.
  - 删除 examples/MiniMind/model/model_adapter.py 的 hf_auto 注册 (registry
    idempotency 不允许两份不同源共存).
  - vendored Qwen3-0.6B tokenizer 文件组入 lighttrain/builtin_plugins/data/
    _q3_tok_baseline/ 含 LICENSE.
  - MiniMind configs 无需改: user_modules 仍导入 model_adapter (现只剩模型注册),
    hf_auto 由 import_all_components walk_packages 自动注册.

## 新增 CLI: prune-tokenizer / check-tokenizer

- 设计来源: 端口并改进 /mnt/.../voca-prune 工具.
- 算法核心: lossless bool 集合裁剪.
  1. 扫 corpus (.json/.jsonl/.txt 递归) encode 出 seen 集合.
  2. (可选) --support-lang 用 langdetect per-token 多语言过滤.
  3. (可选) --inherit-ids 跨批用 seen_ids.json 做 set union.
  4. Fixpoint worklist 递归闭包: 保留 token 的所有 sub-fragment 也加入 seen.
  5. special tokens (索引 >= len(old_bytes_list)) 强制保留.
  6. mapping_new2old 按 old_id 升序输出.
- 端口改进:
  - bool 集合计数替代频次计数 (lossless 不需频次).
  - corpus 支持 .txt 一行一例 (与 lighttrain init 默认 corpus.txt 一致).
  - corpus key 统一七种 (text/prompt/query/response/instruction/input/output),
    去除原 typo "intruction".
  - 去掉 chatml 模板拼接 (prune 工具语义不应混入 chat token 约定).
  - token_mapping.json + seen_ids.json 为 JSON 格式 (torch-无关, 可人工审查).
  - check-tokenizer failure-first: mismatch>0 exit 1, 与 lighttrain 其他 CLI 一致.
- 输出布局:
  - 仅裁 tokenizer: tokenizer 文件组 + token_mapping.json + seen_ids.json.
  - 给 --remap-embed 时追加: config.json + generation_config.json + 切过
    embed_tokens/lm_head 的 .safetensors (+ index).
- extras: 新增 prune=["langdetect>=1.0"], --support-lang 必须安装才可用.

## Activation offload 真模式

- 现 _activation.py mode='offload' 是 warn + fallback recompute, 实为 dead 选项.
- 改动: 自定义 _ActivationOffloadFunction(torch.autograd.Function):
  - forward: 计算 y, 若 x.requires_grad 且 x.is_cuda, 把 x.detach().cpu().pin_memory()
    存到 ctx.x_pinned; CPU 输入直接 raise RuntimeError (fail-loud).
  - backward: x_pinned.to(device, non_blocking=True) 预取; enable_grad 重做 layer
    forward; torch.autograd.grad 拿梯度返回.
- ActivationManager.wrap 改: mode='offload' 且 device 是 CPU 时直接 raise.
- 测试: T6 数值等价性 (gpu marker, 与 mode='recompute' atol=1e-5);
  T7 CPU fail_loud.

## vLLM docstring / extras

- 现状: rl/backends/vllm/__init__.py docstring 自谦 "stub", 实现已完整;
  pyproject vllm=[] 是 no-op.
- 改动:
  - module/class docstring: "stub" → "available when installed" / "Constructor raises ImportError without vllm".
  - pyproject vllm=[] → vllm=["vllm>=0.6"] (实 deps).
  - 删除 pyproject offload=[] extras 与 4 行 comment (纯 torch 路径无 deps).
- 已确认 import_all_components walk_packages 能自动注册 vllm 短名 (未装也能注册,
  构造时 raise); PPO/GRPO 配 rollout_backend: {name: vllm} 直通.

## 验证

新增测试 7 项:

- T1 tests/data/test_tokenizers.py::test_hf_auto_protocol
  验证 HFAutoTokenizer 实现 TokenizerProtocol, 用 vendored Qwen3 tokenizer 实跑.
- T2 tests/data/test_tokenizers.py::test_no_examples_dup
  grep 断言 examples/MiniMind/model/model_adapter.py 不再含 @register("tokenizer".
- T3 tests/cli/test_prune_tokenizer.py::test_lossless_equivalence
  端到端: 造小语料 -> prune-tokenizer -> check-tokenizer -> exit 0.
- T4 tests/cli/test_prune_tokenizer.py::test_recurse_closure
  100-token mini vocab, 断言 fixpoint 后所有 sub-fragment 集 closed.
- T5 tests/cli/test_prune_tokenizer.py::test_support_lang
  pytest.importorskip("langdetect"), mock mini tokenizer 跑 zh 命中.
- T6 tests/layer_offload/test_activation_offload.py::test_offload_equals_recompute_gpu
  @pytest.mark.gpu, 真 GPU, 数值 atol=1e-5 与 recompute 对齐.
- T7 tests/layer_offload/test_activation_offload.py::test_cpu_fail_loud
  CPU 上 mode='offload' raise RuntimeError.

附属测试:

- tests/distributed/test_context_multinode.py::test_multi_node_raises
  NNODES=2 时 from_env raise RuntimeError; finally 还原 env var.
- tests/rl/test_vllm_docstring.py
  grep 断言 vllm/__init__.py 源码不再含 "stub" 字串.

门禁:

- ruff check .: 全净
- mypy lighttrain tests: 0 errors
- pytest: 主干测试 0 失败, 新增 7+2 测试 pass (T6 单跑 -m gpu)
- 隔离子集跑 (experience.md #3 教训):
  pytest tests/distributed tests/data tests/cli
```

**Block G 改动文件**：
```
M  docs/changelog/KNOWN_ISSUES.md
A  docs/changelog/v0/v0.5/v0.5.5.md
```

---

### Block H — 测试撰写清单

| ID | 路径 | marker | 验证内容 |
|---|------|--------|------|
| T1 | `tests/data/test_tokenizers.py` | default | `test_hf_auto_protocol`：用 vendored Qwen3 tokenizer 路径构造 `HFAutoTokenizer`，断言 `isinstance(tok, TokenizerProtocol) == True`、`vocab_size > 0`、`encode("hello")` 返回 `list[int]`、`decode(encode("hello")) == "hello"`、`pad_id >= 0` |
| T2 | 同上 | default | `test_no_examples_dup`：grep 测试 `examples/MiniMind/model/model_adapter.py` 文本断言不含 `@register("tokenizer"` 字串 |
| T3 | `tests/cli/test_prune_tokenizer.py` | default | `test_lossless_equivalence`：端到端——造一个小语料目录（几个 .jsonl/.txt），跑 `prune-tokenizer` → 跑 `check-tokenizer`，断言 `check-tokenizer` 退出码 0 |
| T4 | 同上 | default | `test_recurse_closure`：手造一个 100-token mini tokenizer mock，几条文本 encode 出去的 id 集，断言 fixpoint 后所有 sub-fragment id 都在 seen 集里 |
| T5 | 同上 | `pytest.importorskip("langdetect")` | `test_support_lang`：mock 一个 mini tokenizer，`langdetect` 命中 zh 加入 seen；未命中语言不在；`pytest.importorskip` 自动 skip 未安 langdetect 的环境 |
| T6 | `tests/layer_offload/test_activation_offload.py` | `@pytest.mark.gpu` | `test_offload_equals_recompute_gpu`：真 GPU，确定性 seed 下 mode='offload' 与 mode='recompute' `loss` 数值在 `atol=1e-5` 内对齐 |
| T7 | 同上 | default | `test_cpu_fail_loud`：CPU 上 `ActivationManager(mode='offload', device=torch.device('cpu'))` 调 `wrap` 时断言 raise `RuntimeError` |
| 附1 | `tests/distributed/test_context_multinode.py` | default | `test_multi_node_raises`：`os.environ["NNODES"]="2"` 调 `ParallelContext.from_env`，断言 raise `RuntimeError`；`finally` 删 NNODES 防污染 |
| 附2 | `tests/rl/test_vllm_docstring.py` | default | grep 测试：`vllm/__init__.py` 源码字符串断言不再含 `"stub"` 字串；同时断言 `@register("rl_backend", "vllm")` 字串仍在 |

**测试风格约定**（与现有 lighttrain 一致）：
- 注解优先（与 v0.5.2 ratchet 一致）。
- `assert ... is not None` 用于 Optional narrowing。
- 测试桩如 fake tokenizer 走 `SimpleNamespace` 充当（与 conftest 一致）。
- `pytest.importorskip` 用于 langdetect skip（不要新加 marker，符合 Q18 决议）。

**Block H 改动文件**：
```
A  tests/data/test_tokenizers.py
A  tests/cli/test_prune_tokenizer.py
A  tests/layer_offload/test_activation_offload.py
A  tests/distributed/test_context_multinode.py
A  tests/rl/test_vllm_docstring.py
```

---

### Block I — 质量门禁

按 v0.5.4 验证模式逐条跑：

```bash
# 1. ruff 全净
ruff check .

# 2. mypy 双绿 (lighttrain + tests, CPU-torch parity venv 视角)
mypy lighttrain tests

# 3. pytest 主干无回归 (默认排除 heavy + gpu)
pytest

# 4. pytest prune-tokenizer 隔离子集 (langdetect skip if absent)
pytest tests/cli/test_prune_tokenizer.py tests/data/test_tokenizers.py

# 5. pytest 其它隔离子集 (按 experience.md #3 教训)
pytest tests/distributed tests/rl test_tokenizers.py

# 6. GPU-only tests 单独跑 (--gpu marker, 真装环境才能跑)
pytest -m gpu tests/layer_offload/test_activation_offload.py

# 7. 全量二次确认 (隔离都过后再跑全量)
pytest

# 8. pip install -e . 验证 extras 列表
pip install -e ".[prune,peft]"
pip install -e ".[vllm]"   # 注: 仅断 extras 解析, 不真装 vllm
```

---

## 文件修改全清单

```
Block A:
  M  README.md
  M  README.zh-CN.md
  M  lighttrain/distributed/_context.py
  A  tests/distributed/test_context_multinode.py

Block B:
  M  lighttrain/builtin_plugins/data/core/tokenizers.py
  M  examples/MiniMind/model/model_adapter.py
  A  lighttrain/builtin_plugins/data/_q3_tok_baseline/        (含 tokenizer 文件组 + LICENSE)
  A  tests/data/test_tokenizers.py

Block C:
  A  lighttrain/builtin_plugins/data/prune/__init__.py
  A  lighttrain/builtin_plugins/data/prune/corpus.py
  A  lighttrain/builtin_plugins/data/prune/algorithm.py
  A  lighttrain/builtin_plugins/data/prune/save.py
  A  lighttrain/builtin_plugins/data/prune/remap.py
  A  lighttrain/builtin_plugins/data/prune/langfilter.py
  A  lighttrain/cli/commands/tokenizer.py
  M  lighttrain/cli/_app.py
  A  tests/cli/test_prune_tokenizer.py

Block D:
  M  lighttrain/builtin_plugins/layer_offload/_activation.py
  A  tests/layer_offload/test_activation_offload.py

Block E:
  M  lighttrain/builtin_plugins/rl/backends/vllm/__init__.py
  A  tests/rl/test_vllm_docstring.py

Block F:
  M  pyproject.toml                            (P0.13 删除 + P0.14 新增 与 Block E 合并改)

Block G:
  M  docs/changelog/KNOWN_ISSUES.md
  A  docs/changelog/v0/v0.5/v0.5.5.md

Block H: (已在各 Block 内附韧, 不单独列出)
Block I: 验证脚本运行
```

---

## 执行顺序与每步验证

```
Step 1  Block A (P0.1, P0.2)
        -> 验证: pytest tests/distributed tests/distributed/test_context_multinode.py
        -> 验证: README 双语 diff 人工核对

Step 2  Block B (P0.4-P0.7, 含 vendored Qwen3 tokenizer 落盘)
        -> 验证: pytest tests/data/test_tokenizers.py
        -> 验证: lighttrain dry-run -c examples/MiniMind/configs/pretrain.yaml (MiniMind hf_auto 仍可解析)

Step 3  Block C (P0.8-P0.10)
        -> 验证: pytest tests/cli/test_prune_tokenizer.py (T3, T4)
        -> 验证: pytest tests/cli/test_prune_tokenizer.py::test_support_lang (有 langdetect 时)

Step 4  Block D (P0.11)
        -> 验证: pytest tests/layer_offload/test_activation_offload.py::test_cpu_fail_loud
        -> 验证: pytest -m gpu tests/layer_offload/test_activation_offload.py::test_offload_equals_recompute_gpu (真 GPU)

Step 5  Block E + Block F (P0.12-P0.14)
        -> 验证: ruff check . + mypy lighttrain tests
        -> 验证: pytest tests/rl/test_vllm_docstring.py
        -> 验证: pip install -e ".[prune,peft]"

Step 6  Block G (P0.15, P0.16)
        -> 文档定稿, 人工 review changelog 与 KNOWN_ISSUES tombstone 表述

Step 7  Block I (全量验证)
        -> ruff check .
        -> mypy lighttrain tests
        -> pytest (主干)
        -> pytest tests/distributed tests/data tests/cli (隔离子集)
        -> pytest -m gpu (若真 GPU 可用)
```

每步独立验证通过再下一步，符合 `experience.md` #3「全量绿可能掩盖隔离 bug」教训：
隔离跑子集，不依赖全量绿通过认定。Block B vendored Qwen3 tokenizer 落盘是 Step 2 唯一
需要外部下载的 step，可提前完成或人手提供。

---

## 不在 Phase 0 范围内的 trigger-based 后续工作

这些是 grilling 中识别但**不**纳入本迭代的工作，登记放在 changelog「已决不修」一栏
避免日后重复发掘：

- **lossy target-size 裁剪** — 仅 lossless 已被 voca-prune 验证；若研究室出现具体需求
  再实现 `--target-size` flag（`vocab_save.reduce_to_target_size` 在 voca-prune 中已是
  死代码，端口时不复活）。
- **更多 corpus key** — 当前 7 key 已覆盖 alpaca/sharegpt/openassistant；自定义 schema
  用户应自己整理 corpus。
- **多模态 adapter / 长上下文 / 多机 FSDP/ZeRO 验证** — 与「内部工具」定位不符的红线。
- **recompute_or_offload 探测策略改进** — Block D 仅实现 mode='offload' 真模式；
  recompute_or_offload 现状仍回落到 recompute，不动，不进 changelog。

---

## 红线（始终不动）

- **不重启 tensor/pipeline/expert/sequence parallelism** 的回归（v0.2.3 fail-loud 保持）。
- **不把 RDV 化的 distributed 提升为 production claim**——README 改后只说「单机多卡验证」。
- **不把 RL rollout backend 拓宽到 vLLM/TGI 之外**的「能选中却空转」实现（experience.md #21 原则）。
- **不引入任何「为日后可能」的抽象**——这是 simplicity-first 与内部工具定位的双重要求。

---

## 已决 chokepoint（避免 grilling 决议回退）

未来动手过程中如果遇到边界判断，请先回看此清单——这是已 grilling 通过的决议：

- 算法 bool 化：seen 集合只存"出现过即保留"，不存计数。
- mapping_new2old 按 old_id 升序输出（保持 BPE id 紧凑）。
- special tokens 强制全保留（voca-prune 一致行为）。
- fail-loud 优于 warn-and-fallback：vLLM 未装、CPU 选 offload、多机启动——都 raise。
- output 分层：仅 tokenizer vs +remap-embed 两态。
- check-tokenizer 是门禁工具，exit code 反映等价性。
- 没有 hf_pruned 短名——裁后 tokenizer 仍写 `name: hf_auto, path: pruned_tok/`。
- fingerprint 用 vocab_size（不引入 hash 复杂度，足够 cover 跨批错用场景）。
- 删 offload=[] extras：纯 torch 路径不需要 extras；YAGNI。
- 拼 chatml：prune 工具不做（用户自己拼模板进 corpus）。