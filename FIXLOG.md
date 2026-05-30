# FIXLOG — user_modules seam (root causes A/B) + packing ISSUE-3/4/5

Responds to `experiments/packing/issues.md` (5 issues) and the recurring `user_modules`
import seam that point-fixes (mamba3 #9, v0.1.7, v0.1.9) kept failing to close. This pass
attacks the two **root causes** structurally, fixes the 3 remaining packing issues, and
removes one latent bug found while validating acceptance #1. Suite: **1963 → 1993 passed**,
zero regression.

---

## Root cause A — coverage depended on each command remembering to import

`_import_user_modules` was hand-called in only 3 spots; the `prep` family (via
`build_prep_runner`) never called it → ISSUE-1. **Fix: fold the import into the single
chokepoint `load_config`** that every recipe-eating command flows through.

- New `lighttrain/config/_user_modules.py` — `import_user_modules()` + the dedup set,
  depending only on `importlib`+`Path` (no `config → cli` reverse dependency). The old
  `cli/_runtime.py:_import_user_modules` is now a thin re-export sharing the one dedup set
  (back-compat for existing tests).
- `config/_loader.py:load_config` now imports `root.user_modules` after validation, before
  returning, gated by a new `import_user_modules: bool = True` escape hatch.
- **Scattered calls removed/converted** (so no command must remember anymore):
  - **Removed** `cli/_app.py` dry-run `--build` call — redundant; `load_config` ran already.
  - **Converted** `setup_run_from_config` — the path branch is covered by `load_config`; kept
    one call *only* in the `RootConfig`-direct else-branch (a library bypass of the chokepoint).
  - **Kept** `lab/estimate.py`'s call — `estimate()` is public API called directly with a
    raw dict (`tests/test_estimate.py`), a genuine `load_config` bypass; now imports from
    `config._user_modules` (drops the lab→cli dependency).
  - `--print-config` (`_app.py:239`) passes `import_user_modules=False` — a pure config dump
    must not trigger plugin imports.
- Net: every recipe-eating **CLI command** is covered by the chokepoint; the only remaining
  manual calls are two **library entry points** that bypass `load_config`, each idempotent.

## Root cause B — one file could have two module identities

`user_modules` loads a `.py` by file *stem* (`spec_from_file_location`, not in `sys.modules`);
`_target_` loads it by dotted path. Two module objects ⇒ `@register` ran twice ⇒
`RegistryConflictError` (ISSUE-2). **Fix: make `register()` idempotent by *content identity*.**

- `registry/_core.py`: when `name` already present (and not `force`), compare a
  source-location fingerprint built from the object's code objects —
  `(filenames, __qualname__, {(co_qualname, co_firstlineno)})`. Equal ⇒ same logical
  component ⇒ silent no-op; different file/name/line ⇒ still `raise`. `force=True` unchanged.
- Robust to `sys.modules`: a class loaded outside `sys.modules` (the `user_modules` path)
  defeats `inspect.getfile`, so the fingerprint reads method code objects directly. Two
  *different* lambdas on different lines still conflict (line-level granularity), so existing
  `test_duplicate_raises_conflict` still passes.
- Consequence: duplicate import spellings are harmless; users may delete all `force=True`.
  No real conflict is masked — core modules import once via `_eager_import_components`.

---

## ISSUE-5 — `load` node reads plain `.txt`
`prepgraph/nodes/load.py`: added `_iter_lines()` + the `lines:<path.txt>` scheme →
`{"text": <non-empty line>}` per line. Docstring updated.

## ISSUE-3 — `prep-status --extras` surfaces persisted metrics
- `prepgraph/runner.py`: new `PrepRunner.node_extras()` reads each node's
  `MANIFEST_COMPLETE.json` and returns author metrics (manifest keys minus a
  `_FRAMEWORK_MANIFEST_KEYS` set). Manifest-key knowledge stays in the runner.
- `cli/_app.py`: `prep-status --extras` renders one readable per-node line (not a raw dict).

## ISSUE-4 — built-in `pack`: three strategies + standardized extras + BREAKING default
Ported the packing experiment's parity-verified helpers (`best_fit_decreasing`,
`_units_from_rows`, `_emit_bin`, `_pack_stats`) into `prepgraph/nodes/pack.py`:
- `strategy: concat_chunk | next_fit | best_fit`.
  - **`concat_chunk`** — padding-free baseline. NEW to core; **the new default**.
  - **`next_fit`** — historical greedy-pad-flush, preserved bit-for-bit, **not default**.
  - **`best_fit`** — BFD, opt-in.
- All three emit `truncation_rate / token_utilization / n_truncated_docs / n_sequences`
  (was only `row_count`), so `--extras` is meaningful on built-in pack too.
- Docstring + `docs/registry_and_protocols.md` updated; **BREAKING** documented in
  `docs/changelog/v0.1.11`. Pack recipe users (`recipes/sft_chat.yaml`,
  `recipes/sft_chat_hf.yaml`) recompute on next prep (expected); no test broke
  (`test_prepgraph_runner` asserts only `len(store)>0` + cache reasons).
- The packing experiment keeps its own custom nodes (it exercises the `user_modules` path);
  its 234-bin parity now also cross-checks core's ported BFD.

## Bonus — `cleanup_orphans` KeyError (found validating acceptance #1)
`prep-clean --orphans` crashed `KeyError('raw')` on any multi-node graph: `cleanup_orphans`
ran a bare `_resolve` loop that never populated `_fp_cache` (which `_resolve` reads for
upstream fps). Fixed to call `self.plan()`. Regression test added.

---

## Acceptance evidence (real stdout)

**#1 — documented wiring (`user_modules:` + `kind:`, no `_target_`) across the prep family**
(`experiments/packing/config.yaml`, reverted to documented form):
```
prep:        raw/tok/pack_bf/pack_naive/bf_data/naive_data all RUN → prep complete
             500 docs -> 234 seqs | truncation_rate=0.0000 | token_utilization=0.9891
             500 docs -> 232 seqs | truncation_rate=0.4540 | token_utilization=0.9977
prep-graph:  digraph emitted; edges raw→tok→{pack_bf,pack_naive}→{bf_data,naive_data}
prep-status: all 6 nodes [CACHE] (hit)
prep-clean:  --orphans --dry-run → "nothing to clean"   (post cleanup_orphans fix)
```

**#2 — `user_modules` + `_target_` for the SAME file → no RegistryConflictError**
(file-stem `mynode` + dotted `pkg.mynode`):
```
PrepGraph: 2 nodes, 0 cached, 2 to run
[ RUN ] load/raw ; [ RUN ] _acc2_node/n → prep complete
```

**#3 — register content-identity** (`tests/test_registry.py`): two module identities of one
file → no-op; different class same name → raises; `force=True` overrides. All pass.

**#4 — chokepoint guard** (`tests/test_user_modules_chokepoint.py`): `load_config` registers
the custom node; `import_user_modules=False` skips; `build_prep_runner` resolves it;
parametrized over prep/prep(--dry-run)/prep-status/prep-status --extras/prep-graph/prep-clean
— each resolves the documented-form node (no `NotRegisteredError`).

**#5 — `prep-status --extras`** (experiment):
```
pack_bf:    ... truncation_rate=0    token_utilization=0.9891  strategy=best_fit_decreasing
pack_naive: ... truncation_rate=0.454 token_utilization=0.9977  strategy=concat_chunk
```

**#6 — built-in pack 3 strategies** (`tests/test_pack_strategies_and_extras.py`): default
resolves to `concat_chunk`; concat_chunk util ≥ next_fit, best_fit util ≥ next_fit;
best_fit truncation=0 when all docs fit; `next_fit` preserves legacy fixed-`seq_len` rows.

**#7 — `lines:` load** (test): `lines:<txt>` rows equal the jsonl `{"text": line}` equivalent.

**#8 — experiment full chain** (documented form):
```
prep → 234 (bf) / 232 (naive)
train -c config.yaml --output-summary … → step=20 loss=3.1319 → training complete
parity/bfd_reference.py → PASS LEVEL-1 PARITY — all 234 bins bit-exact across 5 fields
```

**#9 — zero regression**:
```
1993 passed, 7 skipped, 14 deselected, 28 warnings   (baseline 1963 + 30 new tests)
```
