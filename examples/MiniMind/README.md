# MiniMind — lighttrain port

Faithful port of [jingyaogong/minimind](https://github.com/jingyaogong/minimind)
using lighttrain as the training framework.

## What's here

```
examples/MiniMind/
  model/
    model_minimind.py     # 原始模型定义（verbatim copy）
    model_adapter.py      # @register("model", "minimind")
    model_lora.py         # LoRA 工具（verbatim copy）
    tokenizer.json / tokenizer_config.json   # vocab_size=6400 自定义分词器
  dataset/
    lm_dataset.py         # @register datasets: minimind_pretrain/sft/dpo/rlaif
    dataset.md            # 数据集说明
  configs/
    pretrain.yaml         # 预训练 (next_token)
    sft.yaml              # 全参数 SFT
    lora.yaml             # LoRA SFT
    dpo.yaml              # DPO 偏好训练
    grpo.yaml             # GRPO
    ppo.yaml              # PPO（含 reward model）
    distillation.yaml     # 知识蒸馏
  scripts/
    chat_api.py           # OpenAI-compatible 客户端
    web_demo.py           # Streamlit 网页对话
    serve_openai_api.py   # FastAPI OpenAI 兼容 API 服务
    eval_toolcall.py      # 工具调用评测
    convert_model.py      # 转换为 HuggingFace 格式
  trainer/
    trainer_utils.py      # 共享工具（改 imports，适配 lighttrain）
    rollout_engine.py     # PPO/GRPO rollout（verbatim copy）
```

## 模型架构

MiniMind 采用现代 LLM 组件，与 `tiny_lm` 完全不同：
- **RMSNorm**（而非 LayerNorm）
- **RoPE**（支持 YaRN 长文本扩展）
- **GQA**（分组查询注意力，`num_key_value_heads < num_attention_heads`）
- **QK-Norm**（Q/K 上加 RMSNorm）
- **SwiGLU** FFN（`gate_proj * up_proj → down_proj`）
- **MoE**（可选，`use_moe=true`）
- KV-cache 推理 + `GenerationMixin`

## 训练流水线映射

| MiniMind 原始脚本 | lighttrain 对应 |
|---|---|
| `train_pretrain.py` | `pretrain` trainer + `next_token` objective |
| `train_full_sft.py` | 同上，使用 `minimind_sft` 数据集 |
| `train_lora.py` | `lora` model wrapper + `pretrain` trainer |
| `train_dpo.py` | `preference` trainer + `dpo` loss |
| `train_grpo.py` | `grpo` trainer |
| `train_ppo.py` | `ppo` trainer |
| `train_distillation.py` | `pretrain` trainer + `distill` loss |

## 快速开始 — 预训练

```bash
# 1. 准备数据（从 HuggingFace 下载）
#    数据格式 JSONL：{"text": "..."}

# 2. 训练（默认 hidden_size=512, 8 层，~26M 参数）
lighttrain train -c examples/MiniMind/configs/pretrain.yaml \
    ++data.dataset.data_path=/path/to/pretrain.jsonl

# 3. SFT
lighttrain train -c examples/MiniMind/configs/sft.yaml \
    ++data.dataset.jsonl_path=/path/to/sft.jsonl
```

## 推理 / 服务

```bash
# OpenAI 兼容 API 服务（需要 fastapi + uvicorn）
python examples/MiniMind/scripts/serve_openai_api.py \
    --load_from examples/MiniMind/model

# Streamlit 网页对话（需要 streamlit）
streamlit run examples/MiniMind/scripts/web_demo.py

# 命令行对话客户端（服务已启动后）
python examples/MiniMind/scripts/chat_api.py

# 转换为 HuggingFace 格式
python examples/MiniMind/scripts/convert_model.py
```

## MoE 注意事项

设置 `use_moe: true` 时，模型会产生 `aux_loss`（路由负载均衡损失）。
lighttrain 的标准 `next_token` objective 不包含 aux_loss；若需要，可通过自定义
callback 或 objective 在 `ModelOutput.outputs["aux_loss"]` 上叠加。

## 与 nanoGPT 的对比

| | nanoGPT | MiniMind |
|---|---|---|
| 架构 | GPT-2（绝对位置编码，标准 MHA） | Llama 风格（RoPE，GQA，SwiGLU） |
| 分词器 | tiktoken GPT-2 BPE / char-level | 自定义（vocab=6400，中文优化） |
| 训练目标 | 预训练 + 微调 | 预训练 → SFT → RLHF 全流水线 |
| 教育重点 | GPT 核心原理 | 现代小型 LLM 完整训练流程 |
