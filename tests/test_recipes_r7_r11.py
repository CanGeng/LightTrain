"""CPU smoke tests for M7 recipes R7–R11.

These tests instantiate the core components used by each recipe and run a few
steps to confirm end-to-end plumbing works on CPU without a full trainer loop.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# R7 — RWKV stateful language model
# ---------------------------------------------------------------------------

def test_r7_rwkv_smoke():
    from lighttrain.builtin_plugins.architectures.rwkv import TinyRWKVConfig, TinyRWKVModel, rwkv_profile
    cfg = TinyRWKVConfig(vocab_size=32, embed_dim=16, num_layers=2, chunk_size=8)
    model = TinyRWKVModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(5):
        ids = torch.randint(0, 32, (2, 8))
        # Simulate doc boundary reset at step 2
        reset = step == 2
        out = model(input_ids=ids, _reset_state=reset)
        logits = out.outputs["logits"]  # (B, T, V)
        target = torch.randint(0, 32, (2, 8))
        loss = nn.functional.cross_entropy(logits.view(-1, 32), target.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# R8 — Diffusion eps prediction
# ---------------------------------------------------------------------------

def test_r8_diffusion_smoke():
    from lighttrain.builtin_plugins.objectives.diffusion import DiffusionObjective
    from lighttrain.protocols import LossContext, ModelOutput

    obj = DiffusionObjective(target="eps", noise_schedule="linear", timesteps=50)
    # Tiny denoiser
    model = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 8))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(10):
        batch = {"x": torch.randn(4, 8)}
        batch = obj.prepare_batch(batch, step=0, device="cpu")
        noisy = batch["noisy_x"]
        pred = model(noisy)
        mo = ModelOutput(outputs={"pred": pred})
        ld = obj(mo, batch, LossContext())
        loss = ld["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert torch.isfinite(torch.tensor(losses[-1]))


# ---------------------------------------------------------------------------
# R9 — JEPA
# ---------------------------------------------------------------------------

def test_r9_jepa_smoke():
    from lighttrain.builtin_plugins.architectures.jepa import (
        JEPAEncoder, JEPAModelConfig, EMATargetEncoder, JEPAPredictor,
    )
    from lighttrain.builtin_plugins.objectives.jepa import JEPAObjective
    from lighttrain.protocols import LossContext, ModelOutput

    cfg = JEPAModelConfig(patch_dim=16, embed_dim=32, num_heads=2, depth=2, predictor_depth=1)
    enc = JEPAEncoder(cfg)
    target_enc = EMATargetEncoder(enc, momentum=0.99)
    predictor = JEPAPredictor(cfg)
    opt = torch.optim.Adam(list(enc.parameters()) + list(predictor.parameters()), lr=1e-3)

    obj = JEPAObjective(num_context_patches=6, num_target_patches=4)
    obj.set_target_encoder(target_enc)

    for step in range(5):
        patches = torch.randn(2, 16, 16)
        batch = {"patches": patches}
        batch = obj.prepare_batch(batch, step=step, device="cpu")

        ctx_emb = enc(batch["context_patches"])
        tgt_emb = target_enc(batch["target_patches"])

        # Predictor takes context embeddings and target position queries (embed_dim)
        target_pos_queries = torch.randn(2, 4, 32)  # (B, num_target, embed_dim)
        pred_out = predictor(ctx_emb, target_pos_queries)
        mo = ModelOutput(outputs={"pred_embeddings": pred_out}, extras={"target_embeddings": tgt_emb})

        ld = obj(mo, batch, LossContext())
        loss = ld["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        obj.ema_step(enc)

    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# R10a — PCN
# ---------------------------------------------------------------------------

def test_r10a_pcn_smoke():
    from lighttrain.builtin_plugins.update_rules.pcn import PCNUpdateRule
    from lighttrain.engine._context import StepContext

    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    rule = PCNUpdateRule(n_infer=5, lr_weight=0.05)

    ctx = StepContext()
    ctx.model = model
    ctx.bus = None
    ctx.accelerator = None
    ctx.step = 0
    ctx.epoch = 0
    ctx.metrics = {}
    ctx.extras = {}
    ctx.loss_fn = None
    ctx.optimizer = None
    ctx.scheduler = None

    losses = []
    for i in range(5):
        batch = {"x": torch.randn(4, 8), "labels": torch.zeros(4, 4)}
        m = rule.step(model, batch, ctx)
        losses.append(m["loss"])
        ctx.step += 1

    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


# ---------------------------------------------------------------------------
# R10b — ForwardForward
# ---------------------------------------------------------------------------

def test_r10b_ff_smoke():
    from lighttrain.builtin_plugins.update_rules.forward_forward import ForwardForwardUpdateRule
    from lighttrain.engine._context import StepContext

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rule = ForwardForwardUpdateRule(threshold=2.0)

    ctx = StepContext()
    ctx.model = model
    ctx.bus = None
    ctx.accelerator = None
    ctx.step = 0
    ctx.epoch = 0
    ctx.metrics = {}
    ctx.extras = {}
    ctx.loss_fn = None
    ctx.optimizer = opt
    ctx.scheduler = None

    losses = []
    for i in range(5):
        batch = {"x": torch.randn(4, 8)}
        m = rule.step(model, batch, ctx)
        losses.append(m["loss"])
        ctx.step += 1

    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


# ---------------------------------------------------------------------------
# R11 — MeZO
# ---------------------------------------------------------------------------

def test_r11_mezo_smoke():
    from lighttrain.update_rules.mezo import MeZOUpdateRule
    from lighttrain.engine._context import StepContext
    from lighttrain.protocols import ModelOutput

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

        def forward(self, **batch):
            x = batch["input_ids"].float()
            return ModelOutput(outputs={"logits": self.net(x)})

    class FakeLoss:
        def __call__(self, out, batch, ctx):
            logits = out.outputs["logits"]
            labels = batch.get("labels", torch.zeros(logits.shape[0], dtype=torch.long))
            return {"loss": nn.functional.cross_entropy(logits, labels)}

    model = Wrapper()
    loss_fn = FakeLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    rule = MeZOUpdateRule(eps=1e-2)

    ctx = StepContext()
    ctx.model = model
    ctx.bus = None
    ctx.accelerator = None
    ctx.step = 0
    ctx.epoch = 0
    ctx.metrics = {}
    ctx.extras = {}
    ctx.loss_fn = loss_fn
    ctx.optimizer = opt
    ctx.scheduler = None

    losses = []
    for i in range(10):
        batch = {"input_ids": torch.randn(4, 8), "labels": torch.randint(0, 4, (4,))}
        m = rule.step(model, batch, ctx)
        losses.append(m["loss"])
        ctx.step += 1
        # MeZO must never leave gradients
        for p in model.parameters():
            assert p.grad is None

    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
