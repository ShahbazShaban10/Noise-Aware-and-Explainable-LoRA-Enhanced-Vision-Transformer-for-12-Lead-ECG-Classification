"""Architecture, LoRA injection, freezing, and parameter accounting.

The central claims under test:
  * the backbone is exactly 1,648,839 parameters (manuscript Table 8)
  * LoRA r=8 adds exactly 131,072 trainable parameters, 92.05% fewer than full fine-tuning
  * "frozen backbone" means base weights receive no gradient, while gradients still flow
    *through* frozen layers to adapters in earlier blocks
  * zero-initialised B makes the adapted model identical to the backbone at step 0
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ecgvit.config import (CLASS_NAMES, LoRAConfig, ModelConfig, N_LEADS,  # noqa: E402
                           N_SAMPLES)
from ecgvit.lora import LoRALinear, count_parameters, inject_lora  # noqa: E402
from ecgvit.model import LoRAViT, build_model  # noqa: E402

pytestmark = pytest.mark.torch

PAPER_BASE_PARAMS = 1_648_839
PAPER_LORA_TRAINABLE = 131_072
PAPER_LORA_TOTAL = 1_779_911
PAPER_REDUCTION_PCT = 92.05


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
def test_default_backbone_parameter_count_matches_the_manuscript():
    model = LoRAViT(ModelConfig())
    n = sum(p.numel() for p in model.parameters())
    assert n == PAPER_BASE_PARAMS, (
        f"backbone has {n:,} parameters, manuscript Table 8 reports "
        f"{PAPER_BASE_PARAMS:,}. Check embed_dim=128, depth=8, heads=8, patch_len=100."
    )


def test_default_geometry():
    cfg = ModelConfig()
    model = LoRAViT(cfg)
    assert model.n_patches == 50
    assert cfg.depth == 8 and cfg.num_heads == 8 and cfg.embed_dim == 128
    assert model.cls_token is not None
    assert model.pos_embed.shape == (1, 50, 128)


def test_forward_shapes():
    model = LoRAViT(ModelConfig()).eval()
    x = torch.randn(3, N_LEADS, N_SAMPLES)
    with torch.no_grad():
        tokens = model.forward_tokens(x)
        feats = model.forward_features(x)
        logits = model(x)
    assert tokens.shape == (3, 51, 128), "1 CLS + 50 patch tokens"
    assert feats.shape == (3, 128)
    assert logits.shape == (3, len(CLASS_NAMES))


def test_tokeniser_is_two_stage_and_hierarchical():
    """Two convolutions, not one: stage 1 mixes leads into sub-patches, stage 2 pools
    adjacent sub-patches into the 50 temporal tokens."""
    model = LoRAViT(ModelConfig())
    tok = model.tokeniser
    assert tok.conv1.in_channels == 12 and tok.conv1.out_channels == 64
    assert tok.conv1.kernel_size == (50,) and tok.conv1.stride == (50,)
    assert tok.conv2.in_channels == 64 and tok.conv2.out_channels == 128
    assert tok.conv2.kernel_size == (2,) and tok.conv2.stride == (2,)
    with torch.no_grad():
        out = tok(torch.randn(2, 12, 5000))
    assert out.shape == (2, 50, 128)


def test_encoder_is_pre_norm():
    """Pre-norm: the residual branch normalises its input. A post-norm block would leave
    x unchanged when the sublayers output zero *after* the norm -- this test distinguishes
    them by zeroing the sublayers and checking the residual passes through untouched."""
    from ecgvit.model import EncoderBlock

    blk = EncoderBlock(32, 4).eval()
    with torch.no_grad():
        for p in blk.attn.proj.parameters():
            p.zero_()
        for p in blk.mlp.fc2.parameters():
            p.zero_()
        x = torch.randn(2, 5, 32)
        y = blk(x)
    assert torch.allclose(y, x, atol=1e-6), "pre-norm residual did not pass x through"


def test_model_rejects_wrong_input_geometry():
    model = LoRAViT(ModelConfig()).eval()
    with pytest.raises(ValueError):
        model(torch.randn(2, 8, N_SAMPLES))
    with pytest.raises(ValueError):
        model(torch.randn(2, N_LEADS, 3000))


def test_embed_dim_must_divide_heads():
    with pytest.raises(ValueError, match="divisible"):
        LoRAViT(ModelConfig(embed_dim=130, num_heads=8))


# ---------------------------------------------------------------------------
# LoRA parameter accounting
# ---------------------------------------------------------------------------
def test_lora_parameter_counts_match_the_manuscript():
    model, acct = build_model(ModelConfig(), LoRAConfig(rank=8, alpha=16))
    assert acct.trainable_parameters == PAPER_LORA_TRAINABLE, (
        f"{acct.trainable_parameters:,} trainable, manuscript reports "
        f"{PAPER_LORA_TRAINABLE:,}"
    )
    assert acct.total_parameters == PAPER_LORA_TOTAL
    assert acct.baseline_trainable_parameters == PAPER_BASE_PARAMS
    assert acct.trainable_reduction_pct == pytest.approx(PAPER_REDUCTION_PCT, abs=0.01)


def test_lora_targets_every_projection_the_manuscript_lists():
    """W_q, W_k, W_v (fused as qkv), W_o (proj), W_1 (fc1) and W_2 (fc2), in all 8 blocks."""
    model, acct = build_model(ModelConfig(), LoRAConfig())
    assert acct.n_lora_layers == 4 * ModelConfig().depth == 32
    for blk in model.blocks:
        for mod in (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2):
            assert isinstance(mod, LoRALinear)


def test_reduction_exceeds_the_ninety_percent_bar():
    _, acct = build_model(ModelConfig(), LoRAConfig(rank=8))
    assert acct.trainable_reduction_pct >= 90.0


@pytest.mark.parametrize("rank", [1, 2, 4, 8, 16, 32])
def test_trainable_count_scales_linearly_with_rank(rank):
    _, acct = build_model(ModelConfig(), LoRAConfig(rank=rank, alpha=2 * rank))
    assert acct.trainable_parameters == PAPER_LORA_TRAINABLE // 8 * rank


def test_injection_fails_loudly_when_nothing_matches():
    model = LoRAViT(ModelConfig(depth=1, embed_dim=32, num_heads=4))
    with pytest.raises(RuntimeError, match="matched no nn.Linear"):
        inject_lora(model, LoRAConfig(target_modules=("no_such_layer",)))


def test_rank_must_be_positive():
    import torch.nn as nn

    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(8, 8), rank=0)


# ---------------------------------------------------------------------------
# Freezing semantics
# ---------------------------------------------------------------------------
def test_only_lora_matrices_are_trainable():
    model, _ = build_model(ModelConfig(), LoRAConfig())
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    for name in trainable:
        assert "lora_A" in name or "lora_B" in name, f"{name} should be frozen"
    head = [p.requires_grad for n, p in model.named_parameters() if n.startswith("head.")]
    assert not any(head), "the head must be frozen; it comes from the pretrained backbone"


def test_backbone_weights_receive_no_gradient():
    model, _ = build_model(ModelConfig(seq_len=1000, patch_len=100, embed_dim=32,
                                       depth=2, num_heads=4),
                           LoRAConfig(rank=4))
    model.train()
    out = model(torch.randn(2, 12, 1000))
    out.sum().backward()
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue
        assert p.grad is None or torch.all(p.grad == 0), (
            f"frozen parameter {name} received a gradient"
        )


def test_gradients_still_reach_adapters_in_the_first_block():
    """Freezing must not detach the graph. If it did, only the last block's adapters would
    train and the model would barely learn -- a silent, hard-to-spot failure.

    Note which matrix is checked. With B = 0 at init, dL/dA = 0 identically (the LoRA
    branch is x @ Aᵀ @ Bᵀ, so A's gradient carries a factor of B). So a zero A-gradient on
    step 0 is correct LoRA behaviour, not a broken graph -- B's gradient is the signal that
    the backward pass reached this block. A only starts moving once B leaves zero, which
    the second half of this test checks.
    """
    model, _ = build_model(ModelConfig(seq_len=1000, patch_len=100, embed_dim=32,
                                       depth=4, num_heads=4),
                           LoRAConfig(rank=4))
    model.train()
    model(torch.randn(2, 12, 1000)).sum().backward()

    first = model.blocks[0].attn.qkv
    assert isinstance(first, LoRALinear)
    assert first.lora_B.grad is not None and first.lora_B.grad.abs().sum() > 0, (
        "block 0 adapters got no gradient; the backward graph is being cut"
    )
    assert first.lora_A.grad is not None and torch.all(first.lora_A.grad == 0), (
        "with B initialised to zero, A's gradient must be exactly zero on the first step"
    )

    # Once B is non-zero, A must start receiving gradient too.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.05)
    model(torch.randn(2, 12, 1000)).sum().backward()
    assert first.lora_A.grad.abs().sum() > 0, (
        "A receives no gradient even with B non-zero; the adapter is not training"
    )


def test_lora_at_init_is_functionally_identical_to_the_backbone():
    """B is zero-initialised, so delta-W = BA = 0 and the adapted model must reproduce the
    backbone exactly. Without this the 'frozen backbone' claim is untestable."""
    torch.manual_seed(0)
    cfg = ModelConfig(seq_len=1000, patch_len=100, embed_dim=32, depth=2, num_heads=4)
    base = LoRAViT(cfg).eval()
    adapted = LoRAViT(cfg)
    adapted.load_state_dict(base.state_dict())
    inject_lora(adapted, LoRAConfig(rank=4))
    adapted.eval()

    x = torch.randn(4, 12, 1000)
    with torch.no_grad():
        assert torch.allclose(base(x), adapted(x), atol=1e-6)


def test_lora_changes_the_output_once_b_is_nonzero():
    torch.manual_seed(0)
    cfg = ModelConfig(seq_len=1000, patch_len=100, embed_dim=32, depth=2, num_heads=4)
    base = LoRAViT(cfg).eval()
    adapted = LoRAViT(cfg)
    adapted.load_state_dict(base.state_dict())
    inject_lora(adapted, LoRAConfig(rank=4))
    with torch.no_grad():
        for m in adapted.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.05)
    adapted.eval()
    x = torch.randn(4, 12, 1000)
    with torch.no_grad():
        assert not torch.allclose(base(x), adapted(x), atol=1e-4)


def test_scaling_is_alpha_over_rank():
    import torch.nn as nn

    layer = LoRALinear(nn.Linear(16, 16), rank=4, alpha=16)
    assert layer.scaling == pytest.approx(4.0)
    with torch.no_grad():
        layer.lora_A.fill_(1.0)
        layer.lora_B.fill_(1.0)
    dw = layer.delta_weight()
    assert dw.shape == (16, 16)
    assert torch.allclose(dw, torch.full((16, 16), 4.0 * 4))  # (B@A)=rank * scaling


def test_merge_reproduces_the_adapted_forward():
    """Merging folds BA into W for deployment; it must be numerically equivalent."""
    import torch.nn as nn

    torch.manual_seed(1)
    base = nn.Linear(24, 24)
    layer = LoRALinear(base, rank=4, alpha=8).eval()
    with torch.no_grad():
        layer.lora_B.normal_(0, 0.1)
    x = torch.randn(6, 24)
    with torch.no_grad():
        assert torch.allclose(layer(x), layer.merge()(x), atol=1e-5)


def test_lora_disabled_leaves_the_model_fully_trainable():
    model, acct = build_model(ModelConfig(), LoRAConfig(enabled=False))
    assert acct.trainable_parameters == PAPER_BASE_PARAMS
    assert acct.lora_parameters == 0
    assert all(p.requires_grad for p in model.parameters())


def test_state_dict_round_trips_through_save_and_load(tmp_path):
    model, _ = build_model(ModelConfig(), LoRAConfig())
    path = tmp_path / "m.pt"
    torch.save({"state_dict": model.state_dict(), "class_names": list(CLASS_NAMES)}, path)
    reloaded, _ = build_model(ModelConfig(), LoRAConfig())
    ckpt = torch.load(path, map_location="cpu")
    reloaded.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval(); reloaded.eval()
    x = torch.randn(2, N_LEADS, N_SAMPLES)
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x), atol=1e-6)
