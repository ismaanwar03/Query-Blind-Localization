"""Guide §1.3, "the correctness test suite (non-negotiable)". Five tests are specified there;
what each one needs and what this sandbox can actually provide differs per test - read the class
docstrings, they say which of the two situations each test is in.

Run: cd repo && python -m pytest tests/test_correctness.py -v
"""
import torch
import pytest

from adapters.base import attn_probs
from tests.conftest import CHECKPOINTS, random_image

TOL = 1e-4  # the guide's tolerance for tests 1-2

REAL_ATTN_FAMILIES = ["deit3_small", "dinov2_reg_small", "open_clip_vitb32"]


def _make(name):
    adapter, model = CHECKPOINTS[name]()
    adapter.prepare(model)
    return adapter, model


# --------------------------------------------------------------------------------- test 1 and 2
# Runs now, no pretrained weights needed: both tests compare the adapter's hand-rolled q/k/v (or
# attn-branch) recompute against the SAME model's OWN real submodule call on a random-init model.
# Random weights are enough because both sides are the same forward computation done two ways;
# a real checkpoint would make the numbers meaningful but not this equality any tighter.

@pytest.mark.parametrize("name", REAL_ATTN_FAMILIES)
def test_attention_reconstruction(name):
    """Test 1. Our softmax(q.kT/sqrt(d_h)) vs. the model's own attention, captured independently
    of our adapter's q/k extraction (hook for timm, need_weights=True for open_clip)."""
    adapter, model = _make(name)
    x = random_image(adapter, model, seed=0)
    l = adapter.n_blocks(model) // 2

    if name == "open_clip_vitb32":
        z = adapter.residual_in(model, x, l)
        blk = model.transformer.resblocks[l]
        _, real = blk.attn(blk.ln_1(z), blk.ln_1(z), blk.ln_1(z), need_weights=True, average_attn_weights=False)
    else:  # timm families: capture attn_drop's input == the model's real post-softmax probs
        blk = model.blocks[l]
        captured = {}
        h = blk.attn.attn_drop.register_forward_pre_hook(lambda m, inp: captured.setdefault("p", inp[0]))
        adapter.residual_in(model, x, l + 1)  # runs block l for real, triggers the hook
        h.remove()
        real = captured["p"]

    ours = adapter.attention_probs(model, x, l)
    diff = (ours.float() - real.float()).abs().max().item()
    assert diff < TOL, f"{name}: attention reconstruction off by {diff}"


@pytest.mark.parametrize("name", REAL_ATTN_FAMILIES)
def test_attn_branch_reconstruction(name):
    """Test 2 (branch-level, since our adapters never call the block's own .forward() for the
    global token - see adapters/*.py docstrings). global_update() vs. the model's own attn
    submodule + its own LayerScale, called directly rather than through our qkv split."""
    adapter, model = _make(name)
    x = random_image(adapter, model, seed=1)
    l = adapter.n_blocks(model) // 2
    z = adapter.residual_in(model, x, l)

    if name == "open_clip_vitb32":
        blk = model.transformer.resblocks[l]
        real = blk.ls_1(blk.attention(q_x=blk.ln_1(z)))[:, 0]
    else:
        blk = model.blocks[l]
        real = blk.ls1(blk.attn(blk.norm1(z)))[:, adapter._cls(model)]

    ours = adapter.global_update(model, x, l)
    diff = (ours - real.float()).abs().max().item()
    assert diff < TOL, f"{name}: attn-branch reconstruction off by {diff}"


def test_map_head_end_to_end():
    """Tests 1+2 combined for the MAP-head family: there's no attn_drop-style hook point inside
    AttentionPoolLatent (softmax feeds straight into `@ v`, see map_head.py docstring), so the
    independent check is done at the whole-module level instead: global_update() + the head's own
    residual MLP, vs. calling model.attn_pool(z) for real, end to end."""
    adapter, model = _make("siglip_vitb16")
    x = random_image(adapter, model, seed=2)
    l = adapter.n_blocks(model)
    z = adapter._patch_stream(model, x, l)

    ap = model.attn_pool
    assert ap.latent_len == 1, "residual-add shortcut below assumes a single pooling query"
    ours = adapter.global_update(model, x, l)
    if ap.mlp is not None:
        ours = ours + ap.mlp(ap.norm(ours))
    real = ap(z)  # AttentionPoolLatent.forward already pools internally -> (B, C)
    diff = (ours - real.float()).abs().max().item()
    assert diff < TOL, f"siglip_vitb16: MAP-head end-to-end reconstruction off by {diff}"


# ---------------------------------------------------------------------------------------- test 3
# Runs now: uses patch_embed only (a plain strided conv), so it needs no trained weights - see
# adapters/*.py and the test body for why a random-init conv is enough to localise a bright patch.

@pytest.mark.parametrize("name", list(CHECKPOINTS))
def test_token_layout_grid(name):
    """Test 3. Paint one patch white on a black image at a known (row, col); the patch-embedding
    conv output for an all-zero patch is just its bias, while the white patch also gets
    weight_sum * 255, which dominates for typical random init - so the argmax of per-patch
    embedding-norm, reshaped via token_layout()'s grid, should land exactly on that patch. This
    checks the (G, G) reshape / patch_idx indexing, not attention, so it needs no trained weights."""
    adapter, model = _make(name)
    layout = adapter.token_layout(model)
    gh, gw = layout["grid"]
    row, col = gh // 3, gw // 2

    patch_embed = model.patch_embed if hasattr(model, "patch_embed") else model.conv1
    ps = patch_embed.patch_size[0] if hasattr(patch_embed, "patch_size") else patch_embed.kernel_size[0]
    img = torch.zeros(1, 3, gh * ps, gw * ps)
    img[:, :, row * ps:(row + 1) * ps, col * ps:(col + 1) * ps] = 1.0

    with torch.no_grad():
        pe = patch_embed(img)
        pe = pe.flatten(2).transpose(1, 2) if pe.dim() == 4 else pe  # open_clip conv1 is (B,D,H,W)
    norms = pe[0].norm(dim=-1).reshape(gh, gw)
    got = divmod(int(norms.argmax()), gw)
    assert got == (row, col), f"{name}: expected argmax at {(row, col)}, got {got}"


# ---------------------------------------------------------------------------------------- test 4
class TestRegisterIndexing:
    """Test 4. 'Assert the R register tokens have mean L2 norm far above the patch median' is an
    empirical property DINOv2-reg acquires from training (Darcet et al., 2023) - it does not hold
    for a randomly-initialised model, where register and patch tokens are drawn from the same
    init distribution and have no reason to separate. This needs the real dinov2_reg_small
    checkpoint (timm/vit_small_patch14_reg4_dinov2.lvd142m or similar), which this sandbox cannot
    download - see tests/README.md. The check itself (below) is ready to run once weights load."""

    @pytest.mark.skip(reason="needs real pretrained DINOv2-reg weights; see tests/README.md")
    def test_register_norms_exceed_patch_median(self):
        adapter, model = _make("dinov2_reg_small")
        x = random_image(adapter, model, batch=4, seed=3)
        layout = adapter.token_layout(model)
        z = adapter.residual_in(model, x, adapter.n_blocks(model))
        norms = z.norm(dim=-1)
        reg_mean = norms[:, layout["register_idx"]].mean()
        patch_median = norms[:, layout["patch_idx"]].median()
        assert reg_mean > patch_median


# ---------------------------------------------------------------------------------------- test 5
@pytest.mark.skip(reason="needs internet: real DINO ViT-S/16 weights, the DINO-paper images, and "
                         "a LOST implementation + VOC07 to reproduce CorLoc; see tests/README.md")
def test_known_result_replication():
    pass
