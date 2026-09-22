"""Adapter for open_clip VisionTransformer models (guide §1.1, row 1: "OpenCLIP" / §1.2 OpenCLIP bullet).

open_clip's `ResidualAttentionBlock` wraps `nn.MultiheadAttention`, which fuses q/k/v into one
`in_proj_weight` (3E, E) and, when called normally (`need_weights=False`), may dispatch to a
fused SDPA kernel that never materialises attention probabilities (§1.2). We avoid that instead
of fighting it: q, k, v are recomputed manually by splitting `in_proj_weight`/`in_proj_bias` into
three (E, E) blocks, exactly what `nn.MultiheadAttention` does internally. This has been checked
against the module's own eager output (`need_weights=True`) to ~1e-7 max abs difference, well
inside the 1e-4 bound §1.3 test 1 requires — that test is still the thing that guards this at the
per-checkpoint level, since a future open_clip version could change the internals.

Scope: standard CLIP-style ViTs with `pool_type="tok"` (CLS pooling) and the default
`ResidualAttentionBlock` (no qk-norm / scaled-cosine / attentional_pool) — i.e. the OpenCLIP
checkpoints in Sets C and F (LAION / MetaCLIP ViT-B, ViT-L). Attentional-pool OpenCLIP variants
belong with the MAP-head adapter (map_head.py), not here; qk-norm custom blocks raise
NotImplementedError below rather than being silently mishandled.
"""
import torch

from .base import attn_probs, make_layout


class OpenCLIPViTAdapter:
    family = "open_clip_vit"

    # ------------------------------------------------------------------ setup
    def prepare(self, model):
        model.eval()
        vt = model.visual if hasattr(model, "visual") else model
        if getattr(vt, "attentional_pool", None) is not None:
            raise NotImplementedError("attentional-pool OpenCLIP models: use the MAP-head adapter instead")
        for blk in vt.transformer.resblocks:
            if type(blk).__name__ != "ResidualAttentionBlock":
                raise NotImplementedError(
                    f"{type(blk).__name__}: only the default ResidualAttentionBlock (no qk-norm / "
                    "scaled-cosine-attn) is supported by this adapter"
                )
        return model

    def _visual(self, model):
        return model.visual if hasattr(model, "visual") else model

    def n_blocks(self, model):
        return len(self._visual(model).transformer.resblocks)

    # ---------------------------------------------------------------- layout
    def token_layout(self, model):
        vt = self._visual(model)
        gh, gw = vt.grid_size
        assert gh == gw, "the guide assumes a square G x G grid"
        return make_layout(cls_idx=0, register_idx=[], patch_idx=range(1, 1 + gh * gw), grid=(gh, gw))

    # ------------------------------------------------------------ extraction
    @torch.no_grad()
    def residual_in(self, model, x, l):
        """Residual stream entering block l (post ln_pre, pre-block l)."""
        vt = self._visual(model)
        z = vt.conv1(x).reshape(x.shape[0], vt.conv1.out_channels, -1).permute(0, 2, 1)
        cls = vt.class_embedding.to(z.dtype).expand(z.shape[0], 1, -1)
        z = torch.cat([cls, z], dim=1) + vt.positional_embedding.to(z.dtype)
        z = vt.patch_dropout(z)
        z = vt.ln_pre(z)
        for blk in vt.transformer.resblocks[:l]:
            z = blk(z)
        return z

    def attn_scale(self, model, l):
        blk = self._visual(model).transformer.resblocks[l]
        return blk.attn.head_dim ** -0.5

    @torch.no_grad()
    def get_qkv(self, model, x, l):
        """q, k, v of block l, each (B, H, T, d_h), by splitting in_proj_weight/bias (see module docstring)."""
        z = self.residual_in(model, x, l)
        blk = self._visual(model).transformer.resblocks[l]
        a = blk.attn
        B, T, E = z.shape
        H, d = a.num_heads, a.head_dim
        zn = blk.ln_1(z)
        Wq, Wk, Wv = a.in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = a.in_proj_bias.chunk(3, dim=0)
        q = (zn @ Wq.T + bq).reshape(B, T, H, d).transpose(1, 2)
        k = (zn @ Wk.T + bk).reshape(B, T, H, d).transpose(1, 2)
        v = (zn @ Wv.T + bv).reshape(B, T, H, d).transpose(1, 2)
        return q, k, v

    @torch.no_grad()
    def attention_probs(self, model, x, l):
        q, k, _ = self.get_qkv(model, x, l)
        return attn_probs(q, k, self.attn_scale(model, l))

    # ------------------------------------------------- the four guide functions
    def get_global_query(self, model, x, l):
        q, _, _ = self.get_qkv(model, x, l)
        return q[:, :, 0, :]

    def get_keys_values(self, model, x, l):
        _, k, v = self.get_qkv(model, x, l)
        return k, v

    @torch.no_grad()
    def global_update(self, model, x, l):
        """Attn-branch contribution to the CLS token: out_proj(concat_h sum_i a_i^h v_i^h) + ls_1."""
        blk = self._visual(model).transformer.resblocks[l]
        a = blk.attn
        q, k, v = self.get_qkv(model, x, l)
        p = attn_probs(q[:, :, 0:1], k, self.attn_scale(model, l))  # (B,H,1,T), fp32
        o = (p @ v.float()).transpose(1, 2).reshape(p.shape[0], 1, -1)
        o = o.to(v.dtype) @ a.out_proj.weight.T + a.out_proj.bias
        return blk.ls_1(o)[:, 0].float()


_A = OpenCLIPViTAdapter()
prepare = _A.prepare
get_global_query = _A.get_global_query
get_keys_values = _A.get_keys_values
token_layout = _A.token_layout
global_update = _A.global_update
