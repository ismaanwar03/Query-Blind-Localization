"""Adapter for timm VisionTransformer models (guide §1.1).

Covers DeiT-III, DINO, iBOT, MoCo v3, MAE, AugReg, DINOv2 and DINOv2-reg: any timm ViT made of
pre-LN `Block`s with a fused qkv projection.  Token order in timm is
    [CLS] [dist token if DeiT-distilled] [registers if any] [patches, row-major]
The global token is [CLS]; its query q_g is the query the [CLS] token uses in block l.

The four functions the guide asks for are exposed at module level (bottom of file); the class
form exists so the tests can subclass it and inject known bugs.
"""
import torch

from .base import attn_probs, make_layout


class TimmViTAdapter:
    family = "timm_vit"

    # ------------------------------------------------------------------ setup
    def prepare(self, model):
        """§1.2: eval mode + fused attention OFF in every block (fused kernels never build the
        attention matrix, so hooks would silently see nothing)."""
        model.eval()
        for blk in model.blocks:
            if type(blk).__name__ != "Block":
                raise NotImplementedError(
                    f"{type(blk).__name__}: only the pre-LN timm `Block` is supported; "
                    "post-LN / parallel / RoPE variants need their own adapter (see §12: pre-LN vs post-LN)."
                )
            if getattr(blk.attn, "gate", None) is not None:
                raise NotImplementedError("gated attention is not supported")
            blk.attn.fused_attn = False
        return model

    def n_blocks(self, model):
        return len(model.blocks)

    def forward(self, model, x):
        return model.forward_features(x)

    # ---------------------------------------------------------------- layout
    def token_layout(self, model):
        gh, gw = model.patch_embed.grid_size
        assert gh == gw, "the guide assumes a square G x G grid"
        n_prefix = model.num_prefix_tokens
        idx, cls_idx, dist_idx = 0, None, []
        if getattr(model, "cls_token", None) is not None:
            cls_idx, idx = 0, 1
        if getattr(model, "dist_token", None) is not None:
            dist_idx, idx = [idx], idx + 1
        n_reg = model.num_reg_tokens if getattr(model, "reg_token", None) is not None else 0
        reg_idx = list(range(idx, idx + n_reg))
        idx += n_reg
        assert idx == n_prefix, f"prefix tokens: counted {idx}, timm says {n_prefix}"
        return make_layout(cls_idx, reg_idx, range(n_prefix, n_prefix + gh * gw), (gh, gw), dist_idx)

    # ------------------------------------------------------------ extraction
    @torch.no_grad()
    def residual_in(self, model, x, l):
        """Residual stream entering block l.  l == n_blocks gives the last block's output (pre final norm)."""
        z = model.patch_embed(x)
        z = model._pos_embed(z)
        if isinstance(z, tuple):
            raise NotImplementedError("RoPE-style ViTs return (tokens, rope); not supported here")
        z = model.patch_drop(z)
        z = model.norm_pre(z)
        for blk in model.blocks[:l]:
            z = blk(z)
        return z

    def attn_scale(self, model, l):
        return model.blocks[l].attn.scale

    @torch.no_grad()
    def get_qkv(self, model, x, l):
        """q, k, v of block l, each (B, H, T, d_h), exactly as the block feeds them to softmax
        (i.e. after qk-LayerNorm where the model has it)."""
        z = self.residual_in(model, x, l)
        blk = model.blocks[l]
        a = blk.attn
        B, T, _ = z.shape
        qkv = a.qkv(blk.norm1(z)).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = a.q_norm(q), a.k_norm(k)
        return q, k, v

    @torch.no_grad()
    def attention_probs(self, model, x, l):
        """Full (B, H, T, T) attention of block l, recomputed from q, k in fp32 (used by the tests)."""
        q, k, _ = self.get_qkv(model, x, l)
        return attn_probs(q, k, self.attn_scale(model, l))

    # ------------------------------------------------- the four guide functions
    def _cls(self, model):
        c = self.token_layout(model)["cls_idx"]
        if c is None:
            raise ValueError("model has no [CLS] token (MAP-head model?) - it needs the MAP-head adapter")
        return c

    def get_global_query(self, model, x, l):
        """q_g of block l: (B, H, d_h).  NB: the guide writes (model, l); a [CLS] query depends on the
        image, so x is required here.  (For MAP-head models it is a fixed probe and x is unused.)"""
        q, _, _ = self.get_qkv(model, x, l)
        return q[:, :, self._cls(model), :]

    def get_keys_values(self, model, x, l):
        _, k, v = self.get_qkv(model, x, l)
        return k, v

    @torch.no_grad()
    def global_update(self, model, x, l):
        """What block l's attention branch adds to the global token, (B, D):
        LayerScale( W_O concat_h sum_i a_i^h v_i^h + b_O ) - i.e. Delta_g of §4.1 incl. bias and gamma."""
        blk = model.blocks[l]
        a = blk.attn
        c = self._cls(model)
        q, k, v = self.get_qkv(model, x, l)
        p = attn_probs(q[:, :, c:c + 1], k, self.attn_scale(model, l))  # (B,H,1,T), fp32
        o = (p @ v.float()).transpose(1, 2).reshape(p.shape[0], 1, -1)  # heads concatenated
        o = a.proj(a.norm(o.to(v.dtype)))
        return blk.ls1(o)[:, 0].float()


_A = TimmViTAdapter()
prepare = _A.prepare
get_global_query = _A.get_global_query
get_keys_values = _A.get_keys_values
token_layout = _A.token_layout
global_update = _A.global_update
