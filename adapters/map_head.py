"""Adapter for MAP-head (attention-pooling) models — guide §1.1 row 3: SigLIP 2, Perception Encoder.

These have NO [CLS] token: the backbone is plain patch self-attention, and the global
representation is produced once, after the last block, by `AttentionPoolLatent`: a learned
`latent` vector projected to a query, cross-attending over the patch tokens' keys/values (its
own `kv` Linear, not the backbone blocks' own qkv). Per the guide (page 3 note under the family
table): "q_g is a fixed learned vector ... the query is literally the same for every image" —
true regardless of depth too, since it is one nn.Parameter.

ASSUMPTION beyond what the guide specifies (flag this to the supervisor before running at
scale): the guide's four functions are written as if every family has a per-layer q_g, but a
MAP-head model only has ONE real pooling step, at the end. To still get an M1/M3 layer sweep
here, this adapter reuses the pooling head's own (fixed) query and its own kv projection, applied
to the residual stream AT DEPTH l — i.e. "what would this fixed probe attend to if it read out at
layer l instead of the end". At l == n_blocks this is exactly the model's real pooling (post the
backbone's final norm, matching forward_features); at l < n_blocks it is a hypothetical probe with
no independent literature backing yet. Currently built on timm's AttentionPoolLatent
(SigLIP/SigLIP2 in timm); Perception Encoder has not been checked against this interface — it
may need its own kv/latent attribute names if its released code differs.

Backbone mechanics (residual_in / get_qkv / attn_scale) are delegated to TimmViTAdapter since the
blocks themselves are ordinary timm `Block`s here too, just with no cls_token.
"""
import torch

from .base import attn_probs, make_layout
from .timm_vit import TimmViTAdapter


class MapHeadAdapter:
    family = "map_head"

    def __init__(self):
        self._backbone = TimmViTAdapter()

    # ------------------------------------------------------------------ setup
    def prepare(self, model):
        if getattr(model, "attn_pool", None) is None or getattr(model, "cls_token", None) is not None:
            raise ValueError("not a MAP-head model (no attn_pool, or has a [CLS] token) - use TimmViTAdapter")
        self._backbone.prepare(model)  # sets fused_attn=False on every backbone block too
        model.attn_pool.fused_attn = False
        return model

    def n_blocks(self, model):
        return self._backbone.n_blocks(model)

    # ---------------------------------------------------------------- layout
    def token_layout(self, model):
        gh, gw = model.patch_embed.grid_size
        n_reg = model.num_reg_tokens if getattr(model, "reg_token", None) is not None else 0
        return make_layout(cls_idx=None, register_idx=range(n_reg), patch_idx=range(n_reg, n_reg + gh * gw), grid=(gh, gw))

    # ------------------------------------------------------------ extraction
    def _patch_stream(self, model, x, l):
        """Patch-token residual stream at depth l; the real post-backbone norm at l == n_blocks
        (matching forward_features), nothing extra for l < n_blocks (see module docstring)."""
        z = self._backbone.residual_in(model, x, l)
        return model.norm(z) if l == self.n_blocks(model) else z

    @torch.no_grad()
    def get_keys_values(self, model, x, l):
        """k, v the pooling head would use if it read out at depth l: attn_pool's own kv Linear
        applied to the layer-l patch stream. Shapes (B, H, T, d_h)."""
        ap = model.attn_pool
        z = self._patch_stream(model, x, l)
        B, T, _ = z.shape
        kv = ap.kv(z).reshape(B, T, 2, ap.num_heads, ap.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        return ap.k_norm(k), v

    def get_global_query(self, model, x, l):
        """The fixed probe query, (B, H, d_h), broadcast to x's batch size. Independent of x and
        l by construction (see module docstring) - both are accepted only for interface parity."""
        ap = model.attn_pool
        B = x.shape[0]
        q = ap.q(ap.latent).reshape(1, ap.latent_len, ap.num_heads, ap.head_dim).transpose(1, 2)
        return ap.q_norm(q)[:, :, 0, :].expand(B, -1, -1)

    @torch.no_grad()
    def attention_probs(self, model, x, l):
        q = self.get_global_query(model, x, l)
        k, _ = self.get_keys_values(model, x, l)
        return attn_probs(q.unsqueeze(2), k, model.attn_pool.scale).squeeze(2)

    # ------------------------------------------------- the four guide functions
    @torch.no_grad()
    def global_update(self, model, x, l):
        """Attn-branch contribution of the pooling head read out at depth l: proj(concat_h
        sum_i a_i^h v_i^h). At l == n_blocks this is the true attn output the head's residual MLP
        is then added to (AttentionPoolLatent.forward); the MLP branch is excluded here, same
        convention as the other adapters' global_update."""
        ap = model.attn_pool
        q = self.get_global_query(model, x, l)
        k, v = self.get_keys_values(model, x, l)
        p = attn_probs(q.unsqueeze(2), k, ap.scale)  # (B,H,1,T), fp32
        o = (p @ v.float()).transpose(1, 2).reshape(p.shape[0], 1, -1)
        o = ap.proj(o.to(v.dtype))
        return o[:, 0].float()


_A = MapHeadAdapter()
prepare = _A.prepare
get_global_query = _A.get_global_query
get_keys_values = _A.get_keys_values
token_layout = _A.token_layout
global_update = _A.global_update
