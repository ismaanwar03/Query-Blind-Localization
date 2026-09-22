"""Shared helpers for the per-family adapters (guide §1.1-1.2).

Rules enforced here:
  * softmax is always computed in fp32 (§1.2), whatever dtype the model runs in;
  * the 1/sqrt(d_h) scale is applied explicitly and comes from the adapter, never assumed.
"""
import torch


def attn_probs(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    """softmax(q k^T * scale) in fp32.  q: (B,H,Tq,d_h), k: (B,H,Tk,d_h) -> (B,H,Tq,Tk)."""
    logits = (q.float() @ k.float().transpose(-2, -1)) * scale
    return logits.softmax(dim=-1)


def make_layout(cls_idx, register_idx, patch_idx, grid, dist_idx=()):
    """The dict every adapter's token_layout() returns.

    cls_idx      int | None   index of the [CLS] token (None for MAP-head models)
    register_idx list[int]    register tokens (empty if none)
    patch_idx    list[int]    patch tokens, row-major over the (G, G) grid
    grid         (G, G)
    dist_idx     list[int]    DeiT-style distillation token(s): extra non-patch tokens
    """
    return dict(
        cls_idx=cls_idx,
        register_idx=list(register_idx),
        patch_idx=list(patch_idx),
        grid=tuple(grid),
        dist_idx=list(dist_idx),
    )
