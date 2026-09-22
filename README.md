# Step 1 (guide §1): adapters + correctness suite

## What's here
- `adapters/base.py` - shared fp32-softmax helper and the `token_layout()` dict shape.
- `adapters/timm_vit.py` - DeiT-III, DINO, iBOT, MoCo v3, MAE, AugReg, DINOv2, DINOv2-reg
  (any timm ViT built from plain pre-LN `Block`s). Also covers TS-CAM: it's DeiT-S under the
  hood, so it needs no separate file for the q/k/v extraction this step is about.
- `adapters/open_clip_vit.py` - OpenCLIP ViTs (`nn.MultiheadAttention`-based). q/k/v are
  recomputed by splitting `in_proj_weight` rather than fighting the fused SDPA path (§1.2);
  scoped to CLS-pooled models (Sets C/F), not attentional-pool OpenCLIP variants.
- `adapters/map_head.py` - SigLIP 2 / Perception Encoder (MAP head, no [CLS]). **Contains a
  real assumption beyond the guide - see "Needs your/supervisor's sign-off" below.**
- `tests/test_correctness.py` - guide §1.3's five tests. 1-3 run now and pass; 4-5 are `skip`ped
  with a reason, not faked (see "Can't run here" below).
- `configs/checkpoints.yaml` - the §1.4 registry schema. Only the guide's own `dino_vits16`
  example is verified; the other four rows are placeholders (`verify: true`) for the checkpoints
  the test suite uses, not populated from real model cards yet.

Run the suite: `cd repo && python -m pytest tests/test_correctness.py -v` → 11 passed, 2 skipped.

## Needs your/supervisor's sign-off
`map_head.py` extends the guide to get a per-layer QII sweep out of a model that only pools
once. It reuses the MAP head's fixed probe query and its own k/v projection at whatever depth
you probe, on the reading that "same fixed query, different depth's keys" is a sound way to ask
the query-invariance question at that depth. That's my extension, not something the guide states
- flag it before this feeds into real numbers. The docstring in that file has the full reasoning.

## Can't run here
This sandbox has no route to Hugging Face Hub or any weights CDN (see the network allowlist -
pypi/npm/github only), so:
- Tests 4-5 (register-norm separation, DINO/LOST replication) need real trained weights (test 5
  also needs the DINO-paper images and a LOST implementation) that can't be fetched here. Both
  are marked `skip` with why, not stubbed to pass.
- `configs/checkpoints.yaml` can't be filled in against real model cards from here either.

Both are ready to run as-is once there's an environment with internet/HF access - worth deciding
now whether that's Colab, a lab machine, or somewhere else, since it changes how the rest of this
gets built.
