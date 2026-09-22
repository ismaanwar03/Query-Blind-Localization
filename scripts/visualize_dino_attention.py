"""Guide §1.3 test 5, qualitative half: does DINO ViT-S/16's last-layer [CLS] attention show the
DINO segmentation phenomenon, per head?

Not a pytest assertion - the guide asks you to *look at* this and compare to the known result, so
it's a script that saves a figure, not a pass/fail check. (The paper's own Figure 1 is actually
the ViT-S/8 model - see the chat note - so this checks the same phenomenon on the /16 checkpoint
our zoo uses, not a pixel-identical match to that specific figure.)

Uses TimmViTAdapter directly, so a real run here is also an extra real-weights check on that
adapter beyond tests/test_correctness.py's random-init suite.

Usage (needs internet - real DINO weights + these two COCO images):
    python scripts/visualize_dino_attention.py
    python scripts/visualize_dino_attention.py --image-url <url> --out mine.png
"""
import argparse
import io
import urllib.request

import matplotlib.pyplot as plt
import timm
import torch
from PIL import Image
from torchvision import transforms

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adapters.timm_vit import TimmViTAdapter

# Both are standard COCO val2017 images. The cats+remotes one is what HuggingFace's own DINO demo
# space uses (for ViT-S/8); the bear is the one from PyTorch's object-detection tutorial - neither
# is claimed to be literally from the DINO paper's PDF, just clear single/dual-foreground-object
# photos DINO is known to segment well.
DEFAULT_IMAGE_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # two cats + two remotes
    "http://images.cocodataset.org/val2017/000000000285.jpg",  # one bear
]

CHECKPOINT = "vit_small_patch16_224.dino"


def load_image(url_or_path, size):
    if url_or_path.startswith("http"):
        with urllib.request.urlopen(url_or_path) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
    else:
        img = Image.open(url_or_path).convert("RGB")
    # mean/std timm ships for the .dino tag specifically, not a hardcoded ImageNet guess (§8.2's
    # "read normalisation from the checkpoint config" rule, applied here too)
    cfg = timm.data.resolve_data_config({}, model=CHECKPOINT)
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])
    return img, tf(img).unsqueeze(0)


@torch.no_grad()
def last_layer_cls_attention(adapter, model, x):
    """(H, h, w) raw [CLS]->patch attention, last block, upsampled to input resolution."""
    l = adapter.n_blocks(model) - 1
    layout = adapter.token_layout(model)
    probs = adapter.attention_probs(model, x, l)  # (B, H, T, T), fp32
    a = probs[0, :, layout["cls_idx"], layout["patch_idx"]]  # (H, N)
    gh, gw = layout["grid"]
    a = a.reshape(-1, gh, gw)
    return torch.nn.functional.interpolate(a.unsqueeze(0), size=x.shape[-2:], mode="nearest")[0]


def main(urls, out_path, pretrained):
    adapter = TimmViTAdapter()
    model = timm.create_model(CHECKPOINT, pretrained=pretrained)
    adapter.prepare(model)
    size = model.patch_embed.img_size

    fig, axes = plt.subplots(len(urls), 7, figsize=(21, 3 * len(urls)))
    axes = axes.reshape(len(urls), 7)
    for row, url in enumerate(urls):
        img, x = load_image(url, size)
        maps = last_layer_cls_attention(adapter, model, x)
        axes[row, 0].imshow(img.resize(size))
        axes[row, 0].set_title("input")
        for h in range(maps.shape[0]):
            axes[row, h + 1].imshow(maps[h], cmap="inferno")
            axes[row, h + 1].set_title(f"head {h}")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image-url", action="append", dest="urls", default=None)
    p.add_argument("--out", default="dino_attention.png")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = p.parse_args()
    main(args.urls or DEFAULT_IMAGE_URLS, args.out, args.pretrained)
