#!/usr/bin/env python3
"""Measure Amazon search-thumbnail readability for one or more images.

Same two numbers used on 2026-07-29 to kill the composite main-image swap, so
new candidates are compared on an identical basis:

  solid product mass - share of the 172px thumbnail taken by GOLD pixels that
      survive a 3x3 binary erosion. The erosion is the point: thin gold serif
      PRINTING on black packaging vanishes, so only real contiguous gold
      product mass counts.
  brightness - mean luminance of the 172px thumbnail. The competitor field on
      the `under eye patches` results page runs 165 to 207.

172px is the true rendered size of a search-result tile, which is where the
click is won or lost.

    ~/.venvs/crawl4ai/bin/python measure-thumbnail.py <image> [image ...]
"""
import sys, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage

THUMB = 172
HUE_LO, HUE_HI = 20 / 360, 65 / 360   # yellow through orange-gold
SAT_MIN = 0.25
VAL_MIN = 0.35


def measure(path):
    im = Image.open(path).convert("RGB")
    # Pad to square on white first, exactly as Amazon does before it scales a
    # non-square image into a square tile.
    if im.width != im.height:
        s = max(im.size)
        pad = Image.new("RGB", (s, s), (255, 255, 255))
        pad.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
        im = pad
    im = im.resize((THUMB, THUMB), Image.LANCZOS)
    a = np.asarray(im).astype(float) / 255
    hsv = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).convert("HSV"))
    h, s, v = hsv[..., 0] / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    gold = (h >= HUE_LO) & (h <= HUE_HI) & (s >= SAT_MIN) & (v >= VAL_MIN)
    solid = ndimage.binary_erosion(gold, np.ones((3, 3)))
    grey = np.asarray(im.convert("L")).astype(float)
    return solid.mean() * 100, gold.mean() * 100, grey.mean()


def main():
    print(f"{'image':46} {'solid gold %':>12} {'raw gold %':>11} {'brightness':>11}")
    for p in sys.argv[1:]:
        solid, raw, bright = measure(p)
        print(f"{pathlib.Path(p).name[:46]:46} {solid:12.1f} {raw:11.1f} {bright:11.0f}")


if __name__ == "__main__":
    main()
