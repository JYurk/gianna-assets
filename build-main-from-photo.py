#!/usr/bin/env python3
"""Turn Abby's real product photo into an Amazon-compliant MAIN image candidate
for ASIN B0DTGM2181 (Gianna Demure Collagen Power Patches, 24 pairs).

Different job from build-main-image.py, which composited a cutout box and a
cutout patch pair. This one starts from a single real photograph, so all it has
to do is:

  1. cut the product group off the gray studio backdrop,
  2. drop it on pure white RGB 255/255/255,
  3. square it and scale the group to fill ~92 percent of the long side,
  4. write 2000x2000 so Amazon turns zoom on.

Nothing about the product itself is retouched, recoloured, or rescaled relative
to itself. Two crops are produced: the whole group (box plus the sachet fan) and
the sachet fan alone, so they can be measured against the live image and each
other rather than chosen by eye.

    ~/.venvs/crawl4ai/bin/python build-main-from-photo.py <photo.jpg> <outdir>
"""
import sys, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage

CANVAS = 2000
FILL = 0.92          # product long side as a share of the canvas
DARK = 130           # backdrop runs 195-250 and the cast shadow 150-220, so a cut
                     # at 130 takes the box and the foil sachets and leaves the
                     # shadow behind. 175 pulled the shadow in as a gray slab.
SAT = 55             # plus anything strongly coloured, to catch the gold
MIN_BLOB = 4000      # px at full res, drops dust and sensor noise
SPUR = 0.034         # a shadow streak thinner than this share of the frame
                     # height is not product; opened away vertically so the
                     # box corners are left square


def product_mask(a):
    """Boolean mask of the product group in an RGB array."""
    g = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)
    m = (g < DARK) | (sat > SAT)
    m = ndimage.binary_closing(m, np.ones((9, 9)))
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_opening(m, np.ones((max(9, round(a.shape[0] * SPUR)), 1)))
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= MIN_BLOB]
        m = np.isin(lab, keep)
    return m


def on_white(a, m, pad_frac=0.02):
    """Composite the masked product on pure white, cropped to its bounding box."""
    alpha = ndimage.gaussian_filter(m.astype(float), 1.2).clip(0, 1)
    white = np.full_like(a, 255, dtype=float)
    out = a * alpha[..., None] + white * (1 - alpha[..., None])
    ys, xs = np.where(m)
    pad = int(max(m.shape) * pad_frac)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, m.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, m.shape[1])
    return Image.fromarray(out[y0:y1, x0:x1].round().clip(0, 255).astype(np.uint8))


def square(im):
    """Scale to FILL of the canvas long side and centre on pure white."""
    s = CANVAS * FILL / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))),
                   Image.LANCZOS)
    canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    canvas.paste(im, ((CANVAS - im.width) // 2, (CANVAS - im.height) // 2))
    return canvas


def main():
    src, outdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    a = np.asarray(Image.open(src).convert("RGB")).astype(float)
    m = product_mask(a)

    # Whole group: box plus the sachet fan.
    square(on_white(a, m)).save(outdir / "candidate-group.jpg", quality=94,
                                subsampling=0)

    # Sachet fan alone. The box sits on the left of the frame, so split at the
    # widest vertical gap in the mask's column profile inside the middle third.
    cols = m.sum(axis=0)
    lo, hi = int(m.shape[1] * 0.30), int(m.shape[1] * 0.60)
    split = lo + int(np.argmin(cols[lo:hi]))
    m2 = m.copy()
    m2[:, :split] = False
    square(on_white(a, m2)).save(outdir / "candidate-sachets.jpg", quality=94,
                                 subsampling=0)
    print("split column", split, "of", m.shape[1])
    print("wrote candidate-group.jpg and candidate-sachets.jpg to", outdir)


if __name__ == "__main__":
    main()
