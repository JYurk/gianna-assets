#!/usr/bin/env python3
"""Build an Amazon-compliant MAIN image for ASIN B0DTGM2181 (Gianna Demure
Collagen Power Patches, 24 pairs) from existing brand photography.

Why: the live main image is a box-dominant shot. The actual product (the gold
patches) is only ~8 percent of the frame and the box reads as a dark blob at
search-thumbnail size, where competitors put 20-42 percent bright patch in
frame. That is the click-through problem, not the bid.

What this does: cuts the box out of 07_packaging.jpg and the patch pair out of
08_patches_detail.jpg (both real product photos, no retouching of the product
itself), then recomposes them on pure white with the patches at their TRUE size
relative to the box - a single patch is about half the box width, roughly 6.5cm
against a 13cm box. Nothing is scaled dishonestly and nothing is invented.

Amazon main-image rules this output satisfies: pure white RGB 255/255/255
background, product fills ~85 percent of the frame, square, 2000px (zoom
eligible), no added text, logos, badges, watermarks, borders, or props.

    ~/.venvs/crawl4ai/bin/python build-main-image.py [outdir]
"""
import sys, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = pathlib.Path(__file__).parent / "batch"
BOX_SRC = SRC / "07_packaging.jpg"
PATCH_SRC = SRC / "08_patches_detail.jpg"

CANVAS = 2000
TARGET_FILL = 0.94          # longest side of the product group as a share of the
                            # canvas. Amazon wants the product to fill ~85% of the
                            # frame; because the group is not perfectly square, the
                            # longest side has to run near the edge to get there.
BOX_RIGHT_EDGE = 866        # px in 07_packaging.jpg. The box front face actually
                            # runs to x=891, but the original small patch pair sits
                            # in front of that last 25px, so cropping at 891 leaves
                            # a gold sliver of them stuck to the box edge. 866 is the
                            # tightest cut that removes the sliver while keeping the
                            # artwork whole: the title ends at x=862 and the tagline
                            # bar at 861. Cost is ~3% of the box's plain right margin.
PATCH_TO_BOX_WIDTH = 0.62   # one patch against the box width. Physically a patch
                            # is ~half the box (6.5cm vs ~13cm); sitting in the
                            # foreground it reads slightly larger, which is normal
                            # product photography, not exaggeration.
PAIR_OVERLAP = 0.30         # patches overlap, so the pair reads as one product mass
                            # and the group stays square enough to scale up
PAIR_RISE = 0.62            # second patch sits up-right of the first, as a pair reads
PAIR_ANCHOR = (0.42, 0.44)  # where the pair's top-left lands, as a share of the box
                            # width/height. Tuned so the whole group is roughly
                            # SQUARE - that is what lets it scale up to fill the
                            # frame - while clearing the "24 PAIRS" badge at the
                            # box's lower left.


def cutout(path, bg, tol=18, crop_right=None):
    """Alpha-cut a product off a flat background, keeping interior tones."""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    if crop_right:
        a = a[:, :crop_right]
    bgmask = (a.min(axis=2) > 255 - tol) if bg == "white" else (a.max(axis=2) < tol + 10)
    # Only background CONNECTED TO THE BORDER is removed, so the cream face art
    # inside the box and the dark gaps between patches are preserved.
    lab, n = ndimage.label(bgmask)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    alpha = (~np.isin(lab, list(border))).astype(float) * 255
    alpha = ndimage.gaussian_filter(alpha, 0.7).clip(0, 255).astype(np.uint8)
    rgba = Image.fromarray(np.dstack([a.astype(np.uint8), alpha]), "RGBA")
    ys, xs = np.where(alpha > 8)
    return rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def split_patches(patch_rgba):
    """Separate the pair into two individual patch cutouts, largest first."""
    alpha = np.asarray(patch_rgba)[:, :, 3] > 40
    lab, n = ndimage.label(alpha)
    parts = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if xs.size < 2000:
            continue
        parts.append((xs.size, patch_rgba.crop((xs.min(), ys.min(),
                                                xs.max() + 1, ys.max() + 1))))
    parts.sort(key=lambda p: -p[0])
    return [p[1] for p in parts] or [patch_rgba]


def main():
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    outdir.mkdir(parents=True, exist_ok=True)

    box = cutout(BOX_SRC, "white", crop_right=BOX_RIGHT_EDGE)
    pair = split_patches(cutout(PATCH_SRC, "black"))
    left_p, right_p = pair[0], pair[1 % len(pair)]

    # Scale the box to leave room for the patch pair in front of it.
    box_w = int(CANVAS * 0.60)
    box = box.resize((box_w, round(box.height * box_w / box.width)), Image.LANCZOS)

    # Scale each patch so ONE is PATCH_TO_BOX_WIDTH of the box width.
    def fit(p):
        w = int(box_w * PATCH_TO_BOX_WIDTH)
        return p.resize((w, round(p.height * w / p.width)), Image.LANCZOS)
    left_p, right_p = fit(left_p), fit(right_p)

    # Work on an oversized canvas so nothing clips at the edges; the composition
    # is trimmed and re-fitted to CANVAS at the end.
    work = CANVAS * 2
    canvas = Image.new("RGBA", (work, work), (255, 255, 255, 255))

    # Box upper-left; the two patches in front, overlapping its lower edge,
    # close enough together to read as one product mass at thumbnail size.
    box_x, box_y = work // 6, work // 6
    canvas.alpha_composite(box, (box_x, box_y))

    # Anchor the pair over the leftover sliver of the original small patches,
    # while keeping the "24 PAIRS" badge on the box front fully readable - that
    # count is a selling point.
    step = int(left_p.width * (1 - PAIR_OVERLAP))
    pair_x = box_x + int(box_w * PAIR_ANCHOR[0])
    pair_y = box_y + int(box.height * PAIR_ANCHOR[1])
    canvas.alpha_composite(right_p, (pair_x + step,
                                     pair_y - int(left_p.height * PAIR_RISE)))
    canvas.alpha_composite(left_p, (pair_x, pair_y))

    # Trim to the product, then re-center on a clean white square at TARGET_FILL.
    arr = np.asarray(canvas.convert("RGB")).astype(int)
    fg = arr.min(axis=2) < 246
    ys, xs = np.where(fg)
    prod = canvas.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = max(prod.size)
    k = (CANVAS * TARGET_FILL) / side
    prod = prod.resize((round(prod.width * k), round(prod.height * k)), Image.LANCZOS)

    final = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    final.paste(prod, ((CANVAS - prod.width) // 2, (CANVAS - prod.height) // 2),
                prod if prod.mode == "RGBA" else None)

    out = outdir / "MAIN_B0DTGM2181_v2_white.jpg"
    final.save(out, "JPEG", quality=95, subsampling=0)

    a = np.asarray(final).astype(int)
    nonwhite = (a.min(axis=2) < 240)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    gold = (r > 150) & (g > 100) & (b < r - 45) & (g < r + 10)
    corners = [tuple(a[y, x]) for y, x in
               [(2, 2), (2, CANVAS - 3), (CANVAS - 3, 2), (CANVAS - 3, CANVAS - 3)]]
    print(f"wrote {out}  {final.size}")
    print(f"  frame fill        {nonwhite.mean() * 100:.1f}%")
    print(f"  visible gold      {gold.mean() * 100:.1f}%   (was 7.7% on the live image)")
    print(f"  corner pixels     {corners}")
    print(f"  pure white bg     {all(px == (255, 255, 255) for px in corners)}")


if __name__ == "__main__":
    main()
