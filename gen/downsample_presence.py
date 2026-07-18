"""DOWNSAMPLE-BY-PRESENCE detectors.

Output is a small grid that summarizes the input's block/region structure:
each output cell = a summary (dominant color / non-background content /
"clean" color) of the corresponding block of the input, or a collapse of the
input's uniform bands.

Two mechanisms:
  1. REGULAR block grid: input divides evenly into oh x ow equal blocks
     (constant across all train pairs); each output cell = a per-block summary.
     Covers "which region is filled / dominant color of each region".
  2. BAND COLLAPSE (dedupe): the input is made of uniform horizontal/vertical
     bands (possibly separated by lines); collapse consecutive identical
     rows and columns to one, yielding one cell per band. Optionally crop to
     the non-background bounding box first.

Every candidate transform is verified by the engine against all train outputs
before use, so we may offer several rule variants; only fitting ones survive.
"""
import numpy as np


def _bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


# ---------------- regular block-grid summary ----------------
def _block_summary_fn(oh, ow, rule):
    """Return fn(g) that splits g into oh x ow equal blocks and summarizes each."""
    def fn(g):
        try:
            H, W = g.shape
        except Exception:
            return None
        if oh < 1 or ow < 1:
            return None
        if H % oh or W % ow:
            return None
        bh, bw = H // oh, W // ow
        if bh < 1 or bw < 1:
            return None
        vals, cnts = np.unique(g, return_counts=True)
        bg = int(vals[np.argmax(cnts)])
        out = np.zeros((oh, ow), dtype=int)
        for r in range(oh):
            for c in range(ow):
                blk = g[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                bv, bc = np.unique(blk, return_counts=True)
                if rule == "dominant":
                    out[r, c] = int(bv[np.argmax(bc)])
                elif rule == "clean":
                    # uniform single-color block -> that color, else 0
                    out[r, c] = int(bv[0]) if len(bv) == 1 else 0
                elif rule == "nonbg":
                    # dominant non-background color in the block, else bg
                    nz = blk[blk != bg]
                    if nz.size:
                        nv, nc = np.unique(nz, return_counts=True)
                        out[r, c] = int(nv[np.argmax(nc)])
                    else:
                        out[r, c] = bg
                elif rule == "nonbg_zero":
                    # dominant non-background color in the block, else 0
                    nz = blk[blk != bg]
                    if nz.size:
                        nv, nc = np.unique(nz, return_counts=True)
                        out[r, c] = int(nv[np.argmax(nc)])
                    else:
                        out[r, c] = 0
                else:
                    return None
        return out
    return fn


def _regular_block(train, rule):
    """If all train outputs share one shape (oh,ow) and the block-summary
    `rule` reproduces every output, return a transform; else None."""
    try:
        shapes = {o.shape for _, o in train}
    except Exception:
        return None
    if len(shapes) != 1:
        return None
    oh, ow = train[0][1].shape
    if oh < 1 or ow < 1:
        return None
    # require an actual downsample (input strictly larger somewhere)
    if all(i.shape == o.shape for i, o in train):
        return None
    fn = _block_summary_fn(oh, ow, rule)
    try:
        for i, o in train:
            p = fn(i)
            if p is None or p.shape != o.shape or not np.array_equal(p, o):
                return None
    except Exception:
        return None
    return fn


def block_dominant(train):
    return _regular_block(train, "dominant")


def block_clean(train):
    return _regular_block(train, "clean")


def block_nonbg(train):
    return _regular_block(train, "nonbg")


def block_nonbg_zero(train):
    return _regular_block(train, "nonbg_zero")


# ---------------- band collapse (dedupe rows/cols) ----------------
def _dedupe_rows_cols(g):
    if g.ndim != 2 or g.size == 0:
        return g
    keep = [0]
    for r in range(1, g.shape[0]):
        if not np.array_equal(g[r], g[r - 1]):
            keep.append(r)
    g2 = g[keep, :]
    keepc = [0]
    for c in range(1, g2.shape[1]):
        if not np.array_equal(g2[:, c], g2[:, c - 1]):
            keepc.append(c)
    return g2[:, keepc]


def _crop_content(g):
    bg = _bg_color(g)
    mask = g != bg
    if not mask.any():
        return g
    rs, cs = np.where(mask)
    return g[rs.min():rs.max() + 1, cs.min():cs.max() + 1]


def band_collapse(train):
    """Collapse consecutive identical rows and columns -> one cell per band."""
    if all(i.shape == o.shape for i, o in train):
        return None

    def fn(g):
        return _dedupe_rows_cols(g)
    try:
        for i, o in train:
            p = fn(i)
            if p is None or p.shape != o.shape or not np.array_equal(p, o):
                return None
    except Exception:
        return None
    return fn


def crop_band_collapse(train):
    """Crop to non-background bounding box, then collapse uniform bands.

    Only fires when band-collapse actually merges rows/cols on some train input
    (otherwise it is just a plain bbox crop, which the base ensemble already
    covers, and blindly collapsing would misfire on a test with repeated rows)."""
    if all(i.shape == o.shape for i, o in train):
        return None

    def fn(g):
        return _dedupe_rows_cols(_crop_content(g))

    collapses = False
    try:
        for i, o in train:
            cropped = _crop_content(i)
            p = _dedupe_rows_cols(cropped)
            if p is None or p.shape != o.shape or not np.array_equal(p, o):
                return None
            if p.shape != cropped.shape:
                collapses = True
    except Exception:
        return None
    if not collapses:
        return None
    return fn


# ---------------- presence-of-content boolean summary ----------------
def _presence_fn(oh, ow, on, off):
    """Per-block boolean: on-color if block has any non-bg content, else off."""
    def fn(g):
        try:
            H, W = g.shape
        except Exception:
            return None
        if oh < 1 or ow < 1 or H % oh or W % ow:
            return None
        bh, bw = H // oh, W // ow
        vals, cnts = np.unique(g, return_counts=True)
        bg = int(vals[np.argmax(cnts)])
        out = np.zeros((oh, ow), dtype=int)
        for r in range(oh):
            for c in range(ow):
                blk = g[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                out[r, c] = on if np.any(blk != bg) else off
        return out
    return fn


def block_presence(train):
    """Which blocks contain content -> fixed on/off colors."""
    try:
        shapes = {o.shape for _, o in train}
    except Exception:
        return None
    if len(shapes) != 1:
        return None
    if all(i.shape == o.shape for i, o in train):
        return None
    oh, ow = train[0][1].shape
    if oh < 1 or ow < 1:
        return None
    out_colors = set()
    for _, o in train:
        out_colors |= set(np.unique(o).tolist())
    if len(out_colors) > 3:
        return None
    cols = sorted(out_colors)
    for on in cols:
        for off in cols:
            if on == off:
                continue
            fn = _presence_fn(oh, ow, on, off)
            try:
                ok = all(fn(i) is not None and fn(i).shape == o.shape and
                         np.array_equal(fn(i), o) for i, o in train)
            except Exception:
                ok = False
            if ok:
                return fn
    return None


DETECTORS = [
    block_dominant,
    block_nonbg,
    block_clean,
    block_nonbg_zero,
    block_presence,
    band_collapse,
    crop_band_collapse,
]
