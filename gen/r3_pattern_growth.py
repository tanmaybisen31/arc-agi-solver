"""PATTERN GROWTH / EXTRAPOLATION detectors for ARC-AGI.

Family: continue / grow a pattern into a larger output.

  1) FRACTAL SELF-TILING. Output is (H*H) x (W*W). It is an HxW mosaic of
     blocks; block (r,c) is either a (geometrically transformed) copy of the
     whole input or a solid fill block, decided by the color of input cell
     (r,c). We infer, per color, whether it maps to a copy-block or a
     fill-block, plus the single geometric op applied to every copy and the
     fill color. This is the classic "draw the shape inside itself" family
     (007bbfb7, cce03e0d, ...).

  2) FRACTAL via background predicate. Same construction but the copy/blank
     decision is "cell == background" or "cell != background" (background =
     most common color). Handles tasks where the trigger color changes
     between demos but the bg relationship is constant.

Each detector returns a transform reproducing every training pair exactly (the
engine re-verifies), or None. numpy + stdlib only, defensive throughout.
"""
import numpy as np
from collections import Counter


# ------------------------------------------------------------------ helpers --
def _bg(g):
    if g.size == 0:
        return 0
    v, c = np.unique(g, return_counts=True)
    return int(v[np.argmax(c)])


_GEO = {
    "id":  lambda g: g,
    "flr": lambda g: np.fliplr(g),
    "fud": lambda g: np.flipud(g),
    "r180": lambda g: np.rot90(g, 2),
    "r90": lambda g: np.rot90(g, 1),
    "r270": lambda g: np.rot90(g, 3),
    "T":   lambda g: g.T,
    "aT":  lambda g: np.rot90(g, 2).T,
}
_SWAP = {"r90", "r270", "T", "aT"}   # change (h,w) -> only for square inputs


def _self_ratio_ok(train):
    """Every pair must have output shape (h*h, w*w)."""
    for i, o in train:
        h, w = i.shape
        if h == 0 or w == 0:
            return False
        if o.shape != (h * h, w * w):
            return False
    return True


# ------------------------------------------- 1) fractal, per-color trigger ---
def fractal_self_tile(train):
    """Block (r,c) is geo(input) when color(r,c) is in a learned 'copy' set,
    else a solid fill block. Learn geo op, fill color and the per-color roles."""
    if not train or not _self_ratio_ok(train):
        return None
    fills = set()
    for _, o in train:
        fills |= set(np.unique(o).tolist())
    square = all(i.shape[0] == i.shape[1] for i, _ in train)

    for gname, gf in _GEO.items():
        if gname in _SWAP and not square:
            continue
        for fill in sorted(fills):
            color_role = {}          # color -> 'copy' | 'blank'
            ok = True
            for i, o in train:
                h, w = i.shape
                blk = np.asarray(gf(i), dtype=int)
                if blk.shape != (h, w):
                    ok = False
                    break
                const = np.full((h, w), fill, dtype=int)
                for r in range(h):
                    for c in range(w):
                        sub = o[r*h:(r+1)*h, c*w:(c+1)*w]
                        is_copy = np.array_equal(sub, blk)
                        is_blank = np.array_equal(sub, const)
                        if not is_copy and not is_blank:
                            ok = False
                            break
                        if is_copy and is_blank:
                            continue          # ambiguous (blk == fill const)
                        col = int(i[r, c])
                        role = 'copy' if is_copy else 'blank'
                        prev = color_role.get(col)
                        if prev is None:
                            color_role[col] = role
                        elif prev != role:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            if 'copy' not in color_role.values():
                continue
            copy_set = frozenset(k for k, v in color_role.items() if v == 'copy')

            def fn(g, gf=gf, fill=fill, copy_set=copy_set):
                g = np.asarray(g, dtype=int)
                h, w = g.shape
                if h == 0 or w == 0:
                    return None
                blk = np.asarray(gf(g), dtype=int)
                if blk.shape != (h, w):
                    return None
                out = np.full((h*h, w*w), fill, dtype=int)
                for r in range(h):
                    for c in range(w):
                        if int(g[r, c]) in copy_set:
                            out[r*h:(r+1)*h, c*w:(c+1)*w] = blk
                return out
            return fn
    return None


# ---------------------------------- 2) fractal, background-relative trigger --
def fractal_bg_predicate(train):
    """Copy where cell (matches | differs from) the background color.
    Robust when the specific trigger color changes across demos."""
    if not train or not _self_ratio_ok(train):
        return None
    fills = set()
    for _, o in train:
        fills |= set(np.unique(o).tolist())
    square = all(i.shape[0] == i.shape[1] for i, _ in train)

    for gname, gf in _GEO.items():
        if gname in _SWAP and not square:
            continue
        for fill in sorted(fills):
            for pred in ("ne", "eq"):
                ok = True
                for i, o in train:
                    h, w = i.shape
                    bg = _bg(i)
                    m = (i != bg) if pred == "ne" else (i == bg)
                    blk = np.asarray(gf(i), dtype=int)
                    if blk.shape != (h, w):
                        ok = False
                        break
                    out = np.full((h*h, w*w), fill, dtype=int)
                    for r in range(h):
                        for c in range(w):
                            if m[r, c]:
                                out[r*h:(r+1)*h, c*w:(c+1)*w] = blk
                    if not np.array_equal(out, o):
                        ok = False
                        break
                if not ok:
                    continue

                def fn(g, gf=gf, fill=fill, pred=pred):
                    g = np.asarray(g, dtype=int)
                    h, w = g.shape
                    if h == 0 or w == 0:
                        return None
                    bg = _bg(g)
                    m = (g != bg) if pred == "ne" else (g == bg)
                    blk = np.asarray(gf(g), dtype=int)
                    if blk.shape != (h, w):
                        return None
                    out = np.full((h*h, w*w), fill, dtype=int)
                    for r in range(h):
                        for c in range(w):
                            if m[r, c]:
                                out[r*h:(r+1)*h, c*w:(c+1)*w] = blk
                    return out
                return fn
    return None


# ------------------------------ 3) fractal with a 2-color-swapped block ------
def fractal_swap(train):
    """Two-color inputs. Output is the self-tiled mosaic where copy blocks are a
    geo(input) with the two colors optionally swapped, blanks are a solid fill.
    The copy/blank decision is 'cell == a' or 'cell == b' (a<b the two colors).
    Handles negative-image self-tilings (8e2edd66, 0692e18c)."""
    if not train or not _self_ratio_ok(train):
        return None
    if any(len(np.unique(i)) != 2 for i, _ in train):
        return None
    square = all(i.shape[0] == i.shape[1] for i, _ in train)

    def two_colors(g):
        cs = sorted(np.unique(g).tolist())
        return cs[0], cs[1]

    for gname, gf in _GEO.items():
        if gname in _SWAP and not square:
            continue
        for mask_on in ("a", "b"):        # copy where cell == this color
            for swap in (False, True):
                for fill_sel in ("a", "b"):
                    ok = True
                    for i, o in train:
                        h, w = i.shape
                        a, b = two_colors(i)
                        fill = a if fill_sel == "a" else b
                        blk = np.asarray(gf(i), dtype=int)
                        if blk.shape != (h, w):
                            ok = False
                            break
                        if swap:
                            blk = np.where(blk == a, b,
                                           np.where(blk == b, a, blk))
                        on = a if mask_on == "a" else b
                        m = (i == on)
                        out = np.full((h*h, w*w), fill, dtype=int)
                        for r in range(h):
                            for c in range(w):
                                if m[r, c]:
                                    out[r*h:(r+1)*h, c*w:(c+1)*w] = blk
                        if not np.array_equal(out, o):
                            ok = False
                            break
                    if not ok:
                        continue

                    def fn(g, gf=gf, mask_on=mask_on, swap=swap, fill_sel=fill_sel):
                        g = np.asarray(g, dtype=int)
                        h, w = g.shape
                        if h == 0 or w == 0 or len(np.unique(g)) != 2:
                            return None
                        a, b = two_colors(g)
                        fill = a if fill_sel == "a" else b
                        blk = np.asarray(gf(g), dtype=int)
                        if blk.shape != (h, w):
                            return None
                        if swap:
                            blk = np.where(blk == a, b,
                                           np.where(blk == b, a, blk))
                        on = a if mask_on == "a" else b
                        m = (g == on)
                        out = np.full((h*h, w*w), fill, dtype=int)
                        for r in range(h):
                            for c in range(w):
                                if m[r, c]:
                                    out[r*h:(r+1)*h, c*w:(c+1)*w] = blk
                        return out
                    return fn
    return None


# ------------------------------------------------------------------------------
# Counting-driven scale: output = input kron-scaled by an integer k that equals
# a count feature of the input (number of colored cells, distinct colors, or
# connected objects). Growth whose *magnitude* is inferred from the content.
# ------------------------------------------------------------------------------
def _components4(g, bg):
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    n = 0
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            n += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return n


def _count_feature(g, name):
    bg = _bg(g)
    if name == "nonbg":
        return int((g != bg).sum())
    if name == "distinct_all":
        return int(len(np.unique(g)))
    if name == "distinct_fg":
        return int(len([c for c in np.unique(g).tolist() if c != bg]))
    if name == "objs":
        return _components4(g, bg)
    return 0


def scale_by_count(train):
    """Output is np.kron(input, ones(k,k)) where k is a count feature of the
    input (colored cells / distinct colors / objects). k varies per demo but the
    *choice of feature* is fixed, so it generalizes to unseen counts.

    When several features reproduce the demos equally well, prefer the one whose
    value actually VARIES across demos (stronger evidence that the count really
    drives the scale, rather than a coincidental constant factor)."""
    if not train:
        return None
    if any(i.size == 0 for i, _ in train):
        return None
    candidates = []                       # (varies, feat)
    for feat in ("nonbg", "distinct_fg", "distinct_all", "objs"):
        ok = True
        ks = []
        for i, o in train:
            k = _count_feature(i, feat)
            ks.append(k)
            if k < 1 or k > 30:
                ok = False
                break
            exp = np.kron(i, np.ones((k, k), dtype=int))
            if exp.shape != o.shape or not np.array_equal(exp, o):
                ok = False
                break
        if not ok:
            continue
        if all(k <= 1 for k in ks):       # trivial identity — leave to others
            continue
        varies = len(set(ks)) > 1
        candidates.append((varies, feat))
    if not candidates:
        return None
    # varying features first, then the module's generality order
    order = ("nonbg", "distinct_fg", "distinct_all", "objs")
    candidates.sort(key=lambda t: (not t[0], order.index(t[1])))
    feat = candidates[0][1]

    def fn(g, feat=feat):
        g = np.asarray(g, dtype=int)
        k = _count_feature(g, feat)
        if k < 1 or k > 30:
            return None
        return np.kron(g, np.ones((k, k), dtype=int)).astype(int)
    return fn


DETECTORS = [
    fractal_self_tile,
    fractal_bg_predicate,
    fractal_swap,
    scale_by_count,
]
