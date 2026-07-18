"""Region / color operation detectors for ARC-AGI.

Family: split the grid into monochrome regions / connected components and
recolor by a region property, or perform whole-grid colour operations:

  * swap_two_colors        - exchange two colours everywhere (an involution
                             the base per-cell colour_map also covers, kept as
                             a principled, order-independent fallback).
  * keep_only_color        - keep exactly one non-background colour, erase the
                             rest to background (which colour is chosen by a
                             learnable rule: most / least frequent, unique,
                             largest / smallest object).
  * replace_freq_color     - recolour the most- (or least-) frequent non-bg
                             colour to a learned target colour.
  * recolor_to_marker      - a small "key" object (often a single cell of a
                             unique colour) dictates the colour of the other
                             object(s); the key is consumed.
  * recolor_by_border      - two opposite solid border lines colour the grid;
                             each interior mark takes the nearer border colour.
  * region_to_neighbor     - each connected region is recoloured to the colour
                             of its largest (or only) neighbouring region.

Every detector only *proposes* a transform; the engine re-verifies it against
all training pairs and rejects wrong fits. Only numpy + stdlib. Defensive.
"""
import numpy as np
from collections import Counter, defaultdict

# ----------------------------------------------------------------- helpers

def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _global_bg(train):
    c = Counter()
    for i, _ in train:
        for v, n in zip(*np.unique(i, return_counts=True)):
            c[int(v)] += int(n)
    if not c:
        return 0
    if 0 in c:
        return 0
    return c.most_common(1)[0][0]


def _components(g, bg, diag, per_color=True):
    """Connected components of non-bg cells. Returns list of dicts."""
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            col0 = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        if per_color and int(g[ny, nx]) != col0:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append({"cells": cells, "color": col0, "size": len(cells)})
    return comps


def _mask_preserved(train):
    """Output changes only colours; the non-bg occupied mask is preserved
    (per the *task* background, most common overall)."""
    bg = _global_bg(train)
    for i, o in train:
        if i.shape != o.shape:
            return False
        if not np.array_equal(i != bg, o != bg):
            return False
    return True


# =============================================================== detectors

# ----------------------------------------------------------------- 1. swap
def swap_two_colors(train):
    """Exchange exactly two colours everywhere (a <-> b). This is an involution
    special case of a colour map; kept as a principled detector so swaps are
    proposed even when other colours are present and unchanged."""
    if any(i.shape != o.shape for i, o in train):
        return None
    mapping = {}
    for i, o in train:
        for a, b in zip(i.flatten(), o.flatten()):
            a, b = int(a), int(b)
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    changed = {k: v for k, v in mapping.items() if k != v}
    if len(changed) != 2:
        return None
    (a1, b1), (a2, b2) = list(changed.items())
    # must be a genuine 2-cycle: a1->b1==a2 and a2->b2==a1
    if not (b1 == a2 and b2 == a1):
        return None
    x, y = a1, a2

    def fn(g, x=x, y=y):
        out = g.copy()
        out[g == x] = y
        out[g == y] = x
        return out
    return fn


# ----------------------------------------------- 2. keep exactly one colour
def _keep_color_detector(selector, name):
    """Keep the cells of one chosen non-bg colour, erase everything else to bg.
    `selector(colcount, comps, g, bg)` returns the colour to keep, or None."""
    def det(train):
        bg = _global_bg(train)
        # every output must contain (at most) one non-bg colour, a subset of
        # the input's colours, occupying a subset of input's occupied cells.
        for i, o in train:
            if i.shape != o.shape:
                return None
            onz = set(np.unique(o).tolist()) - {bg}
            if len(onz) > 1:
                return None
            # output cells must be exactly the input cells of that colour
            if onz:
                kc = next(iter(onz))
                if not np.array_equal(o == kc, i == kc):
                    return None
                # everything else in output is bg
                if not np.all(o[o != kc] == bg):
                    return None

        def fn(g, bg=bg):
            b = _bg(g) if bg not in np.unique(g) else bg
            # use task bg but guard
            b = bg
            colcount = Counter()
            for v, n in zip(*np.unique(g, return_counts=True)):
                if int(v) != b:
                    colcount[int(v)] = int(n)
            comps = _components(g, b, diag=True, per_color=True)
            kc = selector(colcount, comps, g, b)
            out = np.full_like(g, b)
            if kc is not None:
                out[g == kc] = kc
            return out
        det_ok = True
        try:
            det_ok = all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train)
        except Exception:
            det_ok = False
        if not det_ok:
            return None
        return fn
    det.__name__ = "keep_only_" + name
    return det


def _sel_most_freq(colcount, comps, g, bg):
    if not colcount:
        return None
    m = max(colcount.values())
    cands = [c for c, n in colcount.items() if n == m]
    return cands[0] if len(cands) == 1 else None


def _sel_least_freq(colcount, comps, g, bg):
    if not colcount:
        return None
    m = min(colcount.values())
    cands = [c for c, n in colcount.items() if n == m]
    return cands[0] if len(cands) == 1 else None


def _sel_largest_obj(colcount, comps, g, bg):
    if not comps:
        return None
    m = max(c["size"] for c in comps)
    cands = [c for c in comps if c["size"] == m]
    if len(cands) != 1:
        return None
    return cands[0]["color"]


def _sel_smallest_obj(colcount, comps, g, bg):
    if not comps:
        return None
    m = min(c["size"] for c in comps)
    cands = [c for c in comps if c["size"] == m]
    if len(cands) != 1:
        return None
    return cands[0]["color"]


def _sel_unique_color(colcount, comps, g, bg):
    """The colour that appears exactly once (a single cell) — a common 'marker'."""
    ones = [c for c, n in colcount.items() if n == 1]
    return ones[0] if len(ones) == 1 else None


keep_only_most_freq = _keep_color_detector(_sel_most_freq, "most_freq")
keep_only_least_freq = _keep_color_detector(_sel_least_freq, "least_freq")
keep_only_largest_obj = _keep_color_detector(_sel_largest_obj, "largest_obj")
keep_only_smallest_obj = _keep_color_detector(_sel_smallest_obj, "smallest_obj")


# ------------------------------------------- 3. replace a frequency-extreme colour
def _replace_freq_detector(pick, name):
    """Recolour the most/least frequent non-bg colour to a learned target.
    All other cells unchanged. `pick(colcount)` -> source colour or None."""
    def det(train):
        bg = _global_bg(train)
        target = None
        for i, o in train:
            if i.shape != o.shape:
                return None
            colcount = Counter()
            for v, n in zip(*np.unique(i, return_counts=True)):
                if int(v) != bg:
                    colcount[int(v)] = int(n)
            src = pick(colcount)
            if src is None:
                return None
            # output must equal input with all `src` cells recoloured to one colour
            diff = i != o
            if not diff.any():
                return None
            if not np.all(i[diff] == src):
                return None
            tvals = set(np.unique(o[diff]).tolist())
            if len(tvals) != 1:
                return None
            t = next(iter(tvals))
            if target is None:
                target = t
            elif target != t:
                return None
            # every src cell must have become target (full recolour)
            if not np.all(o[i == src] == target):
                return None
        if target is None:
            return None

        def fn(g, target=target, bg=bg):
            colcount = Counter()
            for v, n in zip(*np.unique(g, return_counts=True)):
                if int(v) != bg:
                    colcount[int(v)] = int(n)
            src = pick(colcount)
            out = g.copy()
            if src is not None:
                out[g == src] = target
            return out
        return fn
    det.__name__ = "replace_" + name + "_color"
    return det


def _pick_most(colcount):
    if not colcount:
        return None
    m = max(colcount.values())
    cands = [c for c, n in colcount.items() if n == m]
    return cands[0] if len(cands) == 1 else None


def _pick_least(colcount):
    if not colcount:
        return None
    m = min(colcount.values())
    cands = [c for c, n in colcount.items() if n == m]
    return cands[0] if len(cands) == 1 else None


replace_most_freq_color = _replace_freq_detector(_pick_most, "most_freq")
replace_least_freq_color = _replace_freq_detector(_pick_least, "least_freq")


# ------------------------------------------------- 4. recolor object(s) to marker
def _least_freq_color(g, bg):
    cc = Counter()
    for v, n in zip(*np.unique(g, return_counts=True)):
        if int(v) != bg:
            cc[int(v)] = int(n)
    if not cc:
        return None
    m = min(cc.values())
    cands = [c for c, n in cc.items() if n == m]
    return cands[0] if len(cands) == 1 else None


def recolor_to_marker(train):
    """A small "marker" (the least-frequent non-bg colour, e.g. a single key
    cell) tells the remaining shape which colour to be. The shape is recoloured
    to the marker colour and the marker cell(s) are erased to background.

    Generalises 'a corner key cell colours the big shape'. After the transform
    the only surviving non-bg colour is the marker colour.
    """
    bg = _global_bg(train)
    for i, o in train:
        if i.shape != o.shape:
            return None
        marker = _least_freq_color(i, bg)
        if marker is None:
            return None
        icols = set(int(v) for v in np.unique(i)) - {bg}
        # need at least a shape colour besides the marker
        if len(icols) < 2:
            return None
        expect = i.copy()
        shape_mask = (i != bg) & (i != marker)
        expect[shape_mask] = marker      # shape adopts marker colour
        expect[i == marker] = bg         # marker consumed
        if not np.array_equal(expect, o):
            return None

    def fn(g, bg=bg):
        marker = _least_freq_color(g, bg)
        if marker is None:
            return g.copy()
        out = g.copy()
        shape_mask = (g != bg) & (g != marker)
        out[shape_mask] = marker
        out[g == marker] = bg
        return out
    return fn


# --------------------------------------------- 5. recolor by nearest border line
def _border_lines(g, bg):
    """Return dict side->colour for full solid edge lines of a single non-bg
    colour. side in {'top','bottom','left','right'}."""
    H, W = g.shape
    lines = {}
    top = g[0, :]
    if top[0] != bg and np.all(top == top[0]):
        lines['top'] = int(top[0])
    bot = g[H - 1, :]
    if bot[0] != bg and np.all(bot == bot[0]):
        lines['bottom'] = int(bot[0])
    left = g[:, 0]
    if left[0] != bg and np.all(left == left[0]):
        lines['left'] = int(left[0])
    right = g[:, W - 1]
    if right[0] != bg and np.all(right == right[0]):
        lines['right'] = int(right[0])
    return lines


def recolor_by_border(train):
    """Two opposite solid border lines colour the grid. Each interior non-bg
    mark is recoloured to the colour of the nearer border (distance measured
    perpendicular to the two lines). The border lines are left untouched.
    """
    bg = _global_bg(train)

    def orientation(lines):
        if 'left' in lines and 'right' in lines and lines['left'] != lines['right']:
            return 'v', lines['left'], lines['right']  # horizontal distance
        if 'top' in lines and 'bottom' in lines and lines['top'] != lines['bottom']:
            return 'h', lines['top'], lines['bottom']  # vertical distance
        return None

    # verify structure on every train pair
    for i, o in train:
        if i.shape != o.shape:
            return None
        lines = _border_lines(i, bg)
        info = orientation(lines)
        if info is None:
            return None

    def fn(g, bg=bg):
        lines = _border_lines(g, bg)
        info = orientation(lines)
        if info is None:
            return g.copy()
        axis, c_lo, c_hi = info
        H, W = g.shape
        out = g.copy()
        if axis == 'v':
            # left border color c_lo at col 0; right border c_hi at col W-1
            for r in range(H):
                for c in range(1, W - 1):
                    if g[r, c] != bg:
                        # keep if it is itself on a border line? interior only
                        dist_left = c
                        dist_right = (W - 1) - c
                        out[r, c] = c_lo if dist_left <= dist_right else c_hi
        else:
            for c in range(W):
                for r in range(1, H - 1):
                    if g[r, c] != bg:
                        dist_top = r
                        dist_bot = (H - 1) - r
                        out[r, c] = c_lo if dist_top <= dist_bot else c_hi
        return out
    return fn


# ------------------------------------- 6. recolor each region to a neighbour colour
def region_to_largest_neighbor(train):
    """Split the grid into connected same-colour regions (background included as
    regions too). Recolour each *foreground* region to the colour of its
    largest orthogonally-adjacent region. Verified against train.
    """
    if not _mask_preserved(train):
        # allow mask change only if bg regions unaffected; keep strict here.
        pass
    bg = _global_bg(train)

    def transform(g):
        H, W = g.shape
        # label every cell into same-colour 4-connected regions
        label = -np.ones((H, W), dtype=int)
        regions = []  # id -> {color,size,cells}
        nid = 0
        for r in range(H):
            for c in range(W):
                if label[r, c] != -1:
                    continue
                col = int(g[r, c])
                stack = [(r, c)]
                label[r, c] = nid
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and label[ny, nx] == -1 and int(g[ny, nx]) == col:
                            label[ny, nx] = nid
                            stack.append((ny, nx))
                regions.append({"color": col, "size": len(cells), "cells": cells})
                nid += 1
        # adjacency: for each region, neighbouring region sizes by color
        neigh = [defaultdict(int) for _ in range(nid)]  # rid -> {neighbor_rid: contact}
        for r in range(H):
            for c in range(W):
                a = label[r, c]
                for dy, dx in ((1, 0), (0, 1)):
                    ny, nx = r + dy, c + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        b = label[ny, nx]
                        if a != b:
                            neigh[a][b] += 1
                            neigh[b][a] += 1
        out = g.copy()
        for rid in range(nid):
            reg = regions[rid]
            if reg["color"] == bg:
                continue
            if not neigh[rid]:
                continue
            # largest neighbouring region (by region size), excluding same colour
            best = None
            best_size = -1
            for nb in neigh[rid]:
                if regions[nb]["color"] == reg["color"]:
                    continue
                if regions[nb]["size"] > best_size:
                    best_size = regions[nb]["size"]
                    best = nb
            if best is None:
                continue
            nc = regions[best]["color"]
            for y, x in reg["cells"]:
                out[y, x] = nc
        return out

    try:
        if all(transform(i).shape == o.shape and np.array_equal(transform(i), o) for i, o in train):
            return transform
    except Exception:
        return None
    return None


DETECTORS = [
    swap_two_colors,
    recolor_to_marker,
    recolor_by_border,
    keep_only_most_freq, keep_only_least_freq,
    keep_only_largest_obj, keep_only_smallest_obj,
    replace_most_freq_color, replace_least_freq_color,
    region_to_largest_neighbor,
]
