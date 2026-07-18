"""Round 5, batch 0 detectors for ARC-AGI.

Each detector: def det(train) -> transform_fn | None
  train = [(input_grid, output_grid), ...]  (numpy int arrays)
  transform_fn: grid -> grid   (engine verifies exact reproduction of all demos)

General, principled rules only. numpy + stdlib. Defensive.
"""
import numpy as np
from collections import deque, Counter


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components_same_color(g, bg, connectivity=8):
    """Connected components of cells sharing the same (non-bg) color."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    if connectivity == 8:
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    comps = []
    for i in range(H):
        for j in range(W):
            if g[i, j] != bg and not seen[i, j]:
                col = g[i, j]
                q = deque([(i, j)])
                seen[i, j] = True
                cells = []
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr, dc in nbrs:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == col and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                rs = [x for x, _ in cells]
                cs = [y for _, y in cells]
                comps.append({
                    "color": int(col),
                    "bbox": (min(rs), max(rs), min(cs), max(cs)),
                    "cells": cells,
                })
    return comps


# ---------------------------------------------------------------------------
# 1) Nested-frame recolor: each object is a set of concentric rectangular
#    frames (separate same-color components).  Group them by bounding-box
#    containment; recolor every descendant to the outermost (root) color.
#    (task 7d1f7ee8)
# ---------------------------------------------------------------------------
def _bbox_inside(inner, outer):
    ir0, ir1, ic0, ic1 = inner
    or0, or1, oc0, oc1 = outer
    return or0 <= ir0 and ir1 <= or1 and oc0 <= ic0 and ic1 <= oc1 and inner != outer


def nested_frame_recolor(train):
    def build(g):
        bg = _bg(g)
        comps = _components_same_color(g, bg, connectivity=8)
        if not comps:
            return g.copy()
        # for each component, find the smallest strictly-containing component
        # (its parent). root = component with no parent.
        n = len(comps)
        parent = [-1] * n
        for i in range(n):
            best = -1
            best_area = None
            for j in range(n):
                if i == j:
                    continue
                if _bbox_inside(comps[i]["bbox"], comps[j]["bbox"]):
                    r0, r1, c0, c1 = comps[j]["bbox"]
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if best_area is None or area < best_area:
                        best_area = area
                        best = j
            parent[i] = best
        # root color for each component
        out = g.copy()
        for i in range(n):
            # walk up to root
            k = i
            guard = 0
            while parent[k] != -1 and guard < n + 2:
                k = parent[k]
                guard += 1
            root_col = comps[k]["color"]
            if root_col != comps[i]["color"]:
                for (r, c) in comps[i]["cells"]:
                    out[r, c] = root_col
        return out

    return build


# ---------------------------------------------------------------------------
# 2) Nested alternating outlines inside a hollow square frame.
#    Input: a hollow rectangular frame (single color, here typically 5).
#    Output: interior filled with concentric square outlines whose colour
#    follows chebyshev-distance from the frame border:
#       d%4==0 -> frame color, ==1 -> second color, ==2 -> frame color,
#       ==3 -> background.
#    We learn the (per-distance) colour map directly from every demo so the
#    exact demo reproduction is guaranteed. (task a3f84088)
# ---------------------------------------------------------------------------
def nested_alt_outline(train):
    # Only single-frame tasks where in/out shapes match and the input is a
    # hollow rectangle of one non-bg colour.
    bg = _bg(train[0][0])
    # learn distance->color map, keyed by (d, is_lone_center) using ALL demos.
    dmap = {}          # (d) -> color  (for ring cells, ambiguous handled below)
    lone_map = {}      # radius -> color for the single center cell of odd shapes
    frames = []
    for ig, og in train:
        if ig.shape != og.shape:
            return None
        nz = np.argwhere(ig != bg)
        if nz.size == 0:
            return None
        r0, c0 = nz.min(0)
        r1, c1 = nz.max(0)
        sub_in = ig[r0:r1 + 1, c0:c1 + 1]
        cols = set(int(v) for v in np.unique(sub_in) if v != bg)
        if len(cols) != 1:
            return None
        fcol = cols.pop()
        # input must be exactly the hollow border of that region
        H, W = sub_in.shape
        if H < 3 or W < 3:
            return None
        border = np.ones((H, W), bool)
        border[1:-1, 1:-1] = False
        expect = np.where(border, fcol, bg)
        if not np.array_equal(sub_in, expect):
            return None
        frames.append((r0, c0, r1, c1, fcol))
        sub_out = og[r0:r1 + 1, c0:c1 + 1]
        # everything outside the frame bbox must be unchanged
        mask = np.ones(og.shape, bool)
        mask[r0:r1 + 1, c0:c1 + 1] = False
        if not np.array_equal(og[mask], ig[mask]):
            return None
        for i in range(H):
            for j in range(W):
                d = min(i, j, H - 1 - i, W - 1 - j)
                v = int(sub_out[i, j])
                # lone center cell of odd shape
                is_lone = (H % 2 == 1 and W % 2 == 1 and i == H // 2 and j == W // 2
                           and d == min(H // 2, W // 2))
                if is_lone:
                    R = d
                    if R in lone_map and lone_map[R] != v:
                        return None
                    lone_map[R] = v
                else:
                    if d in dmap and dmap[d] != v:
                        return None
                    dmap[d] = v
    if not frames:
        return None
    fcolors = set(f[4] for f in frames)

    def build(g):
        # locate the frame by its colour(s), not by majority background
        # (the frame can dominate the grid).
        mask = np.isin(g, list(fcolors))
        nz = np.argwhere(mask)
        if nz.size == 0:
            return g.copy()
        r0, c0 = nz.min(0)
        r1, c1 = nz.max(0)
        H = r1 - r0 + 1
        W = c1 - c0 + 1
        out = g.copy()
        for i in range(H):
            for j in range(W):
                d = min(i, j, H - 1 - i, W - 1 - j)
                is_lone = (H % 2 == 1 and W % 2 == 1 and i == H // 2 and j == W // 2
                           and d == min(H // 2, W // 2))
                if is_lone:
                    R = d
                    if R in lone_map:
                        val = lone_map[R]
                    else:
                        # extend period-4 rule from dmap
                        val = dmap.get(d % 4, dmap.get(d, bg))
                else:
                    val = dmap.get(d, dmap.get(d % 4, bg))
                out[r0 + i, c0 + j] = val
        return out

    return build


# ---------------------------------------------------------------------------
# 3) Periodic marker fill: the output is a perfect tiling of a small
#    fundamental block.  The input equals the output except a "marker" overlay
#    is present in only some tiles.  Replicate the marker to every tile.
#    (task 92e50de0)
# ---------------------------------------------------------------------------
def _smallest_period(g, axis):
    n = g.shape[axis]
    for p in range(1, n + 1):
        if axis == 0:
            base = g[:p]
            reps = int(np.ceil(n / p))
            til = np.tile(base, (reps, 1))[:n]
        else:
            base = g[:, :p]
            reps = int(np.ceil(n / p))
            til = np.tile(base, (1, reps))[:, :n]
        if np.array_equal(til, g):
            return p
    return n


def periodic_marker_fill(train):
    # learn period from the OUTPUT of the first demo, verify it reproduces.
    i0, o0 = train[0]
    if i0.shape != o0.shape:
        return None
    py = _smallest_period(o0, 0)
    px = _smallest_period(o0, 1)
    H0, W0 = o0.shape
    # require a genuine tiling (multiple tiles at least on one axis)
    if py >= H0 and px >= W0:
        return None
    if py < 2 and px < 2:
        return None

    def make(py, px):
        def build(g):
            H, W = g.shape
            bg = _bg(g)
            out = g.copy()
            # for each residue class, gather values across all tiles
            for a in range(py):
                for b in range(px):
                    vals = []
                    for r in range(a, H, py):
                        for c in range(b, W, px):
                            vals.append(int(g[r, c]))
                    if not vals:
                        continue
                    cnt = Counter(vals)
                    base = cnt.most_common(1)[0][0]
                    # marker = any non-background value that differs from base
                    marker = None
                    for v in vals:
                        if v != base and v != bg:
                            marker = v
                            break
                    fill = marker if marker is not None else base
                    for r in range(a, H, py):
                        for c in range(b, W, px):
                            out[r, c] = fill
            return out
        return build

    fn = make(py, px)
    try:
        if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 4) Bounding-box frame of seed dots.  A few isolated seed cells (all one
#    colour) define a rectangle; draw its outline in a learned colour and keep
#    the seeds on top. (task e7639916)
# ---------------------------------------------------------------------------
def seed_bbox_frame(train):
    # learn the frame colour (cells that are 0 in input but non-0 in output).
    frame_col = None
    for ig, og in train:
        if ig.shape != og.shape:
            return None
        added = og[(ig == 0) & (og != 0)]
        if added.size == 0:
            return None
        cols = set(int(v) for v in np.unique(added))
        if len(cols) != 1:
            return None
        fc = cols.pop()
        if frame_col is None:
            frame_col = fc
        elif frame_col != fc:
            return None
    if frame_col is None:
        return None

    def build(g):
        nz = np.argwhere(g != 0)
        if len(nz) < 2:
            return g.copy()
        r0, c0 = nz.min(0)
        r1, c1 = nz.max(0)
        if r1 - r0 < 1 or c1 - c0 < 1:
            return g.copy()
        out = g.copy()
        # draw outline
        for c in range(c0, c1 + 1):
            if out[r0, c] == 0:
                out[r0, c] = frame_col
            if out[r1, c] == 0:
                out[r1, c] = frame_col
        for r in range(r0, r1 + 1):
            if out[r, c0] == 0:
                out[r, c0] = frame_col
            if out[r, c1] == 0:
                out[r, c1] = frame_col
        return out

    try:
        if all(build(i).shape == o.shape and np.array_equal(build(i), o) for i, o in train):
            return build
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 5) Two identical-shape objects; output the rows in which they differ, marking
#    cells 8 where the two objects disagree. (task 2037f2c7)
# ---------------------------------------------------------------------------
def _nonzero_components(g):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    res = []
    for i in range(H):
        for j in range(W):
            if g[i, j] != 0 and not seen[i, j]:
                q = deque([(i, j)])
                seen[i, j] = True
                cells = [(i, j)]
                while q:
                    r, c = q.popleft()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and g[nr, nc] != 0 and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                                cells.append((nr, nc))
                rs = [a for a, _ in cells]
                cs = [b for _, b in cells]
                res.append((min(rs), max(rs), min(cs), max(cs)))
    return res


def two_object_diff_rows(train):
    # learn the marker colour from outputs (single non-zero colour).
    marks = set()
    for _, og in train:
        for v in np.unique(og):
            if v != 0:
                marks.add(int(v))
    if len(marks) != 1:
        return None
    mark = marks.pop()

    def build(g):
        boxes = _nonzero_components(g)
        if len(boxes) != 2:
            return g.copy()
        subs = [g[r0:r1 + 1, c0:c1 + 1] for r0, r1, c0, c1 in boxes]
        if subs[0].shape != subs[1].shape:
            return g.copy()
        a, b = subs
        diff = (a != b)
        rows = [r for r in range(diff.shape[0]) if diff[r].any()]
        if not rows:
            return g.copy()
        out = np.zeros((len(rows), diff.shape[1]), dtype=int)
        for oi, r in enumerate(rows):
            out[oi] = np.where(diff[r], mark, 0)
        return out

    try:
        if all(build(i).shape == o.shape and np.array_equal(build(i), o) for i, o in train):
            return build
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 6) Keep the single large shape, recolour it to the colour of the nearest
#    scattered "noise" cell, and erase all noise. (task 6df30ad6)
# ---------------------------------------------------------------------------
def keep_shape_recolor_by_nearest(train):
    def build(g):
        comps = _components_same_color(g, 0, connectivity=8)
        if len(comps) < 2:
            return g.copy()
        # main shape = component with most cells
        main = max(comps, key=lambda c: len(c["cells"]))
        others = [c for c in comps if c is not main]
        if not others:
            return g.copy()
        rs = [r for r, _ in main["cells"]]
        cs = [c for _, c in main["cells"]]
        cr = sum(rs) / len(rs)
        cc = sum(cs) / len(cs)
        # nearest noise cell to centroid
        best = None
        bd = None
        for comp in others:
            for (r, c) in comp["cells"]:
                d = (r - cr) ** 2 + (c - cc) ** 2
                if bd is None or d < bd:
                    bd = d
                    best = comp["color"]
        out = np.zeros_like(g)
        for (r, c) in main["cells"]:
            out[r, c] = best
        return out

    try:
        if all(build(i).shape == o.shape and np.array_equal(build(i), o) for i, o in train):
            return build
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 7) Key-shape dictionary recolor: two colours present -- a small "key" shape
#    and a large "main" shape.  A fixed dictionary maps the key's silhouette to
#    the colour the main shape should become; the key is then removed.  We
#    learn the dictionary from the demos (works when the test key recurs).
#    (task 009d5c81)
# ---------------------------------------------------------------------------
def _shape_key(mask):
    return (mask.shape, mask.tobytes())


def key_shape_dict_recolor(train):
    # identify, per demo: two colours, small one = key, large one = main.
    mapping = {}
    for ig, og in train:
        cols = [int(v) for v in np.unique(ig) if v != 0]
        if len(cols) != 2:
            return None
        counts = {c: int((ig == c).sum()) for c in cols}
        key_col = min(counts, key=lambda c: counts[c])
        main_col = max(counts, key=lambda c: counts[c])
        if key_col == main_col:
            return None
        kc = np.argwhere(ig == key_col)
        r0, c0 = kc.min(0)
        r1, c1 = kc.max(0)
        kmask = (ig[r0:r1 + 1, c0:c1 + 1] == key_col)
        # output = main shape recoloured, key removed
        outcols = set(int(v) for v in np.unique(og) if v != 0)
        if len(outcols) != 1:
            return None
        ocol = outcols.pop()
        # verify output is exactly the main shape (same mask) in ocol
        main_mask = (ig == main_col)
        if not np.array_equal(og != 0, main_mask):
            return None
        k = _shape_key(kmask)
        if k in mapping and mapping[k] != ocol:
            return None
        mapping[k] = ocol
    if not mapping:
        return None

    def build(g):
        cols = [int(v) for v in np.unique(g) if v != 0]
        if len(cols) != 2:
            return g.copy()
        counts = {c: int((g == c).sum()) for c in cols}
        key_col = min(counts, key=lambda c: counts[c])
        main_col = max(counts, key=lambda c: counts[c])
        kc = np.argwhere(g == key_col)
        r0, c0 = kc.min(0)
        r1, c1 = kc.max(0)
        kmask = (g[r0:r1 + 1, c0:c1 + 1] == key_col)
        k = _shape_key(kmask)
        if k not in mapping:
            return g.copy()
        ocol = mapping[k]
        out = np.zeros_like(g)
        out[g == main_col] = ocol
        return out

    return build


# ---------------------------------------------------------------------------
# 8) Remove the object carrying the fewest "marker" cells.  Objects are made of
#    a body colour speckled with a rarer marker colour; delete the single
#    object whose marker-count is minimal. (task 54db823b)
# ---------------------------------------------------------------------------
def _nonzero_comps_cells(g):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    res = []
    for i in range(H):
        for j in range(W):
            if g[i, j] != 0 and not seen[i, j]:
                q = deque([(i, j)])
                seen[i, j] = True
                cells = [(i, j)]
                while q:
                    r, c = q.popleft()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and g[nr, nc] != 0 and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                                cells.append((nr, nc))
                res.append(cells)
    return res


def remove_fewest_marker_object(train):
    # determine marker colour: the two non-bg colours; marker = the rarer one.
    cols = set()
    for ig, _ in train:
        for v in np.unique(ig):
            if v != 0:
                cols.add(int(v))
    if len(cols) != 2:
        return None
    cols = sorted(cols)
    # rarer colour overall = marker
    tot = {c: 0 for c in cols}
    for ig, _ in train:
        for c in cols:
            tot[c] += int((ig == c).sum())
    marker = min(cols, key=lambda c: tot[c])

    def build(g):
        comps = _nonzero_comps_cells(g)
        if len(comps) < 2:
            return g.copy()
        counts = [sum(1 for (r, c) in cells if g[r, c] == marker) for cells in comps]
        mn = min(counts)
        # require a unique minimum for a well-defined transform
        if counts.count(mn) != 1:
            return g.copy()
        idx = counts.index(mn)
        out = g.copy()
        for (r, c) in comps[idx]:
            out[r, c] = 0
        return out

    try:
        if all(build(i).shape == o.shape and np.array_equal(build(i), o) for i, o in train):
            return build
    except Exception:
        return None
    return None


DETECTORS = [
    nested_frame_recolor,
    nested_alt_outline,
    periodic_marker_fill,
    seed_bbox_frame,
    two_object_diff_rows,
    keep_shape_recolor_by_nearest,
    key_shape_dict_recolor,
    remove_fewest_marker_object,
]
