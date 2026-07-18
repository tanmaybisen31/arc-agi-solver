"""Counting-family rule detectors for ARC-AGI.

The output encodes a *count* derived from the input:
  - number of non-background cells, objects, or distinct colors
  - size of the largest / smallest object
  - per-color occurrence counts (a histogram)
rendered as an N-length bar, an NxN square, or a stack of colored columns/rows.

Each detector infers (count-source, rendering, fill-color) from the training
pairs and returns a transform that reproduces every training output exactly
(the engine re-verifies). Detectors return None whenever the rule doesn't fit.

Only numpy + stdlib. Defensive throughout.
"""
import numpy as np
from collections import Counter


# ---------------- helpers ----------------
def _bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag=False):
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _same_color_components(g, bg, diag=False):
    """Connected components restricted to a single color (per-color blobs)."""
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = g[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < H and 0 <= nx < W and not seen[ny, nx]
                            and g[ny, nx] == col):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((int(col), cells))
    return comps


def _bg_candidates(g):
    """Background hypotheses to try, most likely first: majority color, then 0."""
    cands = [_bg_color(g)]
    if 0 in g and 0 not in cands:
        cands.append(0)
    return cands


def _count_features(g, bg=None):
    """Return a dict of candidate integer count-sources for grid g."""
    try:
        if bg is None:
            bg = _bg_color(g)
        vals, cnts = np.unique(g, return_counts=True)
        nonbg_cells = int((g != bg).sum())
        ncolors = len(vals)
        ncolors_nonbg = len([v for v in vals if int(v) != bg])
        cc4 = _components(g, bg, False)
        cc8 = _components(g, bg, True)
        nobj4 = len(cc4)
        nobj8 = len(cc8)
        sizes4 = [len(c) for c in cc4]
        largest = max(sizes4) if sizes4 else 0
        smallest = min(sizes4) if sizes4 else 0
        nonbg_counts = [int(c) for v, c in zip(vals, cnts) if int(v) != bg]
        maxcc = max(nonbg_counts) if nonbg_counts else 0
        mincc = min(nonbg_counts) if nonbg_counts else 0
        # size of the largest single-color blob
        scc = _same_color_components(g, bg, False)
        blob_sizes = [len(cells) for _, cells in scc]
        largest_blob = max(blob_sizes) if blob_sizes else 0
        return {
            "nonbg_cells": nonbg_cells,
            "ncolors": ncolors,
            "ncolors_nonbg": ncolors_nonbg,
            "nobj4": nobj4,
            "nobj8": nobj8,
            "largest": largest,
            "smallest": smallest,
            "maxcc": maxcc,
            "mincc": mincc,
            "largest_blob": largest_blob,
        }
    except Exception:
        return {}


def _unique_nonbg_color(g, bg=None):
    if bg is None:
        bg = _bg_color(g)
    others = [int(v) for v in np.unique(g) if int(v) != bg]
    if len(others) == 1:
        return others[0]
    return None


_FEATURE_KEYS = ["nonbg_cells", "ncolors_nonbg", "nobj4", "nobj8",
                 "largest", "smallest", "maxcc", "mincc", "largest_blob", "ncolors"]


# ---------------- count -> bar ----------------
def count_bar(train):
    """Output is a 1xN or Nx1 monochrome bar; N is a count of the input, and
    the bar is filled with a fixed color or the input's unique non-bg color."""
    try:
        # every output must be a 1xN or Nx1 monochrome bar
        orient = None  # "row" (1xN) or "col" (Nx1)
        for _, o in train:
            H, W = o.shape
            if H == 1 and W >= 1:
                cur = "row"
            elif W == 1 and H >= 1:
                cur = "col"
            else:
                return None
            if len(np.unique(o)) != 1:
                return None
            if orient is None:
                orient = cur
            elif orient != cur:
                # mixed orientation only ok when the count is 1 (ambiguous 1x1)
                if max(H, W) != 1:
                    return None
        if orient is None:
            return None

        def out_len(o):
            return max(o.shape)

        out_colors = [int(o.flat[0]) for _, o in train]
        fixed_color = out_colors[0] if len(set(out_colors)) == 1 else None

        def make(key, orient, bg_mode, fixed_color, use_unique):
            def fn(g):
                bg = 0 if bg_mode == "zero" else _bg_color(g)
                f = _count_features(g, bg)
                n = f.get(key)
                if n is None or n < 1 or n > 900:
                    return None
                if use_unique:
                    color = _unique_nonbg_color(g, bg)
                    if color is None:
                        color = fixed_color
                elif fixed_color is not None:
                    color = fixed_color
                else:
                    return None
                if color is None:
                    return None
                if orient == "row":
                    return np.full((1, n), color, dtype=int)
                else:
                    return np.full((n, 1), color, dtype=int)
            return fn

        for bg_mode in ("zero", "major"):
            for key in _FEATURE_KEYS:
                # prefer unique-color rule (generalizes better) then fixed color
                for uu, fc in [(True, fixed_color), (False, fixed_color)]:
                    if not uu and fc is None:
                        continue
                    fn = make(key, orient, bg_mode, fc, uu)
                    ok = True
                    for i, o in train:
                        r = fn(i)
                        if r is None or r.shape != o.shape or not np.array_equal(r, o):
                            ok = False
                            break
                    if ok:
                        return fn
        return None
    except Exception:
        return None


# ---------------- count -> square ----------------
def count_square(train):
    """Output is an NxN monochrome square; N is a count of the input."""
    try:
        for _, o in train:
            H, W = o.shape
            if H != W or H < 1:
                return None
            if len(np.unique(o)) != 1:
                return None

        def out_n(o):
            return o.shape[0]

        # require the count to actually vary across train (avoid trivially matching
        # a fixed-size output that constant_output already handles)
        ns = set(out_n(o) for _, o in train)
        if len(ns) < 2:
            return None

        out_colors = [int(o.flat[0]) for _, o in train]
        fixed_color = out_colors[0] if len(set(out_colors)) == 1 else None

        def make(key, bg_mode, fixed_color, use_unique):
            def fn(g):
                bg = 0 if bg_mode == "zero" else _bg_color(g)
                f = _count_features(g, bg)
                n = f.get(key)
                if n is None or n < 1 or n > 30:
                    return None
                if use_unique:
                    color = _unique_nonbg_color(g, bg)
                    if color is None:
                        color = fixed_color
                elif fixed_color is not None:
                    color = fixed_color
                else:
                    return None
                if color is None:
                    return None
                return np.full((n, n), color, dtype=int)
            return fn

        for bg_mode in ("zero", "major"):
            for key in _FEATURE_KEYS:
                for uu, fc in [(True, fixed_color), (False, fixed_color)]:
                    if not uu and fc is None:
                        continue
                    fn = make(key, bg_mode, fc, uu)
                    ok = True
                    for i, o in train:
                        r = fn(i)
                        if r is None or r.shape != o.shape or not np.array_equal(r, o):
                            ok = False
                            break
                    if ok:
                        return fn
        return None
    except Exception:
        return None


# ---------------- per-color histogram ----------------
def color_histogram(train):
    """Output is a histogram of the non-background colors: one column (or row)
    per color, its length equal to that color's count in the input, columns
    ordered by count. Each column is painted with its own color."""
    try:
        def build(g, axis, order, from_top):
            bg = _bg_color(g)
            vals, cnts = np.unique(g, return_counts=True)
            pairs = [(int(v), int(c)) for v, c in zip(vals, cnts) if int(v) != bg]
            if not pairs:
                return None
            if order == "desc":
                pairs.sort(key=lambda p: (-p[1], p[0]))
            else:
                pairs.sort(key=lambda p: (p[1], p[0]))
            maxc = max(c for _, c in pairs)
            k = len(pairs)
            if maxc < 1 or maxc > 60 or k > 20:
                return None
            if axis == "col":
                out = np.zeros((maxc, k), dtype=int)
                for j, (col, c) in enumerate(pairs):
                    if from_top:
                        out[:c, j] = col
                    else:
                        out[maxc - c:, j] = col
                return out
            else:  # rows
                out = np.zeros((k, maxc), dtype=int)
                for j, (col, c) in enumerate(pairs):
                    if from_top:
                        out[j, :c] = col
                    else:
                        out[j, maxc - c:] = col
                return out

        for axis in ("col", "row"):
            for order in ("desc", "asc"):
                for from_top in (True, False):
                    def fn(g, axis=axis, order=order, from_top=from_top):
                        return build(g, axis, order, from_top)
                    ok = True
                    for i, o in train:
                        r = fn(i)
                        if r is None or r.shape != o.shape or not np.array_equal(r, o):
                            ok = False
                            break
                    if ok:
                        return fn
        return None
    except Exception:
        return None


# ---------------- selected-object color rendered as a fixed block ----------------
def _obj_selector_color(g, mode):
    """Return the color of the selected object under `mode`."""
    bg = _bg_color(g)
    if mode in ("largest_obj", "smallest_obj"):
        comps = _components(g, bg, False)
        if not comps:
            return None
        pick = max(comps, key=len) if mode == "largest_obj" else min(comps, key=len)
        y, x = pick[0]
        return int(g[y, x])
    if mode in ("largest_blob", "smallest_blob"):
        scc = _same_color_components(g, bg, False)
        if not scc:
            return None
        pick = (max(scc, key=lambda t: len(t[1])) if mode == "largest_blob"
                else min(scc, key=lambda t: len(t[1])))
        return pick[0]
    if mode in ("most_common", "least_common"):
        vals, cnts = np.unique(g, return_counts=True)
        pairs = [(int(v), int(c)) for v, c in zip(vals, cnts) if int(v) != bg]
        if not pairs:
            return None
        pick = (max(pairs, key=lambda p: p[1]) if mode == "most_common"
                else min(pairs, key=lambda p: p[1]))
        return pick[0]
    return None


def selected_object_color_block(train):
    """Output is a fixed-size monochrome block filled with the color of a
    selected object (largest/smallest object, largest/smallest solid blob,
    or most/least common non-bg color)."""
    try:
        shapes = set(o.shape for _, o in train)
        if len(shapes) != 1:
            return None
        oshape = next(iter(shapes))
        if oshape[0] * oshape[1] <= 1:
            return None
        for _, o in train:
            if len(np.unique(o)) != 1:
                return None
        # the output color must vary across train (else constant_output covers it)
        oc = set(int(o.flat[0]) for _, o in train)
        if len(oc) < 2:
            return None

        modes = ["largest_blob", "largest_obj", "most_common",
                 "smallest_blob", "smallest_obj", "least_common"]
        for mode in modes:
            ok = True
            for i, o in train:
                c = _obj_selector_color(i, mode)
                if c is None or c != int(o.flat[0]):
                    ok = False
                    break
            if ok:
                def fn(g, mode=mode, oshape=oshape):
                    c = _obj_selector_color(g, mode)
                    if c is None:
                        return None
                    return np.full(oshape, c, dtype=int)
                return fn
        return None
    except Exception:
        return None


# ---------------- tile a fixed template by a marker count ----------------
def tile_shape_by_count(train):
    """Output is a fixed shape-template repeated N times along one axis, where N
    equals the number of occurrences (cells or objects) of a 'marker' color in
    the input. The template is learned (and required constant) across train."""
    try:
        if len(train) < 2:
            return None

        def marker_count(g, color, mode):
            if mode == "cells":
                return int((g == color).sum())
            # objects of that color
            H, W = g.shape
            seen = np.zeros((H, W), dtype=bool)
            nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            n = 0
            for r in range(H):
                for c in range(W):
                    if seen[r, c] or g[r, c] != color:
                        continue
                    n += 1
                    st = [(r, c)]
                    seen[r, c] = True
                    while st:
                        y, x = st.pop()
                        for dy, dx in nb:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < H and 0 <= nx < W and not seen[ny, nx]
                                    and g[ny, nx] == color):
                                seen[ny, nx] = True
                                st.append((ny, nx))
            return n

        i0, o0 = train[0]
        # candidate marker colors: any color present in every input
        common = set(int(v) for v in np.unique(i0))
        for i, _ in train[1:]:
            common &= set(int(v) for v in np.unique(i))
        if not common:
            return None

        for axis in ("h", "v"):
            for mode in ("cells", "objects"):
                for marker in sorted(common):
                    # derive N per train, split output into N tiles, require constant template
                    template = None
                    ok = True
                    for i, o in train:
                        n = marker_count(i, marker, mode)
                        if n < 1 or n > 30:
                            ok = False
                            break
                        if axis == "h":
                            if o.shape[1] % n != 0:
                                ok = False
                                break
                            tw = o.shape[1] // n
                            tiles = [o[:, k * tw:(k + 1) * tw] for k in range(n)]
                        else:
                            if o.shape[0] % n != 0:
                                ok = False
                                break
                            th = o.shape[0] // n
                            tiles = [o[k * th:(k + 1) * th, :] for k in range(n)]
                        # all tiles within this output must be identical
                        if any(not np.array_equal(t, tiles[0]) for t in tiles):
                            ok = False
                            break
                        if template is None:
                            template = tiles[0].copy()
                        elif template.shape != tiles[0].shape or not np.array_equal(template, tiles[0]):
                            ok = False
                            break
                    if not ok or template is None:
                        continue
                    # template must be non-trivial (avoid degenerate 1-col cases that
                    # count_bar already handles better) — require >1 distinct value or area>1
                    if template.size == 1 and len(np.unique([t for _, o in train for t in [o]])) < 2:
                        pass
                    tmpl = template

                    def fn(g, axis=axis, mode=mode, marker=marker, tmpl=tmpl):
                        n = marker_count(g, marker, mode)
                        if n < 1 or n > 30:
                            return None
                        if axis == "h":
                            return np.hstack([tmpl] * n)
                        else:
                            return np.vstack([tmpl] * n)

                    good = True
                    for i, o in train:
                        r = fn(i)
                        if r is None or r.shape != o.shape or not np.array_equal(r, o):
                            good = False
                            break
                    if good:
                        return fn

        # variant: tile = the non-marker content cropped to its bounding box,
        # repeated N times (template derived per-input, not constant).
        def content_bbox(g, marker, bg):
            mask = (g != bg) & (g != marker)
            if not mask.any():
                return None
            rs, cs = np.where(mask)
            sub = g[rs.min():rs.max() + 1, cs.min():cs.max() + 1].copy()
            sub[sub == marker] = bg
            return sub

        for axis in ("h", "v"):
            for mode in ("cells", "objects"):
                for marker in sorted(common):
                    def fn(g, axis=axis, mode=mode, marker=marker):
                        bg = _bg_color(g)
                        n = marker_count(g, marker, mode)
                        if n < 1 or n > 30:
                            return None
                        sub = content_bbox(g, marker, bg)
                        if sub is None or sub.size == 0:
                            return None
                        if axis == "h":
                            return np.hstack([sub] * n)
                        else:
                            return np.vstack([sub] * n)
                    good = True
                    for i, o in train:
                        r = fn(i)
                        if r is None or r.shape != o.shape or not np.array_equal(r, o):
                            good = False
                            break
                    if good:
                        return fn
        return None
    except Exception:
        return None


# ---------------- count -> N marks in a fixed template ----------------
def _count_sources(g):
    """Yield (label, value) integer count-sources, including per-(color,size)
    component counts, for a richer 'count of specific objects' space."""
    try:
        bg = _bg_color(g)
        feats = _count_features(g, bg)
        for k, v in feats.items():
            yield ("feat:" + k, v)
        # per-color component-size counts (both 4- and 8-connectivity)
        for diag in (False, True):
            per = {}  # (color, size) -> count
            comps = _same_color_components(g, bg, diag)
            for col, cells in comps:
                per[(col, len(cells))] = per.get((col, len(cells)), 0) + 1
            # also count of components of a given color regardless of size
            per_col = {}
            for col, cells in comps:
                per_col[col] = per_col.get(col, 0) + 1
            for (col, sz), cnt in per.items():
                yield (f"blob{int(diag)}:c{col}:s{sz}", cnt)
            for col, cnt in per_col.items():
                yield (f"blob{int(diag)}:c{col}:any", cnt)
    except Exception:
        return


def count_marks(train):
    """Output is a fixed-shape grid, mostly one background color, with N cells of
    a mark color placed along the main diagonal (or in reading order). N is an
    inferred count of the input."""
    try:
        shapes = set(o.shape for _, o in train)
        if len(shapes) != 1:
            return None
        oshape = next(iter(shapes))
        H, W = oshape
        if H * W <= 1 or H * W > 64:
            return None
        # each output: exactly one non-bg 'mark' color, placed on the diagonal
        # (top-left first) or in reading order; background is the majority color.
        out_bg = None
        mark_color = None
        placement = None  # "diag" or "reading"
        counts = []
        for _, o in train:
            vals, cnts = np.unique(o, return_counts=True)
            b = int(vals[np.argmax(cnts)])
            others = [int(v) for v in vals if int(v) != b]
            if len(others) > 1:
                return None
            mc = others[0] if others else b
            nmark = int((o == mc).sum()) if others else 0
            if out_bg is None:
                out_bg = b
            elif out_bg != b:
                return None
            if others:
                if mark_color is None:
                    mark_color = mc
                elif mark_color != mc:
                    return None
            counts.append(nmark)
            # verify placement
            diag_grid = np.full(oshape, b, dtype=int)
            for k in range(min(nmark, min(H, W))):
                diag_grid[k, k] = mc
            read_grid = np.full(oshape, b, dtype=int)
            flat = read_grid.reshape(-1)
            flat[:nmark] = mc
            read_grid = flat.reshape(oshape)
            is_diag = np.array_equal(diag_grid, o)
            is_read = np.array_equal(read_grid, o)
            if not is_diag and not is_read:
                return None
            if placement is None:
                placement = "diag" if is_diag else "reading"
            else:
                if placement == "diag" and not is_diag:
                    placement = "reading" if is_read else None
                elif placement == "reading" and not is_read:
                    placement = "diag" if is_diag else None
                if placement is None:
                    return None
        if mark_color is None or placement is None:
            return None
        # counts must vary (else constant_output handles it)
        if len(set(counts)) < 2:
            return None

        # find a count-source consistent across all train
        src_maps = [dict(_count_sources(i)) for i, _ in train]
        common_labels = set(src_maps[0].keys())
        for sm in src_maps[1:]:
            common_labels &= set(sm.keys())
        good_label = None
        for label in sorted(common_labels):
            if all(src_maps[k].get(label) == counts[k] for k in range(len(train))):
                good_label = label
                break
        if good_label is None:
            return None

        def render(n):
            g = np.full(oshape, out_bg, dtype=int)
            if placement == "diag":
                for k in range(min(n, min(H, W))):
                    g[k, k] = mark_color
            else:
                flat = g.reshape(-1)
                flat[:min(n, H * W)] = mark_color
                g = flat.reshape(oshape)
            return g

        def fn(g, label=good_label):
            sm = dict(_count_sources(g))
            n = sm.get(label)
            if n is None or n < 0 or n > H * W:
                return None
            return render(n)

        for i, o in train:
            r = fn(i)
            if r is None or not np.array_equal(r, o):
                return None
        return fn
    except Exception:
        return None


DETECTORS = [
    count_bar,
    count_square,
    color_histogram,
    selected_object_color_block,
    tile_shape_by_count,
    count_marks,
]
