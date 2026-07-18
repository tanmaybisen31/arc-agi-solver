"""Object-recoloring rule detectors for ARC-AGI.

Family: find connected components (per-color, 4- and 8-connectivity), then
recolor each object by a learnable property:
  - object cell-count (size -> color map)
  - size rank within the grid (largest -> colorA, ...)
  - shape identity (normalized cell-set -> color, optionally mod symmetry)
  - number of enclosed holes (hole-count -> color)

Every detector only ever recolors cells of existing objects; grid shape and the
occupied-cell mask are preserved.  The engine verifies each transform on all
training pairs before use, so speculative/wrong fits are auto-rejected.
"""
import numpy as np
from collections import Counter, defaultdict

# ----------------------------- helpers -----------------------------

def _bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag, per_color=True):
    """Connected components of non-bg cells.

    per_color: if True, only cells of the SAME color are connected together
    (objects are mono-color). If False, any two adjacent non-bg cells join.
    Returns list of dicts: {cells:[(r,c)..], color:int}.
    """
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
            col0 = g[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            colors = set()
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                colors.add(int(g[y, x]))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        if per_color and g[ny, nx] != col0:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append({"cells": cells, "color": int(col0), "colors": colors})
    return comps


def _norm_shape(cells):
    """Translation-normalized frozenset of cell offsets."""
    rs = min(y for y, _ in cells)
    cs = min(x for _, x in cells)
    return frozenset((y - rs, x - cs) for y, x in cells)


def _shape_canon(cells):
    """Canonical shape key invariant to translation + 8 dihedral symmetries."""
    base = _norm_shape(cells)
    pts = list(base)
    variants = []
    for _ in range(4):  # rotate 4x, each also flipped
        # rotate 90: (y,x)->(x,-y)
        pts = [(x, -y) for (y, x) in pts]
        for flip in (False, True):
            p = [(y, -x) for (y, x) in pts] if flip else list(pts)
            ry = min(y for y, _ in p)
            rx = min(x for _, x in p)
            variants.append(frozenset((y - ry, x - rx) for y, x in p))
    return min(variants, key=lambda s: tuple(sorted(s)))


def _count_holes(cells, diag_bg=False):
    """Number of bg regions fully enclosed by this object's bounding box.

    Fills the object into its bounding box, flood-fills bg from the border;
    remaining bg cells form enclosed holes -> count connected hole regions.
    """
    if not cells:
        return 0
    rs = min(y for y, _ in cells); re = max(y for y, _ in cells)
    cs = min(x for _, x in cells); ce = max(x for _, x in cells)
    H = re - rs + 1
    W = ce - cs + 1
    grid = np.zeros((H, W), dtype=bool)  # True = object cell
    for y, x in cells:
        grid[y - rs, x - cs] = True
    # bg reachable from border
    reach = np.zeros((H, W), dtype=bool)
    stack = []
    for y in range(H):
        for x in (0, W - 1):
            if not grid[y, x] and not reach[y, x]:
                reach[y, x] = True; stack.append((y, x))
    for x in range(W):
        for y in (0, H - 1):
            if not grid[y, x] and not reach[y, x]:
                reach[y, x] = True; stack.append((y, x))
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while stack:
        y, x = stack.pop()
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not grid[ny, nx] and not reach[ny, nx]:
                reach[ny, nx] = True; stack.append((ny, nx))
    # count connected hole regions among unreached bg cells
    hole_seen = np.zeros((H, W), dtype=bool)
    holes = 0
    hn = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag_bg:
        hn = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(H):
        for x in range(W):
            if grid[y, x] or reach[y, x] or hole_seen[y, x]:
                continue
            holes += 1
            st = [(y, x)]; hole_seen[y, x] = True
            while st:
                a, b = st.pop()
                for dy, dx in hn:
                    na, nb = a + dy, b + dx
                    if 0 <= na < H and 0 <= nb < W and not grid[na, nb] and not reach[na, nb] and not hole_seen[na, nb]:
                        hole_seen[na, nb] = True; st.append((na, nb))
    return holes


def _same_shape_all(train):
    return all(i.shape == o.shape for i, o in train)


def _recolor_mask_preserved(train):
    """Output changes only colors; occupied-cell mask (nonbg) preserved."""
    for i, o in train:
        if i.shape != o.shape:
            return False
        bi = _bg_color(i); bo = _bg_color(o)
        if not np.array_equal(i != bi, o != bo):
            return False
    return True


def _apply_recolor(g, comps, color_of):
    """Build output = copy of g with each comp recolored via color_of(comp) (None = keep)."""
    out = g.copy()
    for comp in comps:
        nc = color_of(comp)
        if nc is None:
            continue
        for y, x in comp["cells"]:
            out[y, x] = nc
    return out


# ----------------------------- detectors -----------------------------

def _build_property_detector(prop_fn, per_color, diag, fallback="keep"):
    """Generic: learn a map  property-value -> output-color  from train.

    prop_fn(comp, g) returns a hashable key. For each object we read the
    (single) output color at its cells. If the key->color map is consistent
    across ALL training objects, return a transform.

    fallback controls what an *unseen* property value maps to at test time:
      "keep"     -> leave the object its original color
      "majority" -> the most common output color across training objects
    """
    def det(train):
        try:
            if not _recolor_mask_preserved(train):
                return None
            mapping = {}
            color_counter = Counter()
            for i, o in train:
                bi = _bg_color(i)
                comps = _components(i, bi, diag, per_color)
                if not comps:
                    return None
                for comp in comps:
                    # output color for this object (must be uniform)
                    ocolors = set(int(o[y, x]) for y, x in comp["cells"])
                    if len(ocolors) != 1:
                        return None
                    oc = next(iter(ocolors))
                    key = prop_fn(comp, i)
                    if key is None:
                        return None
                    if key in mapping and mapping[key] != oc:
                        return None
                    mapping[key] = oc
                    color_counter[oc] += 1
            if not mapping:
                return None
            majority = color_counter.most_common(1)[0][0] if fallback == "majority" else None
            def fn(g, mapping=mapping, majority=majority):
                bg = _bg_color(g)
                comps = _components(g, bg, diag, per_color)
                def color_of(comp):
                    key = prop_fn(comp, g)
                    if key in mapping:
                        return mapping[key]
                    return majority  # None if fallback=="keep"
                return _apply_recolor(g, comps, color_of)
            return fn
        except Exception:
            return None
    det.__name__ = "recolor_prop_%s_%s_%s_%s" % (
        getattr(prop_fn, "__name__", "p"),
        "d8" if diag else "d4",
        "any" if not per_color else "col",
        fallback,
    )
    return det


# --- size (cell-count) -> color ---
def _prop_size(comp, g):
    return len(comp["cells"])

recolor_by_size_4 = _build_property_detector(_prop_size, per_color=True, diag=False)
recolor_by_size_8 = _build_property_detector(_prop_size, per_color=True, diag=True)
recolor_by_size_any_4 = _build_property_detector(_prop_size, per_color=False, diag=False)
recolor_by_size_any_8 = _build_property_detector(_prop_size, per_color=False, diag=True)
# majority-fallback: unseen sizes at test time take the dominant output color
recolor_by_size_4_mj = _build_property_detector(_prop_size, per_color=True, diag=False, fallback="majority")
recolor_by_size_8_mj = _build_property_detector(_prop_size, per_color=True, diag=True, fallback="majority")
recolor_by_size_any_4_mj = _build_property_detector(_prop_size, per_color=False, diag=False, fallback="majority")
recolor_by_size_any_8_mj = _build_property_detector(_prop_size, per_color=False, diag=True, fallback="majority")


# --- shape identity (mod symmetry) -> color ---
def _prop_shape_canon(comp, g):
    return _shape_canon(comp["cells"])

recolor_by_shape_4 = _build_property_detector(_prop_shape_canon, per_color=True, diag=False)
recolor_by_shape_8 = _build_property_detector(_prop_shape_canon, per_color=True, diag=True)
recolor_by_shape_any_4 = _build_property_detector(_prop_shape_canon, per_color=False, diag=False)
recolor_by_shape_any_8 = _build_property_detector(_prop_shape_canon, per_color=False, diag=True)


# --- exact shape (translation only) -> color ---
def _prop_shape_exact(comp, g):
    return _norm_shape(comp["cells"])

recolor_by_exact_shape_4 = _build_property_detector(_prop_shape_exact, per_color=True, diag=False)
recolor_by_exact_shape_8 = _build_property_detector(_prop_shape_exact, per_color=True, diag=True)


# --- number of holes -> color ---
def _prop_holes(comp, g):
    return _count_holes(comp["cells"])

recolor_by_holes_4 = _build_property_detector(_prop_holes, per_color=True, diag=False)
recolor_by_holes_8 = _build_property_detector(_prop_holes, per_color=True, diag=True)
recolor_by_holes_any_4 = _build_property_detector(_prop_holes, per_color=False, diag=False)
recolor_by_holes_any_8 = _build_property_detector(_prop_holes, per_color=False, diag=True)
recolor_by_holes_8_mj = _build_property_detector(_prop_holes, per_color=True, diag=True, fallback="majority")
recolor_by_holes_any_8_mj = _build_property_detector(_prop_holes, per_color=False, diag=True, fallback="majority")


# --- size rank within each grid -> color ---
def _build_rank_detector(per_color, diag, ascending):
    """Rank objects by size within each grid; learn rank -> color.

    ascending=False: rank 0 = largest. Objects with equal size share a rank.
    """
    def det(train):
        try:
            if not _recolor_mask_preserved(train):
                return None
            rank_map = {}
            for i, o in train:
                bi = _bg_color(i)
                comps = _components(i, bi, diag, per_color)
                if len(comps) < 2:
                    return None
                sizes = sorted(set(len(c["cells"]) for c in comps), reverse=not ascending)
                size_to_rank = {s: r for r, s in enumerate(sizes)}
                for comp in comps:
                    ocolors = set(int(o[y, x]) for y, x in comp["cells"])
                    if len(ocolors) != 1:
                        return None
                    oc = next(iter(ocolors))
                    rk = size_to_rank[len(comp["cells"])]
                    if rk in rank_map and rank_map[rk] != oc:
                        return None
                    rank_map[rk] = oc
            if len(rank_map) < 2:
                return None
            def fn(g, rank_map=rank_map, ascending=ascending):
                bg = _bg_color(g)
                comps = _components(g, bg, diag, per_color)
                sizes = sorted(set(len(c["cells"]) for c in comps), reverse=not ascending)
                size_to_rank = {s: r for r, s in enumerate(sizes)}
                def color_of(comp):
                    return rank_map.get(size_to_rank[len(comp["cells"])], None)
                return _apply_recolor(g, comps, color_of)
            return fn
        except Exception:
            return None
    det.__name__ = "recolor_rank_%s_%s_%s" % (
        "asc" if ascending else "desc",
        "d8" if diag else "d4",
        "any" if not per_color else "col",
    )
    return det

recolor_rank_desc_4 = _build_rank_detector(per_color=True, diag=False, ascending=False)
recolor_rank_asc_4 = _build_rank_detector(per_color=True, diag=False, ascending=True)
recolor_rank_desc_8 = _build_rank_detector(per_color=True, diag=True, ascending=False)
recolor_rank_asc_8 = _build_rank_detector(per_color=True, diag=True, ascending=True)
recolor_rank_desc_any8 = _build_rank_detector(per_color=False, diag=True, ascending=False)
recolor_rank_asc_any8 = _build_rank_detector(per_color=False, diag=True, ascending=True)


# --- most-common vs rest: largest color-group behavior ---
def _build_size_extreme_detector(per_color, diag, target):
    """Recolor only the single largest (or smallest) object to a learned color,
    leave others unchanged. target in {'largest','smallest'}.
    """
    def det(train):
        try:
            if not _recolor_mask_preserved(train):
                return None
            new_color = None
            for i, o in train:
                bi = _bg_color(i)
                comps = _components(i, bi, diag, per_color)
                if len(comps) < 2:
                    return None
                sizes = [len(c["cells"]) for c in comps]
                if target == "largest":
                    ext = max(sizes)
                else:
                    ext = min(sizes)
                if sizes.count(ext) != 1:
                    return None  # ambiguous
                idx = sizes.index(ext)
                for j, comp in enumerate(comps):
                    ocolors = set(int(o[y, x]) for y, x in comp["cells"])
                    if len(ocolors) != 1:
                        return None
                    oc = next(iter(ocolors))
                    if j == idx:
                        if new_color is None:
                            new_color = oc
                        elif new_color != oc:
                            return None
                    else:
                        if oc != comp["color"]:
                            return None  # others must be unchanged
            if new_color is None:
                return None
            def fn(g, new_color=new_color):
                bg = _bg_color(g)
                comps = _components(g, bg, diag, per_color)
                if len(comps) < 1:
                    return g.copy()
                sizes = [len(c["cells"]) for c in comps]
                ext = max(sizes) if target == "largest" else min(sizes)
                out = g.copy()
                if sizes.count(ext) == 1:
                    idx = sizes.index(ext)
                    for y, x in comps[idx]["cells"]:
                        out[y, x] = new_color
                return out
            return fn
        except Exception:
            return None
    det.__name__ = "recolor_%s_%s_%s" % (
        target, "d8" if diag else "d4", "any" if not per_color else "col",
    )
    return det

recolor_largest_8 = _build_size_extreme_detector(per_color=False, diag=True, target="largest")
recolor_smallest_8 = _build_size_extreme_detector(per_color=False, diag=True, target="smallest")
recolor_largest_4 = _build_size_extreme_detector(per_color=True, diag=False, target="largest")
recolor_smallest_4 = _build_size_extreme_detector(per_color=True, diag=False, target="smallest")


DETECTORS = [
    # shape identity is the most specific/reliable, so list first
    recolor_by_shape_4, recolor_by_shape_8,
    recolor_by_shape_any_4, recolor_by_shape_any_8,
    recolor_by_exact_shape_4, recolor_by_exact_shape_8,
    # holes (number of enclosed regions)
    recolor_by_holes_8_mj, recolor_by_holes_any_8_mj,
    recolor_by_holes_4, recolor_by_holes_8,
    recolor_by_holes_any_4, recolor_by_holes_any_8,
    # size (cell count) — majority-fallback first (generalizes to unseen sizes)
    recolor_by_size_4_mj, recolor_by_size_8_mj,
    recolor_by_size_any_4_mj, recolor_by_size_any_8_mj,
    recolor_by_size_4, recolor_by_size_8,
    recolor_by_size_any_4, recolor_by_size_any_8,
    # size rank
    recolor_rank_desc_4, recolor_rank_asc_4,
    recolor_rank_desc_8, recolor_rank_asc_8,
    recolor_rank_desc_any8, recolor_rank_asc_any8,
    # single-extreme recolor
    recolor_largest_8, recolor_smallest_8,
    recolor_largest_4, recolor_smallest_4,
]
