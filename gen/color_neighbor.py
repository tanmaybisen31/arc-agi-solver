"""Neighbor / position coloring detectors for ARC-AGI.

Family: recolor cells by local rules --
  * enclosed (inside vs outside) region fill
  * count of same/other-colored 4- or 8-neighbors -> color
  * parity of row/col
Each detector infers the map from train and returns a transform_fn|None.
The engine verifies the transform reproduces every train output exactly.
"""
import numpy as np
from collections import Counter

# ------------------------------------------------------------------ helpers
def _bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _flood_outside(mask_free):
    """mask_free: bool grid of cells that flood-fill can pass through (True = passable).
    Returns bool grid of cells reachable from the border through passable cells (4-conn)."""
    H, W = mask_free.shape
    reach = np.zeros((H, W), dtype=bool)
    stack = []
    for r in range(H):
        for c in (0, W - 1):
            if mask_free[r, c] and not reach[r, c]:
                reach[r, c] = True
                stack.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if mask_free[r, c] and not reach[r, c]:
                reach[r, c] = True
                stack.append((r, c))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask_free[ny, nx] and not reach[ny, nx]:
                reach[ny, nx] = True
                stack.append((ny, nx))
    return reach


# ================================================================== detectors

# ---- 1. enclosed-region fill (inside vs outside) -----------------
def enclosed_fill(train):
    """Background cells that are enclosed (cannot reach the border through
    background) get recolored to a learned fill color. Everything else is
    unchanged. Handles both 4- and 8-connectivity for the 'wall'."""
    if any(i.shape != o.shape for i, o in train):
        return None
    # infer background = most common color across all inputs
    bgs = [_bg_color(i) for i, _ in train]
    bg = Counter(bgs).most_common(1)[0][0]

    # The output must only differ from input on cells that are bg in the input.
    fill_colors = set()
    for i, o in train:
        diff = i != o
        if diff.any():
            if not np.all(i[diff] == bg):
                return None  # changed a non-background cell -> not this rule
            fill_colors |= set(np.unique(o[diff]).tolist())
    if len(fill_colors) != 1:
        return None
    fill = int(next(iter(fill_colors)))

    def fn(g):
        b = _bg_color(g)
        free = (g == b)
        reach = _flood_outside(free)
        enclosed = free & (~reach)
        out = g.copy()
        out[enclosed] = fill
        return out

    return fn


# ---- 2. neighbor-count recoloring -------------------------------
def _neighbor_count_map(train, diag, count_mode):
    """Try to learn a mapping (input_color, neighbor_count) -> output_color.
    count_mode:
      'same'  -> count of orthogonal/diag neighbors with the SAME color
      'nonbg' -> count of neighbors that are non-background
      'bg'    -> count of neighbors equal to background
    Returns fn or None."""
    if any(i.shape != o.shape for i, o in train):
        return None
    bgs = [_bg_color(i) for i, _ in train]
    bg = Counter(bgs).most_common(1)[0][0]
    if diag:
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def counts_grid(g, b):
        H, W = g.shape
        cnt = np.zeros((H, W), dtype=int)
        for dy, dx in offs:
            ys0 = max(0, -dy); ys1 = H - max(0, dy)
            xs0 = max(0, -dx); xs1 = W - max(0, dx)
            src = g[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            if count_mode == 'same':
                match = (src == g[ys0:ys1, xs0:xs1])
            elif count_mode == 'nonbg':
                match = (src != b)
            else:  # 'bg'
                match = (src == b)
            cnt[ys0:ys1, xs0:xs1] += match.astype(int)
        return cnt

    # learn mapping from (color, count) -> out_color, only where useful
    mapping = {}
    for i, o in train:
        b = _bg_color(i)
        cnt = counts_grid(i, b)
        H, W = i.shape
        for r in range(H):
            for c in range(W):
                key = (int(i[r, c]), int(cnt[r, c]))
                val = int(o[r, c])
                if key in mapping and mapping[key] != val:
                    return None
                mapping[key] = val
    # require the rule actually recolors something (not pure identity by color)
    changes = any(k[0] != v for k, v in mapping.items())
    if not changes:
        return None

    def fn(g):
        b = _bg_color(g)
        cnt = counts_grid(g, b)
        out = g.copy()
        H, W = g.shape
        for r in range(H):
            for c in range(W):
                key = (int(g[r, c]), int(cnt[r, c]))
                if key in mapping:
                    out[r, c] = mapping[key]
        return out

    return fn


def neighbor_count_recolor(train):
    for diag in (False, True):
        for mode in ('same', 'nonbg', 'bg'):
            try:
                fn = _neighbor_count_map(train, diag, mode)
            except Exception:
                fn = None
            if fn is None:
                continue
            try:
                if all(fn(i).shape == o.shape and np.array_equal(fn(i), o)
                       for i, o in train):
                    return fn
            except Exception:
                continue
    return None


# ---- 3. enclosed fill, per-region color = surrounding wall color -----
def _regions_and_wall_color(g, bg):
    """Return list of (cells, wall_color) for each enclosed background region.
    wall_color is the single non-bg color bordering the region, or None."""
    H, W = g.shape
    free = (g == bg)
    reach = _flood_outside(free)
    enclosed = free & (~reach)
    seen = np.zeros((H, W), dtype=bool)
    regions = []
    for r in range(H):
        for c in range(W):
            if not enclosed[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            wall = set()
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if enclosed[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                        elif g[ny, nx] != bg:
                            wall.add(int(g[ny, nx]))
            wc = next(iter(wall)) if len(wall) == 1 else None
            regions.append((cells, wc))
    return regions


def enclosed_fill_wallcolor(train):
    """Enclosed background regions get filled with the color of the object that
    encloses them (each hole takes its surrounding wall's color)."""
    if any(i.shape != o.shape for i, o in train):
        return None
    bgs = [_bg_color(i) for i, _ in train]
    bg = Counter(bgs).most_common(1)[0][0]
    # verify: every changed cell is a bg cell whose enclosed region wall color
    # matches the output color there.
    saw_change = False
    for i, o in train:
        b = _bg_color(i)
        regions = _regions_and_wall_color(i, b)
        expect = i.copy()
        for cells, wc in regions:
            if wc is None:
                continue
            for y, x in cells:
                expect[y, x] = wc
        if not np.array_equal(expect, o):
            return None
        if (i != o).any():
            saw_change = True
    if not saw_change:
        return None

    def fn(g):
        b = _bg_color(g)
        out = g.copy()
        for cells, wc in _regions_and_wall_color(g, b):
            if wc is None:
                continue
            for y, x in cells:
                out[y, x] = wc
        return out

    return fn


# ---- 4. halo: bg cells adjacent to a target-color cell get recolored -----
def halo_around_color(train):
    """Learn a rule: background cells that touch (4- or 8-neighbour) a cell of a
    specific set of source colors are recolored to a learned color; all other
    cells unchanged. Source-color cells themselves stay put. This generalizes
    'draw a border/halo around every object of color X'."""
    if any(i.shape != o.shape for i, o in train):
        return None
    bgs = [_bg_color(i) for i, _ in train]
    bg = Counter(bgs).most_common(1)[0][0]

    # halo color = the (single) color that appears on changed bg cells
    halo_colors = set()
    for i, o in train:
        diff = i != o
        if diff.any():
            if not np.all(i[diff] == bg):
                return None
            halo_colors |= set(np.unique(o[diff]).tolist())
    if len(halo_colors) != 1:
        return None
    halo = int(next(iter(halo_colors)))

    for diag in (True, False):
        if diag:
            offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # determine which source colors trigger a halo: try "any non-bg,non-halo"
        # and, if that fails, learn the exact set from the first pair.
        def neighbors_of(g, b, srcset):
            H, W = g.shape
            trig = np.zeros((H, W), dtype=bool)
            src = np.isin(g, list(srcset))
            for dy, dx in offs:
                ys0 = max(0, -dy); ys1 = H - max(0, dy)
                xs0 = max(0, -dx); xs1 = W - max(0, dx)
                trig[ys0:ys1, xs0:xs1] |= src[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            return trig & (g == b)

        # candidate source sets
        all_nonbg = set()
        for i, _ in train:
            all_nonbg |= set(int(v) for v in np.unique(i) if int(v) != bg)
        cand_sets = [all_nonbg - {halo}]
        # Principled inference: a colour C is a "source" iff, in every train pair,
        # every background cell adjacent to a C-cell is halo in the output.
        learned = set()
        for cand in all_nonbg - {halo}:
            good = True
            for i, o in train:
                b = _bg_color(i)
                H, W = i.shape
                src = (i == cand)
                adj = np.zeros((H, W), dtype=bool)
                for dy, dx in offs:
                    ys0 = max(0, -dy); ys1 = H - max(0, dy)
                    xs0 = max(0, -dx); xs1 = W - max(0, dx)
                    adj[ys0:ys1, xs0:xs1] |= src[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
                bgadj = adj & (i == b)
                if bgadj.any() and not np.all(o[bgadj] == halo):
                    good = False
                    break
            if good:
                learned.add(cand)
        if learned:
            cand_sets.append(learned)

        for srcset in cand_sets:
            if not srcset:
                continue

            def fn(g, srcset=srcset):
                b = _bg_color(g)
                out = g.copy()
                trig = neighbors_of(g, b, srcset)
                out[trig] = halo
                return out

            try:
                if all(fn(i).shape == o.shape and np.array_equal(fn(i), o)
                       for i, o in train):
                    return fn
            except Exception:
                continue
    return None


# ---- 5. collinear gap fill: bg cell strictly between two same-colored cells --
def between_two_cells(train):
    """A background cell that lies on a straight horizontal or vertical segment
    strictly between two non-background cells of the SAME color, with only
    background in between, is recolored to a learned color. (Classic
    'connect two dots' with a fill color.)"""
    if any(i.shape != o.shape for i, o in train):
        return None
    bgs = [_bg_color(i) for i, _ in train]
    bg = Counter(bgs).most_common(1)[0][0]

    fill_colors = set()
    for i, o in train:
        diff = i != o
        if diff.any():
            if not np.all(i[diff] == bg):
                return None
            fill_colors |= set(np.unique(o[diff]).tolist())
    if len(fill_colors) != 1:
        return None
    fill = int(next(iter(fill_colors)))

    def between_mask(g, b, require_same, max_gap):
        H, W = g.shape
        mask = np.zeros((H, W), dtype=bool)
        # horizontal spans
        for r in range(H):
            c = 0
            while c < W:
                if g[r, c] != b:
                    # find next non-bg in this row
                    c2 = c + 1
                    while c2 < W and g[r, c2] == b:
                        c2 += 1
                    if c2 < W and c2 - c - 1 >= 1 and (max_gap is None or c2 - c - 1 <= max_gap):
                        if (not require_same) or g[r, c] == g[r, c2]:
                            mask[r, c + 1:c2] = True
                    c = c2
                else:
                    c += 1
        # vertical spans
        for cc in range(W):
            r = 0
            while r < H:
                if g[r, cc] != b:
                    r2 = r + 1
                    while r2 < H and g[r2, cc] == b:
                        r2 += 1
                    if r2 < H and r2 - r - 1 >= 1 and (max_gap is None or r2 - r - 1 <= max_gap):
                        if (not require_same) or g[r, cc] == g[r2, cc]:
                            mask[r + 1:r2, cc] = True
                    r = r2
                else:
                    r += 1
        return mask & (g == b)

    for require_same in (True, False):
        for max_gap in (1, None):
            def fn(g, require_same=require_same, max_gap=max_gap):
                b = _bg_color(g)
                out = g.copy()
                m = between_mask(g, b, require_same, max_gap)
                out[m] = fill
                return out
            try:
                if all(fn(i).shape == o.shape and np.array_equal(fn(i), o)
                       for i, o in train):
                    return fn
            except Exception:
                continue
    return None


DETECTORS = [
    enclosed_fill,
    enclosed_fill_wallcolor,
    neighbor_count_recolor,
    halo_around_color,
    between_two_cells,
]
