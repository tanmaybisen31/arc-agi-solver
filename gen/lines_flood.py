"""Detectors for the LINES / RAYS / FLOOD-FILL family.

Patterns covered:
  * flood-fill enclosed background regions (not touching border) with a learned color
  * connect pairs of collinear same-colored markers with a learned line color
  * fill single-cell gaps between two collinear same-colored markers
  * cast straight rays from marker cells in learned direction(s) until edge / obstacle
  * fill the rectangular gap between two aligned solid blocks

All detectors are defensive and only ever help: the engine re-verifies that the
returned transform reproduces every training output exactly before it is used.
"""
import numpy as np
from collections import deque
from itertools import combinations


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


# ----------------------------------------------------------------------------
# Flood-fill enclosed background regions
# ----------------------------------------------------------------------------
def _enclosed_mask(g, bg):
    """bg cells NOT reachable from the border through 4-connected bg cells."""
    H, W = g.shape
    reach = np.zeros((H, W), bool)
    dq = deque()
    for r in range(H):
        for c in (0, W - 1):
            if g[r, c] == bg and not reach[r, c]:
                reach[r, c] = True
                dq.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if g[r, c] == bg and not reach[r, c]:
                reach[r, c] = True
                dq.append((r, c))
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and g[ny, nx] == bg and not reach[ny, nx]:
                reach[ny, nx] = True
                dq.append((ny, nx))
    return (g == bg) & (~reach)


def flood_fill_enclosed(train):
    try:
        fills = set()
        for i, o in train:
            if i.shape != o.shape:
                return None
            bg = _bg(i)
            em = _enclosed_mask(i, bg)
            diff = (i != o)
            if not diff.any():
                return None
            # every changed cell must be an enclosed bg cell, and every enclosed
            # cell must be changed
            if not np.all(em[diff]):
                return None
            if not np.all(diff[em]):
                return None
            fills |= set(o[em].tolist())
        if len(fills) != 1:
            return None
        fill = int(next(iter(fills)))

        def fn(g):
            bg = _bg(g)
            em = _enclosed_mask(g, bg)
            out = g.copy()
            out[em] = fill
            return out
        return fn
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Connect collinear same-colored markers with a line color
# ----------------------------------------------------------------------------
def _connect(g, bg, lc):
    """Connect adjacent same-color collinear markers. lc=None => draw the line in
    the pair's own marker color; otherwise use the fixed color lc."""
    out = g.copy()
    H, W = g.shape
    for r in range(H):
        cols = [c for c in range(W) if g[r, c] != bg]
        for a, b in zip(cols, cols[1:]):
            if g[r, a] == g[r, b] and all(g[r, c] == bg for c in range(a + 1, b)):
                col = int(g[r, a]) if lc is None else lc
                for c in range(a + 1, b):
                    out[r, c] = col
    for c in range(W):
        rows = [r for r in range(H) if g[r, c] != bg]
        for a, b in zip(rows, rows[1:]):
            if g[a, c] == g[b, c] and all(g[r, c] == bg for r in range(a + 1, b)):
                col = int(g[a, c]) if lc is None else lc
                for r in range(a + 1, b):
                    out[r, c] = col
    return out


def connect_collinear(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        if not any((i != o).any() for i, o in train):
            return None
        lcs = set()
        for i, o in train:
            diff = (i != o)
            if diff.any():
                lcs |= set(o[diff].tolist())
        # candidate line colors: each learned added color, plus None (self-color)
        candidates = [int(x) for x in sorted(lcs)] + [None]
        for lc in candidates:
            if all(np.array_equal(_connect(i, _bg(i), lc), o) for i, o in train):
                return (lambda g, lc=lc: _connect(g, _bg(g), lc))
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Fill single-cell gaps between two collinear same-colored markers
# ----------------------------------------------------------------------------
def _gap1(g, bg, lc):
    """Fill single-cell gaps between two same-color markers exactly 2 apart.

    To avoid spurious cross-axis fills (a marker that belongs to a horizontal
    pair should not also be joined vertically to an unrelated marker), a vertical
    fill is only made when neither endpoint has a horizontal partner.
    """
    H, W = g.shape
    out = g.copy()

    def has_h(r, c):
        return ((c + 2 < W and g[r, c + 1] == bg and g[r, c + 2] == g[r, c]) or
                (c - 2 >= 0 and g[r, c - 1] == bg and g[r, c - 2] == g[r, c]))

    for r in range(H):
        for c in range(W):
            if g[r, c] == bg:
                continue
            if c + 2 < W and g[r, c + 1] == bg and g[r, c + 2] == g[r, c]:
                out[r, c + 1] = lc
            if r + 2 < H and g[r + 1, c] == bg and g[r + 2, c] == g[r, c]:
                if not has_h(r, c) and not has_h(r + 2, c):
                    out[r + 1, c] = lc
    return out


def fill_unit_gap(train):
    try:
        lcs = set()
        for i, o in train:
            if i.shape != o.shape:
                return None
            diff = (i != o)
            if diff.any():
                lcs |= set(o[diff].tolist())
        if not lcs:
            return None
        for lc in sorted(lcs):
            lc = int(lc)
            if all(np.array_equal(_gap1(i, _bg(i), lc), o) for i, o in train):
                return (lambda g, lc=lc: _gap1(g, _bg(g), lc))
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Cast straight rays from marker cells until edge / obstacle
# ----------------------------------------------------------------------------
_DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


def _rays(g, bg, dirs):
    out = g.copy()
    H, W = g.shape
    src = [(r, c) for r in range(H) for c in range(W) if g[r, c] != bg]
    for (r, c) in src:
        col = g[r, c]
        for d in dirs:
            dy, dx = _DIRS[d]
            y, x = r + dy, c + dx
            while 0 <= y < H and 0 <= x < W and g[y, x] == bg:
                out[y, x] = col
                y += dy
                x += dx
    return out


def cast_rays(train):
    try:
        alld = list(_DIRS)
        subsets = []
        for k in range(1, 5):
            for combo in combinations(alld, k):
                subsets.append(combo)
        for dirs in subsets:
            if all(i.shape == o.shape and np.array_equal(_rays(i, _bg(i), dirs), o)
                   for i, o in train):
                return (lambda g, dirs=dirs: _rays(g, _bg(g), dirs))
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Intersection grid: edge-row markers give columns, edge-col markers give rows;
# mark every (row, col) crossing with a learned fill color.
# ----------------------------------------------------------------------------
def _grid_fill(g, bg, fill):
    H, W = g.shape
    ys, xs = np.where(g != bg)
    if len(ys) == 0:
        return g.copy()
    cols, rows = set(), set()
    for y, x in zip(ys, xs):
        if y == 0 or y == H - 1:
            cols.add(int(x))
        if x == 0 or x == W - 1:
            rows.add(int(y))
    out = g.copy()
    for r in rows:
        for c in cols:
            if out[r, c] == bg:
                out[r, c] = fill
    return out


def intersection_grid(train):
    try:
        fills = set()
        for i, o in train:
            if i.shape != o.shape:
                return None
            diff = (i != o)
            if diff.any():
                fills |= set(o[diff].tolist())
        if not fills:
            return None
        for f in sorted(fills):
            f = int(f)
            if all(np.array_equal(_grid_fill(i, _bg(i), f), o) for i, o in train):
                return (lambda g, f=f: _grid_fill(g, _bg(g), f))
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Fill the interior of every solid rectangle with a learned color
# ----------------------------------------------------------------------------
def _same_color_components(g, bg):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = g[r, c]
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            out.append(cells)
    return out


def _fill_interior(g, bg, fill):
    out = g.copy()
    for cells in _same_color_components(g, bg):
        ys = [y for y, x in cells]
        xs = [x for y, x in cells]
        r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(cells):
            continue  # not a solid rectangle
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue  # no interior
        out[r0 + 1:r1, c0 + 1:c1] = fill
    return out


def fill_rectangle_interior(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        fills = set()
        for i, o in train:
            diff = (i != o)
            if diff.any():
                fills |= set(o[diff].tolist())
        if not fills:
            return None
        for f in sorted(fills):
            f = int(f)
            if all(np.array_equal(_fill_interior(i, _bg(i), f), o) for i, o in train):
                return (lambda g, f=f: _fill_interior(g, _bg(g), f))
        return None
    except Exception:
        return None


DETECTORS = [
    flood_fill_enclosed,
    connect_collinear,
    fill_unit_gap,
    cast_rays,
    intersection_grid,
    fill_rectangle_interior,
]
