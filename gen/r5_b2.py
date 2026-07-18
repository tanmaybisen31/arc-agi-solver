"""Round 5, batch 2 detectors for ARC-AGI.

Each detector: def det(train) -> transform_fn | None
  train = [(input_grid, output_grid), ...]  (numpy int arrays)
  transform_fn: grid -> grid   (engine verifies exact reproduction of all demos)

Keep detectors defensive & general. numpy + stdlib only.
"""
import numpy as np
from collections import Counter


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag=False):
    """Connected components of non-bg cells (any non-bg color mixes)."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    nb = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
          if diag else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in nb:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            out.append(cells)
    return out


def _mono_components(g, diag=False):
    """Connected components split per single color; returns (color, cells)."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    nb = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
          if diag else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == 0:
                continue
            col = g[r, c]
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in nb:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            out.append((int(col), cells))
    return out


def _norm_shape(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    r0, c0 = min(rs), min(cs)
    return frozenset((r - r0, c - c0) for r, c in cells)


# ---------------------------------------------------------------------------
# 1) f3e62deb : a single hollow shape slides to a grid edge; the direction is
#    determined by the shape's color (learned color->direction from demos).
# ---------------------------------------------------------------------------
def move_shape_by_color(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # learn color -> direction
    color_dir = {}
    for i, o in train:
        ci = np.argwhere(i != 0)
        co = np.argwhere(o != 0)
        if len(ci) == 0 or len(co) == 0:
            return None
        # single object of single color
        cols_i = set(int(v) for v in i[i != 0])
        cols_o = set(int(v) for v in o[o != 0])
        if len(cols_i) != 1 or cols_i != cols_o:
            return None
        col = next(iter(cols_i))
        ir0, ic0 = ci.min(0)
        ir1, ic1 = ci.max(0)
        or0, oc0 = co.min(0)
        or1, oc1 = co.max(0)
        # shape must be preserved (same bbox size + same relative pattern)
        if (ir1 - ir0, ic1 - ic0) != (or1 - or0, oc1 - oc0):
            return None
        dr = or0 - ir0
        dc = oc0 - ic0
        # exactly one axis of movement, sliding to an edge
        if dr != 0 and dc != 0:
            return None
        H, W = i.shape
        if dc == 0 and dr == 0:
            direction = "none"
        elif dc == 0:
            if dr < 0 and or0 == 0:
                direction = "up"
            elif dr > 0 and or1 == H - 1:
                direction = "down"
            else:
                return None
        else:
            if dc < 0 and oc0 == 0:
                direction = "left"
            elif dc > 0 and oc1 == W - 1:
                direction = "right"
            else:
                return None
        if col in color_dir and color_dir[col] != direction:
            return None
        color_dir[col] = direction

    def fn(g):
        out = np.zeros_like(g)
        cols = set(int(v) for v in g[g != 0])
        if len(cols) != 1:
            return None
        col = next(iter(cols))
        if col not in color_dir:
            return None
        d = color_dir[col]
        cc = np.argwhere(g != 0)
        r0, c0 = cc.min(0)
        r1, c1 = cc.max(0)
        H, W = g.shape
        nr0, nc0 = r0, c0
        if d == "up":
            nr0 = 0
        elif d == "down":
            nr0 = H - 1 - (r1 - r0)
        elif d == "left":
            nc0 = 0
        elif d == "right":
            nc0 = W - 1 - (c1 - c0)
        for (r, c) in cc:
            out[nr0 + (r - r0), nc0 + (c - c0)] = g[r, c]
        return out

    try:
        if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 2) 8ee62060 : mirror each object horizontally about the grid center (keep the
#    object's own shape/orientation & rows, only mirror its column position).
#    Also try vertical-position mirror.
# ---------------------------------------------------------------------------
def mirror_object_positions(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def make(axis):
        def fn(g):
            H, W = g.shape
            out = np.zeros_like(g)
            for cells in _components(g, 0, diag=False):
                rs = [r for r, c in cells]
                cs = [c for r, c in cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                for (r, c) in cells:
                    if axis == "col":
                        nr, nc = r, (W - 1 - c1) + (c - c0)
                    else:
                        nr, nc = (H - 1 - r1) + (r - r0), c
                    out[nr, nc] = g[r, c]
            return out
        return fn

    for axis in ("col", "row"):
        fn = make(axis)
        try:
            if all(np.array_equal(fn(i), o) for i, o in train):
                return fn
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# 3) e7dd8335 : recolor the bottom half of each object from color A to color B.
#    Learn (A -> B) and the split fraction (top/bottom or left/right).
# ---------------------------------------------------------------------------
def recolor_object_half(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # find single (from, to) recolor pair
    froms, tos = set(), set()
    for i, o in train:
        d = np.argwhere(i != o)
        for r, c in d:
            froms.add(int(i[r, c]))
            tos.add(int(o[r, c]))
    if len(froms) != 1 or len(tos) != 1:
        return None
    A = next(iter(froms))
    B = next(iter(tos))
    if A == B:
        return None

    def bboxes(g, scope):
        """Yield (r0,r1,c0,c1) regions of color A: 'global' = one bbox over all
        A-cells; 'comp' = per 4-connected component of A."""
        H, W = g.shape
        if scope == "global":
            cc = np.argwhere(g == A)
            if len(cc) == 0:
                return
            yield int(cc[:, 0].min()), int(cc[:, 0].max()), int(cc[:, 1].min()), int(cc[:, 1].max())
            return
        seen = np.zeros((H, W), bool)
        for r in range(H):
            for c in range(W):
                if seen[r, c] or g[r, c] != A:
                    continue
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == A:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                rs = [y for y, x in cells]
                cs = [x for y, x in cells]
                yield min(rs), max(rs), min(cs), max(cs)

    def make(half, scope):
        def fn(g):
            out = g.copy()
            for (r0, r1, c0, c1) in bboxes(g, scope):
                for (y, x) in np.argwhere(g == A):
                    if not (r0 <= y <= r1 and c0 <= x <= c1):
                        continue
                    if half == "bottom" and y > (r0 + r1) / 2.0:
                        out[y, x] = B
                    elif half == "top" and y < (r0 + r1) / 2.0:
                        out[y, x] = B
                    elif half == "right" and x > (c0 + c1) / 2.0:
                        out[y, x] = B
                    elif half == "left" and x < (c0 + c1) / 2.0:
                        out[y, x] = B
            return out
        return fn

    for scope in ("global", "comp"):
        for half in ("bottom", "top", "right", "left"):
            fn = make(half, scope)
            try:
                if all(np.array_equal(fn(i), o) for i, o in train):
                    return fn
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# 4) 72207abc : a short "seed" of colored cells sits against the left wall of a
#    single non-bg row; it is echoed rightward with gaps that grow by one each
#    step, values cycling through the seed sequence.
# ---------------------------------------------------------------------------
def echo_growing_gap(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def get_row(g):
        nz = [r for r in range(g.shape[0]) if np.any(g[r] != 0)]
        if len(nz) != 1:
            return None
        return nz[0]

    # verify all demos are single-row structures
    for i, o in train:
        if get_row(i) is None or get_row(o) is None:
            return None

    def fn(g):
        row = get_row(g)
        if row is None:
            return None
        r = g[row]
        W = len(r)
        seed_cols = [c for c in range(W) if r[c] != 0]
        if not seed_cols:
            return None
        # seed must begin at the left wall
        if seed_cols[0] != 0:
            return None
        seed_vals = [int(r[c]) for c in seed_cols]
        n = len(seed_vals)
        out = np.zeros_like(g)
        # place values cyclically starting at col 0 with gaps 1,2,3,...
        pos = 0
        gap = 1
        k = 0
        while pos < W:
            out[row, pos] = seed_vals[k % n]
            k += 1
            pos = pos + gap
            gap += 1
        return out

    try:
        if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 5) 2a5f8217 : "shape dictionary" recolor. Template objects have distinctive
#    single colors; copy objects (a fixed marker color, usually the most common
#    non-bg copy color) share a template's shape and get recolored to it.
# ---------------------------------------------------------------------------
def shape_dict_recolor(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # determine the copy color: the single color whose cells change
    froms = set()
    for i, o in train:
        for r, c in np.argwhere(i != o):
            froms.add(int(i[r, c]))
    if len(froms) != 1:
        return None
    copy_color = next(iter(froms))

    def fn(g):
        out = g.copy()
        comps = _mono_components(g, diag=False)
        templ = {}
        for col, cells in comps:
            if col != copy_color:
                sh = _norm_shape(cells)
                templ[sh] = col
        for col, cells in comps:
            if col == copy_color:
                sh = _norm_shape(cells)
                if sh in templ:
                    for r, c in cells:
                        out[r, c] = templ[sh]
        return out

    try:
        if all(np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 6) 3ee1011a : each color appears a distinct number of times; output is a set
#    of concentric square rings, the most-frequent color forming the outermost
#    ring and each next-frequent color nested one step inside.
# ---------------------------------------------------------------------------
def concentric_count_rings(train):
    def transform(g):
        vals, cnts = np.unique(g[g != 0], return_counts=True)
        cc = {int(v): int(n) for v, n in zip(vals, cnts)}
        if len(cc) < 2:
            return None
        counts = sorted(cc.values(), reverse=True)
        # counts must be strictly decreasing (unique) so ring order is unambiguous
        if len(set(counts)) != len(counts):
            return None
        N = counts[0]
        order = sorted(cc.items(), key=lambda kv: -kv[1])
        if len(order) - 1 > (N - 1) // 2 + 1:
            return None
        out = np.zeros((N, N), dtype=int)
        for rank, (col, _cnt) in enumerate(order):
            lo, hi = rank, N - 1 - rank
            if lo > hi:
                return None
            out[lo:hi + 1, lo:hi + 1] = col
        return out

    try:
        if all(transform(i) is not None and transform(i).shape == o.shape
               and np.array_equal(transform(i), o) for i, o in train):
            return transform
    except Exception:
        return None
    return None


DETECTORS = [
    move_shape_by_color,
    mirror_object_positions,
    recolor_object_half,
    echo_growing_gap,
    shape_dict_recolor,
    concentric_count_rings,
]
