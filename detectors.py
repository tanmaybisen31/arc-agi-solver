"""Rule detectors for ARC-AGI.

Each detector: def det(train) -> transform_fn | None
  train = [(input_grid, output_grid), ...]  (numpy int arrays)
  transform_fn: grid -> grid   (must reproduce every train output; engine verifies)

Return None if the rule doesn't apply. Keep detectors defensive.
"""
import numpy as np
from collections import Counter

def bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])

# ---------- trivial ----------
def identity(train):
    return (lambda g: g.copy())

def constant_output(train):
    outs = [o for _, o in train]
    first = outs[0]
    if all(o.shape == first.shape and np.array_equal(o, first) for o in outs):
        const = first.copy()
        return (lambda g: const.copy())
    return None

# ---------- geometric ----------
_GEO = {
    "fliplr": np.fliplr, "flipud": np.flipud,
    "rot90": lambda g: np.rot90(g, 1), "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "transpose": lambda g: g.T.copy(),
    "antitranspose": lambda g: np.rot90(g, 2).T.copy(),
}
def geometric(train):
    for op in _GEO.values():
        if all(op(i).shape == o.shape and np.array_equal(op(i), o) for i, o in train):
            return (lambda g, op=op: op(g))
    return None

# ---------- color map ----------
def color_map(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    mapping = {}
    for i, o in train:
        for a, b in zip(i.flatten(), o.flatten()):
            a, b = int(a), int(b)
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    def fn(g):
        out = g.copy()
        for a, b in mapping.items():
            out[g == a] = b
        return out
    return fn

# ---------- tiling ----------
def tile(train):
    i0, o0 = train[0]
    if o0.shape[0] % i0.shape[0] or o0.shape[1] % i0.shape[1]:
        return None
    ah = o0.shape[0] // i0.shape[0]
    aw = o0.shape[1] // i0.shape[1]
    if ah * aw <= 1 or ah > 6 or aw > 6:
        return None
    # try plain tile and mirrored tile variants
    def make(flip_pattern):
        def fn(g):
            rows = []
            for r in range(ah):
                cols = []
                for c in range(aw):
                    blk = g.copy()
                    fh, fv = flip_pattern(r, c)
                    if fh: blk = np.fliplr(blk)
                    if fv: blk = np.flipud(blk)
                    cols.append(blk)
                rows.append(np.hstack(cols))
            return np.vstack(rows)
        return fn
    patterns = [
        lambda r, c: (False, False),                 # plain
        lambda r, c: (c % 2 == 1, r % 2 == 1),       # mirror both
        lambda r, c: (c % 2 == 1, False),            # mirror horiz
        lambda r, c: (False, r % 2 == 1),            # mirror vert
    ]
    for p in patterns:
        fn = make(p)
        if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
            return fn
    return None

# ---------- scaling ----------
def scale_up(train):
    i0, o0 = train[0]
    if i0.shape[0] == 0 or i0.shape[1] == 0:
        return None
    if o0.shape[0] % i0.shape[0] or o0.shape[1] % i0.shape[1]:
        return None
    kh = o0.shape[0] // i0.shape[0]
    kw = o0.shape[1] // i0.shape[1]
    if kh * kw <= 1 or kh > 10 or kw > 10:
        return None
    def fn(g):
        return np.kron(g, np.ones((kh, kw), dtype=int))
    return fn

def scale_down(train):
    i0, o0 = train[0]
    if o0.shape[0] == 0 or o0.shape[1] == 0:
        return None
    if i0.shape[0] % o0.shape[0] or i0.shape[1] % o0.shape[1]:
        return None
    kh = i0.shape[0] // o0.shape[0]
    kw = i0.shape[1] // o0.shape[1]
    if kh * kw <= 1:
        return None
    def fn(g):
        H, W = g.shape[0] // kh, g.shape[1] // kw
        out = np.zeros((H, W), dtype=int)
        for r in range(H):
            for c in range(W):
                blk = g[r*kh:(r+1)*kh, c*kw:(c+1)*kw]
                vals, cnts = np.unique(blk, return_counts=True)
                out[r, c] = int(vals[np.argmax(cnts)])
        return out
    return fn

# ---------- crop to content ----------
def crop_bbox(train):
    def fn(g):
        bg = bg_color(g)
        mask = g != bg
        if not mask.any():
            return g.copy()
        rs, cs = np.where(mask)
        return g[rs.min():rs.max()+1, cs.min():cs.max()+1].copy()
    return fn

# ---------- border ----------
def border_add(train):
    i0, o0 = train[0]
    if o0.shape[0] != i0.shape[0] + 2 or o0.shape[1] != i0.shape[1] + 2:
        return None
    bc = int(o0[0, 0])
    def fn(g):
        out = np.full((g.shape[0]+2, g.shape[1]+2), bc, dtype=int)
        out[1:-1, 1:-1] = g
        return out
    return fn

# ---------- panel logical ops (left/right or top/bottom) ----------
def _split_two(g):
    """Return (A, B, axis) if g splits into two equal halves, optionally via a
    single separator line of a uniform non-background color."""
    H, W = g.shape
    outs = []
    # vertical split (left/right)
    if W % 2 == 0:
        outs.append((g[:, :W//2], g[:, W//2:], "v"))
    if W % 2 == 1:
        mid = W // 2
        col = g[:, mid]
        if len(set(col.tolist())) == 1:
            outs.append((g[:, :mid], g[:, mid+1:], "v"))
    # horizontal split (top/bottom)
    if H % 2 == 0:
        outs.append((g[:H//2, :], g[H//2:, :], "h"))
    if H % 2 == 1:
        mid = H // 2
        row = g[mid, :]
        if len(set(row.tolist())) == 1:
            outs.append((g[:mid, :], g[mid+1:, :], "h"))
    return outs

def panel_logical(train):
    i0, o0 = train[0]
    # find a split of i0 whose halves match o0 shape
    def get_split(g):
        for A, B, ax in _split_two(g):
            if A.shape == o0.shape and B.shape == o0.shape:
                return A, B
        return None
    if get_split(i0) is None:
        return None
    out_colors = set()
    for _, o in train:
        out_colors |= set(np.unique(o).tolist())
    # learn boolean op + on/off colors
    ops = {
        "and": lambda a, b: a & b,
        "or":  lambda a, b: a | b,
        "xor": lambda a, b: a ^ b,
        "diff": lambda a, b: a & (~b),
        "nand": lambda a, b: ~(a & b),
        "nor": lambda a, b: ~(a | b),
        "xnor": lambda a, b: ~(a ^ b),
    }
    for opname, op in ops.items():
        for on in sorted(out_colors):
            for off in sorted(out_colors):
                if on == off:
                    continue
                def fn(g, op=op, on=on, off=off):
                    s = get_split(g)
                    if s is None:
                        return None
                    A, B = s
                    ba = A != bg_color(A)
                    bb = B != bg_color(B)
                    res = op(ba, bb)
                    out = np.where(res, on, off).astype(int)
                    return out
                try:
                    if all(fn(i) is not None and fn(i).shape == o.shape and np.array_equal(fn(i), o)
                           for i, o in train):
                        return fn
                except Exception:
                    continue
    return None

# ---------- single-cell summaries ----------
def most_common_cell(train):
    if not all(o.shape == (1, 1) for _, o in train):
        return None
    def fn(g):
        vals, cnts = np.unique(g, return_counts=True)
        return np.array([[int(vals[np.argmax(cnts)])]], dtype=int)
    return fn

def least_common_cell(train):
    if not all(o.shape == (1, 1) for _, o in train):
        return None
    def fn(g):
        vals, cnts = np.unique(g, return_counts=True)
        return np.array([[int(vals[np.argmin(cnts)])]], dtype=int)
    return fn

# ---------- connected components ----------
def _components(g, bg, diag=False):
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1)]
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
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps

def keep_largest_object(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    def fn(g):
        bg = bg_color(g)
        comps = _components(g, bg)
        if not comps:
            return g.copy()
        big = max(comps, key=len)
        out = np.full_like(g, bg)
        for y, x in big:
            out[y, x] = g[y, x]
        return out
    return fn

def keep_smallest_object(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    def fn(g):
        bg = bg_color(g)
        comps = _components(g, bg)
        if not comps:
            return g.copy()
        small = min(comps, key=len)
        out = np.full_like(g, bg)
        for y, x in small:
            out[y, x] = g[y, x]
        return out
    return fn

# ---------- gravity ----------
def _gravity(g, direction):
    bg = bg_color(g)
    out = np.full_like(g, bg)
    H, W = g.shape
    if direction in ("down", "up"):
        for c in range(W):
            col = [g[r, c] for r in range(H) if g[r, c] != bg]
            if direction == "down":
                for k, v in enumerate(col):
                    out[H-len(col)+k, c] = v
            else:
                for k, v in enumerate(col):
                    out[k, c] = v
    else:
        for r in range(H):
            row = [g[r, c] for c in range(W) if g[r, c] != bg]
            if direction == "right":
                for k, v in enumerate(row):
                    out[r, W-len(row)+k] = v
            else:
                for k, v in enumerate(row):
                    out[r, k] = v
    return out

def gravity(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    for d in ("down", "up", "left", "right"):
        if all(np.array_equal(_gravity(i, d), o) for i, o in train):
            return (lambda g, d=d: _gravity(g, d))
    return None

# Ordered by specificity/reliability (earlier = higher priority in voting ties)
DETECTORS = [
    identity, geometric, color_map, constant_output,
    tile, scale_up, scale_down, crop_bbox, border_add,
    panel_logical, most_common_cell, least_common_cell,
    keep_largest_object, keep_smallest_object, gravity,
]
