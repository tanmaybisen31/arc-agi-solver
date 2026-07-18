"""Round 5, batch 3 detectors.

Each detector: det(train) -> transform_fn | None
Engine verifies exact reproduction of every train pair before use.
numpy + stdlib only.  Keep everything defensive.
"""
import numpy as np
from collections import deque, Counter


# ---------------------------------------------------------------- helpers
def _bg(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[np.argmax(c)])


def _comps(mask, diag=True):
    """Connected components over a boolean mask; returns list of cell lists."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in nbrs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                out.append(cells)
    return out


def _bbox(cells):
    rs = [c[0] for c in cells]
    cs = [c[1] for c in cells]
    return min(rs), min(cs), max(rs), max(cs)


# ================================================================
# 575b1a71: recolor each background pixel by the rank of its column
# among the set of columns that contain any background pixel.
# Background is the majority color; the "holes" are a second color.
# ================================================================
def recolor_by_column_rank(train):
    # every pair same shape, exactly two colors in input (bg + hole)
    for i, o in train:
        if i.shape != o.shape:
            return None
    i0, o0 = train[0]
    in_cols = set()
    for i, o in train:
        in_cols |= set(np.unique(i).tolist())
    if len(in_cols) != 2:
        return None
    # determine which of the two colors is the hole (the one recolored)
    # hole = the color present in input but whose cells change in output
    # bg is majority.
    bg = _bg(i0)
    hole_candidates = in_cols - {bg}
    if len(hole_candidates) != 1:
        return None
    hole = next(iter(hole_candidates))

    def make(axis):
        def fn(g):
            out = g.copy()
            if axis == "col":
                cols = sorted({c for r in range(g.shape[0]) for c in range(g.shape[1])
                               if g[r, c] == hole})
                rank = {c: k + 1 for k, c in enumerate(cols)}
                for r in range(g.shape[0]):
                    for c in range(g.shape[1]):
                        if g[r, c] == hole:
                            out[r, c] = rank[c]
            else:
                rows = sorted({r for r in range(g.shape[0]) for c in range(g.shape[1])
                               if g[r, c] == hole})
                rank = {r: k + 1 for k, r in enumerate(rows)}
                for r in range(g.shape[0]):
                    for c in range(g.shape[1]):
                        if g[r, c] == hole:
                            out[r, c] = rank[r]
            return out
        return fn

    for axis in ("col", "row"):
        fn = make(axis)
        try:
            if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
                return fn
        except Exception:
            pass
    return None


# ================================================================
# e88171ec: find the largest solid axis-aligned rectangle of the
# background color, fill its interior (shrunk by 1 on each side)
# with a fixed fill color.
# ================================================================
def _largest_rect_of(g, val):
    H, W = g.shape
    best = (0, None)
    heights = [0] * W
    for r in range(H):
        for c in range(W):
            heights[c] = heights[c] + 1 if g[r, c] == val else 0
        stack = []
        for c in range(W + 1):
            cur = heights[c] if c < W else 0
            start = c
            while stack and stack[-1][1] > cur:
                s, h = stack.pop()
                area = h * (c - s)
                if area > best[0]:
                    best = (area, (r - h + 1, s, r, c - 1))
                start = s
            stack.append((start, cur))
    return best


def fill_largest_bg_rect_interior(train):
    for i, o in train:
        if i.shape != o.shape:
            return None
    # learn fill color + the color that forms the rectangle (rect_color).
    # The rectangle color is the single color that all changed cells shared
    # in the input.
    fill = None
    rect_color = None
    for i, o in train:
        diff = (i != o)
        if not diff.any():
            return None
        vals = set(np.unique(o[diff]).tolist())
        if len(vals) != 1:
            return None
        v = next(iter(vals))
        if fill is None:
            fill = v
        elif fill != v:
            return None
        srcvals = set(np.unique(i[diff]).tolist())
        if len(srcvals) != 1:
            return None
        rc = next(iter(srcvals))
        if rect_color is None:
            rect_color = rc
        elif rect_color != rc:
            return None
    if fill is None or rect_color is None:
        return None

    def fn(g):
        area, box = _largest_rect_of(g, rect_color)
        out = g.copy()
        if box is None:
            return out
        r0, c0, r1, c1 = box
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            out[r0 + 1:r1, c0 + 1:c1] = fill
        return out
    return fn


# ================================================================
# 9f27f097: an empty solid rectangle of 0s and a decorated block of the
# same size elsewhere.  Copy the decorated block into the empty
# rectangle, applying a single geometric transform learned from all
# training pairs.  Background is majority color; 0 marks the empty slot.
# ================================================================
_GEO = {
    "id": lambda g: g,
    "fliplr": np.fliplr,
    "flipud": np.flipud,
    "rot90": lambda g: np.rot90(g, 1),
    "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "T": lambda g: g.T,
    "aT": lambda g: np.rot90(g, 2).T,
}


def _find_empty_rect(g, empty=0):
    area, box = _largest_rect_of(g, empty)
    if box is None or area < 4:
        return None
    return box


def copy_into_empty_rect(train):
    for i, o in train:
        if i.shape != o.shape:
            return None
    # 0 must be the empty marker distinct from bg
    for i, o in train:
        if _bg(i) == 0:
            return None

    def get_blocks(g):
        bg = _bg(g)
        box = _find_empty_rect(g, 0)
        if box is None:
            return None
        r0, c0, r1, c1 = box
        h, w = r1 - r0 + 1, c1 - c0 + 1
        # source: decorated cells (not bg, not 0) bounding box, excluding empty rect
        mask = (g != bg) & (g != 0)
        mask[r0:r1 + 1, c0:c1 + 1] = False
        if not mask.any():
            return None
        sr0, sc0, sr1, sc1 = _bbox(list(zip(*np.where(mask))))
        src = g[sr0:sr1 + 1, sc0:sc1 + 1]
        return box, src

    # learn transform consistent across all pairs
    for name, f in _GEO.items():
        ok = True
        for i, o in train:
            gb = get_blocks(i)
            if gb is None:
                ok = False
                break
            (r0, c0, r1, c1), src = gb
            h, w = r1 - r0 + 1, c1 - c0 + 1
            try:
                ts = f(src)
            except Exception:
                ok = False
                break
            if ts.shape != (h, w):
                ok = False
                break
            got = o.copy()
            if not np.array_equal(o[r0:r1 + 1, c0:c1 + 1], ts):
                ok = False
                break
            # everything outside the rect must be unchanged
            tmp = i.copy()
            tmp[r0:r1 + 1, c0:c1 + 1] = ts
            if not np.array_equal(tmp, o):
                ok = False
                break
        if ok:
            def fn(g, f=f):
                gb = get_blocks(g)
                if gb is None:
                    return g.copy()
                (r0, c0, r1, c1), src = gb
                h, w = r1 - r0 + 1, c1 - c0 + 1
                ts = f(src)
                if ts.shape != (h, w):
                    return g.copy()
                out = g.copy()
                out[r0:r1 + 1, c0:c1 + 1] = ts
                return out
            return fn
    return None


# ================================================================
# 2b01abd0: a full straight line divides the grid; a two-color shape on
# one side.  Output: swap the two shape colors on the original side, and
# place a mirror reflection of the ORIGINAL shape (original colors) on
# the opposite side of the line.
# ================================================================
def mirror_across_line_swap(train):
    for i, o in train:
        if i.shape != o.shape:
            return None

    def find_line(g):
        H, W = g.shape
        # full row of single non-bg color
        bg = _bg(g)
        for r in range(H):
            row = g[r]
            if row[0] != bg and np.all(row == row[0]):
                return ("row", r, int(row[0]))
        for c in range(W):
            col = g[:, c]
            if col[0] != bg and np.all(col == col[0]):
                return ("col", c, int(col[0]))
        return None

    # each example must have a line and exactly two shape colors; the swap
    # (the two shape colors exchanged) is computed per-grid.
    for i, o in train:
        line = find_line(i)
        if line is None:
            return None
        bg = _bg(i)
        lc = line[2]
        shape_cols = set(np.unique(i[(i != bg) & (i != lc)]).tolist())
        if len(shape_cols) != 2:
            return None

    def fn(g):
        bg = _bg(g)
        line = find_line(g)
        if line is None:
            return g.copy()
        axis, idx, lc = line
        out = g.copy()
        H, W = g.shape
        shape_mask = (g != bg) & (g != lc)
        shape_cols = sorted(set(np.unique(g[shape_mask]).tolist()))
        if len(shape_cols) != 2:
            return g.copy()
        a, b = shape_cols
        swap = {a: b, b: a}
        # recolor original side (swap)
        for r in range(H):
            for c in range(W):
                if shape_mask[r, c] and g[r, c] in swap:
                    out[r, c] = swap[g[r, c]]
        # mirror original shape across the line
        for r in range(H):
            for c in range(W):
                if shape_mask[r, c]:
                    if axis == "row":
                        mr = 2 * idx - r
                        mc = c
                    else:
                        mr = r
                        mc = 2 * idx - c
                    if 0 <= mr < H and 0 <= mc < W:
                        out[mr, mc] = g[r, c]
        return out

    return fn


# ================================================================
# 604001fa: objects = 8-connected non-bg blobs, each containing an
# L-tromino "key" of one distinct color plus a body of another color.
# The key's orientation (which corner of its 2x2 bbox is missing)
# selects a recolor for the body; the key is removed.
# ================================================================
def _l_tromino_corner(cells):
    """cells: list of (r,c) forming an L-tromino. Returns which corner
    of the 2x2 bounding box is EMPTY, as one of 'TL','TR','BL','BR', or
    None if not a 3-cell L in a 2x2 box."""
    if len(cells) != 3:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    present = {(r - r0, c - c0) for r, c in cells}
    allc = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing = allc - present
    if len(missing) != 1:
        return None
    m = next(iter(missing))
    return {(0, 0): "TL", (0, 1): "TR", (1, 0): "BL", (1, 1): "BR"}[m]


def key_orientation_recolor(train):
    for i, o in train:
        if i.shape != o.shape:
            return None
    # identify the key color: the color that becomes 0 in output on the
    # object; the body color(s) get recolored.
    # Learn mapping corner -> output color.  Also determine key color.
    corner_map = {}
    key_color = None
    for i, o in train:
        bg = _bg(i)
        objs = _comps(i != bg, diag=True)
        for cells in objs:
            colors = Counter(int(i[r, c]) for r, c in cells)
            if len(colors) != 2:
                return None
            # the key is the color forming an L-tromino of size 3
            found = None
            for col in colors:
                col_cells = [(r, c) for r, c in cells if i[r, c] == col]
                corner = _l_tromino_corner(col_cells)
                if corner is not None and len(col_cells) == 3:
                    found = (col, corner, col_cells)
                    break
            if found is None:
                return None
            kc, corner, kcells = found
            if key_color is None:
                key_color = kc
            elif key_color != kc:
                # different key color per task is fine only if consistent
                return None
            body_cells = [(r, c) for r, c in cells if i[r, c] != kc]
            oc = set(int(o[r, c]) for r, c in body_cells)
            if len(oc) != 1:
                return None
            ocol = next(iter(oc))
            if corner in corner_map and corner_map[corner] != ocol:
                return None
            corner_map[corner] = ocol
            # key cells must become bg in output
            for r, c in kcells:
                if o[r, c] != bg:
                    return None
    if key_color is None or not corner_map:
        return None

    def fn(g):
        bg = _bg(g)
        out = g.copy()
        objs = _comps(g != bg, diag=True)
        for cells in objs:
            kcells = [(r, c) for r, c in cells if g[r, c] == key_color]
            if len(kcells) != 3:
                continue
            corner = _l_tromino_corner(kcells)
            if corner is None or corner not in corner_map:
                continue
            ocol = corner_map[corner]
            for r, c in kcells:
                out[r, c] = bg
            for r, c in cells:
                if g[r, c] != key_color:
                    out[r, c] = ocol
        return out

    return fn


# ================================================================
# c97c0139: each maximal straight segment (H or V) of a marker color
# grows two perpendicular isoceles triangles of a fill color (one on
# each side), tapering from the segment toward an apex.
# ================================================================
def perpendicular_triangles(train):
    for i, o in train:
        if i.shape != o.shape:
            return None
    # marker color: appears in input, unchanged; fill color: new in output
    fill = None
    marker = None
    for i, o in train:
        diff = (i != o)
        if not diff.any():
            return None
        fvals = set(np.unique(o[diff]).tolist())
        if len(fvals) != 1:
            return None
        v = next(iter(fvals))
        if fill is None:
            fill = v
        elif fill != v:
            return None
        # changed cells must have been background (0)
        if not np.all(i[diff] == 0):
            return None
        # marker = the non-bg color in input (should be single)
        mvals = set(np.unique(i[i != 0]).tolist())
        if len(mvals) != 1:
            return None
        mv = next(iter(mvals))
        if marker is None:
            marker = mv
        elif marker != mv:
            return None
    if fill is None or marker is None or fill == 0:
        return None

    def fn(g):
        H, W = g.shape
        out = g.copy()
        # horizontal runs
        for r in range(H):
            c = 0
            while c < W:
                if g[r, c] == marker:
                    c2 = c
                    while c2 < W and g[r, c2] == marker:
                        c2 += 1
                    if c2 - c >= 2:
                        c0, c1 = c, c2 - 1
                        d = 1
                        while c0 + d <= c1 - d:
                            for rr in (r - d, r + d):
                                if 0 <= rr < H:
                                    for cc in range(c0 + d, c1 - d + 1):
                                        if out[rr, cc] == 0:
                                            out[rr, cc] = fill
                            d += 1
                    c = c2
                else:
                    c += 1
        # vertical runs
        for c in range(W):
            r = 0
            while r < H:
                if g[r, c] == marker:
                    r2 = r
                    while r2 < H and g[r2, c] == marker:
                        r2 += 1
                    if r2 - r >= 2:
                        r0, r1 = r, r2 - 1
                        d = 1
                        while r0 + d <= r1 - d:
                            for cc in (c - d, c + d):
                                if 0 <= cc < W:
                                    for rr in range(r0 + d, r1 - d + 1):
                                        if out[rr, cc] == 0:
                                            out[rr, cc] = fill
                            d += 1
                    r = r2
                else:
                    r += 1
        return out

    return fn


DETECTORS = [
    recolor_by_column_rank,
    fill_largest_bg_rect_interior,
    copy_into_empty_rect,
    mirror_across_line_swap,
    key_orientation_recolor,
    perpendicular_triangles,
]
