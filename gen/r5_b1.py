"""Round 5, batch 1 detectors for ARC-AGI.

Each detector: def det(train) -> transform_fn | None
  train = [(input_grid, output_grid), ...]  (numpy int arrays)
  transform_fn: grid -> grid   (engine verifies exact reproduction of all demos)

Keep detectors defensive & general. numpy + stdlib only.
"""
import numpy as np
from collections import Counter, deque
from itertools import combinations


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _comps(g, bg=0, diag=False):
    """Connected components of non-bg cells (same-color runs not required)."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = []
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg and not seen[r, c]:
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


def _color_comps(g, col, diag=False):
    """Connected components of a specific color."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = []
    for r in range(H):
        for c in range(W):
            if g[r, c] == col and not seen[r, c]:
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
                out.append(cells)
    return out


# ---------------------------------------------------------------------------
# 1) Concentric-ring stamp from a top legend row (task 9356391f).
#    Row 0 encodes, per column index d, the color of the square ring at
#    Chebyshev distance d from a single seed cell located below a solid
#    separator row.  Legend colors detached from the leading run are consumed
#    (overwritten by the separator color) after being used.
# ---------------------------------------------------------------------------
def concentric_ring_legend(train):
    def solve(g):
        H, W = g.shape
        # separator row: first row (>=1) that is constant & nonzero
        seprow = None
        sepcol = None
        for r in range(1, H):
            row = g[r]
            v = set(int(x) for x in row)
            if len(v) == 1 and 0 not in v:
                seprow = r
                sepcol = int(row[0])
                break
        if seprow is None:
            return None
        legend = g[0]
        if not (legend != 0).any():
            return None
        seeds = [(r, c) for r in range(seprow + 1, H) for c in range(W) if g[r, c] != 0]
        if len(seeds) != 1:
            return None
        sr, sc = seeds[0]
        out = g.copy()
        for d in range(W):
            col = int(legend[d])
            if col == 0:
                continue
            for r in range(seprow + 1, H):
                for c in range(W):
                    if max(abs(r - sr), abs(c - sc)) == d:
                        out[r, c] = col
        # rewrite row0: an *isolated* single detached legend cell (0 on both
        # sides, not part of the leading run) is a spent "reach marker" -> sepcol.
        # Contiguous legend runs are kept as-is.
        row0 = g[0].copy()
        e = 0
        while e < W and row0[e] != 0:
            e += 1  # end of leading run
        for c in range(e, W):
            if row0[c] == 0:
                continue
            left0 = (c == 0) or row0[c - 1] == 0
            right0 = (c == W - 1) or row0[c + 1] == 0
            if left0 and right0:
                row0[c] = sepcol
        out[0] = row0
        return out

    try:
        if all(solve(i) is not None and np.array_equal(solve(i), o) for i, o in train):
            return solve
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 2) Collect the hollow (frame) objects, drop the solid ones, concatenate them
#    in reading order along the axis in which they are more spread (a680ac02).
# ---------------------------------------------------------------------------
def collect_hollow_objects(train):
    def objs(g):
        res = []
        for cells in _comps(g, 0, diag=True):
            rs = [y for y, x in cells]
            cs = [x for y, x in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            hollow = len(cells) < h * w
            res.append(dict(r0=r0, c0=c0, h=h, w=w, hollow=hollow,
                            sub=g[r0:r1 + 1, c0:c1 + 1].copy()))
        return res

    def solve(g):
        o = objs(g)
        keep = [x for x in o if x['hollow']]
        if len(keep) < 1:
            return None
        # all kept must share the same shape (so they concat cleanly)
        shp = keep[0]['sub'].shape
        if any(x['sub'].shape != shp for x in keep):
            return None
        rmin = min(x['r0'] for x in keep)
        rmax = max(x['r0'] for x in keep)
        cmin = min(x['c0'] for x in keep)
        cmax = max(x['c0'] for x in keep)
        if (cmax - cmin) >= (rmax - rmin):
            keep.sort(key=lambda x: x['c0'])
            return np.hstack([x['sub'] for x in keep])
        else:
            keep.sort(key=lambda x: x['r0'])
            return np.vstack([x['sub'] for x in keep])

    try:
        if all(solve(i) is not None and np.array_equal(solve(i), o) for i, o in train):
            return solve
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 3) Bridge aligned same-color rectangles with a connector color (f3b10344).
#    For any two solid rectangles of the same color whose spans overlap on one
#    axis and are separated by a *clear* gap on the other, fill the gap over the
#    interior of the overlap with the connector color.
# ---------------------------------------------------------------------------
def _rect_boxes(g):
    boxes = []
    for cells in _comps(g, 0, diag=False):
        # split into color-uniform? components are non-bg connected; require
        # single color and rectangular fill.
        cols = set(int(g[y, x]) for y, x in cells)
        if len(cols) != 1:
            continue
        col = cols.pop()
        rs = [y for y, x in cells]
        cs = [x for y, x in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
            boxes.append((col, r0, r1, c0, c1))
    return boxes


def bridge_aligned_rects(train):
    # learn the connector color = the single new color in outputs
    new_cols = set()
    for i, o in train:
        if i.shape != o.shape:
            return None
        for r, c in np.argwhere(i != o):
            new_cols.add(int(o[r, c]))
    if len(new_cols) != 1:
        return None
    fill = next(iter(new_cols))

    def centered_contain(lo1, hi1, lo2, hi2):
        # one span contains the other with equal margins (symmetric nesting)
        if lo1 <= lo2 and hi2 <= hi1:
            return (lo2 - lo1) == (hi1 - hi2)
        if lo2 <= lo1 and hi1 <= hi2:
            return (lo1 - lo2) == (hi2 - hi1)
        return False

    def solve(g):
        out = g.copy()
        boxes = _rect_boxes(g)
        for a, b in combinations(boxes, 2):
            ca, ar0, ar1, ac0, ac1 = a
            cb, br0, br1, bc0, bc1 = b
            if ca != cb:
                continue
            # horizontal bridge: col gap, row-spans nested & centered
            if ac1 < bc0 or bc1 < ac0:
                ro0, ro1 = max(ar0, br0), min(ar1, br1)
                if ro0 <= ro1 and ro1 - ro0 >= 2 and centered_contain(ar0, ar1, br0, br1):
                    g0, g1 = (ac1 + 1, bc0 - 1) if ac1 < bc0 else (bc1 + 1, ac0 - 1)
                    if g1 >= g0 and (g[ro0:ro1 + 1, g0:g1 + 1] == 0).all():
                        out[ro0 + 1:ro1, g0:g1 + 1] = fill
            # vertical bridge: row gap, col-spans nested & centered
            if ar1 < br0 or br1 < ar0:
                co0, co1 = max(ac0, bc0), min(ac1, bc1)
                if co0 <= co1 and co1 - co0 >= 2 and centered_contain(ac0, ac1, bc0, bc1):
                    h0, h1 = (ar1 + 1, br0 - 1) if ar1 < br0 else (br1 + 1, ar0 - 1)
                    if h1 >= h0 and (g[h0:h1 + 1, co0:co1 + 1] == 0).all():
                        out[h0:h1 + 1, co0 + 1:co1] = fill
        return out

    try:
        if all(np.array_equal(solve(i), o) for i, o in train):
            return solve
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 4) Shift every object by its own bounding-box size along one axis (64a7c07e).
# ---------------------------------------------------------------------------
def shift_by_own_size(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def make(along, dr, dc):
        def solve(g):
            H, W = g.shape
            out = np.zeros_like(g)
            for cells in _comps(g, 0, diag=True):
                rs = [y for y, x in cells]
                cs = [x for y, x in cells]
                amt = (max(cs) - min(cs) + 1) if along == 'w' else (max(rs) - min(rs) + 1)
                for y, x in cells:
                    ny, nx = y + dr * amt, x + dc * amt
                    if 0 <= ny < H and 0 <= nx < W:
                        out[ny, nx] = g[y, x]
            return out
        return solve

    for along in ('w', 'h'):
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            fn = make(along, dr, dc)
            try:
                if all(np.array_equal(fn(i), o) for i, o in train):
                    return fn
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# 5) Outline closed loops: fill the cells hugging a closed loop from inside
#    with one color and from outside with another; open shapes are untouched
#    (d931c21c).  Colors learned from the two new colors in the outputs.
# ---------------------------------------------------------------------------
def _adj8_mask(mask):
    H, W = mask.shape
    m = np.zeros((H, W), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.zeros((H, W), bool)
            ys = slice(max(0, dy), H + min(0, dy))
            xs = slice(max(0, dx), W + min(0, dx))
            yt = slice(max(0, -dy), H + min(0, -dy))
            xt = slice(max(0, -dx), W + min(0, -dx))
            sh[yt, xt] = mask[ys, xs]
            m |= sh
    return m


def outline_closed_loops(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    loop_cands = set()
    new_cols = set()
    for i, o in train:
        loop_cands |= set(int(v) for v in np.unique(i) if v != 0)
        for r, c in np.argwhere(i != o):
            new_cols.add(int(o[r, c]))
    if len(loop_cands) != 1 or len(new_cols) != 2:
        return None
    loopc = next(iter(loop_cands))
    if loopc in new_cols:
        return None

    def make(inc, outc):
        def solve(g):
            H, W = g.shape
            out = g.copy()
            outside = np.zeros((H, W), bool)
            dq = deque()
            for r in range(H):
                for c in range(W):
                    if (r in (0, H - 1) or c in (0, W - 1)) and g[r, c] == 0:
                        outside[r, c] = True
                        dq.append((r, c))
            while dq:
                y, x = dq.popleft()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not outside[ny, nx] and g[ny, nx] == 0:
                        outside[ny, nx] = True
                        dq.append((ny, nx))
            enclosed_all = (g == 0) & (~outside)
            two_total = np.zeros((H, W), bool)
            for cells in _color_comps(g, loopc, diag=True):
                cmask = np.zeros((H, W), bool)
                for y, x in cells:
                    cmask[y, x] = True
                near = _adj8_mask(cmask)
                enc = enclosed_all & near
                if not enc.any():
                    continue
                out[enc] = inc
                two = (g == 0) & outside & near
                rs = [y for y, x in cells]
                cs = [x for y, x in cells]
                r0, r1 = min(rs) - 1, max(rs) + 1
                c0, c1 = min(cs) - 1, max(cs) + 1
                frame = np.zeros((H, W), bool)
                for c in range(max(0, c0), min(W, c1 + 1)):
                    if 0 <= r0 < H:
                        frame[r0, c] = True
                    if 0 <= r1 < H:
                        frame[r1, c] = True
                for r in range(max(0, r0), min(H, r1 + 1)):
                    if 0 <= c0 < W:
                        frame[r, c0] = True
                    if 0 <= c1 < W:
                        frame[r, c1] = True
                changed = True
                while changed:
                    changed = False
                    for y, x in np.argwhere(frame & outside & (g == 0) & ~two):
                        up = two[y - 1, x] if y > 0 else False
                        dn = two[y + 1, x] if y < H - 1 else False
                        lf = two[y, x - 1] if x > 0 else False
                        rt = two[y, x + 1] if x < W - 1 else False
                        if (up or dn) and (lf or rt):
                            two[y, x] = True
                            changed = True
                two_total |= two
            out[two_total] = outc
            return out
        return solve

    a, b = sorted(new_cols)
    for inc, outc in ((a, b), (b, a)):
        fn = make(inc, outc)
        try:
            if all(np.array_equal(fn(i), o) for i, o in train):
                return fn
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# 6) Two-glyph swap (3391f8c0).  The grid holds two colors, each with a single
#    canonical glyph shape (same bounding box).  Replace every glyph with the
#    *other* color's glyph, anchored at the same bounding-box top-left.
# ---------------------------------------------------------------------------
def _canon_glyph(g, col):
    cs = _color_comps(g, col, diag=True)
    if not cs:
        return None
    shapes = set()
    for cells in cs:
        rs = [y for y, x in cells]
        xs = [x for y, x in cells]
        r0, c0 = min(rs), min(xs)
        shapes.add(frozenset((y - r0, x - c0) for y, x in cells))
    if len(shapes) != 1:
        return None
    return next(iter(shapes))


def two_glyph_swap(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def solve(g):
        cols = sorted(int(v) for v in np.unique(g) if v != 0)
        if len(cols) != 2:
            return None
        a, b = cols
        ga, gb = _canon_glyph(g, a), _canon_glyph(g, b)
        if ga is None or gb is None:
            return None

        def dims(s):
            rs = [y for y, x in s]
            xs = [x for y, x in s]
            return max(rs) + 1, max(xs) + 1

        if dims(ga) != dims(gb):
            return None
        other = {a: b, b: a}
        glyph = {a: ga, b: gb}
        out = np.zeros_like(g)
        for col in cols:
            for cells in _color_comps(g, col, diag=True):
                rs = [y for y, x in cells]
                xs = [x for y, x in cells]
                r0, c0 = min(rs), min(xs)
                for (yy, xx) in glyph[other[col]]:
                    out[r0 + yy, c0 + xx] = other[col]
        return out

    try:
        if all(solve(i) is not None and np.array_equal(solve(i), o) for i, o in train):
            return solve
    except Exception:
        return None
    return None


DETECTORS = [
    concentric_ring_legend,
    collect_hollow_objects,
    bridge_aligned_rects,
    shift_by_own_size,
    outline_closed_loops,
    two_glyph_swap,
]
