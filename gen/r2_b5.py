"""Round-2 batch-5 detectors for ARC-AGI.

Each detector: def det(train) -> transform_fn | None
The engine verifies the returned fn reproduces EVERY train pair exactly.
Import only numpy + stdlib. Keep everything defensive.
"""
import numpy as np
from collections import Counter


# ---------------- helpers ----------------
def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag=False):
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            comps.append(cells)
    return comps


def _components_of(g, color, diag=False):
    """Connected components of cells equal to `color`."""
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] != color:
                continue
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == color:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            comps.append(cells)
    return comps


# ============================================================
# 1. FRACTAL self-tiling: NxN -> (N*H)x(N*W)
#    place a small pattern in each block where a selector holds.
# ============================================================
def fractal_selftile(train):
    i0, o0 = train[0]
    H, W = i0.shape
    if H == 0 or W == 0:
        return None
    if o0.shape != (H * H, W * W):
        return None
    if not all(i.shape == (H, W) and o.shape == (H * H, W * W) for i, o in train):
        # allow varying sizes but ratio must be N*H, N*W
        for i, o in train:
            h, w = i.shape
            if o.shape != (h * h, w * w):
                return None

    def build(sel, content):
        def fn(g):
            h, w = g.shape
            nz = [int(v) for v in np.unique(g) if v != 0]
            col = nz[0] if len(nz) == 1 else None
            out = np.zeros((h * h, w * w), dtype=int)
            for r in range(h):
                for c in range(w):
                    v = int(g[r, c])
                    if sel == "nz":
                        place = v != 0
                    elif sel == "z":
                        place = v == 0
                    elif sel == "mode":
                        vals, cnts = np.unique(g, return_counts=True)
                        nzvals = [(int(vv), int(cc)) for vv, cc in zip(vals, cnts) if vv != 0]
                        if not nzvals:
                            place = False
                        else:
                            m = max(nzvals, key=lambda t: t[1])[0]
                            place = v == m
                    else:  # fixed specific color, sel is an int
                        place = v == sel
                    if not place:
                        continue
                    if content == "orig":
                        patt = g
                    else:  # inverse
                        if col is None:
                            return None
                        patt = np.where(g == 0, col, 0)
                    out[r * h:(r + 1) * h, c * w:(c + 1) * w] = patt
            return out
        return fn

    # candidate selectors, ordered by generalization reliability:
    # structural (nz/z) first, then specific fixed colors, then fragile "mode".
    allcolors = set()
    for i, o in train:
        allcolors |= set(int(v) for v in np.unique(i))
    sels = ["nz", "z"] + [c for c in sorted(allcolors) if c != 0] + ["mode"]
    for content in ("orig", "inv"):
        for sel in sels:
            fn = build(sel, content)
            try:
                if all(fn(i) is not None and fn(i).shape == o.shape and np.array_equal(fn(i), o)
                       for i, o in train):
                    return fn
            except Exception:
                continue
    return None


# ============================================================
# 2. per-object 2-color swap (45737921)
# ============================================================
def object_two_color_swap(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def fn(g):
        bg = _bg(g)
        out = g.copy()
        for comp in _components(g, bg, diag=True):
            cols = sorted({int(g[y, x]) for y, x in comp})
            if len(cols) == 2:
                a, b = cols
                for y, x in comp:
                    out[y, x] = b if g[y, x] == a else a
        return out
    return fn


# ============================================================
# 3. diagonal legend intersection stamp (a406ac07)
#    bottom row = per-column color legend, right col = per-row legend.
#    at cells where they agree, paint that color.
# ============================================================
def legend_intersection(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def make(bottom_is_col):
        def fn(g):
            H, W = g.shape
            if bottom_is_col:
                colleg = g[H - 1, :]
                rowleg = g[:, W - 1]
            else:
                colleg = g[0, :]
                rowleg = g[:, 0]
            out = np.zeros_like(g)
            if bottom_is_col:
                out[H - 1, :] = g[H - 1, :]
                out[:, W - 1] = g[:, W - 1]
            else:
                out[0, :] = g[0, :]
                out[:, 0] = g[:, 0]
            for r in range(H):
                for c in range(W):
                    if rowleg[r] != 0 and rowleg[r] == colleg[c]:
                        out[r, c] = rowleg[r]
            return out
        return fn

    for flag in (True, False):
        fn = make(flag)
        try:
            if all(np.array_equal(fn(i), o) for i, o in train):
                return fn
        except Exception:
            continue
    return None


# ============================================================
# 4. fill column-intersection between two panels (770cc55f)
# ============================================================
def panel_intersection_fill(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # learn the fill color from the diff
    fill = None
    for i, o in train:
        d = o[o != i]
        if d.size:
            u = np.unique(d)
            if u.size != 1:
                return None
            if fill is None:
                fill = int(u[0])
            elif fill != int(u[0]):
                return None
    if fill is None:
        return None

    def sep_axis(g):
        H, W = g.shape
        # a separator is a uniform non-zero interior full line
        rows = [r for r in range(1, H - 1)
                if len(set(g[r, :].tolist())) == 1 and g[r, 0] != 0]
        cols = [c for c in range(1, W - 1)
                if len(set(g[:, c].tolist())) == 1 and g[0, c] != 0]
        if len(rows) == 1 and len(cols) == 0:
            return ("h", rows[0])
        if len(cols) == 1 and len(rows) == 0:
            return ("v", cols[0])
        return None

    def fn(g):
        H, W = g.shape
        sa = sep_axis(g)
        if sa is None:
            return None
        axis, s = sa
        if axis == "h":
            top = g[0, :]
            bot = g[H - 1, :]
            tset = set(c for c in range(W) if top[c] != 0)
            bset = set(c for c in range(W) if bot[c] != 0)
            inter = tset & bset
            out = g.copy()
            if len(tset) >= len(bset):
                for r in range(1, s):
                    for c in inter:
                        out[r, c] = fill
            else:
                for r in range(s + 1, H - 1):
                    for c in inter:
                        out[r, c] = fill
            return out
        else:
            left = g[:, 0]
            right = g[:, W - 1]
            lset = set(r for r in range(H) if left[r] != 0)
            rset = set(r for r in range(H) if right[r] != 0)
            inter = lset & rset
            out = g.copy()
            if len(lset) >= len(rset):
                for c in range(1, s):
                    for r in inter:
                        out[r, c] = fill
            else:
                for c in range(s + 1, W - 1):
                    for r in inter:
                        out[r, c] = fill
            return out

    try:
        if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ============================================================
# 5. recolor solid squares into a fixed 4-quadrant frame pattern (639f5a19)
# ============================================================
def square_quadrant_frame(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # detect the source color (color that forms solid rectangles) and learn palette
    src_candidates = set()
    for i, o in train:
        for v in np.unique(i):
            if v != 0:
                src_candidates.add(int(v))
    # try each source color and a couple frame widths
    for src in sorted(src_candidates):
        for fw in (2, 1, 3):
            fn = _square_quadrant_fn(src, fw)
            try:
                if all(np.array_equal(fn(i), o) for i, o in train):
                    return fn
            except Exception:
                continue
    return None


def _square_quadrant_fn(src, fw):
    # learned palette (fixed for this family)
    TL, TR, BL, BR, INNER = 6, 1, 2, 3, 4

    def fn(g):
        out = g.copy()
        for cells in _components_of(g, src):
            ys = [y for y, x in cells]
            xs = [x for y, x in cells]
            r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            # must be a solid rectangle
            if len(cells) != h * w:
                return None
            if h < 2 * fw + 1 or w < 2 * fw + 1:
                return None
            for (y, x) in cells:
                rr = y - r0
                cc = x - c0
                infr = (rr < fw or rr >= h - fw or cc < fw or cc >= w - fw)
                if not infr:
                    out[y, x] = INNER
                else:
                    top = rr < h / 2
                    left = cc < w / 2
                    out[y, x] = TL if (top and left) else (TR if (top and not left) else (BL if (not top and left) else BR))
        return out
    return fn


# ============================================================
# 6. color -> destination column projection (f45f5ca7)
#    each colored cell moves horizontally to a column fixed per color.
# ============================================================
def color_to_column(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    mp = {}
    for i, o in train:
        H, W = i.shape
        for r in range(H):
            src = [(c, int(i[r, c])) for c in range(W) if i[r, c] != 0]
            dst = [(c, int(o[r, c])) for c in range(W) if o[r, c] != 0]
            if len(src) == 0 and len(dst) == 0:
                continue
            if len(src) != 1 or len(dst) != 1:
                return None
            _, col = src[0]
            dc, dcol = dst[0]
            if col != dcol:
                return None
            if col in mp and mp[col] != dc:
                return None
            mp[col] = dc
    if not mp:
        return None

    def fn(g):
        H, W = g.shape
        out = np.zeros_like(g)
        for r in range(H):
            for c in range(W):
                if g[r, c] != 0:
                    col = mp.get(int(g[r, c]))
                    if col is None or col >= W:
                        return None
                    out[r, col] = g[r, c]
        return out

    try:
        if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ============================================================
# 7. most-common color from above a separator, placed at bottom-center (27a77e38)
# ============================================================
def mode_above_sep_to_bottom(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def fn(g):
        H, W = g.shape
        seps = [r for r in range(H)
                if len(set(g[r, :].tolist())) == 1 and g[r, 0] != 0]
        if not seps:
            return None
        sep = seps[0]
        sepcol = int(g[sep, 0])
        top = g[:sep, :]
        cnt = Counter(int(v) for v in top.flatten() if v != 0 and v != sepcol)
        if not cnt:
            return None
        mc = cnt.most_common(1)[0][0]
        out = g.copy()
        out[H - 1, W // 2] = mc
        return out

    try:
        if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ============================================================
# 8. quadrant color extraction from a framed rectangle (19bb5feb)
#    inside the frame (color 8 or other), each colored 2x2 block maps to
#    a 2x2 output by its quadrant.
# ============================================================
def frame_quadrant_extract(train):
    outs = [o for _, o in train]
    if not all(o.shape == (2, 2) for o in outs):
        return None
    # detect the frame color = the most common non-zero color in inputs
    frame_cands = Counter()
    for i, _ in train:
        vals, cnts = np.unique(i, return_counts=True)
        for v, c in zip(vals, cnts):
            if v != 0:
                frame_cands[int(v)] += int(c)
    if not frame_cands:
        return None

    def make(frame):
        def fn(g):
            mask = g == frame
            if not mask.any():
                return None
            ys, xs = np.where(mask)
            r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
            cr = (r0 + r1) / 2
            cc = (c0 + c1) / 2
            out = np.zeros((2, 2), dtype=int)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    v = int(g[r, c])
                    if v == 0 or v == frame:
                        continue
                    qr = 0 if r < cr else 1
                    qc = 0 if c < cc else 1
                    out[qr, qc] = v
            return out
        return fn

    for frame, _ in frame_cands.most_common():
        fn = make(frame)
        try:
            if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
                return fn
        except Exception:
            continue
    return None


DETECTORS = [
    fractal_selftile,
    object_two_color_swap,
    legend_intersection,
    panel_intersection_fill,
    square_quadrant_frame,
    color_to_column,
    mode_above_sep_to_bottom,
    frame_quadrant_extract,
]
