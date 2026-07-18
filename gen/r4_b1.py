"""Round-4 batch-1 detectors for ARC-AGI eval tasks.

Interface: each detector def det(train) -> transform_fn | None
  train = [(input_grid, output_grid), ...]  numpy int 2D arrays
The engine verifies the returned fn reproduces every train pair exactly.
numpy + stdlib only; keep everything defensive.
"""
import numpy as np
from collections import Counter, defaultdict


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


# ---------------------------------------------------------------------------
# 1. Color -> fixed block substitution.
#    Each input cell of color k expands into a fixed kh x kw block (learned).
#    Output shape = (H*kh, W*kw).  (task 2072aba6)
# ---------------------------------------------------------------------------
def color_block_subst(train):
    i0, o0 = train[0]
    ih, iw = i0.shape
    oh, ow = o0.shape
    if ih == 0 or iw == 0 or oh % ih or ow % iw:
        return None
    kh, kw = oh // ih, ow // iw
    if kh * kw <= 1 or kh > 6 or kw > 6:
        return None
    # learn mapping color -> block (kh,kw) consistent across all train pairs
    mapping = {}
    for i, o in train:
        H, W = i.shape
        if o.shape != (H * kh, W * kw):
            return None
        for r in range(H):
            for c in range(W):
                k = int(i[r, c])
                blk = o[r*kh:(r+1)*kh, c*kw:(c+1)*kw]
                if k in mapping:
                    if not np.array_equal(mapping[k], blk):
                        return None
                else:
                    mapping[k] = blk.copy()

    def fn(g):
        H, W = g.shape
        out = np.zeros((H * kh, W * kw), dtype=int)
        for r in range(H):
            for c in range(W):
                k = int(g[r, c])
                if k in mapping:
                    out[r*kh:(r+1)*kh, c*kw:(c+1)*kw] = mapping[k]
                # unseen color -> leave as zeros (defensive)
        return out
    return fn


# ---------------------------------------------------------------------------
# 2. Mark diagonal neighbour of each colored cell, then tile.
#    For every non-bg cell at (r,c) place a fixed mark-color at
#    ((r+dr)%H, (c+dc)%W) when that target is background; then tile the
#    modified grid by (th, tw).  (task 310f3251)
# ---------------------------------------------------------------------------
def _tiled(m, th, tw):
    return np.tile(m, (th, tw))


def diag_mark_tile(train):
    i0, o0 = train[0]
    ih, iw = i0.shape
    oh, ow = o0.shape
    if ih == 0 or iw == 0 or oh % ih or ow % iw:
        return None
    th, tw = oh // ih, ow // iw
    if th < 1 or tw < 1 or (th == 1 and tw == 1):
        return None
    for i, o in train:
        if o.shape != (i.shape[0]*th, i.shape[1]*tw):
            return None

    bgs = [_bg(i) for i, _ in train]
    # candidate offsets (diagonals + orthogonals)
    offs = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
    # candidate mark colors from output cells not in input
    mark_cands = set()
    for i, o in train:
        mark_cands |= set(int(x) for x in np.unique(o))
    for dr, dc in offs:
        for wrap in (True, False):
            for mc in sorted(mark_cands):
                ok = True
                for (i, o), bg in zip(train, bgs):
                    H, W = i.shape
                    m = i.copy()
                    for r in range(H):
                        for c in range(W):
                            if i[r, c] == bg:
                                continue
                            nr, nc = r + dr, c + dc
                            if wrap:
                                nr %= H; nc %= W
                            elif not (0 <= nr < H and 0 <= nc < W):
                                continue
                            if i[nr, nc] == bg:  # only stamp onto bg
                                m[nr, nc] = mc
                    if not np.array_equal(_tiled(m, th, tw), o):
                        ok = False
                        break
                if ok:
                    def fn(g, dr=dr, dc=dc, wrap=wrap, mc=mc, th=th, tw=tw):
                        bg = _bg(g)
                        H, W = g.shape
                        m = g.copy()
                        for r in range(H):
                            for c in range(W):
                                if g[r, c] == bg:
                                    continue
                                nr, nc = r + dr, c + dc
                                if wrap:
                                    nr %= H; nc %= W
                                elif not (0 <= nr < H and 0 <= nc < W):
                                    continue
                                if g[nr, nc] == bg:
                                    m[nr, nc] = mc
                        return _tiled(m, th, tw)
                    return fn
    return None


# ---------------------------------------------------------------------------
# 3. Recolor solid monochrome rectangles to a fixed color.
#    Find maximal solid rectangles (single color, fully filled) of size >=
#    minarea and recolor them all to a learned color.  (task 25094a63)
# ---------------------------------------------------------------------------
def _rects_of_color(g, col, minside):
    """All maximal solid rectangles filled with `col`, both sides >= minside.
    Robust to noise cells of the same color elsewhere in the grid (does not
    rely on connectivity)."""
    H, W = g.shape
    mask = (g == col)
    cands = []
    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0]:
                continue
            maxw = 0
            while c0 + maxw < W and mask[r0, c0 + maxw]:
                maxw += 1
            for ww in range(minside, maxw + 1):
                hh = 0
                while r0 + hh < H and mask[r0 + hh, c0:c0 + ww].all():
                    hh += 1
                if hh >= minside:
                    cands.append((r0, r0 + hh - 1, c0, c0 + ww - 1))
    cands = list(set(cands))

    def contains(a, b):
        return a[0] <= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] >= b[3] and a != b
    return [c for c in cands if not any(contains(o, c) for o in cands)]


def _solid_rects(g, minside=4):
    """Return (r0,r1,c0,c1,color) for maximal solid monochrome rectangles."""
    rects = []
    for col in np.unique(g):
        for (r0, r1, c0, c1) in _rects_of_color(g, int(col), minside):
            rects.append((r0, r1, c0, c1, int(col)))
    return rects


def recolor_solid_rects(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # learn target color from changed cells
    tgt = None
    for i, o in train:
        d = i != o
        if not d.any():
            continue
        tos = set(int(x) for x in o[d])
        if len(tos) != 1:
            return None
        tc = tos.pop()
        if tgt is None:
            tgt = tc
        elif tgt != tc:
            return None
    if tgt is None:
        return None

    # determine minimum side that makes the rule reproduce train
    for minside in (4, 5, 6, 3):
        def fn(g, tgt=tgt, minside=minside):
            out = g.copy()
            for r0, r1, c0, c1, col in _solid_rects(g, minside):
                if col == tgt:
                    continue
                out[r0:r1+1, c0:c1+1] = tgt
            return out
        if all(np.array_equal(fn(i), o) for i, o in train):
            return fn
    return None


# ---------------------------------------------------------------------------
# 4. Left key-block copied to the far edge, gap filled by its edge column.
#    A pattern occupies the leftmost k columns (rest background). Output keeps
#    that block on the left, copies it to the rightmost k columns, and fills
#    the middle columns with the block's last column value (per row).
#    Symmetric top-edge variant handled too.  (task 62b74c02)
# ---------------------------------------------------------------------------
def _block_extend_axis(train, axis):
    """axis=1 -> horizontal (block on left, extend right).
    axis=0 -> vertical (block on top, extend down)."""
    def apply(g):
        gg = g if axis == 1 else g.T
        H, W = gg.shape
        bg = _bg(gg)
        # block = leading columns until the first all-bg column
        k = 0
        while k < W and np.any(gg[:, k] != bg):
            k += 1
        if k == 0 or k * 2 > W:
            return None
        out = gg.copy()
        edge = gg[:, k-1]
        for c in range(k, W - k):
            out[:, c] = edge
        out[:, W-k:] = gg[:, :k]
        return out if axis == 1 else out.T
    if all((apply(i) is not None and apply(i).shape == o.shape and np.array_equal(apply(i), o))
           for i, o in train):
        return apply
    return None


def left_block_extend(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    for axis in (1, 0):
        fn = _block_extend_axis(train, axis)
        if fn is not None:
            return fn
    return None


# ---------------------------------------------------------------------------
# 5. Marker cross-propagation over a periodic foreground.
#    A periodic 2-color (bg + fg) pattern has a small block of a special
#    color S. Every foreground cell whose row is in S's rows, or whose
#    column is in S's columns, becomes S.  (task 4f537728)
# ---------------------------------------------------------------------------
def _cross_prop(g, S, bg):
    rows = set(np.where(g == S)[0].tolist())
    cols = set(np.where(g == S)[1].tolist())
    out = g.copy()
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg and (r in rows or c in cols):
                out[r, c] = S
    return out


def _pick_special(g, bg):
    """The special marker = the least-common non-background color, provided
    there is a distinct more-common foreground color."""
    cc = Counter(int(x) for x in g.flatten())
    nz = [c for c in cc if c != bg]
    if len(nz) < 2:
        return None
    fgmain = max(nz, key=lambda c: cc[c])
    others = [c for c in nz if c != fgmain]
    S = min(others, key=lambda c: cc[c])
    return S


def marker_cross_prop(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def fn(g):
        b = _bg(g)
        S = _pick_special(g, b)
        if S is None:
            return g.copy()
        return _cross_prop(g, S, b)
    if all(np.array_equal(fn(i), o) for i, o in train):
        return fn
    return None


# ---------------------------------------------------------------------------
# 6. Mark plus-shaped holes of background with a color.
#    Every background cell whose 4 orthogonal neighbours are all background
#    (i.e. the centre of a plus of background) gets marked, together with the
#    4 arm cells, using a learned mark color.  (task 7e02026e)
# ---------------------------------------------------------------------------
def mark_plus_holes(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # learn mark color and the "hole" (background) color from changes
    mark = None
    hole = None
    for i, o in train:
        d = i != o
        if not d.any():
            continue
        tos = set(int(x) for x in o[d])
        froms = set(int(x) for x in i[d])
        if len(tos) != 1 or len(froms) != 1:
            return None
        tc = tos.pop(); fc = froms.pop()
        if mark is None:
            mark, hole = tc, fc
        elif mark != tc or hole != fc:
            return None
    if mark is None:
        return None

    def fn(g, mark=mark, bg=hole):
        H, W = g.shape
        out = g.copy()
        for r in range(1, H-1):
            for c in range(1, W-1):
                if (g[r, c] == bg and g[r-1, c] == bg and g[r+1, c] == bg
                        and g[r, c-1] == bg and g[r, c+1] == bg):
                    out[r, c] = mark
                    out[r-1, c] = mark
                    out[r+1, c] = mark
                    out[r, c-1] = mark
                    out[r, c+1] = mark
        return out
    if all(np.array_equal(fn(i), o) for i, o in train):
        return fn
    return None


# ---------------------------------------------------------------------------
# 7. Recolor fill cells by the marker of their separator-delimited region.
#    Grid is divided by full-line separators into a block grid. One "fill"
#    color marks cells to be recolored; each region carries a unique marker
#    color; fill cells adopt the marker of their column-region (else
#    row-region).  (task ef26cbf6)
# ---------------------------------------------------------------------------
def _regions(n, seps):
    regs = []; cur = []
    for i in range(n):
        if i in seps:
            if cur:
                regs.append(cur); cur = []
        else:
            cur.append(i)
    if cur:
        regs.append(cur)
    return regs


def _region_of(idx, regs):
    for k, rg in enumerate(regs):
        if idx in rg:
            return k
    return None


def region_marker_recolor(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # learn fill color from changes (input value that gets recolored)
    fills = set()
    for i, o in train:
        d = i != o
        if d.any():
            fills |= set(int(x) for x in i[d])
    if len(fills) != 1:
        return None
    fill = fills.pop()

    def fn(g, fill=fill):
        H, W = g.shape
        cc = Counter(int(x) for x in g.flatten())
        bg = max(cc, key=lambda c: cc[c])
        sep = None
        for col in sorted(cc):
            if col in (bg, fill):
                continue
            fr = any(np.all(g[r, :] == col) for r in range(H))
            fcl = any(np.all(g[:, c] == col) for c in range(W))
            if fr or fcl:
                sep = col; break
        if sep is None:
            return g.copy()
        sep_rows = set(r for r in range(H) if np.all(g[r, :] == sep))
        sep_cols = set(c for c in range(W) if np.all(g[:, c] == sep))
        colregs = _regions(W, sep_cols)
        rowregs = _regions(H, sep_rows)
        markers = [c for c in cc if c not in (bg, sep, fill)]
        mk_col = {}; mk_row = {}
        if markers:
            ys, xs = np.where(np.isin(g, markers))
            for y, x in zip(ys, xs):
                v = int(g[y, x])
                cr = _region_of(x, colregs); rr = _region_of(y, rowregs)
                mk_col.setdefault(cr, set()).add(v)
                mk_row.setdefault(rr, set()).add(v)
        out = g.copy()
        fy, fx = np.where(g == fill)
        for y, x in zip(fy, fx):
            cr = _region_of(x, colregs); rr = _region_of(y, rowregs)
            m = None
            if cr in mk_col and len(mk_col[cr]) == 1:
                m = next(iter(mk_col[cr]))
            elif rr in mk_row and len(mk_row[rr]) == 1:
                m = next(iter(mk_row[rr]))
            if m is not None:
                out[y, x] = m
        return out
    if all(np.array_equal(fn(i), o) for i, o in train):
        return fn
    return None


# ---------------------------------------------------------------------------
# 8. Punch a checkerboard into the interior of every solid rectangle.
#    Each solid monochrome rectangle keeps its border ring; interior cells
#    whose (r+c) parity matches the top-left corner are set to background.
#    (task ba9d41b8)
# ---------------------------------------------------------------------------
def checker_punch_rects(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    i0, _ = train[0]
    bg = _bg(i0)
    for minside in (3, 4):
        def fn(g, bg=bg, minside=minside):
            out = g.copy()
            for col in np.unique(g):
                if int(col) == bg:
                    continue
                for (r0, r1, c0, c1) in _rects_of_color(g, int(col), minside):
                    par = (r0 + c0) % 2
                    for r in range(r0 + 1, r1):
                        for c in range(c0 + 1, c1):
                            if (r + c) % 2 == par:
                                out[r, c] = bg
            return out
        if all(np.array_equal(fn(i), o) for i, o in train):
            return fn
    return None


# ---------------------------------------------------------------------------
# 9. Draw a square around each centre marker sized by its distance to a ruler.
#    Two marker colours: a "centre" C and a "ruler" R. Around each C, draw a
#    filled square of side 2d-1 (d = chebyshev distance to the nearest R),
#    centred on C, using a fill colour F (over background only), keeping C.
#    (task ff72ca3e)
# ---------------------------------------------------------------------------
def dist_square_stamp(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # fill = color appearing in outputs but not inputs
    in_cols = set(); out_cols = set()
    for i, o in train:
        in_cols |= set(int(x) for x in np.unique(i))
        out_cols |= set(int(x) for x in np.unique(o))
    new_cols = out_cols - in_cols
    if len(new_cols) != 1:
        return None
    F = new_cols.pop()
    i0, _ = train[0]
    bg = _bg(i0)
    specials = [c for c in in_cols if c != bg and c != F]
    if len(specials) < 2:
        return None

    def make(C, R, F=F, bg=bg):
        def fn(g):
            H, W = g.shape
            out = g.copy()
            cs = list(zip(*np.where(g == C)))
            rs = list(zip(*np.where(g == R)))
            if not cs or not rs:
                return g.copy()
            for (cy, cx) in cs:
                d = min(max(abs(cy-ry), abs(cx-rx)) for (ry, rx) in rs)
                rad = d - 1
                for r in range(cy-rad, cy+rad+1):
                    for c in range(cx-rad, cx+rad+1):
                        if 0 <= r < H and 0 <= c < W and out[r, c] == bg:
                            out[r, c] = F
                out[cy, cx] = C
            return out
        return fn

    for C in specials:
        for R in specials:
            if C == R:
                continue
            fn = make(C, R)
            if all(np.array_equal(fn(i), o) for i, o in train):
                return fn
    return None


# ---------------------------------------------------------------------------
# 10. Fill loop interiors and leak a ray through the wall gap.
#     Each rectangular loop (outline of a foreground colour) has its interior
#     filled with a fill colour; the single gap in the wall leaks a straight
#     ray of the fill colour outward to the grid edge.  (task 292dd178)
# ---------------------------------------------------------------------------
def _comps(g, colorset):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) not in colorset:
                continue
            st = [(r, c)]; seen[r, c] = True; cells = [(r, c)]
            while st:
                y, x = st.pop()
                for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and int(g[ny, nx]) in colorset:
                        seen[ny, nx] = True; st.append((ny, nx)); cells.append((ny, nx))
            out.append(cells)
    return out


def loop_fill_leak(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    fills = set()
    for i, o in train:
        d = i != o
        if d.any():
            fills |= set(int(x) for x in o[d])
    if len(fills) != 1:
        return None
    fill = fills.pop()

    def make(fg, fill=fill):
        def fn(g):
            H, W = g.shape
            bg = _bg(g)
            out = g.copy()
            for cs in _comps(g, {fg}):
                ys = [y for y, x in cs]; xs = [x for y, x in cs]
                r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
                if r1 - r0 < 2 or c1 - c0 < 2:
                    continue
                for r in range(r0+1, r1):
                    for c in range(c0+1, c1):
                        if g[r, c] == bg:
                            out[r, c] = fill
                for c in range(c0, c1+1):
                    if g[r0, c] == bg:
                        out[r0, c] = fill
                        for rr in range(r0-1, -1, -1):
                            out[rr, c] = fill
                    if g[r1, c] == bg:
                        out[r1, c] = fill
                        for rr in range(r1+1, H):
                            out[rr, c] = fill
                for r in range(r0, r1+1):
                    if g[r, c0] == bg:
                        out[r, c0] = fill
                        for cc in range(c0-1, -1, -1):
                            out[r, cc] = fill
                    if g[r, c1] == bg:
                        out[r, c1] = fill
                        for cc in range(c1+1, W):
                            out[r, cc] = fill
            return out
        return fn

    i0, _ = train[0]
    bg0 = _bg(i0)
    fgs = set()
    for i, _ in train:
        fgs |= set(int(x) for x in np.unique(i))
    fgs -= {bg0, fill}
    for fg in sorted(fgs):
        fn = make(fg)
        if all(np.array_equal(fn(i), o) for i, o in train):
            return fn
    return None


# ---------------------------------------------------------------------------
# 11. Stamp a learned template at each single-cell marker, plus an edge ray.
#     Each isolated marker cell expands into a fixed (2r+1)x(2r+1) template
#     learned from the outputs; additionally the whole grid row at the
#     template's bottom edge is painted with the ray colour before stamping.
#     (task 3f23242b)
# ---------------------------------------------------------------------------
def _single_cells(g, color):
    H, W = g.shape
    pts = []
    for (y, x) in zip(*np.where(g == color)):
        y, x = int(y), int(x)
        iso = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < H and 0 <= nx < W and g[ny, nx] == color:
                    iso = False
        if iso:
            pts.append((y, x))
    return pts


def marker_template_stamp(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    i0, _ = train[0]
    bg = _bg(i0)
    cols = set()
    for i, _ in train:
        cols |= set(int(x) for x in np.unique(i))
    cand_markers = [c for c in cols if c != bg]

    for M in sorted(cand_markers):
        # need every train input to have marker M as isolated cells
        if not all(len(_single_cells(i, M)) >= 1 for i, _ in train):
            continue
        for r in (1, 2, 3):
            # learn template from a clean, fully in-bounds, isolated marker
            tmpl = None
            for i, o in train:
                H, W = i.shape
                pts = _single_cells(i, M)
                for (my, mx) in pts:
                    if not (my-r >= 0 and my+r < H and mx-r >= 0 and mx+r < W):
                        continue
                    # skip if another marker contaminates the template window
                    if any((oy, ox) != (my, mx) and max(abs(oy-my), abs(ox-mx)) <= 2*r
                           for (oy, ox) in pts):
                        continue
                    # skip if another marker shares this marker's ray row
                    if any((oy, ox) != (my, mx) and abs(oy-my) <= r for (oy, ox) in pts):
                        continue
                    cand = o[my-r:my+r+1, mx-r:mx+r+1].copy()
                    if cand[r, r] == M:
                        tmpl = cand
                        break
                if tmpl is not None:
                    break
            if tmpl is None:
                continue
            # candidate ray colours: side-frame colours of the template
            ray_cands = set()
            for rr in range(2*r+1):
                ray_cands.add(int(tmpl[rr, 0]))
                ray_cands.add(int(tmpl[rr, 2*r]))
            ray_cands.discard(bg)
            ray_cands.add(None)

            found = None
            for ray_col in ray_cands:
                def fn(g, M=M, r=r, tmpl=tmpl, ray_col=ray_col):
                    H, W = g.shape
                    out = g.copy()
                    ms = _single_cells(g, M)
                    if ray_col is not None:
                        for (my, mx) in ms:
                            rr = my + r
                            if 0 <= rr < H:
                                out[rr, :] = ray_col
                    for (my, mx) in ms:
                        for dr in range(-r, r+1):
                            for dc in range(-r, r+1):
                                y, x = my+dr, mx+dc
                                if 0 <= y < H and 0 <= x < W:
                                    v = int(tmpl[dr+r, dc+r])
                                    if v != bg:
                                        out[y, x] = v
                    return out
                if all(np.array_equal(fn(i), o) for i, o in train):
                    found = fn
                    break
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
DETECTORS = [
    color_block_subst,
    diag_mark_tile,
    recolor_solid_rects,
    left_block_extend,
    marker_cross_prop,
    mark_plus_holes,
    region_marker_recolor,
    checker_punch_rects,
    dist_square_stamp,
    loop_fill_leak,
    marker_template_stamp,
]
