"""Round 5, batch 4 detectors for ARC-AGI.

Each detector: def det(train) -> transform_fn | None
  train = [(input_grid, output_grid), ...]  (numpy int arrays)
  transform_fn: grid -> grid   (engine verifies exact reproduction of all demos)

General, principled rules only. numpy + stdlib.
"""
import numpy as np
from collections import Counter


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


# ---------------------------------------------------------------------------
# reverse_color_bands: the grid is a vertical stack of horizontal "bands", each
# band being all cells of one color. Output reverses the vertical order of the
# bands while preserving each band's internal shape.  (e21a174a)
# ---------------------------------------------------------------------------
def reverse_color_bands(train):
    def solve(inp):
        n, m = inp.shape
        colors = [c for c in np.unique(inp) if c != 0]
        if len(colors) < 2:
            return None
        bands = []
        for c in colors:
            rows = np.where((inp == c).any(axis=1))[0]
            if rows.size == 0:
                return None
            bands.append((int(rows.min()), int(rows.max()), int(c)))
        bands.sort()
        # bands must be non-overlapping and contiguous (each color occupies a
        # distinct horizontal slab)
        for k in range(len(bands) - 1):
            if bands[k][1] >= bands[k + 1][0]:
                return None
        out = np.zeros_like(inp)
        cur = bands[0][0]
        for (t0, t1, c) in bands[::-1]:
            h = t1 - t0 + 1
            src = inp[t0:t1 + 1]
            dst = out[cur:cur + h]
            dst[src == c] = c
            cur += h
        return out
    try:
        for i, o in train:
            if i.shape != o.shape:
                return None
            r = solve(i)
            if r is None or not np.array_equal(r, o):
                return None
    except Exception:
        return None
    return solve


# ---------------------------------------------------------------------------
# frame_interior_largest: input contains one or more single-colour rectangular
# frames (the four bbox borders of that colour are complete). Output is the
# interior of the frame with the largest interior area.  (1a6449f1)
# ---------------------------------------------------------------------------
def _find_frames(inp):
    n, m = inp.shape
    res = []
    for c in np.unique(inp):
        if c == 0:
            continue
        mask = (inp == c)
        if mask.sum() < 8:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        # candidate frames: search over rows that are (partly) c on both ends.
        r_cand = list(rows)
        c_cand = list(cols)
        for a in range(len(r_cand)):
            r0 = r_cand[a]
            for b in range(a + 1, len(r_cand)):
                r1 = r_cand[b]
                if r1 - r0 < 2:
                    continue
                for x in range(len(c_cand)):
                    c0 = c_cand[x]
                    for y in range(x + 1, len(c_cand)):
                        c1 = c_cand[y]
                        if c1 - c0 < 2:
                            continue
                        if (mask[r0, c0:c1 + 1].all() and mask[r1, c0:c1 + 1].all()
                                and mask[r0:r1 + 1, c0].all() and mask[r0:r1 + 1, c1].all()):
                            interior = inp[r0 + 1:r1, c0 + 1:c1]
                            area = interior.shape[0] * interior.shape[1]
                            res.append((area, int(c), (r0, r1, c0, c1), interior))
    return res


def frame_interior_largest(train):
    def solve(inp):
        frames = _find_frames(inp)
        if not frames:
            return None
        frames.sort(key=lambda t: -t[0])
        return frames[0][3].copy()
    try:
        for i, o in train:
            r = solve(i)
            if r is None or r.shape != o.shape or not np.array_equal(r, o):
                return None
    except Exception:
        return None
    return solve


# ---------------------------------------------------------------------------
# fill_frame_keep_marker_window: a single rectangular frame encloses a region
# that is background except for a few "marker" cells (distinct colour). Output
# fills the interior with the frame colour, but preserves the axis-aligned
# bounding box of the markers unchanged.  (d37a1ef5)
# ---------------------------------------------------------------------------
def _one_frame(inp):
    n, m = inp.shape
    for c in np.unique(inp):
        if c == 0:
            continue
        ys, xs = np.where(inp == c)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        mask = (inp == c)
        if (mask[r0, c0:c1 + 1].all() and mask[r1, c0:c1 + 1].all()
                and mask[r0:r1 + 1, c0].all() and mask[r0:r1 + 1, c1].all()):
            return int(c), (int(r0), int(r1), int(c0), int(c1))
    return None


def fill_frame_keep_marker_window(train):
    def solve(inp):
        fr = _one_frame(inp)
        if fr is None:
            return None
        fc, (r0, r1, c0, c1) = fr
        interior = inp[r0 + 1:r1, c0 + 1:c1]
        mys, mxs = np.where(interior != 0)
        out = inp.copy()
        out[r0 + 1:r1, c0 + 1:c1] = fc
        if mys.size > 0:
            mr0, mr1 = mys.min(), mys.max()
            mc0, mc1 = mxs.min(), mxs.max()
            ar0, ac0 = r0 + 1 + mr0, c0 + 1 + mc0
            ar1, ac1 = r0 + 1 + mr1, c0 + 1 + mc1
            out[ar0:ar1 + 1, ac0:ac1 + 1] = inp[ar0:ar1 + 1, ac0:ac1 + 1]
        return out
    try:
        for i, o in train:
            if i.shape != o.shape:
                return None
            r = solve(i)
            if r is None or not np.array_equal(r, o):
                return None
    except Exception:
        return None
    return solve


# ---------------------------------------------------------------------------
# quadrant_overlay: grid split by one all-colour separator row and one all-colour
# separator column into 4 quadrants of equal size. Output overlays them in a
# fixed positional priority order, learned from the demos.  (e99362f0)
# ---------------------------------------------------------------------------
def _quadrants(inp):
    n, m = inp.shape
    seprows = [r for r in range(n) if len(set(inp[r].tolist())) == 1 and inp[r, 0] != 0]
    sepcols = [c for c in range(m) if len(set(inp[:, c].tolist())) == 1 and inp[0, c] != 0]
    if len(seprows) != 1 or len(sepcols) != 1:
        return None
    r, c = seprows[0], sepcols[0]
    q = [inp[:r, :c], inp[:r, c + 1:], inp[r + 1:, :c], inp[r + 1:, c + 1:]]
    shp = q[0].shape
    if any(x.shape != shp or x.size == 0 for x in q):
        return None
    return q


def quadrant_overlay(train):
    import itertools
    tr = []
    for i, o in train:
        q = _quadrants(i)
        if q is None or q[0].shape != o.shape:
            return None
        tr.append((q, o))
    good = None
    for perm in itertools.permutations(range(4)):
        ok = True
        for q, o in tr:
            res = np.zeros_like(o)
            filled = np.zeros(o.shape, bool)
            for idx in perm:
                qi = q[idx]
                take = (~filled) & (qi != 0)
                res[take] = qi[take]
                filled |= (qi != 0)
            if not np.array_equal(res, o):
                ok = False
                break
        if ok:
            good = perm
            break
    if good is None:
        return None

    def solve(inp):
        q = _quadrants(inp)
        if q is None:
            return None
        res = np.zeros_like(q[0])
        filled = np.zeros(q[0].shape, bool)
        for idx in good:
            qi = q[idx]
            take = (~filled) & (qi != 0)
            res[take] = qi[take]
            filled |= (qi != 0)
        return res
    return solve


# ---------------------------------------------------------------------------
# concentric_ring_number: each solid single-colour rectangle is re-coloured with
# concentric-ring values. We learn a mapping from the (folded chebyshev) ring
# index k to the output value from the demos, and apply it to every block.
#   (516b51b7)
# ---------------------------------------------------------------------------
def _solid_blocks(inp):
    n, m = inp.shape
    seen = np.zeros_like(inp, bool)
    res = []
    for i in range(n):
        for j in range(m):
            if inp[i, j] != 0 and not seen[i, j]:
                col = inp[i, j]
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < n and 0 <= nb < m and not seen[na, nb] and inp[na, nb] == col:
                            seen[na, nb] = True
                            stack.append((na, nb))
                rs = [c[0] for c in cells]
                cs = [c[1] for c in cells]
                r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
                if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
                    res.append((r0, r1, c0, c1, int(col)))
    return res


def _ring_index(r, c, H, W):
    rd = min(r, H - 1 - r)
    cd = min(c, W - 1 - c)
    return min(rd, cd)


def concentric_ring_number(train):
    table = {}
    for i, o in train:
        if i.shape != o.shape:
            return None
        blks = _solid_blocks(i)
        if not blks:
            return None
        for (r0, r1, c0, c1, col) in blks:
            H, W = r1 - r0 + 1, c1 - c0 + 1
            if H < 2 or W < 2:
                return None
            for r in range(H):
                for c in range(W):
                    k = _ring_index(r, c, H, W)
                    v = int(o[r0 + r, c0 + c])
                    if v == 0:
                        return None
                    if k in table and table[k] != v:
                        return None
                    table[k] = v
    if not table or set(table.keys()) == {0}:
        return None

    def solve(inp):
        out = inp.copy()
        for (r0, r1, c0, c1, col) in _solid_blocks(inp):
            H, W = r1 - r0 + 1, c1 - c0 + 1
            for r in range(H):
                for c in range(W):
                    k = _ring_index(r, c, H, W)
                    out[r0 + r, c0 + c] = table.get(k, col)
        return out
    return solve


# ---------------------------------------------------------------------------
# periodic_shape_recolor: shapes (connected components) are laid out in reading
# order; every k-th shape (offset o) is recoloured to a fixed target colour.
# We learn k, o and the target colour from the demos.  (22a4bbc2)
# ---------------------------------------------------------------------------
def _components(inp):
    n, m = inp.shape
    seen = np.zeros_like(inp, bool)
    res = []
    for i in range(n):
        for j in range(m):
            if inp[i, j] != 0 and not seen[i, j]:
                col = inp[i, j]
                st = [(i, j)]
                seen[i, j] = True
                cells = []
                while st:
                    a, b = st.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < n and 0 <= nb < m and not seen[na, nb] and inp[na, nb] == col:
                            seen[na, nb] = True
                            st.append((na, nb))
                r0 = min(a for a, b in cells)
                c0 = min(b for a, b in cells)
                res.append((r0, c0, int(col), cells))
    res.sort(key=lambda x: (x[0], x[1]))
    return res


def periodic_shape_recolor(train):
    # Determine, per demo, which shape indices got fully recoloured and to what.
    target = None
    changed_sets = []
    counts = []
    for i, o in train:
        if i.shape != o.shape:
            return None
        comps = _components(i)
        counts.append(len(comps))
        chg = []
        for idx, (r0, c0, col, cells) in enumerate(comps):
            ovals = set(int(o[a, b]) for a, b in cells)
            if len(ovals) == 1 and next(iter(ovals)) != col:
                tv = next(iter(ovals))
                if target is None:
                    target = tv
                elif target != tv:
                    return None
                chg.append(idx)
            else:
                # unchanged cells must equal input exactly
                if any(int(o[a, b]) != col for a, b in cells):
                    return None
        # also everything outside shapes must be unchanged
        changed_sets.append(set(chg))
    if target is None:
        return None
    # find period k and offset that reproduces every demo's changed set
    for k in range(1, 8):
        for off in range(k):
            ok = True
            for cnt, chg in zip(counts, changed_sets):
                pred = set(idx for idx in range(cnt) if idx % k == off)
                if pred != chg:
                    ok = False
                    break
            if ok:
                kk, oo = k, off

                def solve(inp, kk=kk, oo=oo, target=target):
                    out = inp.copy()
                    for idx, (r0, c0, col, cells) in enumerate(_components(inp)):
                        if idx % kk == oo:
                            for a, b in cells:
                                out[a, b] = target
                    return out
                return solve
    return None


# ---------------------------------------------------------------------------
# legend_stack_template: a "legend" (a small block whose every row -- or every
# column -- is the identical ordered sequence of distinct colours) specifies a
# colour order. A single template shape appears elsewhere (one or more copies in
# some of those colours). Output removes the legend and stacks one copy of the
# template per legend colour, in legend order, at a fixed pitch aligned with the
# existing copies.  (94be5b80)
# ---------------------------------------------------------------------------
def _find_legend(inp):
    n, m = inp.shape
    best = None
    for r0 in range(n):
        for r1 in range(r0 + 1, n):
            block = inp[r0:r1 + 1]
            colmask = [(block[:, c] == block[0, c]).all() and block[0, c] != 0 for c in range(m)]
            c = 0
            while c < m:
                if colmask[c]:
                    s = c
                    while c < m and colmask[c]:
                        c += 1
                    seq = [int(block[0, x]) for x in range(s, c)]
                    if len(seq) >= 2 and len(set(seq)) == len(seq):
                        area = (r1 - r0 + 1) * (c - s)
                        if best is None or area > best[6]:
                            best = (r0, r1, s, c - 1, seq, 'h', area)
                else:
                    c += 1
    for c0 in range(m):
        for c1 in range(c0 + 1, m):
            block = inp[:, c0:c1 + 1]
            rowmask = [(block[r, :] == block[r, 0]).all() and block[r, 0] != 0 for r in range(n)]
            r = 0
            while r < n:
                if rowmask[r]:
                    s = r
                    while r < n and rowmask[r]:
                        r += 1
                    seq = [int(block[x, 0]) for x in range(s, r)]
                    if len(seq) >= 2 and len(set(seq)) == len(seq):
                        area = (r - s) * (c1 - c0 + 1)
                        if best is None or area > best[6]:
                            best = (s, r - 1, c0, c1, seq, 'v', area)
                else:
                    r += 1
    return best


def legend_stack_template(train):
    def solve(inp):
        leg = _find_legend(inp)
        if leg is None:
            return None
        lr0, lr1, lc0, lc1, seq, orient, _ = leg
        n, m = inp.shape
        out = inp.copy()
        out[lr0:lr1 + 1, lc0:lc1 + 1] = 0
        shapes = {}
        for c in seq:
            ys, xs = np.where(out == c)
            if len(ys) == 0:
                continue
            y0, x0 = int(ys.min()), int(xs.min())
            norm = frozenset((int(a - y0), int(b - x0)) for a, b in zip(ys, xs))
            shapes[c] = (y0, x0, norm)
        if not shapes:
            return None
        norms = set(s[2] for s in shapes.values())
        if len(norms) != 1:
            return None
        template = next(iter(norms))
        h = max(a for a, b in template) + 1
        w = max(b for a, b in template) + 1
        present = sorted((seq.index(c), shapes[c][0], shapes[c][1]) for c in shapes)
        if len(present) >= 2:
            (i0, y0, x0), (i1, y1, x1) = present[0], present[1]
            if orient == 'h':
                if (y1 - y0) % (i1 - i0) != 0:
                    return None
                pitch = (y1 - y0) // (i1 - i0)
                base_y, base_x = y0 - i0 * pitch, x0
            else:
                if (x1 - x0) % (i1 - i0) != 0:
                    return None
                pitch = (x1 - x0) // (i1 - i0)
                base_x, base_y = x0 - i0 * pitch, y0
        else:
            i0, y0, x0 = present[0]
            if orient == 'h':
                pitch, base_y, base_x = h, y0 - i0 * h, x0
            else:
                pitch, base_x, base_y = w, x0 - i0 * w, y0
        res = np.zeros_like(inp)
        for idx, c in enumerate(seq):
            if orient == 'h':
                ty, tx = base_y + idx * pitch, base_x
            else:
                ty, tx = base_y, base_x + idx * pitch
            for (a, b) in template:
                yy, xx = ty + a, tx + b
                if 0 <= yy < n and 0 <= xx < m:
                    res[yy, xx] = c
        return res
    try:
        for i, o in train:
            r = solve(i)
            if r is None or r.shape != o.shape or not np.array_equal(r, o):
                return None
    except Exception:
        return None
    return solve


# ---------------------------------------------------------------------------
# shape_dict_recolor: a legend panel (top-left, cut off by a separator row and
# column of a single "wall" colour) holds several coloured template shapes. In
# the field, mono-colour shapes of a single "query" colour are recoloured to
# match the legend template with the same geometry.  (845d6e51)
# ---------------------------------------------------------------------------
def _norm_shape(cells):
    ys = [a for a, b in cells]
    xs = [b for a, b in cells]
    y0, x0 = min(ys), min(xs)
    return frozenset((a - y0, b - x0) for a, b in cells)


def _renorm(s):
    ys = [a for a, b in s]
    xs = [b for a, b in s]
    y0, x0 = min(ys), min(xs)
    return frozenset((a - y0, b - x0) for a, b in s)


def _dihedral(shape):
    res = set()
    s = shape
    for _ in range(4):
        s = _renorm(frozenset((c, -r) for r, c in s))  # rot90
        res.add(s)
        res.add(_renorm(frozenset((r, -c) for r, c in s)))  # + mirror
    return res


def _comps_color(g, col, region=None):
    n, m = g.shape
    seen = np.zeros((n, m), bool)
    res = []
    for i in range(n):
        for j in range(m):
            if region is not None and not region(i, j):
                continue
            if g[i, j] == col and not seen[i, j]:
                st = [(i, j)]
                seen[i, j] = True
                cells = []
                while st:
                    a, b = st.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if (0 <= na < n and 0 <= nb < m and not seen[na, nb]
                                and (region is None or region(na, nb)) and g[na, nb] == col):
                            seen[na, nb] = True
                            st.append((na, nb))
                res.append(cells)
    return res


def shape_dict_recolor(train):
    def build(inp):
        n, m = inp.shape
        # wall colour: the colour that forms a long horizontal run in one row.
        best = None
        for wc in np.unique(inp):
            if wc == 0:
                continue
            for r in range(n):
                cnt = int((inp[r] == wc).sum())
                if cnt >= 4 and (best is None or cnt > best[0]):
                    best = (cnt, int(wc), r)
        if best is None:
            return None
        _, wall, seprow = best
        # separator column: the column whose top portion (above seprow) is all wall
        sepcol = None
        for c in range(m):
            if seprow > 0 and (inp[:seprow, c] == wall).all():
                sepcol = c
                break
        if sepcol is None:
            return None
        r, c = seprow, sepcol
        # legend region: the quadrant above-left of the separators
        legcolors = set(int(v) for v in inp[:r, :c].flatten()) - {0, int(wall)}
        legmap = {}
        for col in legcolors:
            for cells in _comps_color(inp, col, region=lambda a, b, r=r, c=c: a < r and b < c):
                legmap[_norm_shape(cells)] = col
        if not legmap:
            return None
        # precompute dihedral variant -> colour, but only when unambiguous
        variant_map = {}
        for shp, col in legmap.items():
            for v in _dihedral(shp):
                if v in variant_map and variant_map[v] != col:
                    variant_map[v] = None  # ambiguous
                else:
                    variant_map.setdefault(v, col)
        return wall, r, c, legmap, variant_map

    info = build(train[0][0])
    if info is None:
        return None

    def solve(inp):
        b = build(inp)
        if b is None:
            return None
        wall, r, c, legmap, variant_map = b
        n, m = inp.shape
        res = inp.copy()
        legcols = set(legmap.values())
        qcolors = set(int(v) for v in np.unique(inp)) - {0, int(wall)} - legcols
        for qc in qcolors:
            for cells in _comps_color(inp, qc):
                if all(a < r and bb < c for a, bb in cells):
                    continue
                nm = _norm_shape(cells)
                tgt = legmap.get(nm)
                if tgt is None:
                    tgt = variant_map.get(nm)
                if tgt is not None:
                    for a, bb in cells:
                        res[a, bb] = tgt
        return res
    try:
        for i, o in train:
            rr = solve(i)
            if rr is None or rr.shape != o.shape or not np.array_equal(rr, o):
                return None
    except Exception:
        return None
    return solve


# ---------------------------------------------------------------------------
# diagonal_extend: the input holds a diagonally-periodic motif (translational
# symmetry along a shift vector (dr,dc)). The output is a larger grid (scaled by
# a learned factor) in which the motif block is tiled along that diagonal.
#   (cad67732)
# ---------------------------------------------------------------------------
def _diag_shift(inp):
    N, M = inp.shape
    cands = []
    for dr in range(1, N):
        for dc in range(-M + 1, M):
            if dc == 0:
                continue
            ok = True
            overlap = 0
            for r in range(N):
                for c in range(M):
                    r2, c2 = r - dr, c - dc
                    if 0 <= r2 < N and 0 <= c2 < M:
                        overlap += 1
                        if inp[r, c] != inp[r2, c2]:
                            ok = False
                            break
                if not ok:
                    break
            if ok and overlap >= N:
                cands.append((dr, dc, overlap))
    cands.sort(key=lambda x: (abs(x[0]) + abs(x[1])))
    return cands


def diagonal_extend(train):
    i0, o0 = train[0]
    if o0.shape[0] % i0.shape[0] or o0.shape[1] % i0.shape[1]:
        return None
    sh = o0.shape[0] // i0.shape[0]
    sw = o0.shape[1] // i0.shape[1]
    if sh < 2 or sw < 2 or sh > 4 or sw > 4:
        return None

    def solve(inp):
        N, M = inp.shape
        cands = _diag_shift(inp)
        if not cands:
            return None
        dr, dc, _ = cands[0]
        OH, OW = N * sh, M * sw
        out = np.zeros((OH, OW), int)
        ar = 0
        ac = 0 if dc > 0 else OW - M
        for k in range(-(OH + OW), OH + OW):
            rr = ar + k * dr
            cc = ac + k * dc
            for i in range(N):
                for j in range(M):
                    y, x = rr + i, cc + j
                    if 0 <= y < OH and 0 <= x < OW and inp[i, j] != 0:
                        out[y, x] = inp[i, j]
        return out
    try:
        for i, o in train:
            r = solve(i)
            if r is None or r.shape != o.shape or not np.array_equal(r, o):
                return None
    except Exception:
        return None
    return solve


# ---------------------------------------------------------------------------
# collapse_to_line: a straight line of colour 8 (partial run) splits the grid.
# Scattered single-colour markers on each side of the line collapse onto the
# cell immediately adjacent to the line, per row (vertical line) or per column
# (horizontal line), but only where the line actually exists; markers beyond the
# line's span are discarded.  (f83cb3f6)
# ---------------------------------------------------------------------------
def collapse_to_line(train):
    def solve(inp):
        n, m = inp.shape
        cols = set(int(v) for v in np.unique(inp)) - {0, 8}
        if len(cols) != 1:
            return None
        mk = cols.pop()
        rows8 = [r for r in range(n) if (inp[r] == 8).sum() >= 3]
        cols8 = [c for c in range(m) if (inp[:, c] == 8).sum() >= 3]
        out = np.zeros_like(inp)
        if len(cols8) == 1 and len(rows8) == 0:
            lc = cols8[0]
            if lc == 0 or lc == m - 1:
                return None
            for r in range(n):
                out[r, lc] = inp[r, lc]
                if inp[r, lc] != 8:
                    continue
                if (inp[r, :lc] == mk).any():
                    out[r, lc - 1] = mk
                if (inp[r, lc + 1:] == mk).any():
                    out[r, lc + 1] = mk
            return out
        if len(rows8) == 1 and len(cols8) == 0:
            lr = rows8[0]
            if lr == 0 or lr == n - 1:
                return None
            for c in range(m):
                out[lr, c] = inp[lr, c]
                if inp[lr, c] != 8:
                    continue
                if (inp[:lr, c] == mk).any():
                    out[lr - 1, c] = mk
                if (inp[lr + 1:, c] == mk).any():
                    out[lr + 1, c] = mk
            return out
        return None
    try:
        for i, o in train:
            r = solve(i)
            if r is None or r.shape != o.shape or not np.array_equal(r, o):
                return None
    except Exception:
        return None
    return solve


# ---------------------------------------------------------------------------
# sort_panels_by_count: the grid is a matrix of equal panels separated by all-0
# rows/columns. The panels are reordered by their count of a key colour, filling
# the panel matrix in a fixed traversal order. We learn the key colour, sort
# direction and traversal order from the demos.  (dc2aa30b)
# ---------------------------------------------------------------------------
def _panel_segments(g):
    n, m = g.shape
    seprows = [r for r in range(n) if (g[r] == 0).all()]
    sepcols = [c for c in range(m) if (g[:, c] == 0).all()]

    def segs(seps, tot):
        s = []
        prev = 0
        for x in seps + [tot]:
            if x > prev:
                s.append((prev, x))
            prev = x + 1
        return s
    return segs(seprows, n), segs(sepcols, m)


def sort_panels_by_count(train):
    rseg0, cseg0 = _panel_segments(train[0][0])
    R, C = len(rseg0), len(cseg0)
    if R * C < 4 or R < 2 or C < 2:
        return None
    colors = set()
    for i, o in train:
        colors |= set(int(v) for v in np.unique(i)) - {0}
    # candidate traversal orders of positions
    def make_orders(R, C):
        orders = {}
        rows_tb = list(range(R))
        rows_bt = list(range(R - 1, -1, -1))
        cols_lr = list(range(C))
        cols_rl = list(range(C - 1, -1, -1))
        for rn, rs in (('tb', rows_tb), ('bt', rows_bt)):
            for cn, cs in (('lr', cols_lr), ('rl', cols_rl)):
                orders['r' + rn + cn] = [(rr, cc) for rr in rs for cc in cs]
                orders['c' + cn + rn] = [(rr, cc) for cc in cs for rr in rs]
        return orders

    def build(inp, key, desc, posorder):
        rseg, cseg = _panel_segments(inp)
        if len(rseg) != R or len(cseg) != C:
            return None
        panels = []
        for (a, b) in rseg:
            for (c, d) in cseg:
                panels.append(((a, b, c, d), inp[a:b, c:d]))
        counts = [int((p[1] == key).sum()) for p in panels]
        if len(set(counts)) != len(counts):
            return None
        order = sorted(range(len(panels)), key=lambda k: counts[k], reverse=desc)
        out = inp.copy()
        for rank, pi in enumerate(order):
            rr, cc = posorder[rank]
            a, b = rseg[rr]
            c, d = cseg[cc]
            out[a:b, c:d] = panels[pi][1]
        return out

    orders = make_orders(R, C)
    for key in colors:
        for desc in (False, True):
            for oname, posorder in orders.items():
                ok = True
                for i, o in train:
                    try:
                        r = build(i, key, desc, posorder)
                    except Exception:
                        r = None
                    if r is None or r.shape != o.shape or not np.array_equal(r, o):
                        ok = False
                        break
                if ok:
                    kk, dd, pp = key, desc, posorder

                    def solve(inp, kk=kk, dd=dd, pp=pp):
                        return build(inp, kk, dd, pp)
                    return solve
    return None


# ---------------------------------------------------------------------------
# template_by_dominant_color: the output is a small fixed template selected by
# the most frequent (non-zero) colour in the input. We learn the colour->output
# mapping from the demos and apply it.  (9110e3c5)
# ---------------------------------------------------------------------------
def template_by_dominant_color(train):
    def dom(inp):
        vals, cnts = np.unique(inp, return_counts=True)
        best = None
        for v, c in zip(vals, cnts):
            if v == 0:
                continue
            if best is None or c > best[1]:
                best = (int(v), int(c))
        return None if best is None else best[0]

    mapping = {}
    outshape = train[0][1].shape
    for i, o in train:
        if o.shape != outshape:
            return None
        d = dom(i)
        if d is None:
            return None
        key = o.tobytes()
        if d in mapping and mapping[d][0] != key:
            return None
        mapping[d] = (key, o.copy())
    # require the mapping to be a genuine lookup (more than one distinct output OR
    # a clear color->template relation); avoid trivially firing on constant tasks
    if len(set(v[0] for v in mapping.values())) < 2:
        return None

    def solve(inp):
        d = dom(inp)
        if d in mapping:
            return mapping[d][1].copy()
        return None
    return solve


DETECTORS = [
    reverse_color_bands,
    frame_interior_largest,
    fill_frame_keep_marker_window,
    quadrant_overlay,
    concentric_ring_number,
    periodic_shape_recolor,
    legend_stack_template,
    shape_dict_recolor,
    diagonal_extend,
    collapse_to_line,
    sort_panels_by_count,
    template_by_dominant_color,
]
