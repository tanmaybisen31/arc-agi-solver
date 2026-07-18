"""Round 4 batch 3 detectors.

Each detector: def det(train) -> transform_fn | None
Engine verifies the returned fn reproduces every training pair exactly.
numpy + stdlib only. Defensive: never raise, return None when unsure.
"""
import numpy as np
from collections import Counter


# ---------------- helpers ----------------
def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _nonzero_pts(g, bg=0):
    H, W = g.shape
    return [(r, c, int(g[r, c])) for r in range(H) for c in range(W) if g[r, c] != bg]


def _components(g, bg, diag=False):
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


# ================================================================
# 45bbe264 : each isolated dot casts its color along its full row and
# full column; where two different dots' cross-lines intersect the cell
# takes a fixed "cross" color; the dot's own crossing keeps the dot color.
# ================================================================
def crosshair_project(train):
    # learn cross color and background
    bg = 0
    # every input must have all nonzero cells being isolated single dots
    cross_candidates = None
    for a, b in train:
        if a.shape != b.shape:
            return None
        pts = _nonzero_pts(a, bg)
        if len(pts) < 1:
            return None
        rows = set(r for r, c, v in pts)
        cols = set(c for r, c, v in pts)
        # dots must have distinct rows and distinct cols (so lines are well-defined)
        if len(rows) != len(pts) or len(cols) != len(pts):
            return None
        # figure the cross color from an intersection of two different dots
        rowcol = {}
        colcol = {}
        for r, c, v in pts:
            rowcol[r] = v
            colcol[c] = v
        found = None
        for r1, c1, v1 in pts:
            for r2, c2, v2 in pts:
                if (r1, c1) == (r2, c2):
                    continue
                found = int(b[r1, c2])
                break
            if found is not None:
                break
        if found is None:
            return None
        if cross_candidates is None:
            cross_candidates = found
        elif cross_candidates != found:
            return None
    cross = cross_candidates
    if cross is None:
        return None

    def fn(g):
        H, W = g.shape
        pts = _nonzero_pts(g, bg)
        if not pts:
            return g.copy()
        rows = {}
        cols = {}
        for r, c, v in pts:
            rows[r] = v
            cols[c] = v
        out = np.full((H, W), bg, dtype=int)
        for r in range(H):
            for c in range(W):
                inr = r in rows
                inc = c in cols
                if inr and inc:
                    out[r, c] = cross
                elif inr:
                    out[r, c] = rows[r]
                elif inc:
                    out[r, c] = cols[c]
        for r, c, v in pts:
            out[r, c] = v
        return out

    return fn


# ================================================================
# e5790162 : a "source" dot emits a beam moving in a fixed start dir,
# drawing a path color; two marker colors deflect it clockwise / ccw.
# Learn source color, path color, marker colors and start dir from data.
# ================================================================
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N,E,S,W


def _beam_sim(a, src_color, path_color, cw, ccw, start_dir, bg=0):
    H, W = a.shape
    srcs = [(r, c) for r in range(H) for c in range(W) if a[r, c] == src_color]
    if len(srcs) != 1:
        return None
    out = a.copy()
    r, c = srcs[0]
    d = start_dir
    for _ in range(2 * (H + W) + 10):
        dr, dc = _DIRS[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            break
        v = int(a[nr, nc])
        if v == cw:
            d = (d + 1) % 4
            continue
        if v == ccw:
            d = (d - 1) % 4
            continue
        if v != bg:
            break
        r, c = nr, nc
        if out[r, c] == bg:
            out[r, c] = path_color
    return out


def beam_deflect(train):
    bg = 0
    # collect all colors
    colors = set()
    for a, b in train:
        if a.shape != b.shape:
            return None
        colors |= set(np.unique(a).tolist())
        colors |= set(np.unique(b).tolist())
    colors.discard(bg)
    colors = sorted(colors)
    if not colors:
        return None
    # source color: appears once per input, and path color = a color that grows in output
    # Try all reasonable assignments.
    # Candidate source colors: colors appearing exactly once in each input
    def count_once(col):
        for a, b in train:
            if int((a == col).sum()) != 1:
                return False
        return True
    src_cands = [c for c in colors if count_once(c)]
    # path color: the color whose count increases most from in->out
    inc = {}
    for c in colors:
        d = 0
        for a, b in train:
            d += int((b == c).sum()) - int((a == c).sum())
        inc[c] = d
    path_cands = sorted([c for c in colors if inc[c] > 0], key=lambda c: -inc[c])
    if not path_cands:
        path_cands = colors
    marker_cands = colors
    for src in (src_cands or colors):
        for path in path_cands:
            for cw in marker_cands:
                for ccw in marker_cands:
                    if cw == ccw:
                        continue
                    if cw in (src,) or ccw in (src,):
                        continue
                    for sd in range(4):
                        ok = True
                        for a, b in train:
                            o = _beam_sim(a, src, path, cw, ccw, sd, bg)
                            if o is None or not np.array_equal(o, b):
                                ok = False
                                break
                        if ok:
                            return (lambda g, s=src, p=path, x=cw, y=ccw, d=sd:
                                    (_beam_sim(g, s, p, x, y, d, bg)
                                     if _beam_sim(g, s, p, x, y, d, bg) is not None else g.copy()))
    return None


# ================================================================
# e872b94a : output is a single column of (num_objects + k) background
# cells. Learn the offset k from the object count relation.
# ================================================================
def count_column(train):
    bg = None
    # all outputs must be width-1 and constant value
    outvals = set()
    for a, b in train:
        if b.shape[1] != 1:
            return None
        vals = set(np.unique(b).tolist())
        if len(vals) != 1:
            return None
        outvals |= vals
    if len(outvals) != 1:
        return None
    outval = outvals.pop()
    # try connectivity 4 and 8, several background choices, and offset
    for diag in (False, True):
        for bgmode in ("common", "zero"):
            offs = set()
            ok = True
            for a, b in train:
                bgc = _bg(a) if bgmode == "common" else 0
                n = len(_components(a, bgc, diag))
                off = b.shape[0] - n
                offs.add(off)
            if len(offs) == 1:
                off = offs.pop()

                def fn(g, diag=diag, bgmode=bgmode, off=off, outval=outval):
                    bgc = _bg(g) if bgmode == "common" else 0
                    n = len(_components(g, bgc, diag))
                    h = max(1, n + off)
                    return np.full((h, 1), outval, dtype=int)

                # verify
                if all(np.array_equal(fn(a), b) for a, b in train):
                    return fn
    return None


# ================================================================
# 5289ad53 : output is a small fixed-size grid acting as a histogram.
# Count objects per color; fill the grid row-major with each color
# repeated by its count, colors in descending order, pad with 0.
# ================================================================
def object_histogram(train):
    # output shape constant
    shapes = set(b.shape for _, b in train)
    if len(shapes) != 1:
        return None
    OH, OW = shapes.pop()
    cap = OH * OW
    if cap < 1 or cap > 30:
        return None

    def counts(a):
        bg = _bg(a)
        cs = _components(a, bg, False)
        cc = Counter()
        for cells in cs:
            cc[int(a[cells[0][0], cells[0][1]])] += 1
        return cc

    # determine order: try descending color, ascending color, by-count-desc
    def build(a, order):
        cc = counts(a)
        cols = sorted(cc.keys())
        if order == "cdesc":
            cols = sorted(cc.keys(), reverse=True)
        elif order == "casc":
            cols = sorted(cc.keys())
        elif order == "ndesc":
            cols = sorted(cc.keys(), key=lambda k: (-cc[k], -k))
        seq = []
        for col in cols:
            seq += [col] * cc[col]
        if len(seq) > cap:
            return None
        seq = seq + [0] * (cap - len(seq))
        return np.array(seq, dtype=int).reshape(OH, OW)

    for order in ("cdesc", "casc", "ndesc"):
        ok = True
        for a, b in train:
            o = build(a, order)
            if o is None or not np.array_equal(o, b):
                ok = False
                break
        if ok:
            return (lambda g, order=order: build(g, order)
                    if build(g, order) is not None
                    else np.zeros((OH, OW), dtype=int))
    return None


# ================================================================
# 7c9b52a0 : several equal-sized panels (connected non-bg regions with
# an internal 0-background) get overlaid into one panel; non-zero cells
# stack (later region wins).
# ================================================================
def panel_overlay_regions(train):
    # output shape must equal the panel (comp bbox) size
    def panels(a):
        bg = _bg(a)
        cs = _components(a, bg, False)
        subs = []
        for cells in cs:
            ys = [y for y, x in cells]
            xs = [x for y, x in cells]
            r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
            subs.append(a[r0:r1 + 1, c0:c1 + 1].copy())
        return subs

    def build(a):
        subs = panels(a)
        if len(subs) < 2:
            return None
        shapes = set(s.shape for s in subs)
        if len(shapes) != 1:
            return None
        H, W = subs[0].shape
        out = np.zeros((H, W), dtype=int)
        for s in subs:
            m = s != 0
            out[m] = s[m]
        return out

    for a, b in train:
        o = build(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return None
    return (lambda g: build(g) if build(g) is not None else g.copy())


# ================================================================
# b1fc8b8e : count total marker cells N; produce a small grid whose 4
# corners each hold a 2x2 tile filled with N/4 markers (in a canonical
# fill order) separated by a blank cross.
# ================================================================
def _tile_fill(k, color):
    t = np.zeros((2, 2), dtype=int)
    order = [(1, 1), (1, 0), (0, 1), (0, 0)]
    for i in range(min(k, 4)):
        t[order[i]] = color
    return t


def quad_tile_count(train):
    # outputs are 5x5-ish with blank separator row/col in the middle
    shapes = set(b.shape for _, b in train)
    if len(shapes) != 1:
        return None
    OH, OW = shapes.pop()
    if OH != 5 or OW != 5:
        return None
    # marker color: single non-bg color in input
    def marker(a):
        vals = [v for v in np.unique(a).tolist() if v != 0]
        return vals[0] if len(vals) == 1 else None

    def build(a):
        col = marker(a)
        if col is None:
            return None
        n = int((a == col).sum())
        if n % 4 != 0:
            return None
        k = n // 4
        if k < 1 or k > 4:
            return None
        tile = _tile_fill(k, col)
        out = np.zeros((5, 5), dtype=int)
        out[0:2, 0:2] = tile
        out[0:2, 3:5] = tile
        out[3:5, 0:2] = tile
        out[3:5, 3:5] = tile
        return out

    for a, b in train:
        o = build(a)
        if o is None or not np.array_equal(o, b):
            return None
    return (lambda g: build(g) if build(g) is not None else np.zeros((5, 5), dtype=int))


# ================================================================
# ea959feb : the grid is a noisy periodic tiling; restore the clean
# periodicity by majority-voting each tile position.  The smallest
# period (ph,pw) whose per-position majority is strong and whose
# reconstruction agrees with most of the input is chosen.
# ================================================================
def _period_reconstruct(a, ph, pw):
    H, W = a.shape
    tile = np.zeros((ph, pw), dtype=int)
    worst = 1.0
    for i in range(ph):
        for j in range(pw):
            vals = [int(a[r, c]) for r in range(i, H, ph) for c in range(j, W, pw)]
            cc = Counter(vals)
            best, cnt = cc.most_common(1)[0]
            tile[i, j] = best
            worst = min(worst, cnt / len(vals))
    reps_h = H // ph + 2
    reps_w = W // pw + 2
    rec = np.tile(tile, (reps_h, reps_w))[:H, :W]
    agree = int((rec == a).sum()) / a.size
    return rec, worst, agree


def _best_period(a, wthr=0.6, athr=0.85):
    H, W = a.shape
    best = None
    for ph in range(1, H + 1):
        for pw in range(1, W + 1):
            if ph == H and pw == W:
                continue
            rec, worst, agree = _period_reconstruct(a, ph, pw)
            if worst >= wthr and agree >= athr:
                if best is None or ph * pw < best[0]:
                    best = (ph * pw, rec)
    return best[1] if best else None


def periodic_repair(train):
    if any(a.shape != b.shape for a, b in train):
        return None
    # must actually be a repair (some cells differ) at least once
    if all(np.array_equal(a, b) for a, b in train):
        return None
    for a, b in train:
        rec = _best_period(a)
        if rec is None or not np.array_equal(rec, b):
            return None
    return (lambda g: (_best_period(g) if _best_period(g) is not None else g.copy()))


# ================================================================
# cb227835 : two dots of a fixed color; connect them with two staircase
# paths (a "lens"): one goes straight-along-major-axis then diagonal,
# the other diagonal then straight.  Fill with a path color.
# ================================================================
def _lens_paths(a, dotcol, pathcol):
    H, W = a.shape
    pts = [(r, c) for r in range(H) for c in range(W) if a[r, c] == dotcol]
    if len(pts) != 2:
        return None
    (r1, c1), (r2, c2) = pts
    dr = r2 - r1
    dc = c2 - c1
    sr = 1 if dr >= 0 else -1
    sc = 1 if dc >= 0 else -1
    adr = abs(dr)
    adc = abs(dc)
    ndiag = min(adr, adc)
    nstr = max(adr, adc) - ndiag
    major = 'R' if adr >= adc else 'C'
    out = a.copy()

    def walk(diag_first):
        r, c = r1, c1
        cells = []
        seq = (['G'] * ndiag + ['S'] * nstr) if diag_first else (['S'] * nstr + ['G'] * ndiag)
        for mv in seq:
            if mv == 'G':
                r += sr
                c += sc
            elif major == 'R':
                r += sr
            else:
                c += sc
            cells.append((r, c))
        return cells[:-1]

    for pa in (walk(False), walk(True)):
        for r, c in pa:
            if 0 <= r < H and 0 <= c < W and out[r, c] == 0:
                out[r, c] = pathcol
    return out


def lens_connect(train):
    # infer dot color (appears exactly twice, same in every train) and path color
    dot_cands = None
    for a, b in train:
        if a.shape != b.shape:
            return None
        cnt = Counter(v for v in a.flatten().tolist() if v != 0)
        twos = {c for c, n in cnt.items() if n == 2}
        dot_cands = twos if dot_cands is None else (dot_cands & twos)
    if not dot_cands:
        return None
    # path color = color that appears in output but not input (grows)
    path_cands = set()
    for a, b in train:
        newcols = set(np.unique(b).tolist()) - set(np.unique(a).tolist())
        path_cands = newcols if not path_cands else (path_cands & newcols)
    for dot in sorted(dot_cands):
        pcands = sorted(path_cands) if path_cands else [c for c in range(1, 10) if c != dot]
        for path in pcands:
            ok = True
            for a, b in train:
                o = _lens_paths(a, dot, path)
                if o is None or not np.array_equal(o, b):
                    ok = False
                    break
            if ok:
                return (lambda g, d=dot, p=path:
                        (_lens_paths(g, d, p) if _lens_paths(g, d, p) is not None else g.copy()))
    return None


DETECTORS = [
    crosshair_project,
    beam_deflect,
    count_column,
    object_histogram,
    panel_overlay_regions,
    quad_tile_count,
    periodic_repair,
    lens_connect,
]
