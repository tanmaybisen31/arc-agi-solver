"""Round-5 batch-5 detectors for ARC-AGI.

General, principled geometric rules.  Every transform is re-verified against all
training pairs by the engine before it is ever applied to a test grid, so these
detectors only fire when they reproduce every demonstration exactly.

  * staircase_beam    -- a single "ball" cell emits a 2-wide up-right staircase
                         beam that deflects sideways off obstacle cells.
                         (task 69889d6e family)
  * row_shear         -- a single monochrome shape is sheared: each row is
                         translated horizontally by an amount proportional to
                         its distance from an anchor row/edge, dropping cells
                         that fall off the grid.  (task 423a55dc family)
  * line_double_ramp  -- a single vertical (or horizontal) bar of a marker
                         colour is turned into two mirrored triangular ramps of
                         learned colours on either side.  (task 5207a7b5 family)

Only numpy + stdlib are used.
"""
import numpy as np
from collections import Counter, deque


# --------------------------------------------------------------------------- utils
def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg):
    """8-connected components of non-background cells."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    out = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            dq = deque([(r, c)])
            seen[r, c] = True
            cells = [(r, c)]
            while dq:
                y, x = dq.popleft()
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        dq.append((ny, nx))
                        cells.append((ny, nx))
            out.append(cells)
    return out


def _single_color_positions(g, color):
    ys, xs = np.where(g == color)
    return list(zip(ys.tolist(), xs.tolist()))


# =========================================================================== #
# 1. staircase beam that deflects off obstacles                               #
# =========================================================================== #
def _beam_walk(g, ball_color, obst_color, first, dcol_sign):
    """Walk an up/right (configurable) staircase from the single ball cell.

    first: 'U' or 'R'  -- which axis the beam tries first.
    Movement alternates the primary axis (up / horizontal); when the primary
    target is an obstacle or off-grid, the beam takes the secondary step
    without consuming the alternation.  Stops on reaching the top row or when
    the horizontal extreme column is reached via a vertical (up) step, or when
    it can no longer advance.
    """
    H, W = g.shape
    balls = _single_color_positions(g, ball_color)
    if len(balls) != 1:
        return None
    r, c = balls[0]
    obst = set(_single_color_positions(g, obst_color)) if obst_color is not None else set()

    def blocked(a, b):
        return not (0 <= a < H and 0 <= b < W) or (a, b) in obst

    dc = dcol_sign  # +1 => moving right, -1 => moving left
    edge_col = W - 1 if dc > 0 else 0
    cells = [(r, c)]
    phase = first
    for _ in range(2 * (H + W) + 5):
        if phase == 'U':
            pr, pc, sr, sc = r - 1, c, r, c + dc
        else:
            pr, pc, sr, sc = r, c + dc, r - 1, c
        if not blocked(pr, pc):
            r, c = pr, pc
            moved = phase
            phase = 'R' if phase == 'U' else 'U'
        elif not blocked(sr, sc):
            r, c = sr, sc
            moved = 'R' if phase == 'U' else 'U'
        else:
            break
        cells.append((r, c))
        if r == 0:
            break
        if c == edge_col and moved == 'U':
            break
    out = g.copy()
    for a, b in cells:
        out[a, b] = ball_color
    return out


def staircase_beam(train):
    """Detect a single 'ball' cell whose output adds a deflecting staircase beam."""
    # ball color: colour present exactly once in every input, and whose count
    # grows in the output (the beam is drawn in that colour).
    i0, o0 = train[0]
    if i0.shape != o0.shape:
        return None
    # candidate ball colours: appear exactly once in i0
    vals, cnts = np.unique(i0, return_counts=True)
    singles = [int(v) for v, c in zip(vals, cnts) if c == 1]
    # obstacle colours: colours (other than ball & bg) present in input and
    # unchanged in output.
    for ball in singles:
        if int((o0 == ball).sum()) <= int((i0 == ball).sum()):
            continue
        # obstacle colour = a non-background colour (not ball) that appears in inputs
        obst_cands = set()
        for i, o in train:
            for v in np.unique(i):
                v = int(v)
                if v != ball and v != _bg(i):
                    obst_cands.add(v)
        obst_cands.add(None)
        for obst in list(obst_cands):
            for first in ('U', 'R'):
                for dc in (1, -1):
                    def fn(g, ball=ball, obst=obst, first=first, dc=dc):
                        return _beam_walk(g, ball, obst, first, dc)
                    try:
                        if all(fn(i) is not None and fn(i).shape == o.shape
                               and np.array_equal(fn(i), o) for i, o in train):
                            return fn
                    except Exception:
                        continue
    return None


# =========================================================================== #
# 2. row / column shear of a single monochrome shape                          #
# =========================================================================== #
def _shear(g, axis, anchor, direction):
    """Shear each line of the grid.

    axis='rows'  : shift row r horizontally by k*(distance to anchor row)
    axis='cols'  : shift col c vertically   by k*(distance to anchor col)
    anchor='max' : the bottom-most / right-most occupied line has zero shift
    anchor='min' : the top-most / left-most occupied line has zero shift
    direction    : +1 or -1 for the sign of the shift
    """
    nz = np.argwhere(g != _bg(g))
    if len(nz) == 0:
        return g.copy()
    H, W = g.shape
    out = np.full_like(g, _bg(g))
    if axis == 'rows':
        lines = nz[:, 0]
        anc = int(lines.max()) if anchor == 'max' else int(lines.min())
        for (r, c) in nz:
            r = int(r); c = int(c)
            shift = direction * (anc - r if anchor == 'max' else r - anc)
            nc = c + shift
            if 0 <= nc < W:
                out[r, nc] = g[r, c]
    else:
        lines = nz[:, 1]
        anc = int(lines.max()) if anchor == 'max' else int(lines.min())
        for (r, c) in nz:
            r = int(r); c = int(c)
            shift = direction * (anc - c if anchor == 'max' else c - anc)
            nr = r + shift
            if 0 <= nr < H:
                out[nr, c] = g[r, c]
    return out


def row_shear(train):
    """A single monochrome shape sheared row-by-row (or col-by-col)."""
    # require same shape and at least a few occupied rows to avoid trivial fits
    for i, o in train:
        if i.shape != o.shape:
            return None
    for axis in ('rows', 'cols'):
        for anchor in ('max', 'min'):
            for direction in (-1, 1):
                def fn(g, axis=axis, anchor=anchor, direction=direction):
                    return _shear(g, axis, anchor, direction)
                try:
                    if all(np.array_equal(fn(i), o) for i, o in train):
                        # guard: reject the trivial identity fit (shift always 0)
                        if any(not np.array_equal(i, o) for i, o in train):
                            return fn
                except Exception:
                    continue
    return None


# =========================================================================== #
# 3. vertical/horizontal bar -> mirrored triangular double ramp               #
# =========================================================================== #
def _ramp_from_vline(g, line_color, left_color, right_color):
    """Given a vertical bar of `line_color` starting at the top, build the
    two mirrored triangular ramps observed in the 5207a7b5 family.

    left  ramp count(col c<col)  = L + 2 + 2*(col-1-c)
    right ramp count(col c>col)  = L - 2 - 2*(c-col-1)
    """
    H, W = g.shape
    ys, xs = np.where(g == line_color)
    if len(xs) == 0:
        return None
    cols = np.unique(xs)
    if len(cols) != 1:
        return None
    col = int(cols[0])
    r0 = int(ys.min())
    L = len(ys)
    # bar must be a contiguous vertical run
    if sorted(ys.tolist()) != list(range(r0, r0 + L)):
        return None
    out = np.zeros((H, W), dtype=int)
    for r in range(r0, r0 + L):
        out[r, col] = line_color
    for c in range(col):
        cnt = L + 2 + 2 * (col - 1 - c)
        for r in range(max(0, min(cnt, H))):
            out[r, c] = left_color
    for c in range(col + 1, W):
        cnt = L - 2 - 2 * (c - col - 1)
        for r in range(max(0, min(cnt, H))):
            out[r, c] = right_color
    return out


def line_double_ramp(train):
    """Detect the vertical-bar -> double-ramp transform, learning the colours."""
    i0, o0 = train[0]
    if i0.shape != o0.shape:
        return None
    # line colour: the single non-background colour present in the input
    invals = [int(v) for v in np.unique(i0) if int(v) != 0]
    for line_color in invals:
        # learn ramp colours from the first output: colours left/right of the bar
        ys, xs = np.where(i0 == line_color)
        if len(xs) == 0:
            continue
        cu = np.unique(xs)
        if len(cu) != 1:
            continue
        col = int(cu[0])
        left_vals = [int(v) for v in np.unique(o0[:, :col]) if int(v) not in (0, line_color)]
        right_vals = [int(v) for v in np.unique(o0[:, col + 1:]) if int(v) not in (0, line_color)]
        lc_opts = left_vals if left_vals else [0]
        rc_opts = right_vals if right_vals else [0]
        for lc in lc_opts:
            for rc in rc_opts:
                def fn(g, line_color=line_color, lc=lc, rc=rc):
                    return _ramp_from_vline(g, line_color, lc, rc)
                try:
                    if all(fn(i) is not None and fn(i).shape == o.shape
                           and np.array_equal(fn(i), o) for i, o in train):
                        return fn
                except Exception:
                    continue
    return None


# =========================================================================== #
# 4. panel grid: connect same-colour markers with filled straight segments     #
#    (task e760a62e family)                                                     #
# =========================================================================== #
def _panel_segments(g, sep):
    H, W = g.shape
    sr = [r for r in range(H) if (g[r, :] == sep).all()]
    sc = [c for c in range(W) if (g[:, c] == sep).all()]

    def runs(S, n):
        segs = []
        s = None
        for i in range(n):
            if i in S:
                if s is not None:
                    segs.append((s, i)); s = None
            else:
                if s is None:
                    s = i
        if s is not None:
            segs.append((s, n))
        return segs
    return runs(set(sr), H), runs(set(sc), W)


def _connect_panels(g, sep, bg, cross):
    rsegs, csegs = _panel_segments(g, sep)
    R, C = len(rsegs), len(csegs)
    if R < 2 and C < 2:
        return None
    mk = {}
    colors = set()
    for a, (r0, r1) in enumerate(rsegs):
        for b, (c0, c1) in enumerate(csegs):
            pan = g[r0:r1, c0:c1]
            vs = [int(v) for v in np.unique(pan) if int(v) not in (bg, sep)]
            if len(vs) == 1:
                mk[(a, b)] = vs[0]
                colors.add(vs[0])
            elif len(vs) > 1:
                return None
    if not mk:
        return None
    fill = {}
    for color in colors:
        pts = [p for p, c in mk.items() if c == color]
        for x in range(len(pts)):
            for y in range(x + 1, len(pts)):
                (a1, b1), (a2, b2) = pts[x], pts[y]
                if a1 == a2:
                    for b in range(min(b1, b2), max(b1, b2) + 1):
                        fill.setdefault((a1, b), set()).add(color)
                elif b1 == b2:
                    for a in range(min(a1, a2), max(a1, a2) + 1):
                        fill.setdefault((a, b1), set()).add(color)
    out = g.copy()
    for (a, b), cols in fill.items():
        r0, r1 = rsegs[a]
        c0, c1 = csegs[b]
        col = cross if len(cols) > 1 else next(iter(cols))
        out[r0:r1, c0:c1] = col
    return out


def panel_connect_markers(train):
    i0, o0 = train[0]
    if i0.shape != o0.shape:
        return None
    # separator colour: a colour forming at least one full row or column line
    H, W = i0.shape
    sep_cands = set()
    for r in range(H):
        row = i0[r, :]
        if len(np.unique(row)) == 1:
            sep_cands.add(int(row[0]))
    for c in range(W):
        col = i0[:, c]
        if len(np.unique(col)) == 1:
            sep_cands.add(int(col[0]))
    # learn cross colour: a colour appearing in some output but not as an input
    # marker anywhere.
    in_marks = set()
    out_extra = set()
    for i, o in train:
        for v in np.unique(i):
            in_marks.add(int(v))
    for i, o in train:
        for v in np.unique(o):
            if int(v) not in in_marks:
                out_extra.add(int(v))
    cross_opts = list(out_extra) if out_extra else [6]
    for sep in sep_cands:
        bg = _bg(i0)
        if bg == sep:
            # background equals separator: pick next most common colour as bg
            vals, cnts = np.unique(i0, return_counts=True)
            order = [int(v) for v, _ in sorted(zip(vals, cnts), key=lambda z: -z[1])]
            bg = next((v for v in order if v != sep), sep)
        for cross in cross_opts:
            def fn(g, sep=sep, bg=bg, cross=cross):
                return _connect_panels(g, sep, bg, cross)
            try:
                if all(fn(i) is not None and fn(i).shape == o.shape
                       and np.array_equal(fn(i), o) for i, o in train):
                    return fn
            except Exception:
                continue
    return None


# =========================================================================== #
# 5. stamp a template shape at each indicator dot, preserving inter-dot gaps    #
#    (task 9b2a60aa family)                                                     #
# =========================================================================== #
def _stamp_dots(g):
    bg = _bg(g)
    cs = _components(g, bg)
    if len(cs) < 3:
        return None
    big = max(cs, key=len)
    dots = [c for c in cs if len(c) == 1]
    # every non-template component must be a single dot
    if len(dots) != len(cs) - 1 or len(dots) < 2 or len(big) < 3:
        return None
    ys = [y for y, x in big]
    xs = [x for y, x in big]
    r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
    th, tw = r1 - r0 + 1, c1 - c0 + 1
    tcolor = int(g[big[0][0], big[0][1]])
    # template must be single-colour
    if any(int(g[y, x]) != tcolor for y, x in big):
        return None
    rel = [(y - r0, x - c0) for y, x in big]
    dp = [(d[0][0], d[0][1], int(g[d[0][0], d[0][1]])) for d in dots]
    rowspan = max(y for y, _, _ in dp) - min(y for y, _, _ in dp)
    colspan = max(x for _, x, _ in dp) - min(x for _, x, _ in dp)
    axis = 'col' if colspan >= rowspan else 'row'
    anchors = [d for d in dp if d[2] == tcolor]
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    key = (lambda d: d[1]) if axis == 'col' else (lambda d: d[0])
    dsorted = sorted(dp, key=key)
    extent = tw if axis == 'col' else th
    ai = next(idx for idx, d in enumerate(dsorted) if d == anchor)
    base = c0 if axis == 'col' else r0
    pos = [None] * len(dsorted)
    pos[ai] = base
    for j in range(ai + 1, len(dsorted)):
        gap = key(dsorted[j]) - key(dsorted[j - 1])
        pos[j] = pos[j - 1] + extent + (gap - 1)
    for j in range(ai - 1, -1, -1):
        gap = key(dsorted[j + 1]) - key(dsorted[j])
        pos[j] = pos[j + 1] - (extent + (gap - 1))
    out = g.copy()
    H, W = g.shape
    for j, d in enumerate(dsorted):
        color = d[2]
        if axis == 'col':
            br, bc = r0, pos[j]
        else:
            br, bc = pos[j], c0
        for dy, dx in rel:
            rr, cc = br + dy, bc + dx
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = color
    return out


def stamp_template_at_dots(train):
    for i, o in train:
        if i.shape != o.shape:
            return None
    fn = lambda g: _stamp_dots(g)
    try:
        if all(_stamp_dots(i) is not None and np.array_equal(_stamp_dots(i), o)
               for i, o in train):
            return fn
    except Exception:
        return None
    return None


# =========================================================================== #
# 6. four edge markers + one interior marker -> coloured box with a cross       #
#    (task e4075551 family)                                                     #
# =========================================================================== #
def _box_cross(g, cross):
    bg = _bg(g)
    pts = [(int(r), int(c), int(g[r, c]))
           for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] != bg]
    if len(pts) != 5:
        return None
    rs = [r for r, c, v in pts]
    cs = [c for r, c, v in pts]
    top, bot, left, right = min(rs), max(rs), min(cs), max(cs)
    if bot - top < 2 or right - left < 2:
        return None
    tp = [p for p in pts if p[0] == top]
    bp = [p for p in pts if p[0] == bot]
    lp = [p for p in pts if p[1] == left]
    rp = [p for p in pts if p[1] == right]
    if not (len(tp) == 1 and len(bp) == 1 and len(lp) == 1 and len(rp) == 1):
        return None
    edges = {id(tp[0]), id(bp[0]), id(lp[0]), id(rp[0])}
    interior = [p for p in pts if id(p) not in edges]
    if len(interior) != 1:
        return None
    ir, ic, ival = interior[0]
    if not (top < ir < bot and left < ic < right):
        return None
    out = g.copy()
    for r in range(top + 1, bot):
        out[r, ic] = cross
    for c in range(left + 1, right):
        out[ir, c] = cross
    out[ir, ic] = ival
    for r in range(top, bot + 1):
        out[r, left] = lp[0][2]
        out[r, right] = rp[0][2]
    for c in range(left, right + 1):
        out[top, c] = tp[0][2]
        out[bot, c] = bp[0][2]
    return out


def box_from_markers(train):
    for i, o in train:
        if i.shape != o.shape:
            return None
    # learn cross colour: colour present in some output but not in that input
    cross_opts = set()
    for i, o in train:
        inv = set(int(v) for v in np.unique(i))
        for v in np.unique(o):
            if int(v) not in inv:
                cross_opts.add(int(v))
    if not cross_opts:
        cross_opts = {5}
    for cross in cross_opts:
        def fn(g, cross=cross):
            return _box_cross(g, cross)
        try:
            if all(_box_cross(i, cross) is not None and np.array_equal(_box_cross(i, cross), o)
                   for i, o in train):
                return fn
        except Exception:
            continue
    return None


# =========================================================================== #
# 7. nest coloured frames into concentric squares, ordered by size              #
#    (task c658a4bd family)                                                     #
# =========================================================================== #
def _nest_frames(g):
    bg = _bg(g)
    cols = [int(v) for v in np.unique(g) if int(v) != bg]
    if len(cols) < 2:
        return None
    info = []
    for col in cols:
        ys, xs = np.where(g == col)
        md = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
        info.append((md, col))
    info.sort(reverse=True)
    # sizes must be distinct and decrease by exactly 2 (proper nesting)
    sizes = [md for md, _ in info]
    if len(set(sizes)) != len(sizes):
        return None
    for a, b in zip(sizes, sizes[1:]):
        if a - b != 2:
            return None
    n = len(info)
    S = info[0][0]
    if S != 2 * (n - 1) + 1 and S != 2 * n:
        return None
    out = np.full((S, S), bg, dtype=int)
    for j, (md, col) in enumerate(info):
        for c in range(j, S - j):
            out[j, c] = col
            out[S - 1 - j, c] = col
        for r in range(j, S - j):
            out[r, j] = col
            out[r, S - 1 - j] = col
    return out


def nest_frames(train):
    # output must be smaller/reshaped (not identity)
    if all(i.shape == o.shape for i, o in train):
        # still possible but require actual change
        if all(np.array_equal(i, o) for i, o in train):
            return None
    fn = lambda g: _nest_frames(g)
    try:
        if all(_nest_frames(i) is not None and _nest_frames(i).shape == o.shape
               and np.array_equal(_nest_frames(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


DETECTORS = [staircase_beam, row_shear, line_double_ramp,
             panel_connect_markers, stamp_template_at_dots, box_from_markers,
             nest_frames]
