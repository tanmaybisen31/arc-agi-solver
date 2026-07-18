"""Round-2 batch-4 detectors for ARC-AGI eval tasks.

Each detector: def det(train) -> transform_fn | None
Only numpy + stdlib. Defensive. Engine verifies every train pair reproduces.
"""
import numpy as np
from collections import Counter, defaultdict


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag=True):
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
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


def _color_components(g, bg, diag=True):
    """Connected components where cells must share the SAME color."""
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1)]
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            color = g[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


# ============================================================
# self_tile : output = input tiled (H copies vertically, W copies
# horizontally) where each "cell" of the tiling is the whole input.
# Actually the classic form: output shape = (H*H, W*W); the input is
# tiled H x W times.  Generalizes because the tiling factor is derived
# from the input's own dimensions, not a fixed train amplification.
# ============================================================
def self_tile(train):
    def fn(g):
        H, W = g.shape
        return np.tile(g, (H, W))
    try:
        if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ============================================================
# concentric_rings : a hollow rectangular box (single color outline,
# interior = background).  Fill the interior with concentric rings.
# The ring color sequence is learned from the training outputs (the
# colors encountered peeling from outside inward).  Handles both the
# "alternating 2 colors" case and the "key sequence" case.
# ============================================================
def _find_box(g, bg):
    """Return (r0, c0, r1, c1, color) of a hollow rectangle outline, or None."""
    comps = _components(g, bg, diag=False)
    best = None
    for cells in comps:
        ys = [y for y, x in cells]; xs = [x for y, x in cells]
        r0, r1 = min(ys), max(ys); c0, c1 = min(xs), max(xs)
        h = r1 - r0 + 1; w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        colors = set(int(g[y, x]) for y, x in cells)
        if len(colors) != 1:
            continue
        col = colors.pop()
        cellset = set(cells)
        # must be a full outline rectangle, interior all bg
        outline_ok = True
        for y in range(r0, r1+1):
            for x in range(c0, c1+1):
                on_edge = (y == r0 or y == r1 or x == c0 or x == c1)
                if on_edge:
                    if (y, x) not in cellset:
                        outline_ok = False
                if not on_edge:
                    if g[y, x] != bg:
                        outline_ok = False
            if not outline_ok:
                break
        if outline_ok:
            area = h * w
            if best is None or area > best[5]:
                best = (r0, c0, r1, c1, col, area)
    if best is None:
        return None
    return best[:5]


def _ring_colors_from_output(inp, out, bg):
    box = _find_box(inp, bg)
    if box is None:
        return None
    r0, c0, r1, c1, col = box
    seq = []
    k = 0
    while r0 + k <= r1 - k and c0 + k <= c1 - k:
        # ring k: cells at distance k from box border
        yy0, yy1 = r0 + k, r1 - k
        xx0, xx1 = c0 + k, c1 - k
        # sample a border cell of this ring in output
        val = int(out[yy0, xx0])
        seq.append(val)
        k += 1
    return box, seq


def concentric_rings(train):
    bg = _bg(train[0][0])
    parsed = []
    for i, o in train:
        r = _ring_colors_from_output(i, o, bg)
        if r is None:
            return None
        parsed.append(r)
    # Determine a rule for the ring color sequence.
    # Case A: fixed sequence of colors (same for all boxes) -> use directly,
    #         extended by repeating the tail-2 alternation if box is bigger.
    seqs = [seq for (_box, seq) in parsed]

    def build_fn(seq_rule):
        def fn(g):
            b = _find_box(g, bg)
            if b is None:
                return g.copy()
            r0, c0, r1, c1, col = b
            out = g.copy()
            k = 0
            while r0 + k <= r1 - k and c0 + k <= c1 - k:
                color = seq_rule(k, col)
                yy0, yy1 = r0 + k, r1 - k
                xx0, xx1 = c0 + k, c1 - k
                out[yy0, xx0:xx1+1] = color
                out[yy1, xx0:xx1+1] = color
                out[yy0:yy1+1, xx0] = color
                out[yy0:yy1+1, xx1] = color
                k += 1
            return out
        return fn

    # try: fixed literal sequence (max length), with alternation fallback
    maxlen = max(len(s) for s in seqs)
    # consistency: for each ring index k that appears in >=1 seq, all agree
    ring_color = {}
    consistent = True
    for s in seqs:
        for k, v in enumerate(s):
            if k in ring_color and ring_color[k] != v:
                consistent = False
                break
            ring_color[k] = v
        if not consistent:
            break
    if consistent:
        longest = max(seqs, key=len)
        # detect the minimal repeating period of `longest`
        def min_period(seq):
            n = len(seq)
            for p in range(1, n + 1):
                if all(seq[i] == seq[i % p] for i in range(n)):
                    return p
            return n
        # Try both: (a) pure periodic sequence, (b) per-index literal then
        # periodic tail using the detected period.
        candidates = []
        if longest:
            p = min_period(longest)
            base = longest[:p]
            def seq_rule_periodic(k, col, base=base):
                return base[k % len(base)]
            candidates.append(seq_rule_periodic)
        # per-index literal, extend by whole-sequence period
        if longest:
            p = min_period(longest)
            def seq_rule_literal(k, col, longest=longest, p=p):
                if k < len(longest):
                    return longest[k]
                return longest[k % p]
            candidates.append(seq_rule_literal)
        for sr in candidates:
            fn = build_fn(sr)
            try:
                if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
                    return fn
            except Exception:
                pass

    # Case B: sequence derived from a key column/diagonal of colored cells
    # near the box.  Ring[0]=box color, ring[k]=key[k-1].
    def find_key(g, bg, box_color):
        # collect isolated colored cells (not part of the big box)
        b = _find_box(g, bg)
        if b is None:
            return None
        r0, c0, r1, c1, col = b
        pts = []
        H, W = g.shape
        for y in range(H):
            for x in range(W):
                v = int(g[y, x])
                if v == bg:
                    continue
                # skip cells belonging to the box outline
                if r0 <= y <= r1 and c0 <= x <= c1:
                    on_edge = (y == r0 or y == r1 or x == c0 or x == c1)
                    if on_edge and v == col:
                        continue
                pts.append((y, x, v))
        if not pts:
            return None
        # order points by distance from top-left corner (nearest first)
        pts.sort(key=lambda p: (p[0] + p[1], p[0], p[1]))
        return [col] + [v for _, _, v in pts]

    def seq_rule_key_factory(g):
        keyseq = find_key(g, bg, None)
        return keyseq

    def fn_key(g):
        b = _find_box(g, bg)
        if b is None:
            return g.copy()
        r0, c0, r1, c1, col = b
        keyseq = find_key(g, bg, col)
        if keyseq is None:
            return g.copy()
        out = g.copy()
        k = 0
        while r0 + k <= r1 - k and c0 + k <= c1 - k:
            color = keyseq[k] if k < len(keyseq) else keyseq[-1]
            yy0, yy1 = r0 + k, r1 - k
            xx0, xx1 = c0 + k, c1 - k
            out[yy0, xx0:xx1+1] = color
            out[yy1, xx0:xx1+1] = color
            out[yy0:yy1+1, xx0] = color
            out[yy0:yy1+1, xx1] = color
            k += 1
        return out
    try:
        if all(fn_key(i).shape == o.shape and np.array_equal(fn_key(i), o) for i, o in train):
            return fn_key
    except Exception:
        pass
    return None


# ============================================================
# gravity_to_wall : a full row or column of one color acts as a wall.
# All other non-bg cells slide toward that wall (per perpendicular line),
# stacking and preserving order.  Wall stays fixed.
# ============================================================
def _full_lines(g):
    """Return list of (kind, index, color) for rows/cols entirely one non-bg color."""
    H, W = g.shape
    res = []
    for r in range(H):
        vals = set(int(x) for x in g[r])
        if len(vals) == 1:
            c = vals.pop()
            res.append(("row", r, c))
    for c in range(W):
        vals = set(int(x) for x in g[:, c])
        if len(vals) == 1:
            cc = vals.pop()
            res.append(("col", c, cc))
    return res


def gravity_to_wall(train):
    bg = _bg(train[0][0])

    def detect_wall(g):
        H, W = g.shape
        best = None
        for kind, idx, col in _full_lines(g):
            if col == bg:
                continue
            # must be an edge line
            if kind == "row" and idx in (0, H - 1):
                d = "down" if idx == H - 1 else "up"
                best = (kind, idx, col, d)
            elif kind == "col" and idx in (0, W - 1):
                d = "right" if idx == W - 1 else "left"
                best = (kind, idx, col, d)
        return best

    DVEC = {"down": (1, 0), "up": (-1, 0), "right": (0, 1), "left": (0, -1)}

    def fn(g):
        H, W = g.shape
        w = detect_wall(g)
        if w is None:
            return g.copy()
        kind, idx, col, d = w
        dy, dx = DVEC[d]
        # gather objects excluding the wall line
        gg = g.copy()
        if kind == "row":
            wallset = set((idx, c) for c in range(W))
        else:
            wallset = set((r, idx) for r in range(H))
        # mask out wall from component detection
        gmask = g.copy()
        for (y, x) in wallset:
            gmask[y, x] = bg
        comps = _color_components(gmask, bg, diag=True)
        # occupancy: wall cells are blockers
        occ = np.zeros((H, W), dtype=bool)
        for (y, x) in wallset:
            occ[y, x] = True
        # order objects by proximity to wall (closest settles first)
        def key(cells):
            if d == "down":
                return -max(y for y, x in cells)
            if d == "up":
                return min(y for y, x in cells)
            if d == "right":
                return -max(x for y, x in cells)
            return min(x for y, x in cells)
        comps.sort(key=key)
        out = np.full_like(g, bg)
        for (y, x) in wallset:
            out[y, x] = col
        for cells in comps:
            # move this rigid object step by step until collision
            shift = 0
            while True:
                ok = True
                for (y, x) in cells:
                    ny, nx = y + dy * (shift + 1), x + dx * (shift + 1)
                    if not (0 <= ny < H and 0 <= nx < W):
                        ok = False
                        break
                    if occ[ny, nx]:
                        ok = False
                        break
                if not ok:
                    break
                shift += 1
            for (y, x) in cells:
                ny, nx = y + dy * shift, x + dx * shift
                out[ny, nx] = g[y, x]
                occ[ny, nx] = True
        return out
    try:
        if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ============================================================
# single_color_gravity : exactly one "moving" color slides in a fixed
# direction; every other non-bg color is a static obstacle.  Moving cells
# stack against obstacles / the wall, preserving order along the axis.
# (9c56f360: color 3 slides left, 8 = obstacle.)
# ============================================================
def single_color_gravity(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    bg = _bg(train[0][0])
    # find the color that moves: cells that changed position.
    # candidate moving color = color present in input whose cells differ in output
    moving_candidates = set()
    for i, o in train:
        for v in np.unique(i):
            v = int(v)
            if v == bg:
                continue
            if not np.array_equal(i == v, o == v):
                moving_candidates.add(v)
    if len(moving_candidates) != 1:
        return None
    mv = moving_candidates.pop()

    def slide(g, direction):
        H, W = g.shape
        out = g.copy()
        # remove moving cells, treat rest as obstacles
        out[out == mv] = bg
        if direction in ("left", "right"):
            for r in range(H):
                # segments between obstacles
                obstacles = [c for c in range(W) if g[r, c] != bg and g[r, c] != mv]
                bounds = [-1] + obstacles + [W]
                # count moving cells per segment
                for si in range(len(bounds) - 1):
                    lo, hi = bounds[si] + 1, bounds[si + 1]  # [lo, hi)
                    cnt = sum(1 for c in range(lo, hi) if g[r, c] == mv)
                    if cnt == 0:
                        continue
                    if direction == "left":
                        for k in range(cnt):
                            out[r, lo + k] = mv
                    else:
                        for k in range(cnt):
                            out[r, hi - 1 - k] = mv
        else:  # up/down
            for c in range(W):
                obstacles = [r for r in range(H) if g[r, c] != bg and g[r, c] != mv]
                bounds = [-1] + obstacles + [H]
                for si in range(len(bounds) - 1):
                    lo, hi = bounds[si] + 1, bounds[si + 1]
                    cnt = sum(1 for r in range(lo, hi) if g[r, c] == mv)
                    if cnt == 0:
                        continue
                    if direction == "up":
                        for k in range(cnt):
                            out[lo + k, c] = mv
                    else:
                        for k in range(cnt):
                            out[hi - 1 - k, c] = mv
        return out

    for d in ("left", "right", "up", "down"):
        fn = (lambda g, d=d: slide(g, d))
        try:
            if all(np.array_equal(fn(i), o) for i, o in train):
                return fn
        except Exception:
            continue
    return None


# ============================================================
# grid_cell_ring : a grid of cells separated by full lines of one color.
# Each cell's interior is filled based on its position in the cell-grid:
# border-ring cells one color, interior cells another (learned from train).
# Generalizes to any grid size / any two fill colors.
# ============================================================
def grid_cell_ring(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def analyze(g):
        H, W = g.shape
        # separator color: appears as full rows AND full cols
        row_colors = {}
        for r in range(H):
            vals = set(int(x) for x in g[r])
            if len(vals) == 1:
                row_colors[r] = vals.pop()
        col_colors = {}
        for c in range(W):
            vals = set(int(x) for x in g[:, c])
            if len(vals) == 1:
                col_colors[c] = vals.pop()
        # separator = color common to a full row and a full col
        cand = set(row_colors.values()) & set(col_colors.values())
        if not cand:
            return None
        sep = None
        for s in cand:
            fr = [r for r, cc in row_colors.items() if cc == s]
            fc = [c for c, cc in col_colors.items() if cc == s]
            if fr and fc:
                sep = s
                break
        if sep is None:
            return None
        sep_rows = sorted(r for r, cc in row_colors.items() if cc == sep)
        sep_cols = sorted(c for c, cc in col_colors.items() if cc == sep)
        # cell row bands = gaps between consecutive sep rows
        def bands(seps, N):
            b = []
            prev = -1
            allsep = sorted(seps)
            # boundaries include edges
            markers = allsep
            # bands are ranges strictly between adjacent separators
            pts = [-1] + markers + [N]
            for a, b2 in zip(pts, pts[1:]):
                if b2 - a > 1:
                    b.append((a + 1, b2 - 1))
            return b
        rbands = bands(sep_rows, H)
        cbands = bands(sep_cols, W)
        if len(rbands) < 1 or len(cbands) < 1:
            return None
        return sep, rbands, cbands

    a0 = analyze(train[0][0])
    if a0 is None:
        return None

    # learn border_fill / interior_fill from train outputs
    border_fill = set()
    interior_fill = set()
    for i, o in train:
        a = analyze(i)
        if a is None:
            return None
        sep, rbands, cbands = a
        nr, nc = len(rbands), len(cbands)
        for ri, (r0, r1) in enumerate(rbands):
            for ci, (c0, c1) in enumerate(cbands):
                is_border = (ri == 0 or ri == nr - 1 or ci == 0 or ci == nc - 1)
                block = o[r0:r1+1, c0:c1+1]
                vals = set(int(x) for x in block.flatten())
                if len(vals) != 1:
                    return None
                v = vals.pop()
                if is_border:
                    border_fill.add(v)
                else:
                    interior_fill.add(v)
    if len(border_fill) != 1:
        return None
    bf = border_fill.pop()
    itf = interior_fill.pop() if len(interior_fill) == 1 else bf

    def fn(g):
        a = analyze(g)
        if a is None:
            return g.copy()
        sep, rbands, cbands = a
        nr, nc = len(rbands), len(cbands)
        out = g.copy()
        for ri, (r0, r1) in enumerate(rbands):
            for ci, (c0, c1) in enumerate(cbands):
                is_border = (ri == 0 or ri == nr - 1 or ci == 0 or ci == nc - 1)
                out[r0:r1+1, c0:c1+1] = bf if is_border else itf
        return out
    try:
        if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ============================================================
# count_inside_frame : a rectangular frame of one color; scattered dots of
# another color.  Output is a fixed-size grid filled (row-major) with N cells
# of the dot color, where N = number of dots inside the frame.
# ============================================================
def count_inside_frame(train):
    oshape = train[0][1].shape
    if any(o.shape != oshape for _, o in train):
        return None

    def analyze(g):
        # frame color: color whose cells form the largest bbox with most cells
        # heuristic: the color forming a rectangle outline (the border color)
        colors = [int(c) for c in np.unique(g) if c != 0]
        if len(colors) < 2:
            return None
        # frame = color with the largest bounding-box perimeter coverage
        best_frame = None
        best_score = -1
        for c in colors:
            ys, xs = np.where(g == c)
            if len(ys) < 4:
                continue
            r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            perim = 2 * (h + w) - 4
            # fraction of perimeter occupied by this color
            on_perim = sum(1 for y, x in zip(ys, xs)
                           if y == r0 or y == r1 or x == c0 or x == c1)
            score = on_perim / max(1, perim)
            if score > 0.6 and len(ys) > best_score:
                best_score = len(ys)
                best_frame = (c, r0, r1, c0, c1)
        if best_frame is None:
            return None
        fc, r0, r1, c0, c1 = best_frame
        dot_colors = [c for c in colors if c != fc]
        if len(dot_colors) != 1:
            return None
        dc = dot_colors[0]
        dy, dx = np.where(g == dc)
        inside = sum(1 for y, x in zip(dy, dx) if r0 < y < r1 and c0 < x < c1)
        return dc, inside

    a0 = analyze(train[0][0])
    if a0 is None:
        return None

    def fn(g):
        a = analyze(g)
        out = np.zeros(oshape, dtype=int)
        if a is None:
            return out
        dc, n = a
        H, W = oshape
        k = 0
        for r in range(H):
            for c in range(W):
                if k < n:
                    out[r, c] = dc
                    k += 1
        return out
    try:
        if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ============================================================
# legend_broadcast : grid split by full separator lines into a matrix of
# cells.  One band (row-band or col-band) is a legend giving one color per
# index.  In the other cells, a marker color is repainted with the legend
# color of that cell's index.  (ef26cbf6)
# ============================================================
def legend_broadcast(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def structure(g):
        H, W = g.shape
        # separator color = a color forming >=1 full row AND full col, OR
        # the most frequent color among full lines
        colors = [int(c) for c in np.unique(g)]
        sep = None
        for c in colors:
            fr = [r for r in range(H) if np.all(g[r] == c)]
            fc = [cc for cc in range(W) if np.all(g[:, cc] == c)]
            if fr or fc:
                # prefer a color with both, else any full line
                if fr and fc:
                    sep = c
                    break
                if sep is None:
                    sep = c
        if sep is None:
            return None
        frows = [r for r in range(H) if np.all(g[r] == sep)]
        fcols = [c for c in range(W) if np.all(g[:, c] == sep)]

        def bands(seps, N):
            pts = [-1] + sorted(seps) + [N]
            b = []
            for a, bb in zip(pts, pts[1:]):
                if bb - a > 1:
                    b.append((a + 1, bb - 1))
            return b
        rb = bands(frows, H)
        cb = bands(fcols, W)
        if len(rb) < 2 and len(cb) < 2:
            return None
        return sep, rb, cb

    s0 = structure(train[0][0])
    if s0 is None:
        return None

    # Determine marker color + legend orientation by comparing in/out.
    # marker = a color that in some query cell is replaced.
    def cell_dominant_nonbg(g, r0, r1, c0, c1, ignore):
        block = g[r0:r1+1, c0:c1+1]
        vals = [int(v) for v in block.flatten() if int(v) not in ignore]
        if not vals:
            return None
        return Counter(vals).most_common(1)[0][0]

    bg = _bg(train[0][0])

    # infer marker: color present in input that disappears/changes in output
    marker_cands = set()
    for i, o in train:
        for v in np.unique(i):
            v = int(v)
            if v == bg:
                continue
            if not np.array_equal(i == v, o == v) and (o == v).sum() < (i == v).sum():
                marker_cands.add(v)
    # marker most likely the one whose cells got recolored
    if not marker_cands:
        return None

    def build(marker, orient):
        def fn(g):
            st = structure(g)
            if st is None:
                return g.copy()
            sep, rb, cb = st
            out = g.copy()
            ignore = {bg, sep, marker}
            if orient == "col":
                # legend is one row-band; color per col index
                # find legend row-band = the one with special colors, few markers
                legend_ri = None
                for ri, (r0, r1) in enumerate(rb):
                    has_special = False
                    for (c0, c1) in cb:
                        d = cell_dominant_nonbg(g, r0, r1, c0, c1, {bg, sep, marker})
                        if d is not None:
                            has_special = True
                    if has_special:
                        legend_ri = ri
                        break
                if legend_ri is None:
                    return g.copy()
                r0L, r1L = rb[legend_ri]
                legend = {}
                for ci, (c0, c1) in enumerate(cb):
                    legend[ci] = cell_dominant_nonbg(g, r0L, r1L, c0, c1, {bg, sep, marker})
                for ri, (r0, r1) in enumerate(rb):
                    if ri == legend_ri:
                        continue
                    for ci, (c0, c1) in enumerate(cb):
                        col = legend.get(ci)
                        if col is None:
                            continue
                        blk = out[r0:r1+1, c0:c1+1]
                        blk[blk == marker] = col
                        out[r0:r1+1, c0:c1+1] = blk
            else:
                legend_ci = None
                for ci, (c0, c1) in enumerate(cb):
                    has_special = False
                    for (r0, r1) in rb:
                        d = cell_dominant_nonbg(g, r0, r1, c0, c1, {bg, sep, marker})
                        if d is not None:
                            has_special = True
                    if has_special:
                        legend_ci = ci
                        break
                if legend_ci is None:
                    return g.copy()
                c0L, c1L = cb[legend_ci]
                legend = {}
                for ri, (r0, r1) in enumerate(rb):
                    legend[ri] = cell_dominant_nonbg(g, r0, r1, c0L, c1L, {bg, sep, marker})
                for ci, (c0, c1) in enumerate(cb):
                    if ci == legend_ci:
                        continue
                    for ri, (r0, r1) in enumerate(rb):
                        col = legend.get(ri)
                        if col is None:
                            continue
                        blk = out[r0:r1+1, c0:c1+1]
                        blk[blk == marker] = col
                        out[r0:r1+1, c0:c1+1] = blk
            return out
        return fn

    for marker in sorted(marker_cands):
        for orient in ("col", "row"):
            fn = build(marker, orient)
            try:
                if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
                    return fn
            except Exception:
                continue
    return None


DETECTORS = [self_tile, concentric_rings, gravity_to_wall,
             single_color_gravity, grid_cell_ring, count_inside_frame,
             legend_broadcast]
