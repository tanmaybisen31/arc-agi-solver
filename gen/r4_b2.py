"""Round 4 batch 2 detectors for ARC-AGI eval tasks.

Interface: each detector def det(train) -> transform_fn | None.
train = [(in_np2d, out_np2d), ...]; transform_fn(grid)->grid.
Engine verifies exact reproduction of every demo. numpy+stdlib only. Defensive.
"""
import numpy as np
from collections import Counter, defaultdict


def bg_color(g):
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


# ---------------------------------------------------------------------------
# fc754716 : single dot -> hollow rectangle frame around the whole grid
# ---------------------------------------------------------------------------
def frame_around_grid(train):
    # Rule: input has a single non-background colored cell; output is a
    # 1-cell-thick frame of that color around the entire grid perimeter.
    for i, o in train:
        if i.shape != o.shape:
            return None
        bg = bg_color(i)
        nz = np.argwhere(i != bg)
        if len(nz) == 0:
            return None
        cols = set(int(i[r, c]) for r, c in nz)
        if len(cols) != 1:
            return None
    def fn(g):
        bg = bg_color(g)
        nz = np.argwhere(g != bg)
        if len(nz) == 0:
            return g.copy()
        color = int(g[nz[0][0], nz[0][1]])
        out = np.full_like(g, bg)
        out[0, :] = color
        out[-1, :] = color
        out[:, 0] = color
        out[:, -1] = color
        return out
    return fn


# ---------------------------------------------------------------------------
# 0b17323b : arithmetic diagonal of dots -> extend forward in second color
# ---------------------------------------------------------------------------
def extend_dot_sequence(train):
    # All input non-bg cells are single dots of one color forming an arithmetic
    # progression (constant step). Output keeps them and continues the sequence
    # (same step) with a NEW single color until falling off the grid.
    new_color = None
    for i, o in train:
        if i.shape != o.shape:
            return None
        bg = bg_color(i)
        in_pts = [tuple(p) for p in np.argwhere(i != bg)]
        if len(in_pts) < 2:
            return None
        in_colors = set(int(i[r, c]) for r, c in in_pts)
        if len(in_colors) != 1:
            return None
        # kept cells must match; new cells all one new color
        out_pts = [tuple(p) for p in np.argwhere(o != bg)]
        added = [p for p in out_pts if p not in set(in_pts)]
        if not added:
            return None
        acs = set(int(o[r, c]) for r, c in added)
        if len(acs) != 1:
            return None
        nc = next(iter(acs))
        if new_color is None:
            new_color = nc
        elif new_color != nc:
            return None
    def fn(g):
        bg = bg_color(g)
        pts = sorted(tuple(p) for p in np.argwhere(g != bg))
        if len(pts) < 2:
            return g.copy()
        # verify constant step
        steps = set()
        for a, b in zip(pts, pts[1:]):
            steps.add((b[0] - a[0], b[1] - a[1]))
        if len(steps) != 1:
            return g.copy()
        dr, dc = next(iter(steps))
        if dr == 0 and dc == 0:
            return g.copy()
        out = g.copy()
        H, W = g.shape
        r, c = pts[-1]
        r += dr
        c += dc
        while 0 <= r < H and 0 <= c < W:
            out[r, c] = new_color
            r += dr
            c += dc
        return out
    return fn


# ---------------------------------------------------------------------------
# 8ba14f53 : two outline shapes -> row-major bars of their interior-hole counts
# ---------------------------------------------------------------------------
def _interior_holes(mask):
    # mask: bool array of the shape's cells within its bounding box.
    H, W = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    stack = []
    for r in range(H):
        for c in (0, W - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    return int(((~mask) & (~outside)).sum())


def holes_barchart(train):
    out_shape = None
    for i, o in train:
        if out_shape is None:
            out_shape = o.shape
        elif out_shape != o.shape:
            return None
    if out_shape is None:
        return None

    def build(g):
        bg = bg_color(g)
        cols = [c for c in sorted(set(np.unique(g).tolist())) if c != bg]
        items = []  # (leftcol, color, holes)
        for col in cols:
            ys, xs = np.where(g == col)
            if len(ys) == 0:
                continue
            r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
            mask = (g[r0:r1 + 1, c0:c1 + 1] == col)
            h = _interior_holes(mask)
            items.append((int(c0), int(col), h))
        items.sort()
        H, W = out_shape
        out = np.full(out_shape, bg, dtype=int)
        row = 0
        for _, color, h in items:
            if h <= 0:
                # still advance to a fresh row so ordering is preserved
                if row < H:
                    row += 0  # zero-hole shape contributes nothing, no row used
                continue
            col = 0
            placed = 0
            while placed < h and row < H:
                out[row, col] = color
                placed += 1
                col += 1
                if col >= W:
                    col = 0
                    row += 1
            if col != 0:
                row += 1  # move to fresh row for next shape
        return out

    for i, o in train:
        r = build(i)
        if r.shape != o.shape or not np.array_equal(r, o):
            return None
    return build


# ---------------------------------------------------------------------------
# 5b526a93 : rows of identical box-shapes; complete each row to the template
#            using a new color for the added boxes.
# ---------------------------------------------------------------------------
def complete_box_rows(train):
    new_color = None
    for i, o in train:
        if i.shape != o.shape:
            return None
        added = (i != o)
        if added.any():
            vals = set(int(o[r, c]) for r, c in np.argwhere(added))
            # cells that changed: input was bg there
            if len(vals) != 1:
                return None
            nc = next(iter(vals))
            if new_color is None:
                new_color = nc
            elif new_color != nc:
                return None
    if new_color is None:
        return None

    def build(g):
        bg = bg_color(g)
        comps = _components(g, bg, diag=True)
        if not comps:
            return g.copy()
        boxes = []
        for cc in comps:
            ys = [y for y, x in cc]
            xs = [x for y, x in cc]
            boxes.append((min(ys), min(xs), max(ys), max(xs), cc))
        # all boxes must share the same size/shape
        h0 = boxes[0][2] - boxes[0][0]
        w0 = boxes[0][3] - boxes[0][1]
        for (r0, c0, r1, c1, cc) in boxes:
            if (r1 - r0) != h0 or (c1 - c0) != w0:
                return g.copy()
        bh, bw = h0 + 1, w0 + 1
        # canonical shape mask from first box
        r0, c0, r1, c1, cc0 = boxes[0]
        shape = np.zeros((bh, bw), dtype=bool)
        for y, x in cc0:
            shape[y - r0, x - c0] = True
        # group by band (rmin)
        bands = defaultdict(list)
        for (r0, c0, r1, c1, cc) in boxes:
            bands[r0].append(c0)
        # template columns = band with most boxes
        template = max(bands.values(), key=len)
        template_cols = sorted(set(template))
        out = g.copy()
        H, W = g.shape
        for band_r, cols in bands.items():
            present = set(cols)
            for tc in template_cols:
                if tc in present:
                    continue
                if band_r + bh > H or tc + bw > W:
                    continue
                for dy in range(bh):
                    for dx in range(bw):
                        if shape[dy, dx]:
                            out[band_r + dy, tc + dx] = new_color
        return out

    for i, o in train:
        r = build(i)
        if r.shape != o.shape or not np.array_equal(r, o):
            return None
    return build


# ---------------------------------------------------------------------------
# 642248e4 : dots gain an adjacent marker in the color of the nearer of two
#            opposite uniform borders.
# ---------------------------------------------------------------------------
def dots_toward_border(train):
    def analyze(g):
        H, W = g.shape
        # horizontal mode: top and bottom rows uniform, differing colors
        top = set(g[0].tolist())
        bot = set(g[-1].tolist())
        left = set(g[:, 0].tolist())
        right = set(g[:, -1].tolist())
        if len(top) == 1 and len(bot) == 1 and top != bot:
            return ("h", next(iter(top)), next(iter(bot)))
        if len(left) == 1 and len(right) == 1 and left != right:
            return ("v", next(iter(left)), next(iter(right)))
        return None

    def build(g):
        info = analyze(g)
        if info is None:
            return g.copy()
        mode, ca, cb = info
        H, W = g.shape
        bg = bg_color(g)
        out = g.copy()
        for r in range(H):
            for c in range(W):
                v = g[r, c]
                if v == bg:
                    continue
                if mode == "h":
                    if r == 0 or r == H - 1:
                        continue
                    if v == ca or v == cb:
                        continue
                    dtop = r
                    dbot = H - 1 - r
                    if dtop <= dbot:
                        nr, nc, col = r - 1, c, ca
                    else:
                        nr, nc, col = r + 1, c, cb
                else:
                    if c == 0 or c == W - 1:
                        continue
                    if v == ca or v == cb:
                        continue
                    dleft = c
                    dright = W - 1 - c
                    if dleft <= dright:
                        nr, nc, col = r, c - 1, ca
                    else:
                        nr, nc, col = r, c + 1, cb
                if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == bg:
                    out[nr, nc] = col
        return out

    for i, o in train:
        if i.shape != o.shape:
            return None
    for i, o in train:
        r = build(i)
        if r.shape != o.shape or not np.array_equal(r, o):
            return None
    return build


# ---------------------------------------------------------------------------
# 94414823 : fill a hollow box interior with two corner-marker colors placed
#            in diagonal quadrant pairs.
# ---------------------------------------------------------------------------
def box_interior_diagonals(train):
    # box color = the color forming a hollow rectangle ring; markers = the
    # two other non-bg colors located at grid corners.
    box_color = None
    for i, o in train:
        if i.shape != o.shape:
            return None

    def build(g):
        H, W = g.shape
        bg = bg_color(g)
        # box color: most frequent non-bg with a rectangular hollow ring
        cand = None
        for col in sorted(set(np.unique(g).tolist()) - {bg}):
            ys, xs = np.where(g == col)
            if len(ys) < 8:
                continue
            r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            # check ring: all perimeter cells == col, interior mostly bg
            sub = g[r0:r1 + 1, c0:c1 + 1]
            ring = np.zeros_like(sub, dtype=bool)
            ring[0, :] = True; ring[-1, :] = True
            ring[:, 0] = True; ring[:, -1] = True
            if np.all(sub[ring] == col):
                cand = (col, r0, r1, c0, c1)
                break
        if cand is None:
            return g.copy()
        col5, r0, r1, c0, c1 = cand
        ir0, ir1, ic0, ic1 = r0 + 1, r1 - 1, c0 + 1, c1 - 1
        ih = ir1 - ir0 + 1
        iw = ic1 - ic0 + 1
        if ih < 2 or iw < 2 or ih % 2 or iw % 2:
            return g.copy()
        markers = []
        for mc in sorted(set(np.unique(g).tolist()) - {bg, col5}):
            mys, mxs = np.where(g == mc)
            # use first cell
            y, x = int(mys[0]), int(mxs[0])
            top = y < H / 2
            leftside = x < W / 2
            main_diag = (top and leftside) or ((not top) and (not leftside))
            markers.append((mc, main_diag))
        if len(markers) != 2:
            return g.copy()
        main_col = None; anti_col = None
        for mc, md in markers:
            if md:
                main_col = mc
            else:
                anti_col = mc
        if main_col is None or anti_col is None:
            return g.copy()
        out = g.copy()
        hh = ih // 2
        hw = iw // 2
        for dr in range(ih):
            for dc in range(iw):
                q_top = dr < hh
                q_left = dc < hw
                is_main = (q_top and q_left) or ((not q_top) and (not q_left))
                out[ir0 + dr, ic0 + dc] = main_col if is_main else anti_col
        return out

    for i, o in train:
        r = build(i)
        if r.shape != o.shape or not np.array_equal(r, o):
            return None
    return build


# ---------------------------------------------------------------------------
# 29700607 : top-row keys drop vertical lines to matching edge markers, then
#            run a horizontal line to that marker's edge.
# ---------------------------------------------------------------------------
def key_drop_to_marker(train):
    def build(g):
        H, W = g.shape
        bg = bg_color(g)
        keys = [(c, int(g[0, c])) for c in range(W) if g[0, c] != bg]
        if not keys:
            return g.copy()
        # markers: non-bg cells not in top row, located on left/right edges
        markers = {}
        for r in range(1, H):
            for c in (0, W - 1):
                v = int(g[r, c])
                if v != bg:
                    markers.setdefault(v, (r, c))
        out = g.copy()
        for kc, color in keys:
            m = markers.get(color)
            if m is None:
                # straight down to bottom
                for r in range(0, H):
                    out[r, kc] = color
            else:
                mr, mcedge = m
                for r in range(0, mr + 1):
                    out[r, kc] = color
                if mcedge == 0:
                    lo, hi = 0, kc
                else:
                    lo, hi = kc, W - 1
                for c in range(lo, hi + 1):
                    out[mr, c] = color
        return out

    for i, o in train:
        if i.shape != o.shape:
            return None
    for i, o in train:
        r = build(i)
        if r.shape != o.shape or not np.array_equal(r, o):
            return None
    return build


# ---------------------------------------------------------------------------
# 7ee1c6ea : inside a rectangular frame, swap the two non-bg figure colors.
# ---------------------------------------------------------------------------
def frame_interior_swap(train):
    def find_frame(g):
        bg = bg_color(g)
        H, W = g.shape
        best = None
        for col in sorted(set(np.unique(g).tolist()) - {bg}):
            ys, xs = np.where(g == col)
            if len(ys) < 2 * (H + W) - 4 - 8:
                pass
            r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            sub = g[r0:r1 + 1, c0:c1 + 1]
            ring = np.zeros_like(sub, dtype=bool)
            ring[0, :] = True; ring[-1, :] = True
            ring[:, 0] = True; ring[:, -1] = True
            ringcells = sub[ring]
            frac = np.mean(ringcells == col)
            if frac >= 0.85:
                score = (r1 - r0) * (c1 - c0)
                if best is None or score > best[0]:
                    best = (score, col, r0, r1, c0, c1)
        if best is None:
            return None
        return best[1:]

    def build(g):
        f = find_frame(g)
        if f is None:
            return g.copy()
        col, r0, r1, c0, c1 = f
        interior = g[r0 + 1:r1, c0 + 1:c1]
        # 0 is the untouched background inside the frame
        figs = sorted(set(np.unique(interior).tolist()) - {col, 0})
        if len(figs) != 2:
            return g.copy()
        a, b = figs
        out = g.copy()
        sub = out[r0 + 1:r1, c0 + 1:c1]
        ma = sub == a
        mb = sub == b
        sub[ma] = b
        sub[mb] = a
        out[r0 + 1:r1, c0 + 1:c1] = sub
        return out

    for i, o in train:
        if i.shape != o.shape:
            return None
    for i, o in train:
        r = build(i)
        if r.shape != o.shape or not np.array_equal(r, o):
            return None
    return build


# ---------------------------------------------------------------------------
# 13713586 : colored segments extend toward a full-edge "wall" line; nearer
#            segments override farther ones.
# ---------------------------------------------------------------------------
def segments_extend_to_wall(train):
    def find_wall(g):
        H, W = g.shape
        bg = bg_color(g)
        # a full edge line of one uniform non-bg color
        edges = [
            ("up", g[0, :]),      # wall at top -> fill upward
            ("down", g[-1, :]),   # wall at bottom -> fill downward
            ("left", g[:, 0]),    # wall at left -> fill leftward
            ("right", g[:, -1]),  # wall at right -> fill rightward
        ]
        for direction, line in edges:
            vals = set(line.tolist())
            if len(vals) == 1:
                v = next(iter(vals))
                if v != bg:
                    return direction, v
        return None

    def build(g):
        w = find_wall(g)
        if w is None:
            return g.copy()
        direction, wall_color = w
        H, W = g.shape
        bg = bg_color(g)
        # collect colored cells (not bg, not on the wall edge)
        cells = []
        for r in range(H):
            for c in range(W):
                v = g[r, c]
                if v == bg:
                    continue
                # skip wall edge cells
                if direction == "up" and r == 0:
                    continue
                if direction == "down" and r == H - 1:
                    continue
                if direction == "left" and c == 0:
                    continue
                if direction == "right" and c == W - 1:
                    continue
                if v == wall_color:
                    continue
                cells.append((r, c, int(v)))
        # distance to wall
        def dist(r, c):
            if direction == "up":
                return r
            if direction == "down":
                return H - 1 - r
            if direction == "left":
                return c
            return W - 1 - c
        # farthest first so nearest overrides
        cells.sort(key=lambda t: -dist(t[0], t[1]))
        out = g.copy()
        for r, c, v in cells:
            if direction == "up":
                for rr in range(1, r + 1):
                    out[rr, c] = v
            elif direction == "down":
                for rr in range(r, H - 1):
                    out[rr, c] = v
            elif direction == "left":
                for cc in range(1, c + 1):
                    out[r, cc] = v
            else:
                for cc in range(c, W - 1):
                    out[r, cc] = v
        return out

    for i, o in train:
        if i.shape != o.shape:
            return None
    for i, o in train:
        r = build(i)
        if r.shape != o.shape or not np.array_equal(r, o):
            return None
    return build


# ---------------------------------------------------------------------------
# d2acf2cb : columns/rows bracketed by a marker color get a learned color swap
#            applied to their interior cells.
# ---------------------------------------------------------------------------
def bracketed_line_swap(train):
    # learn: marker color + swap map + orientation from training diffs
    swap = {}
    marker = None
    orient = None  # "col" or "row"
    for i, o in train:
        if i.shape != o.shape:
            return None
        H, W = i.shape
        diff = np.argwhere(i != o)
        if len(diff) == 0:
            continue
        rows = set(int(r) for r, c in diff)
        cols = set(int(c) for r, c in diff)
        # determine orientation: if changes are confined to some columns fully
        # (vertical bands) -> col mode; else row mode
        for r, c in diff:
            a, b = int(i[r, c]), int(o[r, c])
            if a in swap and swap[a] != b:
                return None
            swap[a] = b
    if not swap:
        return None
    # ensure swap is an involution-ish mapping we can apply
    # infer marker/orientation per-grid at apply time.

    swap_colors = set(swap.keys()) | set(swap.values())

    def build(g):
        H, W = g.shape
        # marker candidates: non-bg colors not involved in the swap
        cands = [c for c in sorted(set(np.unique(g).tolist()) - {0})
                 if c not in swap_colors]
        if not cands:
            return g.copy()
        # pick marker as the one whose cells all lie on opposite grid edges
        marker = None
        mode = None
        best_lines = None
        for m in cands:
            pos = np.argwhere(g == m)
            rs = set(int(r) for r, c in pos)
            cs = set(int(c) for r, c in pos)
            # column mode: all markers on rows 0 and H-1
            if rs <= {0, H - 1}:
                cols_top = set(int(c) for r, c in pos if r == 0)
                cols_bot = set(int(c) for r, c in pos if r == H - 1)
                lines = sorted(cols_top & cols_bot)
                if lines and (best_lines is None or len(lines) > len(best_lines)):
                    marker, mode, best_lines = m, "col", lines
            # row mode: all markers on cols 0 and W-1
            if cs <= {0, W - 1}:
                rows_left = set(int(r) for r, c in pos if c == 0)
                rows_right = set(int(r) for r, c in pos if c == W - 1)
                lines = sorted(rows_left & rows_right)
                if lines and (best_lines is None or len(lines) > len(best_lines)):
                    marker, mode, best_lines = m, "row", lines
        if marker is None:
            return g.copy()
        out = g.copy()
        if mode == "col":
            for c in best_lines:
                for r in range(1, H - 1):
                    v = int(g[r, c])
                    if v in swap:
                        out[r, c] = swap[v]
        else:
            for r in best_lines:
                for c in range(1, W - 1):
                    v = int(g[r, c])
                    if v in swap:
                        out[r, c] = swap[v]
        return out

    for i, o in train:
        r = build(i)
        if r.shape != o.shape or not np.array_equal(r, o):
            return None
    return build


DETECTORS = [
    frame_around_grid,
    extend_dot_sequence,
    holes_barchart,
    complete_box_rows,
    dots_toward_border,
    box_interior_diagonals,
    key_drop_to_marker,
    frame_interior_swap,
    segments_extend_to_wall,
    bracketed_line_swap,
]
