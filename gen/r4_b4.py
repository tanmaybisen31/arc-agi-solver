"""Round-4 batch-4 rule detectors for ARC-AGI.

Each detector:  def det(train) -> transform_fn | None
  train = [(input_grid, output_grid), ...]  (numpy int arrays)
  transform_fn: grid -> grid  (must reproduce EVERY train output; engine verifies)

numpy + stdlib only.  Defensive against ragged / degenerate inputs.
"""
import numpy as np
from collections import Counter


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag=False, samecolor=False):
    """Connected components of non-bg cells. Returns list of lists of (r,c)."""
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        if samecolor and int(g[ny, nx]) != col:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _verify(fn, train):
    try:
        for i, o in train:
            r = fn(i)
            if r is None or r.shape != o.shape or not np.array_equal(r, o):
                return False
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# per-object vertical / horizontal flip in place
# --------------------------------------------------------------------------- #
def per_color_flip(train):
    """Each colour's cells are reflected within that colour's bounding box.

    Tries vertical flip, horizontal flip, and 180-rotation.  The whole grid
    keeps its shape; the background is preserved.
    """
    if any(i.shape != o.shape for i, o in train):
        return None

    def apply(g, mode):
        bg = _bg_color(g)
        out = np.full_like(g, bg)
        colors = [int(v) for v in np.unique(g) if int(v) != bg]
        for c in colors:
            ys, xs = np.where(g == c)
            if len(ys) == 0:
                continue
            r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
            sub = (g[r0:r1 + 1, c0:c1 + 1] == c)
            if mode == "v":
                sub = sub[::-1, :]
            elif mode == "h":
                sub = sub[:, ::-1]
            else:  # rot180
                sub = sub[::-1, ::-1]
            yy, xx = np.where(sub)
            out[r0 + yy, c0 + xx] = c
        return out

    for mode in ("v", "h", "r"):
        fn = (lambda g, m=mode: apply(g, m))
        if _verify(fn, train):
            return fn
    return None


# --------------------------------------------------------------------------- #
# per-object vertical/horizontal flip using connected components (any colour)
# --------------------------------------------------------------------------- #
def per_component_flip(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def apply(g, mode, diag):
        bg = _bg_color(g)
        out = np.full_like(g, bg)
        for cells in _components(g, bg, diag=diag):
            ys = [y for y, x in cells]
            xs = [x for y, x in cells]
            r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            for (y, x) in cells:
                ly, lx = y - r0, x - c0
                if mode == "v":
                    ny, nx = h - 1 - ly, lx
                elif mode == "h":
                    ny, nx = ly, w - 1 - lx
                else:
                    ny, nx = h - 1 - ly, w - 1 - lx
                out[r0 + ny, c0 + nx] = g[y, x]
        return out

    for diag in (False, True):
        for mode in ("v", "h", "r"):
            fn = (lambda g, m=mode, d=diag: apply(g, m, d))
            if _verify(fn, train):
                return fn
    return None


# --------------------------------------------------------------------------- #
# recolor shape cells by a "key" colour found in the same row (column 0)
# --------------------------------------------------------------------------- #
def recolor_by_row_key(train):
    """A 'shape' colour is painted across the grid; each row has a key colour in
    the left-most column.  Every shape cell is recoloured to its row's key.
    Generalises to key in column 0 or the last column."""
    if any(i.shape != o.shape for i, o in train):
        return None

    def find_shape_color(g):
        # shape colour = colour whose cells change to something else
        return None

    # learn which colour is the "shape" colour (constant across demos)
    shape_col = None
    for i, o in train:
        diff = i != o
        if not diff.any():
            return None
        sc = set(i[diff].tolist())
        if len(sc) != 1:
            return None
        if shape_col is None:
            shape_col = sc.pop()
        elif shape_col not in sc:
            return None
    if shape_col is None:
        return None

    for keycol in (0, -1):
        def fn(g, sc=shape_col, kc=keycol):
            out = g.copy()
            H, W = g.shape
            keyidx = kc if kc >= 0 else W + kc
            for r in range(H):
                key = g[r, keyidx]
                if key == 0 or key == sc:
                    continue
                out[r, g[r] == sc] = key
            return out
        if _verify(fn, train):
            return fn

    # also try: key is nearest non-bg,non-shape colour on the same row
    def fn2(g, sc=shape_col):
        out = g.copy()
        H, W = g.shape
        for r in range(H):
            row = g[r]
            keys = [int(v) for v in row if v != 0 and v != sc]
            if len(set(keys)) == 1:
                out[r, row == sc] = keys[0]
        return out
    if _verify(fn2, train):
        return fn2
    return None


# --------------------------------------------------------------------------- #
# draw the symmetry-axis line of a single symmetric shape
# --------------------------------------------------------------------------- #
def symmetry_axis_line(train):
    """A single shape that is mirror-symmetric.  A line of a fixed colour is
    drawn across the whole grid through its axis of symmetry."""
    if any(i.shape != o.shape for i, o in train):
        return None

    # learn line colour
    line_col = None
    for i, o in train:
        diff = i != o
        if not diff.any():
            return None
        nc = set(o[diff].tolist())
        if len(nc) != 1:
            return None
        c = nc.pop()
        if line_col is None:
            line_col = c
        elif line_col != c:
            return None
    if line_col is None:
        return None

    def apply(g, lc):
        bg = _bg_color(g)
        mask = g != bg
        if not mask.any():
            return None
        ys, xs = np.where(mask)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        cands = []
        for r in range(r0 + 1, r1):
            if (not mask[r, :].any() and mask[r0:r, :].any()
                    and mask[r + 1:r1 + 1, :].any()):
                cands.append(("row", r))
        for c in range(c0 + 1, c1):
            if (not mask[:, c].any() and mask[:, c0:c].any()
                    and mask[:, c + 1:c1 + 1].any()):
                cands.append(("col", c))
        if len(cands) != 1:
            return None
        kind, idx = cands[0]
        out = g.copy()
        if kind == "row":
            out[idx, :] = lc
        else:
            out[:, idx] = lc
        return out

    fn = (lambda g, lc=line_col: apply(g, lc))
    if _verify(fn, train):
        return fn
    return None


# --------------------------------------------------------------------------- #
# add nested frames around each object, coloured/counted by its interior dots
# --------------------------------------------------------------------------- #
def frame_by_noise_count(train):
    """Each object is a solid rectangle of one colour with a few 'dot' cells of
    a second colour.  Wrap it in N nested rectangular frames, where N = number
    of dots and the frame colour = the dot colour."""
    if any(i.shape != o.shape for i, o in train):
        return None

    def apply(g):
        bg = _bg_color(g)
        out = g.copy()
        H, W = g.shape
        for cells in _components(g, bg, diag=True):
            ys = [y for y, x in cells]
            xs = [x for y, x in cells]
            r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
            vals = Counter(int(g[y, x]) for y, x in cells)
            if len(vals) < 2:
                continue
            main = vals.most_common(1)[0][0]
            noise = {k: v for k, v in vals.items() if k != main}
            if len(noise) != 1:
                return None
            nc, ncount = next(iter(noise.items()))
            if ncount < 1:
                continue
            for d in range(1, ncount + 1):
                nr0, nr1, nc0, nc1 = r0 - d, r1 + d, c0 - d, c1 + d
                if nr0 < 0 or nc0 < 0 or nr1 >= H or nc1 >= W:
                    return None
                out[nr0, nc0:nc1 + 1] = nc
                out[nr1, nc0:nc1 + 1] = nc
                out[nr0:nr1 + 1, nc0] = nc
                out[nr0:nr1 + 1, nc1] = nc
        return out

    fn = apply
    if _verify(fn, train):
        return fn
    return None


# --------------------------------------------------------------------------- #
# periodic pattern with a defect band -> shifted, gap-filled periodic tiling
# --------------------------------------------------------------------------- #
def periodic_shift_fill(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def defect_color(g):
        H, W = g.shape
        cand = Counter()
        for r in range(H):
            vals = set(g[r, :].tolist())
            if len(vals) == 1:
                cand[list(vals)[0]] += 1
        for c in range(W):
            vals = set(g[:, c].tolist())
            if len(vals) == 1:
                cand[list(vals)[0]] += 1
        return cand.most_common(1)[0][0] if cand else None

    def get_tile(g):
        H, W = g.shape
        defect = defect_color(g)
        if defect is None:
            return None
        mask = g != defect
        if not mask.any():
            return None
        for ph in range(1, min(H, 6) + 1):
            for pw in range(1, min(W, 6) + 1):
                tile = [[None] * pw for _ in range(ph)]
                ok = True
                for r in range(H):
                    for c in range(W):
                        if not mask[r, c]:
                            continue
                        v = int(g[r, c])
                        rr, cc = r % ph, c % pw
                        if tile[rr][cc] is None:
                            tile[rr][cc] = v
                        elif tile[rr][cc] != v:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and all(all(x is not None for x in row) for row in tile):
                    if ph * pw <= 1:
                        continue
                    return ph, pw, tile
        return None

    def build(g, dr, dc):
        res = get_tile(g)
        if res is None:
            return None
        ph, pw, tile = res
        H, W = g.shape
        return np.array([[tile[(r + dr) % ph][(c + dc) % pw]
                          for c in range(W)] for r in range(H)], dtype=int)

    # learn a single (dr,dc) shift consistent with all demos
    for dr in range(6):
        for dc in range(6):
            fn = (lambda g, a=dr, b=dc: build(g, a, b))
            if _verify(fn, train):
                return fn
    return None


# --------------------------------------------------------------------------- #
# each column: fill upward from nearest marker at-or-below
# --------------------------------------------------------------------------- #
def column_fill_up_from_below(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def apply(g):
        out = g.copy()
        H, W = g.shape
        for c in range(W):
            cur = 0
            for r in range(H - 1, -1, -1):
                if g[r, c] != 0:
                    cur = g[r, c]
                elif cur != 0:
                    out[r, c] = cur
        return out

    fn = apply
    if _verify(fn, train):
        return fn
    # also downward / left / right variants
    def apply_dir(g, d):
        out = g.copy()
        H, W = g.shape
        if d in ("up", "down"):
            for c in range(W):
                cur = 0
                rng = range(H) if d == "up" else range(H - 1, -1, -1)
                for r in rng:
                    if g[r, c] != 0:
                        cur = g[r, c]
                    elif cur != 0:
                        out[r, c] = cur
        else:
            for r in range(H):
                cur = 0
                rng = range(W) if d == "left" else range(W - 1, -1, -1)
                for c in rng:
                    if g[r, c] != 0:
                        cur = g[r, c]
                    elif cur != 0:
                        out[r, c] = cur
        return out
    for d in ("up", "down", "left", "right"):
        fn = (lambda g, dd=d: apply_dir(g, dd))
        if _verify(fn, train):
            return fn
    return None


# --------------------------------------------------------------------------- #
# complete a partial Latin square (each row / col a permutation of the values)
# --------------------------------------------------------------------------- #
def latin_square_complete(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # square grids with a fixed value set; 0 = blank
    for i, o in train:
        if i.shape[0] != i.shape[1]:
            return None

    def solve(g):
        n = g.shape[0]
        vals = sorted(int(v) for v in np.unique(g) if int(v) != 0)
        if len(vals) != n:
            return None
        grid = g.copy()
        # backtracking
        cells = [(r, c) for r in range(n) for c in range(n) if grid[r, c] == 0]

        def ok(r, c, v):
            if v in grid[r, :]:
                return False
            if v in grid[:, c]:
                return False
            return True

        def bt(k):
            if k == len(cells):
                return True
            r, c = cells[k]
            for v in vals:
                if ok(r, c, v):
                    grid[r, c] = v
                    if bt(k + 1):
                        return True
                    grid[r, c] = 0
            return False

        if bt(0):
            return grid
        return None

    fn = solve
    if _verify(fn, train):
        return fn
    return None


# --------------------------------------------------------------------------- #
# 2x2 corner legend defines colour-swap pairs applied to the rest of the grid
# --------------------------------------------------------------------------- #
def corner_legend_swap(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def legend_pos(g):
        H, W = g.shape
        corners = {
            "tl": (0, 1, 0, 1),
            "tr": (0, 1, W - 2, W - 1),
            "bl": (H - 2, H - 1, 0, 1),
            "br": (H - 2, H - 1, W - 2, W - 1),
        }
        return corners

    def build(g, corner, mode):
        cs = legend_pos(g)
        r0, r1, c0, c1 = cs[corner]
        a, b = int(g[r0, c0]), int(g[r0, c1])
        c, d = int(g[r1, c0]), int(g[r1, c1])
        if 0 in (a, b, c, d):
            return None
        if mode == "col":  # swap columns of legend
            swap = {a: b, b: a, c: d, d: c}
        else:              # swap rows of legend
            swap = {a: c, c: a, b: d, d: b}
        # bijection check
        if len(set([a, b, c, d])) < 2:
            return None
        out = g.copy()
        H, W = g.shape
        for r in range(H):
            for x in range(W):
                if r0 <= r <= r1 and c0 <= x <= c1:
                    continue
                v = int(g[r, x])
                if v in swap:
                    out[r, x] = swap[v]
        return out

    for corner in ("tl", "tr", "bl", "br"):
        for mode in ("col", "row"):
            fn = (lambda g, cc=corner, mm=mode: build(g, cc, mm))
            if _verify(fn, train):
                return fn
    return None


# --------------------------------------------------------------------------- #
# a framed 'key' colour marks which scattered colour to erase everywhere else
# --------------------------------------------------------------------------- #
def erase_keyed_color(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def comps_color(g, color):
        H, W = g.shape
        seen = np.zeros((H, W), bool)
        out = []
        for r in range(H):
            for c in range(W):
                if g[r, c] == color and not seen[r, c]:
                    st = [(r, c)]
                    seen[r, c] = True
                    cells = [(r, c)]
                    while st:
                        y, x = st.pop()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < H and 0 <= nx < W and g[ny, nx] == color
                                        and not seen[ny, nx]):
                                    seen[ny, nx] = True
                                    st.append((ny, nx))
                                    cells.append((ny, nx))
                    out.append(cells)
        return out

    def find_marker(g):
        H, W = g.shape
        best = None
        for color in set(int(v) for v in np.unique(g)) - {0}:
            for cells in comps_color(g, color):
                if len(cells) < 3:
                    continue
                ys = [y for y, x in cells]
                xs = [x for y, x in cells]
                r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
                inside = [(y, x) for y in range(r0, r1 + 1) for x in range(c0, c1 + 1)
                          if g[y, x] != 0 and g[y, x] != color]
                if not inside:
                    continue
                keys = set(int(g[y, x]) for y, x in inside)
                if len(keys) == 1:
                    if best is not None:
                        return None  # ambiguous
                    best = (r0, r1, c0, c1, keys.pop())
        return best

    def apply(g):
        m = find_marker(g)
        if m is None:
            return None
        r0, r1, c0, c1, key = m
        out = g.copy()
        H, W = g.shape
        for r in range(H):
            for c in range(W):
                if r0 <= r <= r1 and c0 <= c <= c1:
                    continue
                if g[r, c] == key:
                    out[r, c] = 0
        return out

    fn = apply
    if _verify(fn, train):
        return fn
    return None


# --------------------------------------------------------------------------- #
# histogram bars driven by a count marker + a code row (below a divider line)
# --------------------------------------------------------------------------- #
def counted_code_bars(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def apply(g):
        H, W = g.shape
        rows5 = [r for r in range(H) if len(set(g[r, :].tolist())) == 1 and g[r, 0] != 0]
        if not rows5:
            return None
        fr = rows5[0]
        marker = int(g[fr, 0])
        # count of the "count colour" in the header (row 0), assume top row
        n = int((g[0, :] != 0).sum())
        # use the count of the most common non-zero colour in row 0
        top = [int(v) for v in g[0, :] if v != 0]
        if not top:
            return None
        n = len(top)
        if n < 1 or n > fr - 1:
            return None
        # code row: first non-empty row strictly below the divider
        coderow = None
        for r in range(fr + 1, H):
            if (g[r, :] != 0).any():
                coderow = r
                break
        if coderow is None:
            return None
        code = g[coderow, :]
        cc = Counter(int(v) for v in code if v != 0)
        targets = [d for d, cnt in cc.items() if cnt == n and d != marker]
        if not targets:
            return None
        out = g.copy()
        for d in targets:
            for c in range(W):
                if int(code[c]) == d:
                    for k in range(n):
                        rr = fr - 1 - k
                        if rr < 0:
                            return None
                        out[rr, c] = d
        return out

    fn = apply
    if _verify(fn, train):
        return fn
    return None


# --------------------------------------------------------------------------- #
# reverse the concentric-ring colour sequence of each object
# --------------------------------------------------------------------------- #
def reverse_concentric_rings(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def apply(g):
        bg = _bg_color(g)
        out = g.copy()
        changed = False
        for cells in _components(g, bg, diag=True):
            ys = [y for y, x in cells]
            xs = [x for y, x in cells]
            r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
            layer_color = {}
            consistent = True
            for (y, x) in cells:
                d = min(y - r0, r1 - y, x - c0, c1 - x)
                v = int(g[y, x])
                if d not in layer_color:
                    layer_color[d] = v
                elif layer_color[d] != v:
                    consistent = False
                    break
            if not consistent:
                return None
            maxd = max(layer_color)
            seq = []
            for d in range(maxd + 1):
                if d in layer_color:
                    if not seq or seq[-1] != layer_color[d]:
                        seq.append(layer_color[d])
            if len(seq) < 2:
                continue
            rev = seq[::-1]
            cmap = {a: b for a, b in zip(seq, rev)}
            for (y, x) in cells:
                nv = cmap.get(int(g[y, x]))
                if nv is not None:
                    out[y, x] = nv
            changed = True
        if not changed:
            return None
        return out

    fn = apply
    if _verify(fn, train):
        return fn
    return None


# --------------------------------------------------------------------------- #
# fill each hollow frame with a checkerboard of the noise colour whose dot-count
# matches the frame's interior checkerboard size; delete the loose dots
# --------------------------------------------------------------------------- #
def frame_checkerboard_fill(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def comps_color(g, color):
        H, W = g.shape
        seen = np.zeros((H, W), bool)
        out = []
        for r in range(H):
            for c in range(W):
                if g[r, c] == color and not seen[r, c]:
                    st = [(r, c)]
                    seen[r, c] = True
                    cells = [(r, c)]
                    while st:
                        y, x = st.pop()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < H and 0 <= nx < W and g[ny, nx] == color
                                        and not seen[ny, nx]):
                                    seen[ny, nx] = True
                                    st.append((ny, nx))
                                    cells.append((ny, nx))
                    out.append(cells)
        return out

    def apply(g):
        H, W = g.shape
        cnt = Counter(int(v) for v in g.flatten() if v != 0)
        frames = []
        noises = {}
        for color, n in cnt.items():
            comps = comps_color(g, color)
            if len(comps) == 1 and len(comps[0]) >= 8:
                cells = comps[0]
                ys = [y for y, x in cells]
                xs = [x for y, x in cells]
                r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
                # must be a hollow rectangle outline
                per = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
                if len(cells) == per and (r1 - r0) >= 2 and (c1 - c0) >= 2:
                    frames.append((color, r0, r1, c0, c1))
                else:
                    noises[color] = n
            else:
                noises[color] = n
        if not frames or not noises:
            return None
        out = g.copy()
        for color in noises:
            out[g == color] = 0
        used = set()
        for (color, r0, r1, c0, c1) in frames:
            cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            done = False
            for phase in (0, 1):
                cb = [(r, c) for (r, c) in cells
                      if ((r - (r0 + 1)) + (c - (c0 + 1))) % 2 == phase]
                match = [col for col, nn in noises.items()
                         if nn == len(cb) and col not in used]
                if match:
                    fillcol = match[0]
                    used.add(fillcol)
                    for (r, c) in cb:
                        out[r, c] = fillcol
                    done = True
                    break
            if not done:
                return None
        return out

    fn = apply
    if _verify(fn, train):
        return fn
    return None


# --------------------------------------------------------------------------- #
# translate all foreground by a fixed vector, repaint background a new colour
# --------------------------------------------------------------------------- #
def translate_and_repaint_bg(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # the new background colour (learned): value covering most of output where
    # input was background
    fills = set()
    for i, o in train:
        bg = _bg_color(i)
        vals = o[i == bg]
        if vals.size == 0:
            return None
        u = set(int(v) for v in np.unique(vals))
        fills = fills | u if not fills else (fills & u) if fills & u else fills | u
    # determine a single fill colour consistent across demos
    common = None
    for i, o in train:
        bg = _bg_color(i)
        vals = [int(v) for v in o[i == bg]]
        c = Counter(vals).most_common(1)[0][0]
        if common is None:
            common = c
        elif common != c:
            return None
    if common is None:
        return None

    def build(g, dr, dc, fill):
        H, W = g.shape
        bg = _bg_color(g)
        out = np.full((H, W), fill, dtype=int)
        for r in range(H):
            for c in range(W):
                if g[r, c] != bg:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr, nc] = g[r, c]
        return out

    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            fn = (lambda g, a=dr, b=dc, f=common: build(g, a, b, f))
            if _verify(fn, train):
                return fn
    return None


# --------------------------------------------------------------------------- #
# fill a hollow box interior with concentric (Chebyshev) rings around a seed dot
# --------------------------------------------------------------------------- #
def seed_ring_fill(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def box_and_seed(g, boxcol):
        ys, xs = np.where(g == boxcol)
        if len(ys) == 0:
            return None
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        if r1 - r0 < 3 or c1 - c0 < 3:
            return None
        seed = None
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if g[r, c] == boxcol:
                    if seed is not None:
                        return None
                    seed = (r, c)
        if seed is None:
            return None
        return r0, r1, c0, c1, seed

    def build(g, boxcol, ca, cb):
        bs = box_and_seed(g, boxcol)
        if bs is None:
            return None
        r0, r1, c0, c1, (sr, sc) = bs
        out = g.copy()
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                d = max(abs(r - sr), abs(c - sc))
                if d == 0:
                    continue
                out[r, c] = ca if d % 2 == 1 else cb
        return out

    # candidate box colour = the colour forming the outline
    boxcols = set()
    for i, o in train:
        boxcols |= set(int(v) for v in np.unique(i) if int(v) != _bg_color(i))
    fill_colors = set()
    for i, o in train:
        fill_colors |= set(int(v) for v in np.unique(o) if int(v) != _bg_color(o))
    for boxcol in sorted(boxcols):
        for ca in sorted(fill_colors):
            for cb in sorted(fill_colors):
                if ca == cb:
                    continue
                fn = (lambda g, bc=boxcol, a=ca, b=cb: build(g, bc, a, b))
                if _verify(fn, train):
                    return fn
    return None


# --------------------------------------------------------------------------- #
# connect aligned same-colour rectangles with a bridge (colour 8)
# --------------------------------------------------------------------------- #
def connect_aligned_rects(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def rects(g):
        H, W = g.shape
        seen = np.zeros((H, W), bool)
        out = []
        for r in range(H):
            for c in range(W):
                if g[r, c] != 0 and not seen[r, c]:
                    col = int(g[r, c])
                    st = [(r, c)]
                    seen[r, c] = True
                    cells = [(r, c)]
                    while st:
                        y, x = st.pop()
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < H and 0 <= nx < W and g[ny, nx] == col
                                    and not seen[ny, nx]):
                                seen[ny, nx] = True
                                st.append((ny, nx))
                                cells.append((ny, nx))
                    ys = [y for y, x in cells]
                    xs = [x for y, x in cells]
                    r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
                    # only accept solid rectangles
                    if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
                        out.append((col, r0, r1, c0, c1))
        return out

    def ov(a0, a1, b0, b1):
        return max(a0, b0), min(a1, b1)

    def apply(g):
        out = g.copy()
        rs = rects(g)
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                ca, ar0, ar1, ac0, ac1 = rs[a]
                cb, br0, br1, bc0, bc1 = rs[b]
                if ca != cb:
                    continue
                oc0, oc1 = ov(ac0, ac1, bc0, bc1)
                or0, or1 = ov(ar0, ar1, br0, br1)
                if or0 > or1 and oc0 <= oc1:  # vertical bridge
                    top, bot = sorted([(ar0, ar1), (br0, br1)])
                    gr0, gr1 = top[1] + 1, bot[0] - 1
                    if gr0 > gr1:
                        continue
                    blocked = False
                    for k in range(len(rs)):
                        if k in (a, b) or rs[k][0] != ca:
                            continue
                        kk = rs[k]
                        if not (kk[2] < gr0 or kk[1] > gr1):
                            k0, k1 = ov(kk[3], kk[4], oc0, oc1)
                            if k0 <= k1:
                                blocked = True
                                break
                    if blocked:
                        continue
                    for r in range(gr0, gr1 + 1):
                        for c in range(oc0 + 1, oc1):
                            if out[r, c] == 0:
                                out[r, c] = 8
                if oc0 > oc1 and or0 <= or1:  # horizontal bridge
                    left, right = sorted([(ac0, ac1), (bc0, bc1)])
                    gc0, gc1 = left[1] + 1, right[0] - 1
                    if gc0 > gc1:
                        continue
                    blocked = False
                    for k in range(len(rs)):
                        if k in (a, b) or rs[k][0] != ca:
                            continue
                        kk = rs[k]
                        if not (kk[4] < gc0 or kk[3] > gc1):
                            k0, k1 = ov(kk[1], kk[2], or0, or1)
                            if k0 <= k1:
                                blocked = True
                                break
                    if blocked:
                        continue
                    for r in range(or0 + 1, or1):
                        for c in range(gc0, gc1 + 1):
                            if out[r, c] == 0:
                                out[r, c] = 8
        return out

    fn = apply
    if _verify(fn, train):
        return fn
    return None


# Ordered by specificity
DETECTORS = [
    per_color_flip,
    per_component_flip,
    recolor_by_row_key,
    symmetry_axis_line,
    frame_by_noise_count,
    periodic_shift_fill,
    column_fill_up_from_below,
    latin_square_complete,
    corner_legend_swap,
    erase_keyed_color,
    counted_code_bars,
    reverse_concentric_rings,
    frame_checkerboard_fill,
    translate_and_repaint_bg,
    seed_ring_fill,
]
