"""Round-2 batch-0 detectors for previously-unsolved ARC-AGI eval tasks.

Each detector: def det(train) -> transform_fn | None
The engine verifies the returned fn reproduces every train pair exactly before
using it, so detectors infer a general rule and stay defensive.
"""
import numpy as np
from collections import Counter, deque


def bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


# =====================================================================
# Family: dilate a seed color into its 0-neighbors (3x3), keep others.
#   e.g. f0df5ff0 — every `1` cell floods its background neighbourhood
#   with 1s but does not overwrite other non-background colours.
# =====================================================================
def dilate_seed_color(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # Determine which color got dilated: cells that changed all went 0->X
    seed = None
    for i, o in train:
        diff = (i != o)
        if not diff.any():
            continue
        olds = set(int(i[r, c]) for r, c in zip(*np.where(diff)))
        news = set(int(o[r, c]) for r, c in zip(*np.where(diff)))
        if olds != {0}:
            return None
        if len(news) != 1:
            return None
        s = next(iter(news))
        if seed is None:
            seed = s
        elif seed != s:
            return None
    if seed is None:
        return None

    def make(diag):
        if diag:
            offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1)]
        else:
            offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def fn(g, seed=seed, offs=offs):
            out = g.copy()
            H, W = g.shape
            src = np.argwhere(g == seed)
            for r, c in src:
                for dr, dc in offs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == 0:
                        out[nr, nc] = seed
            return out
        return fn

    for diag in (True, False):
        fn = make(diag)
        if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
            return fn
    return None


# =====================================================================
# Family: invert foreground/background mask + recolor.
#   Output cell = mapped-color where input was background, else background.
#   e.g. 6ea4a07e (3x3 grids, 8->2, 3->1, 5->4 recoloring on inversion).
# =====================================================================
def invert_recolor(train):
    # Background is taken as 0 (the "empty" color). Output places a recoloured
    # foreground where the input was empty and empties where input was filled.
    if any(i.shape != o.shape for i, o in train):
        return None
    cmap = {}          # input-fg-color -> output-fg-color
    for i, o in train:
        bg = 0
        # inverted mask: input==0 cells are the new foreground.
        inmask = (i == bg)
        outfg = (o != bg)
        if not np.array_equal(inmask, outfg):
            return None
        in_fg_cols = set(np.unique(i[i != bg]).tolist())
        out_fg_cols = set(np.unique(o[o != bg]).tolist())
        if len(in_fg_cols) != 1 or len(out_fg_cols) != 1:
            return None
        a = next(iter(in_fg_cols))
        b = next(iter(out_fg_cols))
        if a in cmap and cmap[a] != b:
            return None
        cmap[a] = b
    if not cmap:
        return None

    def fn(g, cmap=cmap):
        bg = 0
        fg = set(np.unique(g[g != bg]).tolist())
        if len(fg) != 1:
            return None
        a = next(iter(fg))
        b = cmap.get(a)
        if b is None:
            if len(set(cmap.values())) == 1:
                b = next(iter(set(cmap.values())))
            else:
                return None
        out = np.full_like(g, bg)
        out[g == bg] = b
        return out
    return fn


# =====================================================================
# Family: output depends only on position parity/period, independent of the
#   (blank / uniform) input.  Learn out[r,c] as a function of (r%ph, c%pw).
#   e.g. 332efdb3 — blank input -> alternating grid of 1s.
# =====================================================================
def positional_periodic(train):
    # Only makes sense when the input is uniform (no information) but output
    # varies over position.
    for i, o in train:
        if i.shape != o.shape:
            return None
        if len(np.unique(i)) != 1:
            return None
    # find smallest periods that explain every output
    def find_period(axis_len, getval, maxp):
        for p in range(1, maxp + 1):
            ok = True
            for k in range(axis_len):
                if getval(k) != getval(k % p):
                    # can't check like this; handled below
                    pass
            # We instead just try p and verify tiling per-output later.
            return None
    # Simpler: learn a lookup table keyed by (r%PH, c%PW) shared across all
    # outputs, searching small PH,PW.
    for PH in range(1, 7):
        for PW in range(1, 7):
            table = {}
            consistent = True
            for _, o in train:
                H, W = o.shape
                for r in range(H):
                    for c in range(W):
                        k = (r % PH, c % PW)
                        v = int(o[r, c])
                        if k in table and table[k] != v:
                            consistent = False
                            break
                        table[k] = v
                    if not consistent:
                        break
                if not consistent:
                    break
            if not consistent:
                continue
            # require the pattern to actually be periodic (not degenerate:
            # PH*PW should be smaller than a single output area, else it's
            # just memorizing the whole grid)
            if PH * PW >= min(o.shape[0] * o.shape[1] for _, o in train):
                continue

            def fn(g, table=dict(table), PH=PH, PW=PW):
                H, W = g.shape
                out = np.zeros((H, W), dtype=int)
                for r in range(H):
                    for c in range(W):
                        v = table.get((r % PH, c % PW))
                        if v is None:
                            return None
                        out[r, c] = v
                return out
            if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
                return fn
    return None


# =====================================================================
# Family: tile a corner "stamp" to fill each framed box.
#   Background is uniform (e.g. 8). Enclosed rectangular regions contain a
#   small stamp (bbox of nonzero cells) in a corner; the rest is 0. Output
#   tiles the stamp across the whole box. e.g. a57f2f04.
# =====================================================================
def _regions_nonbg(g, bg):
    """Bounding boxes of 4-connected components of cells != bg (treating 0 as
    part of the content so a box's empty interior stays with its stamp)."""
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out = []
    for r in range(H):
        for c in range(W):
            if not seen[r, c] and g[r, c] != bg:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = [(r, c)]
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                            cells.append((ny, nx))
                ys = [y for y, x in cells]
                xs = [x for y, x in cells]
                out.append((min(ys), max(ys), min(xs), max(xs)))
    return out


def tile_stamp_in_box(train):
    # background = most common color, expected uniform border color.
    def apply(g):
        bg = bg_color(g)
        out = g.copy()
        H, W = g.shape
        # Content = cells that are not bg and not 0 (the actual stamp colors).
        content = (g != bg) & (g != 0)
        if not content.any():
            return None
        # Boxes = connected regions of (bg-complement); interior is 0.
        for r0, r1, c0, c1 in _regions_nonbg(g, bg):
            box = g[r0:r1 + 1, c0:c1 + 1]
            nz = np.argwhere(box != 0)
            if len(nz) == 0:
                continue
            sr0, sc0 = nz.min(0)
            sr1, sc1 = nz.max(0)
            # stamp anchored at box origin: dimensions from nonzero extent.
            sh = sr1 + 1
            sw = sc1 + 1
            # require the stamp to sit in the top-left corner of the box.
            if sr0 != 0 or sc0 != 0:
                # try to anchor stamp by its own bbox and tile from box origin
                sh = sr1 - sr0 + 1
                sw = sc1 - sc0 + 1
                stamp = box[sr0:sr1 + 1, sc0:sc1 + 1]
            else:
                stamp = box[0:sh, 0:sw]
            bh, bw = box.shape
            filled = np.zeros_like(box)
            for rr in range(bh):
                for cc in range(bw):
                    filled[rr, cc] = stamp[rr % sh, cc % sw]
            out[r0:r1 + 1, c0:c1 + 1] = filled
        return out

    try:
        if all(apply(i) is not None and apply(i).shape == o.shape and np.array_equal(apply(i), o)
               for i, o in train):
            return apply
    except Exception:
        return None
    return None


# =====================================================================
# Family: stamp a template shape onto lone anchor markers.
#   One multicell component contains a single anchor-colored cell; other
#   lone cells of the anchor color are targets. The template is copied so its
#   anchor aligns with each target, EXCLUDING the anchor cell itself.
#   e.g. 2c737e39.
# =====================================================================
def _components_diag(g, bg):
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out = []
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r in range(H):
        for c in range(W):
            if not seen[r, c] and g[r, c] != bg:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = [(r, c)]
                while q:
                    y, x = q.popleft()
                    for dy, dx in nbrs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                            cells.append((ny, nx))
                out.append(cells)
    return out


def stamp_on_anchor(train):
    def apply(g, anchor):
        H, W = g.shape
        comps = _components_diag(g, 0)
        template = None
        targets = []
        for c in comps:
            colset = set(int(g[y, x]) for y, x in c)
            if len(c) == 1 and int(g[c[0][0], c[0][1]]) == anchor:
                targets.append(c[0])
            elif anchor in colset and len(c) > 1:
                if template is not None:
                    return None      # ambiguous
                template = c
        if template is None or not targets:
            return None
        ap = [(y, x) for y, x in template if g[y, x] == anchor]
        if len(ap) != 1:
            return None
        ay, ax = ap[0]
        out = g.copy()
        for ty, tx in targets:
            out[ty, tx] = 0           # erase the lone marker
            dr, dc = ty - ay, tx - ax
            for (y, x) in template:
                if g[y, x] == anchor:
                    continue          # don't copy the marker itself
                ny, nx = y + dr, x + dc
                if 0 <= ny < H and 0 <= nx < W:
                    out[ny, nx] = g[y, x]
        return out

    # pick the anchor color that both forms lone cells and sits inside a shape
    cand = set()
    for i, _ in train:
        for col in range(10):
            if col == 0:
                continue
            cand.add(col)
    for anchor in sorted(cand):
        try:
            if all(apply(i, anchor) is not None and apply(i, anchor).shape == o.shape
                   and np.array_equal(apply(i, anchor), o) for i, o in train):
                return (lambda g, a=anchor: apply(g, a))
        except Exception:
            continue
    return None


# =====================================================================
# Family: markers embedded in a 1-line grow perpendicular columns.
#   A straight line of 1s contains a few non-1 markers. Each marker sprouts a
#   perpendicular bar of 1s of a fixed per-color length, capped by the marker
#   color at the far end. e.g. 72a961c9.
# =====================================================================
def line_marker_growth(train):
    # learn per-color bar length and growth direction from the demos.
    def find_line(g):
        H, W = g.shape
        # a full-ish straight line made only of {0-excluded} 1s + markers
        for r in range(H):
            row = g[r]
            nz = row[row != 0]
            if len(nz) >= 3 and 1 in set(nz.tolist()) and (row != 0).sum() >= W - 1:
                return ("h", r)
        for c in range(W):
            col = g[:, c]
            nz = col[col != 0]
            if len(nz) >= 3 and 1 in set(nz.tolist()) and (col != 0).sum() >= H - 1:
                return ("v", c)
        return None

    lengths = {}   # marker color -> bar length (cells beyond the line)

    def learn(g, o):
        fl = find_line(g)
        if fl is None:
            return False
        axis, idx = fl
        H, W = g.shape
        if axis == "h":
            for c in range(W):
                v = int(g[idx, c])
                if v in (0, 1):
                    continue
                up = [r for r in range(idx) if o[r, c] != 0]
                dn = [r for r in range(idx + 1, H) if o[r, c] != 0]
                seg = up if len(up) >= len(dn) else dn
                L = len(seg)
                if v in lengths and lengths[v] != L:
                    return False
                lengths[v] = L
        else:
            for r in range(H):
                v = int(g[r, idx])
                if v in (0, 1):
                    continue
                lf = [c for c in range(idx) if o[r, c] != 0]
                rt = [c for c in range(idx + 1, W) if o[r, c] != 0]
                seg = lf if len(lf) >= len(rt) else rt
                L = len(seg)
                if v in lengths and lengths[v] != L:
                    return False
                lengths[v] = L
        return True

    for i, o in train:
        if i.shape != o.shape:
            return None
        if not learn(i, o):
            return None
    if not lengths:
        return None

    def grow(g, up_dir):
        fl = find_line(g)
        if fl is None:
            return None
        axis, idx = fl
        H, W = g.shape
        out = g.copy()
        if axis == "h":
            for c in range(W):
                v = int(g[idx, c])
                if v in (0, 1):
                    continue
                L = lengths.get(v)
                if L is None or L == 0:
                    continue
                if up_dir:
                    cells = list(range(idx - 1, idx - 1 - L, -1))
                else:
                    cells = list(range(idx + 1, idx + 1 + L))
                for k, rr in enumerate(cells):
                    if 0 <= rr < H:
                        out[rr, c] = v if k == L - 1 else 1
        else:
            for r in range(H):
                v = int(g[r, idx])
                if v in (0, 1):
                    continue
                L = lengths.get(v)
                if L is None or L == 0:
                    continue
                if up_dir:
                    cells = list(range(idx - 1, idx - 1 - L, -1))
                else:
                    cells = list(range(idx + 1, idx + 1 + L))
                for k, cc in enumerate(cells):
                    if 0 <= cc < W:
                        out[r, cc] = v if k == L - 1 else 1
        return out

    for up_dir in (True, False):
        fn = (lambda g, u=up_dir: grow(g, u))
        try:
            if all(fn(i) is not None and fn(i).shape == o.shape and np.array_equal(fn(i), o)
                   for i, o in train):
                return fn
        except Exception:
            continue
    return None


# =====================================================================
# Family: two panels separated by a uniform line; merge if their non-zero
#   masks are disjoint, else fall back to one panel. e.g. bbb1b8b6.
# =====================================================================
def _split_panels(g):
    H, W = g.shape
    # vertical separator column
    for c in range(1, W - 1):
        col = g[:, c]
        if len(set(col.tolist())) == 1 and col[0] != 0:
            L, R = g[:, :c], g[:, c + 1:]
            if L.shape == R.shape and L.size > 0:
                return L, R
    # horizontal separator row
    for r in range(1, H - 1):
        row = g[r]
        if len(set(row.tolist())) == 1 and row[0] != 0:
            T, B = g[:r], g[r + 1:]
            if T.shape == B.shape and T.size > 0:
                return T, B
    # plain halves
    if W % 2 == 0:
        return g[:, :W // 2], g[:, W // 2:]
    if H % 2 == 0:
        return g[:H // 2, :], g[H // 2:, :]
    return None


def conditional_panel_merge(train):
    i0, o0 = train[0]
    sp = _split_panels(i0)
    if sp is None or sp[0].shape != o0.shape:
        return None

    def build(base_idx):
        def fn(g, base_idx=base_idx):
            s = _split_panels(g)
            if s is None:
                return None
            A, B = s
            if A.shape != B.shape:
                return None
            disjoint = not ((A != 0) & (B != 0)).any()
            if disjoint:
                out = A.copy()
                out[B != 0] = B[B != 0]
                return out
            return (A if base_idx == 0 else B).copy()
        return fn

    for base_idx in (0, 1):
        fn = build(base_idx)
        try:
            if all(fn(i) is not None and fn(i).shape == o.shape and np.array_equal(fn(i), o)
                   for i, o in train):
                return fn
        except Exception:
            continue
    return None


# =====================================================================
# Family: output is the single color that forms a full constant line.
#   Grid has stripes; exactly one color fully fills a whole row or column.
#   Output is a 1x1 grid of that color. e.g. 1a2e2828.
# =====================================================================
def full_line_color(train):
    if not all(o.shape == (1, 1) for _, o in train):
        return None

    def color_of(g):
        H, W = g.shape
        cols = set()
        for r in range(H):
            row = g[r]
            if row[0] != 0 and len(set(row.tolist())) == 1:
                cols.add(int(row[0]))
        for c in range(W):
            col = g[:, c]
            if col[0] != 0 and len(set(col.tolist())) == 1:
                cols.add(int(col[0]))
        if len(cols) == 1:
            return next(iter(cols))
        return None

    def fn(g):
        v = color_of(g)
        if v is None:
            return None
        return np.array([[v]], dtype=int)

    if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
        return fn
    return None


# =====================================================================
# Family: point-reflection of all cells through a unique pivot cell.
#   One color appears exactly once and marks the pivot; every other cell is
#   reflected 180 degrees around it. e.g. 90347967.
# =====================================================================
def point_reflection_pivot(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # find the pivot color: appears exactly once in every input and is fixed.
    counts_per = []
    for i, _ in train:
        vals, cnts = np.unique(i[i != 0], return_counts=True)
        singles = set(int(v) for v, c in zip(vals, cnts) if c == 1)
        counts_per.append(singles)
    common_singles = set.intersection(*counts_per) if counts_per else set()
    if not common_singles:
        return None

    def make(pivot):
        def fn(g, pivot=pivot):
            pos = np.argwhere(g == pivot)
            if len(pos) != 1:
                return None
            py, px = pos[0]
            H, W = g.shape
            out = np.zeros_like(g)
            for y, x in np.argwhere(g != 0):
                ny, nx = 2 * py - y, 2 * px - x
                if 0 <= ny < H and 0 <= nx < W:
                    out[ny, nx] = g[y, x]
            return out
        return fn

    for pivot in sorted(common_singles):
        fn = make(pivot)
        try:
            if all(fn(i) is not None and fn(i).shape == o.shape and np.array_equal(fn(i), o)
                   for i, o in train):
                return fn
        except Exception:
            continue
    return None


DETECTORS = [
    dilate_seed_color,
    invert_recolor,
    positional_periodic,
    tile_stamp_in_box,
    stamp_on_anchor,
    line_marker_growth,
    conditional_panel_merge,
    full_line_color,
    point_reflection_pivot,
]
