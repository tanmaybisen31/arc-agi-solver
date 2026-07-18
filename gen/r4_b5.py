"""Round-4 batch-5 detectors for ARC-AGI.

Two general families, both defensive; the engine re-verifies exact reproduction
of every training pair before a transform is ever applied to a test input.

  * concat_objects   -- the grid contains several equal-shaped foreground
                        objects laid out along one axis (a row or a column of
                        objects).  The output is those object crops
                        concatenated in reading order along that axis.
  * block_dictionary -- the grid tiles into an R x C array of equal blocks and
                        every output block is a deterministic function of the
                        corresponding input block (a learnt block -> block
                        dictionary).  Requires genuine reuse of block values so
                        it generalises rather than memorising a single grid.

Only numpy + stdlib are used.
"""
import numpy as np
from collections import deque


# --------------------------------------------------------------------------- utils
def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag=True):
    """Connected components of non-background cells (colour agnostic)."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
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
            comps.append(cells)
    return comps


def _bbox(cells):
    ys = [y for y, _ in cells]
    xs = [x for _, x in cells]
    return min(ys), max(ys), min(xs), max(xs)


# ------------------------------------------------------ 1. concatenate equal objects
def _gather_equal_crops(g, diag):
    """Return list of (r0, c0, crop) for every foreground component if they all
    share the same bounding-box shape; else None."""
    bg = _bg(g)
    comps = _components(g, bg, diag)
    if len(comps) < 2:
        return None
    crops = []
    for cc in comps:
        r0, r1, c0, c1 = _bbox(cc)
        crops.append((r0, c0, g[r0:r1 + 1, c0:c1 + 1].copy()))
    shp = crops[0][2].shape
    if any(cr.shape != shp for _, _, cr in crops):
        return None
    return crops


def _arrangement_axis(crops):
    """'h' if the objects are spread out mostly along columns, else 'v'."""
    r_centers = [(r0 + cr.shape[0] / 2.0) for r0, _, cr in crops]
    c_centers = [(c0 + cr.shape[1] / 2.0) for _, c0, cr in crops]
    rspread = max(r_centers) - min(r_centers)
    cspread = max(c_centers) - min(c_centers)
    return 'h' if cspread >= rspread else 'v'


def _concat(crops, axis):
    if axis == 'h':
        ordered = sorted(crops, key=lambda t: (t[1], t[0]))
        return np.hstack([cr for _, _, cr in ordered])
    ordered = sorted(crops, key=lambda t: (t[0], t[1]))
    return np.vstack([cr for _, _, cr in ordered])


def concat_objects(train):
    """Equal-shaped objects concatenated along their arrangement axis."""
    try:
        for diag in (True, False):
            def fn(g, diag=diag):
                crops = _gather_equal_crops(g, diag)
                if crops is None:
                    return None
                axis = _arrangement_axis(crops)
                return _concat(crops, axis)

            ok = True
            for i, o in train:
                out = fn(i)
                if out is None or out.shape != o.shape or not np.array_equal(out, o):
                    ok = False
                    break
            if ok:
                return fn
        return None
    except Exception:
        return None


# --------------------------------------------------------- 2. block -> block dictionary
def _divisors(n):
    return [d for d in range(2, n + 1) if n % d == 0]


def block_dictionary(train):
    """Grid divides into an R x C array of >=2x2 blocks; each output block is a
    deterministic function of the corresponding input block.  A learnt
    dictionary maps input-block bytes to output blocks.  We require genuine
    reuse (total blocks seen >= 2 x distinct) so the rule generalises."""
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        H, W = train[0][0].shape
        if any(i.shape != (H, W) for i, o in train):
            return None
        for bh in _divisors(H):
            for bw in _divisors(W):
                nR, nC = H // bh, W // bw
                if nR * nC < 3:
                    continue
                mp = {}
                ok = True
                total = 0
                for i, o in train:
                    for r in range(nR):
                        for c in range(nC):
                            ib = i[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                            ob = o[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                            key = ib.tobytes()
                            total += 1
                            if key in mp:
                                if not np.array_equal(mp[key], ob):
                                    ok = False
                                    break
                            else:
                                mp[key] = ob.copy()
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                if total < 2 * len(mp):        # demand real reuse -> generalisation
                    continue
                # skip pure identity dictionaries (base identity covers those)
                if all(np.array_equal(
                        np.frombuffer(k, dtype=train[0][0].dtype).reshape(bh, bw), v)
                        for k, v in mp.items()):
                    continue

                def fn(g, bh=bh, bw=bw, mp=dict(mp)):
                    H2, W2 = g.shape
                    if H2 % bh or W2 % bw:
                        return None
                    out = g.copy()
                    for r in range(H2 // bh):
                        for c in range(W2 // bw):
                            key = g[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw].tobytes()
                            if key not in mp:
                                return None
                            out[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw] = mp[key]
                    return out

                if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
                    return fn
        return None
    except Exception:
        return None


# ------------------------------------------------------------- 3. colour histogram
def color_barchart(train):
    """Count each foreground colour and draw a bottom-anchored bar chart: the
    colours (sorted) occupy columns 0, 1, 2, ... with bar height equal to their
    count.  Output has the same shape as the input; everything else is
    background."""
    try:
        if any(i.shape != o.shape for i, o in train):
            return None

        def fn(g):
            H, W = g.shape
            vals = g[g != 0]
            if vals.size == 0:
                return None
            colors = sorted(set(int(v) for v in vals.tolist()))
            if len(colors) > W:
                return None
            out = np.zeros((H, W), dtype=int)
            for ci, col in enumerate(colors):
                h = int((g == col).sum())
                if h > H:
                    return None
                for k in range(h):
                    out[H - 1 - k, ci] = col
            return out

        for i, o in train:
            out = fn(i)
            if out is None or not np.array_equal(out, o):
                return None
        # guard against degenerate identity/no-op fits
        if all(np.array_equal(i, o) for i, o in train):
            return None
        return fn
    except Exception:
        return None


# ------------------------------------------------- 4. two dots -> cross + filled box
def two_dot_cross_box(train):
    """Exactly two foreground dots of the same colour: draw full-width/height
    lines through both dot rows and both dot columns in the dot colour, and fill
    the interior rectangle strictly between them with a learnt fill colour."""
    try:
        if any(i.shape != o.shape for i, o in train):
            return None

        def dots(g):
            ys, xs = np.where(g != 0)
            if len(ys) != 2:
                return None
            if g[ys[0], xs[0]] != g[ys[1], xs[1]]:
                return None
            return (int(ys[0]), int(xs[0])), (int(ys[1]), int(xs[1])), int(g[ys[0], xs[0]])

        def build(g, fill):
            d = dots(g)
            if d is None:
                return None
            (r0, c0), (r1, c1), col = d
            H, W = g.shape
            out = np.zeros((H, W), dtype=int)
            out[r0, :] = col
            out[r1, :] = col
            out[:, c0] = col
            out[:, c1] = col
            rr0, rr1 = sorted((r0, r1))
            cc0, cc1 = sorted((c0, c1))
            if rr1 - rr0 > 1 and cc1 - cc0 > 1:
                out[rr0 + 1:rr1, cc0 + 1:cc1] = fill
            return out

        # learn a fill colour that reproduces all training pairs
        for fill in range(0, 10):
            if all(build(i, fill) is not None and np.array_equal(build(i, fill), o)
                   for i, o in train):
                return (lambda g, fill=fill: build(g, fill))
        return None
    except Exception:
        return None


# ------------------------------------------ 5. template boxes filled by matched blobs
def _crop_mask(m):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def template_box_fill(train):
    """Input = a framed 'template' half beside a half holding coloured blobs.
    Each template box (a row-band framed by full frame-colour rows) has a hollow
    interior; its interior shape matches exactly one coloured blob, and the box
    is filled with that blob's colour.  Output is the filled template half."""
    try:
        # output width equals half the input width (left template kept)
        ows = set(o.shape[1] for _, o in train)
        if len(ows) != 1:
            # allow per-pair; use input-derived width instead
            pass
        for i, o in train:
            if o.shape[0] != i.shape[0]:
                return None
            if i.shape[1] != 2 * o.shape[1]:
                return None

        def solve(a, ow):
            H, W = a.shape
            if W < 2 * ow:
                return None
            left = a[:, :ow]
            right = a[:, ow:2 * ow]
            nz = [int(v) for v in np.unique(left) if v != 0]
            if len(nz) != 1:
                return None
            fc = nz[0]
            fullrows = [r for r in range(H) if bool((left[r] == fc).all())]
            segs = []
            for k in range(len(fullrows) - 1):
                if fullrows[k + 1] - fullrows[k] > 1:
                    segs.append((fullrows[k] + 1, fullrows[k + 1]))
            if not segs:
                return None
            rshapes = {}
            for cval in [int(c) for c in np.unique(right) if c != 0]:
                rshapes[cval] = _crop_mask(right == cval)
            out = left.copy()
            used = set()
            for (r0, r1) in segs:
                zm = (left[r0:r1, :] == 0)
                cm = _crop_mask(zm)
                if cm is None:
                    continue
                matched = None
                for cval, shape in rshapes.items():
                    if cval in used or shape is None:
                        continue
                    if shape.shape == cm.shape and np.array_equal(shape.astype(bool), cm):
                        matched = cval
                        break
                if matched is None:
                    return None
                used.add(matched)
                for rr in range(r0, r1):
                    for cc in range(ow):
                        if left[rr, cc] == 0:
                            out[rr, cc] = matched
            return out

        def fn(g):
            ow = g.shape[1] // 2
            return solve(g, ow)

        for i, o in train:
            out = fn(i)
            if out is None or out.shape != o.shape or not np.array_equal(out, o):
                return None
        return fn
    except Exception:
        return None


DETECTORS = [concat_objects, block_dictionary, color_barchart,
             two_dot_cross_box, template_box_fill]
