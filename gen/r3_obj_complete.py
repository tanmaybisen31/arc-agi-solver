"""Object-completion detectors for ARC-AGI.

Family: complete partial / occluded objects.

  * hole_fill_map / hole_fill_parity - each object is a closed outline (a box, a
    ring, a loop). The cells *enclosed* by the outline (background cells that
    cannot reach the grid border) are its interior; the task fills that interior
    with a color. The fill color is either constant or a function of the object
    (its bounding-box smaller side, larger side, height, width, area, hole
    count, or the parity of its smaller side). We learn whichever feature maps
    consistently to a fill color across every training object.

  * bbox_fill - a partial object (an L, a broken ring, a diagonal) is completed
    into a solid filled rectangle: every background cell inside the object's
    bounding box is painted a single learned fill color.

  * obj_local_symmetry - each object carries its own local mirror / rotational
    symmetry that is only partly drawn; the object is completed by reflecting
    its existing marks into the empty cells of its own bounding box.

All detectors are same-shape, additive (they only turn background cells into
non-background), and the harness verifies every training pair exactly before a
transform is ever used, so a wrong inference is auto-rejected. numpy + stdlib
only, defensive throughout.
"""
import numpy as np
from collections import deque, Counter


# --------------------------------------------------------------- helpers
def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag):
    """Connected components of non-bg cells. Returns list of cell lists."""
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    if diag:
        nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
                for dy, dx in nb:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _bbox(cells):
    ys = [y for y, x in cells]
    xs = [x for y, x in cells]
    return min(ys), max(ys), min(xs), max(xs)


def _enclosed_mask(g, bg, diag):
    """Background cells that cannot reach the grid border (interior holes)."""
    H, W = g.shape
    reach = np.zeros((H, W), dtype=bool)
    dq = deque()
    for r in range(H):
        for c in (0, W - 1):
            if g[r, c] == bg and not reach[r, c]:
                reach[r, c] = True
                dq.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if g[r, c] == bg and not reach[r, c]:
                reach[r, c] = True
                dq.append((r, c))
    if diag:
        nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while dq:
        y, x = dq.popleft()
        for dy, dx in nb:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and g[ny, nx] == bg and not reach[ny, nx]:
                reach[ny, nx] = True
                dq.append((ny, nx))
    return (g == bg) & (~reach)


def _feats(h, w, hole):
    return {
        "min": min(h, w), "max": max(h, w),
        "h": h, "w": w, "area": h * w, "hole": hole,
        "par_min": min(h, w) % 2, "par_max": max(h, w) % 2,
    }

# Feature keys tried in order.  Constant / structural features first so that a
# task with a single fill color (any feature works) locks onto the most
# generalizable rule, and parity features last as robust fallbacks that cover
# object sizes unseen in training.
_FEAT_KEYS = ["par_min", "par_max", "min", "max", "h", "w", "hole", "area"]


# ----------------------------------------------- 1/2. enclosed-interior fill
def _learn_hole_map(train, prefer_keys):
    """Collect (features -> fill_color) samples over all training objects, verify
    that non-hole cells are unchanged, and return (key, mapping) for the first
    feature key (from prefer_keys) that maps consistently, else None."""
    samples = []
    for i, o in train:
        if i.shape != o.shape:
            return None
        bg = _bg(i)
        gm = _enclosed_mask(i, bg, diag=False)
        if not gm.any():
            return None
        # every cell that is NOT an interior hole must be unchanged
        if not np.array_equal(i[~gm], o[~gm]):
            return None
        for cells in _components(i, bg, diag=False):
            r0, r1, c0, c1 = _bbox(cells)
            si = i[r0:r1 + 1, c0:c1 + 1]
            so = o[r0:r1 + 1, c0:c1 + 1]
            m = _enclosed_mask(si, bg, diag=False)
            if not m.any():
                continue
            fills = set(int(v) for v in so[m].tolist())
            if len(fills) != 1:
                return None
            fc = fills.pop()
            if fc == bg:
                return None
            samples.append((_feats(r1 - r0 + 1, c1 - c0 + 1, int(m.sum())), fc))
    if not samples:
        return None
    for key in prefer_keys:
        mp = {}
        ok = True
        for fd, fc in samples:
            k = fd[key]
            if k in mp and mp[k] != fc:
                ok = False
                break
            mp[k] = fc
        if ok:
            return key, mp
    return None


def _make_hole_fill(key, mp):
    def fn(g):
        g = np.asarray(g, dtype=int)
        bg = _bg(g)
        out = g.copy()
        for cells in _components(g, bg, diag=False):
            r0, r1, c0, c1 = _bbox(cells)
            sub = out[r0:r1 + 1, c0:c1 + 1]
            m = _enclosed_mask(sub, bg, diag=False)
            if not m.any():
                continue
            fd = _feats(r1 - r0 + 1, c1 - c0 + 1, int(m.sum()))
            k = fd[key]
            if k not in mp:
                return None
            sub[m] = mp[k]
        return out
    return fn


def hole_fill_map(train):
    """Fill each object's enclosed interior; color = consistent feature map,
    preferring constant/structural features (min side, area, ...)."""
    try:
        res = _learn_hole_map(train, ["min", "max", "h", "w", "hole", "area"])
        if res is None:
            return None
        fn = _make_hole_fill(*res)
        for i, o in train:
            r = fn(i)
            if r is None or not np.array_equal(r, o):
                return None
        if all(np.array_equal(i, o) for i, o in train):
            return None
        return fn
    except Exception:
        return None


def hole_fill_parity(train):
    """Same as hole_fill_map but prefers parity-of-side features, which extend to
    object sizes not present in training (odd side -> A, even side -> B)."""
    try:
        res = _learn_hole_map(train, ["par_min", "par_max"])
        if res is None:
            return None
        fn = _make_hole_fill(*res)
        for i, o in train:
            r = fn(i)
            if r is None or not np.array_equal(r, o):
                return None
        if all(np.array_equal(i, o) for i, o in train):
            return None
        return fn
    except Exception:
        return None


# --------------------------------------------------- 3. bounding-box completion
def bbox_fill(train):
    """Complete each partial object into a solid filled rectangle: every
    background cell inside the object's bounding box is painted one learned
    fill color. Uses 8-connectivity so diagonal fragments count as one object."""
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        for diag in (True, False):
            bg0 = _bg(train[0][0])
            fills = set()
            consistent = True
            for i, o in train:
                bg = _bg(i)
                for cells in _components(i, bg, diag):
                    r0, r1, c0, c1 = _bbox(cells)
                    ii = i[r0:r1 + 1, c0:c1 + 1]
                    oo = o[r0:r1 + 1, c0:c1 + 1]
                    fills |= set(int(v) for v in oo[ii == bg].tolist())
            fills.discard(bg0)
            if len(fills) != 1:
                continue
            fc = fills.pop()

            def fn(g, diag=diag, fc=fc):
                g = np.asarray(g, dtype=int)
                bg = _bg(g)
                out = g.copy()
                for cells in _components(g, bg, diag):
                    r0, r1, c0, c1 = _bbox(cells)
                    sub = out[r0:r1 + 1, c0:c1 + 1]
                    sub[sub == bg] = fc
                return out

            ok = True
            for i, o in train:
                if not np.array_equal(fn(i), o):
                    ok = False
                    break
            if ok and not all(np.array_equal(i, o) for i, o in train):
                return fn
        return None
    except Exception:
        return None


# ------------------------------------------------ 4. per-object local symmetry
def _sym_maps(H, W):
    v = [lambda r, c: (r, W - 1 - c),
         lambda r, c: (H - 1 - r, c),
         lambda r, c: (H - 1 - r, W - 1 - c)]
    if H == W:
        v += [lambda r, c: (c, r),
              lambda r, c: (W - 1 - c, H - 1 - r),
              lambda r, c: (c, H - 1 - r),
              lambda r, c: (W - 1 - c, r)]
    return v


def _symmetrize_sub(sub, bg):
    """Complete a partially drawn symmetric patch: for each cell take the unique
    non-bg value across its symmetry orbit under whichever global mirror/rotation
    maps are consistent with the existing marks. Returns None if no non-trivial
    symmetry holds."""
    H, W = sub.shape
    nz = sub != bg
    if nz.sum() < 2:
        return None
    good = []
    for f in _sym_maps(H, W):
        ok = True
        touched = False
        for r in range(H):
            for c in range(W):
                if not nz[r, c]:
                    continue
                r2, c2 = f(r, c)
                if 0 <= r2 < H and 0 <= c2 < W:
                    if nz[r2, c2]:
                        if sub[r2, c2] != sub[r, c]:
                            ok = False
                            break
                        touched = True
            if not ok:
                break
        if ok and touched:
            good.append(f)
    if not good:
        return None
    out = sub.copy()
    for r in range(H):
        for c in range(W):
            seen = {(r, c)}
            dq = deque([(r, c)])
            val = None
            conflict = False
            while dq:
                y, x = dq.popleft()
                if sub[y, x] != bg:
                    if val is None:
                        val = sub[y, x]
                    elif val != sub[y, x]:
                        conflict = True
                for f in good:
                    ny, nx = f(y, x)
                    if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        dq.append((ny, nx))
            if val is not None and not conflict:
                out[r, c] = val
    return out


def obj_local_symmetry(train):
    """Complete each object by its own local mirror/rotational symmetry, working
    on the bounding box of each connected object (8-connectivity)."""
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        for diag in (True, False):
            def fn(g, diag=diag):
                g = np.asarray(g, dtype=int)
                bg = _bg(g)
                out = g.copy()
                for cells in _components(g, bg, diag):
                    r0, r1, c0, c1 = _bbox(cells)
                    if (r1 - r0 + 1) * (c1 - c0 + 1) < 4:
                        continue
                    sub = out[r0:r1 + 1, c0:c1 + 1]
                    s = _symmetrize_sub(sub, bg)
                    if s is not None:
                        out[r0:r1 + 1, c0:c1 + 1] = s
                return out

            ok = True
            for i, o in train:
                if not np.array_equal(fn(i), o):
                    ok = False
                    break
            if ok and not all(np.array_equal(i, o) for i, o in train):
                return fn
        return None
    except Exception:
        return None


DETECTORS = [hole_fill_map, hole_fill_parity, bbox_fill, obj_local_symmetry]
