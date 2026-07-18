"""Round 2, batch 2 detectors for ARC-AGI eval tasks.

General rule families discovered from unsolved eval tasks. Each detector infers
a rule from train pairs and returns a transform; the engine verifies it
reproduces every train demo before use.
"""
import numpy as np
from collections import Counter


def _bg(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[np.argmax(c)])


def _comps(mask, diag=True):
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    if diag:
        nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in nb:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append(cells)
    return out


# ------------------------------------------------------------------
# Family: grid divided by uniform separator lines -> downsample each
# cell to its majority color (output shape = #row-bands x #col-bands)
# ------------------------------------------------------------------
def grid_downsample(train):
    def bands(seps, N):
        b = []
        start = 0
        for s in sorted(seps) + [N]:
            if s > start:
                b.append((start, s))
            start = s + 1
        return b

    def fn(g):
        H, W = g.shape
        rs = [r for r in range(H) if len(set(g[r, :].tolist())) == 1]
        cs = [c for c in range(W) if len(set(g[:, c].tolist())) == 1]
        rb = bands(rs, H)
        cb = bands(cs, W)
        if not rb or not cb or (len(rb) == 1 and len(cb) == 1):
            return None
        out = np.zeros((len(rb), len(cb)), int)
        for ri, (r0, r1) in enumerate(rb):
            for ci, (c0, c1) in enumerate(cb):
                blk = g[r0:r1, c0:c1]
                v, c = np.unique(blk, return_counts=True)
                out[ri, ci] = int(v[np.argmax(c)])
        return out

    i0, o0 = train[0]
    if o0.shape[0] >= i0.shape[0] and o0.shape[1] >= i0.shape[1]:
        return None
    return fn


# ------------------------------------------------------------------
# Family: solid rectangle with 4 corner markers on its diagonals ->
# replace rectangle with 4 quadrants each filled with the nearest
# corner marker's color; erase markers.
# ------------------------------------------------------------------
def box_corner_quadrants(train):
    def fn(g):
        bg = _bg(g)
        out = g.copy()
        applied = False
        for col in np.unique(g):
            if col == bg:
                continue
            for cell in _comps(g == col, diag=False):
                ys = [y for y, x in cell]
                xs = [x for y, x in cell]
                r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
                h, w = r1 - r0 + 1, c1 - c0 + 1
                if h < 2 or w < 2 or len(cell) != h * w:
                    continue

                def marker(rr, cc):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            y, x = rr + dy, cc + dx
                            if 0 <= y < g.shape[0] and 0 <= x < g.shape[1] and g[y, x] != bg and g[y, x] != col:
                                return int(g[y, x]), (y, x)
                    return None

                mtl = marker(r0 - 1, c0 - 1)
                mtr = marker(r0 - 1, c1 + 1)
                mbl = marker(r1 + 1, c0 - 1)
                mbr = marker(r1 + 1, c1 + 1)
                if None in (mtl, mtr, mbl, mbr):
                    continue
                for y in range(r0, r1 + 1):
                    for x in range(c0, c1 + 1):
                        topq = y < r0 + (h + 1) // 2 if h % 2 == 1 else y < r0 + h // 2
                        leftq = x < c0 + (w + 1) // 2 if w % 2 == 1 else x < c0 + w // 2
                        m = mtl if (topq and leftq) else mtr if topq else mbl if leftq else mbr
                        out[y, x] = m[0]
                for m in (mtl, mtr, mbl, mbr):
                    out[m[1]] = bg
                applied = True
        return out if applied else None

    if any(i.shape != o.shape for i, o in train):
        return None
    return fn


# ------------------------------------------------------------------
# Family: several objects; move each along one axis so it aligns to an
# "anchor" colored object (color learned from train). Other axis fixed.
# ------------------------------------------------------------------
def align_objects_to_anchor(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def make(anchor, axis):
        def fn(g):
            bg = _bg(g)
            cs = _comps(g != bg, diag=True)
            if not cs:
                return None
            aval = None
            for cell in cs:
                cols = set(int(g[y, x]) for y, x in cell)
                if anchor in cols:
                    aval = min(y for y, x in cell) if axis == 0 else min(x for y, x in cell)
                    break
            if aval is None:
                return None
            out = np.full_like(g, bg)
            for cell in cs:
                if axis == 0:
                    d = aval - min(y for y, x in cell)
                    for y, x in cell:
                        ny = y + d
                        if 0 <= ny < g.shape[0]:
                            out[ny, x] = g[y, x]
                else:
                    d = aval - min(x for y, x in cell)
                    for y, x in cell:
                        nx = x + d
                        if 0 <= nx < g.shape[1]:
                            out[y, nx] = g[y, x]
            return out
        return fn

    colors = set()
    for i, _ in train:
        colors |= set(int(v) for v in np.unique(i) if v != _bg(i))
    for axis in (0, 1):
        for anchor in sorted(colors):
            fn = make(anchor, axis)
            try:
                if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
                    return fn
            except Exception:
                continue
    return None


# ------------------------------------------------------------------
# Family: two-color grid -> swap the two colors then tile NxM
# (color-inverted input, all tiles identical).
# ------------------------------------------------------------------
def swap_colors_tile(train):
    i0, o0 = train[0]
    if i0.shape[0] == 0 or i0.shape[1] == 0:
        return None
    if o0.shape[0] % i0.shape[0] or o0.shape[1] % i0.shape[1]:
        return None
    ah = o0.shape[0] // i0.shape[0]
    aw = o0.shape[1] // i0.shape[1]
    if ah * aw <= 1 or ah > 6 or aw > 6:
        return None
    # every train input must have exactly two colors
    if any(len(np.unique(i)) != 2 for i, _ in train):
        return None

    def fn(g):
        u = np.unique(g)
        if len(u) != 2:
            return None
        a, b = int(u[0]), int(u[1])
        sw = g.copy()
        sw[g == a] = b
        sw[g == b] = a
        return np.tile(sw, (ah, aw))

    return fn


# ------------------------------------------------------------------
# Family: a "main" shape + a small arrow marker. Mirror the main shape
# in the direction the arrow points (arrow = smaller-count color; its
# pointing direction = COM offset rotated 90 deg CW).
# ------------------------------------------------------------------
def mirror_by_arrow(train):
    if any(i.shape != o.shape for i, o in train):
        return None

    def fn(g):
        bg = _bg(g)
        nb = [int(v) for v in np.unique(g) if v != bg]
        if len(nb) != 2:
            return None
        arrow = min(nb, key=lambda c: int((g == c).sum()))
        main = [c for c in nb if c != arrow][0]
        a = np.argwhere(g == arrow)
        if len(a) == 0:
            return None
        r0, c0 = a.min(0)
        r1, c1 = a.max(0)
        cy = (r0 + r1) / 2.0
        cx = (c0 + c1) / 2.0
        offy = a[:, 0].mean() - cy
        offx = a[:, 1].mean() - cx
        rdy, rdx = offx, -offy  # rotate offset 90 deg CW
        if abs(rdy) >= abs(rdx):
            direction = 'down' if rdy > 0 else 'up'
        else:
            direction = 'right' if rdx > 0 else 'left'
        mm = np.argwhere(g == main)
        if len(mm) == 0:
            return None
        mr0, mc0 = mm.min(0)
        mr1, mc1 = mm.max(0)
        sub = (g[mr0:mr1 + 1, mc0:mc1 + 1] == main)
        h, w = sub.shape
        out = np.full_like(g, bg)
        for y, x in mm:
            out[y, x] = main
        if direction in ('down', 'up'):
            fl = np.flipud(sub)
            base = mr1 + 1 if direction == 'down' else mr0 - h
            for dy in range(h):
                for dx in range(w):
                    if fl[dy, dx]:
                        yy = base + dy
                        if 0 <= yy < g.shape[0]:
                            out[yy, mc0 + dx] = main
        else:
            fl = np.fliplr(sub)
            base = mc1 + 1 if direction == 'right' else mc0 - w
            for dy in range(h):
                for dx in range(w):
                    if fl[dy, dx]:
                        xx = base + dx
                        if 0 <= xx < g.shape[1]:
                            out[mr0 + dy, xx] = main
        return out

    return fn


# ------------------------------------------------------------------
# Family: a framed box (hollow rectangle of one color) with a small
# marker sub-grid inside -> crop to box, scale the marker sub-grid to
# fill the whole interior.
# ------------------------------------------------------------------
def box_expand_markers(train):
    def fn(g):
        bg = _bg(g)
        best = None
        for col in np.unique(g):
            if col == bg:
                continue
            pos = np.argwhere(g == col)
            r0, c0 = pos.min(0)
            r1, c1 = pos.max(0)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            bok = (all(g[r0, c] == col for c in range(c0, c1 + 1)) and
                   all(g[r1, c] == col for c in range(c0, c1 + 1)) and
                   all(g[r, c0] == col for r in range(r0, r1 + 1)) and
                   all(g[r, c1] == col for r in range(r0, r1 + 1)))
            if bok and (best is None or h * w > best[0]):
                best = (h * w, int(col), r0, r1, c0, c1)
        if best is None:
            return None
        _, fc, r0, r1, c0, c1 = best
        crop = g[r0:r1 + 1, c0:c1 + 1].copy()
        H, W = crop.shape
        iH, iW = H - 2, W - 2
        if iH <= 0 or iW <= 0:
            return None
        mk = [(y, x, int(crop[y, x])) for y in range(1, H - 1) for x in range(1, W - 1)
              if crop[y, x] != bg and crop[y, x] != fc]
        if not mk:
            return None
        mys = [y for y, x, v in mk]
        mxs = [x for y, x, v in mk]
        br0, br1, bc0, bc1 = min(mys), max(mys), min(mxs), max(mxs)
        bh, bw = br1 - br0 + 1, bc1 - bc0 + 1
        sub = np.full((bh, bw), bg)
        for y, x, v in mk:
            sub[y - br0, x - bc0] = v
        out = crop.copy()
        for yy in range(iH):
            for xx in range(iW):
                sy = min(bh - 1, yy * bh // iH)
                sx = min(bw - 1, xx * bw // iW)
                out[1 + yy, 1 + xx] = sub[sy, sx]
        return out

    return fn


DETECTORS = [
    grid_downsample,
    box_corner_quadrants,
    align_objects_to_anchor,
    swap_colors_tile,
    mirror_by_arrow,
    box_expand_markers,
]
