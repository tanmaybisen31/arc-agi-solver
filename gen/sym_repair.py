"""Symmetry & occlusion-repair detectors for ARC-AGI.

Family: many tasks hide a region behind a uniform "occluder" color, and the
hidden content can be reconstructed from the grid's global symmetry (mirror,
point/rotational, diagonal, translational/periodic). The output is either the
full repaired grid or just the recovered rectangular patch. A related case is
"symmetrize a partial grid": complete a symmetric shape drawn on a background.

All detectors infer a fixed occluder color from the training pairs and return a
transform that generalizes to the test input. The harness verifies every train
output exactly before a transform is ever used, so a mis-fit is auto-rejected.
"""
import numpy as np
from collections import Counter, deque


# ---------------------------------------------------------------- symmetries
def _find_syms(grid, known, ms):
    """Coordinate maps (r,c)->(r2,c2) that are consistent on all *known* cells
    with at least `ms` supporting known-known matches. Covers generalized
    mirror axes, diagonal (square grids) and translations. Point/rotational
    symmetry emerges from the orbit closure of these when filling, so it need
    not be searched directly (which keeps this O(H*W) per candidate)."""
    H, W = grid.shape
    maps = []
    RR, CC = np.mgrid[0:H, 0:W]

    def add(R2, C2, f):
        valid = (R2 >= 0) & (R2 < H) & (C2 >= 0) & (C2 < W)
        m = known & valid
        if not m.any():
            return
        km = m & known[np.clip(R2, 0, H - 1), np.clip(C2, 0, W - 1)]
        if km.sum() < ms:
            return
        if not np.array_equal(grid[km], grid[R2[km], C2[km]]):
            return
        maps.append(f)

    # vertical mirror about column axis a/2:  c -> a-c
    for a in range(1, 2 * W - 2):
        add(RR, a - CC, lambda r, c, a=a: (r, a - c))
    # horizontal mirror about row axis a/2:  r -> a-r
    for a in range(1, 2 * H - 2):
        add(a - RR, CC, lambda r, c, a=a: (a - r, c))
    if H == W:
        add(CC.copy(), RR.copy(), lambda r, c: (c, r))              # transpose
        add(W - 1 - CC, H - 1 - RR, lambda r, c: (W - 1 - c, H - 1 - r))  # anti
    # translations (periodicity) along each axis
    for p in range(1, H):
        add(RR + p, CC, lambda r, c, p=p: (r + p, c))
        add(RR - p, CC, lambda r, c, p=p: (r - p, c))
    for p in range(1, W):
        add(RR, CC + p, lambda r, c, p=p: (r, c + p))
        add(RR, CC - p, lambda r, c, p=p: (r, c - p))
    return maps


def _fill(grid, occ, ms):
    """Fill cells equal to `occ` using detected symmetries. Returns (out, known)
    where `known` marks cells that were successfully determined."""
    grid = grid.astype(int)
    H, W = grid.shape
    known = grid != occ
    if not (~known).any():
        return grid.copy(), known.copy()
    maps = _find_syms(grid, known, ms)
    if not maps:
        return None, None
    out = grid.copy()
    kn = known.copy()
    for _ in range(8):
        changed = False
        for r in range(H):
            for c in range(W):
                if kn[r, c]:
                    continue
                votes = Counter()
                seen = {(r, c)}
                dq = deque([(r, c)])
                steps = 0
                while dq and steps < 4000:
                    steps += 1
                    y, x = dq.popleft()
                    for f in maps:
                        ny, nx = f(y, x)
                        if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in seen:
                            seen.add((ny, nx))
                            if kn[ny, nx]:
                                votes[out[ny, nx]] += 1
                            else:
                                dq.append((ny, nx))
                if votes:
                    out[r, c] = votes.most_common(1)[0][0]
                    kn[r, c] = True
                    changed = True
        if not changed:
            break
    return out, kn


def _ms_for(grid, frac=16):
    H, W = grid.shape
    return max(6, (H * W) // frac)


def _occluder_candidates(train):
    """Colors present in some input but never in any output (classic occluder)."""
    in_all = set()
    out_all = set()
    for i, o in train:
        in_all |= set(np.unique(i).tolist())
        out_all |= set(np.unique(o).tolist())
    return sorted(in_all - out_all)


def _diff_colors(train):
    """Colors located at cells that differ between input and output (same-shape)."""
    cols = set()
    for i, o in train:
        if i.shape != o.shape:
            return None
        d = i != o
        if d.any():
            cols |= set(np.unique(i[d]).tolist())
    return sorted(cols)


def _solid_rect(g, occ):
    """Bounding box of `occ` cells if they form a completely filled rectangle."""
    m = g == occ
    if not m.any():
        return None
    rs, cs = np.where(m)
    r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
    if m[r0:r1 + 1, c0:c1 + 1].all():
        return int(r0), int(r1), int(c0), int(c1)
    return None


# -------------------------------------------------- 1. full-grid occlusion repair
def sym_occlusion_full(train):
    """Occluder color marks a hidden region; output = full grid with region
    reconstructed from symmetry. Handles both a distinct noise color and the
    case where background 0 marks the hole."""
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        cands = _occluder_candidates(train)
        if not cands:
            dc = _diff_colors(train)
            cands = dc if dc else []
        cands = [c for c in cands if 0 <= c <= 9]
        for occ in cands:
            fracs = (16, 8, 32)
            for fr in fracs:
                ok = True
                for i, o in train:
                    out, kn = _fill(i, occ, _ms_for(i, fr))
                    if out is None or not kn.all() or not np.array_equal(out, o):
                        ok = False
                        break
                if ok:
                    def fn(g, occ=occ, fr=fr):
                        out, kn = _fill(g, occ, _ms_for(g, fr))
                        if out is None or not kn.all():
                            return None
                        return out
                    return fn
        return None
    except Exception:
        return None


# ----------------------------------------------- 2. recovered-patch extraction
def sym_occlusion_patch(train):
    """Occluder color forms a solid rectangle; output = the reconstructed patch
    (same shape as that rectangle)."""
    try:
        # determine a single occluder color consistent across all pairs
        i0, o0 = train[0]
        occ = None
        for c in np.unique(i0):
            rect = _solid_rect(i0, int(c))
            if rect and (rect[1] - rect[0] + 1, rect[3] - rect[2] + 1) == o0.shape:
                occ = int(c)
                break
        if occ is None:
            return None
        for fr in (16, 8, 32):
            ok = True
            for i, o in train:
                rect = _solid_rect(i, occ)
                if rect is None:
                    ok = False
                    break
                r0, r1, c0, c1 = rect
                if (r1 - r0 + 1, c1 - c0 + 1) != o.shape:
                    ok = False
                    break
                out, kn = _fill(i, occ, _ms_for(i, fr))
                if out is None or not kn[r0:r1 + 1, c0:c1 + 1].all():
                    ok = False
                    break
                if not np.array_equal(out[r0:r1 + 1, c0:c1 + 1], o):
                    ok = False
                    break
            if ok:
                def fn(g, occ=occ, fr=fr):
                    rect = _solid_rect(g, occ)
                    if rect is None:
                        return None
                    r0, r1, c0, c1 = rect
                    out, kn = _fill(g, occ, _ms_for(g, fr))
                    if out is None or not kn[r0:r1 + 1, c0:c1 + 1].all():
                        return None
                    return out[r0:r1 + 1, c0:c1 + 1].copy()
                return fn
        return None
    except Exception:
        return None


# --------------------------------------------------- 3. symmetrize partial grid
def _global_syms(H, W):
    v = [lambda r, c: (r, W - 1 - c),
         lambda r, c: (H - 1 - r, c),
         lambda r, c: (H - 1 - r, W - 1 - c)]
    if H == W:
        v += [lambda r, c: (c, r),
              lambda r, c: (W - 1 - c, H - 1 - r),
              lambda r, c: (c, H - 1 - r),
              lambda r, c: (W - 1 - c, r)]
    return v


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _symmetrize(grid, bg):
    """Complete a symmetric pattern: for each cell take the (unique) non-bg
    value across its symmetry orbit; leave bg where the whole orbit is bg."""
    grid = grid.astype(int)
    H, W = grid.shape
    nz = grid != bg
    good = []
    for f in _global_syms(H, W):
        ok = True
        for r in range(H):
            for c in range(W):
                if not nz[r, c]:
                    continue
                r2, c2 = f(r, c)
                if 0 <= r2 < H and 0 <= c2 < W and nz[r2, c2] and grid[r2, c2] != grid[r, c]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            good.append(f)
    if not good:
        return None
    out = grid.copy()
    for r in range(H):
        for c in range(W):
            seen = {(r, c)}
            dq = deque([(r, c)])
            val = None
            conflict = False
            while dq:
                y, x = dq.popleft()
                if grid[y, x] != bg:
                    if val is None:
                        val = grid[y, x]
                    elif val != grid[y, x]:
                        conflict = True
                for f in good:
                    ny, nx = f(y, x)
                    if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        dq.append((ny, nx))
            if val is not None and not conflict:
                out[r, c] = val
    return out


def symmetrize_grid(train):
    """Same-shape task where a symmetric drawing on a background is completed by
    mirroring/rotating existing marks into empty (background) cells."""
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        # background must be plausible & the transform must add, not remove, marks
        bgs = set()
        for i, o in train:
            bgs.add(_bg(i))
        if len(bgs) != 1:
            return None
        bg = bgs.pop()
        for i, o in train:
            r = _symmetrize(i, bg)
            if r is None or not np.array_equal(r, o):
                return None
        # guard against trivial identity (would duplicate the base identity detector)
        if all(np.array_equal(i, o) for i, o in train):
            return None
        return lambda g, bg=bg: (_symmetrize(g, bg) if _symmetrize(g, bg) is not None else None)
    except Exception:
        return None


DETECTORS = [sym_occlusion_full, sym_occlusion_patch, symmetrize_grid]
