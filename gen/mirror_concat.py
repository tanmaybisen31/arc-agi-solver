"""MIRROR / CONCATENATION detectors for ARC-AGI.

Family: output = input tiled into an (rh x rw) mosaic of geometrically
transformed copies of the input (identity / fliplr / flipud / rot90 / rot180 /
rot270 / transpose / anti-transpose). Covers horizontal doubling, vertical
doubling, 2x2 kaleidoscopes, and 3x3 mosaics.

Strategy: infer the doubling/tripling factor from the output shape, then learn a
per-block geometric transform grid directly from the training pairs. Because the
grid is learned from the data (rather than a small fixed set of hard-coded
patterns), this generalizes to arbitrary reflection/rotation mosaics. The engine
verifies exact reproduction of every training output before the transform is
ever used on a test input.
"""
import numpy as np

# Geometric primitives. Ordered by "simplicity": identity first, then the
# reflections/180 that preserve shape, then the 90-degree / transpose ops that
# require a square block. When several ops reproduce the same block (e.g. a
# symmetric input where id == fliplr) we keep the earliest, which tends to match
# the intended construction and generalizes to asymmetric test inputs.
def _id(g):    return g
def _flr(g):   return np.fliplr(g)
def _fud(g):   return np.flipud(g)
def _r180(g):  return np.rot90(g, 2)
def _r90(g):   return np.rot90(g, 1)
def _r270(g):  return np.rot90(g, 3)
def _T(g):     return g.T
def _aT(g):    return np.rot90(g, 2).T

# (name, fn, keeps_shape) — keeps_shape ops don't change (h,w); the others swap
# them and so only apply when the block is square.
_SHAPE_KEEP = [("id", _id), ("flr", _flr), ("fud", _fud), ("r180", _r180)]
_SHAPE_SWAP = [("r90", _r90), ("r270", _r270), ("T", _T), ("aT", _aT)]


def _ratio(train):
    """Return (rh, rw) if every pair has the same integer output/input ratio."""
    ratios = set()
    for i, o in train:
        ih, iw = i.shape
        oh, ow = o.shape
        if ih == 0 or iw == 0:
            return None
        if oh % ih or ow % iw:
            return None
        ratios.add((oh // ih, ow // iw))
    if len(ratios) != 1:
        return None
    rh, rw = ratios.pop()
    if rh * rw <= 1 or rh > 4 or rw > 4:
        return None
    return rh, rw


def _learn_grid(train, rh, rw):
    """For each block position, find a single geometric op that reproduces it in
    EVERY training pair. Returns a list-of-lists of op fns, or None if any block
    has no consistent op."""
    # squareness of blocks (all train inputs must agree for swap ops to be usable)
    can_swap = all(i.shape[0] == i.shape[1] for i, _ in train)
    ops = list(_SHAPE_KEEP)
    if can_swap:
        ops = list(_SHAPE_KEEP) + list(_SHAPE_SWAP)

    grid = []
    for r in range(rh):
        row = []
        for c in range(rw):
            chosen = None
            for _, op in ops:
                ok = True
                for i, o in train:
                    bh, bw = i.shape
                    if o.shape[0] != bh * rh or o.shape[1] != bw * rw:
                        ok = False
                        break
                    blk = o[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                    t = op(i)
                    if t.shape != blk.shape or not np.array_equal(t, blk):
                        ok = False
                        break
                if ok:
                    chosen = op
                    break
            if chosen is None:
                return None
            row.append(chosen)
        grid.append(row)
    return grid


def geo_mosaic(train):
    """Output is an (rh x rw) mosaic of geometric transforms of the input."""
    try:
        r = _ratio(train)
        if r is None:
            return None
        rh, rw = r
        grid = _learn_grid(train, rh, rw)
        if grid is None:
            return None

        def fn(g, grid=grid, rh=rh, rw=rw):
            rows = []
            for rr in range(rh):
                cols = []
                for cc in range(rw):
                    cols.append(np.asarray(grid[rr][cc](g), dtype=int))
                rows.append(np.hstack(cols))
            return np.vstack(rows)
        return fn
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Concatenation with a per-copy color remap (e.g. one half is a mirror of the
# other with two colors swapped). Handles the "mirror + recolor" doubling tasks
# that a pure geometric mosaic cannot express.
# ---------------------------------------------------------------------------
def _learn_block_recolor(train, rh, rw):
    """Like _learn_grid but each block may be geo-op(input) followed by a fixed
    color permutation learned from the data. Returns list-of-lists of
    (op, mapping-dict) or None."""
    can_swap = all(i.shape[0] == i.shape[1] for i, _ in train)
    ops = list(_SHAPE_KEEP)
    if can_swap:
        ops = ops + list(_SHAPE_SWAP)

    grid = []
    for r in range(rh):
        row = []
        for c in range(rw):
            chosen = None
            for _, op in ops:
                mapping = {}
                ok = True
                for i, o in train:
                    bh, bw = i.shape
                    if o.shape[0] != bh * rh or o.shape[1] != bw * rw:
                        ok = False
                        break
                    blk = o[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                    t = op(i)
                    if t.shape != blk.shape:
                        ok = False
                        break
                    for a, b in zip(t.flatten(), blk.flatten()):
                        a, b = int(a), int(b)
                        if a in mapping and mapping[a] != b:
                            ok = False
                            break
                        mapping[a] = b
                    if not ok:
                        break
                if ok:
                    chosen = (op, dict(mapping))
                    break
            if chosen is None:
                return None
            row.append(chosen)
        grid.append(row)
    return grid


def geo_mosaic_recolor(train):
    """Mosaic where each block is a geometric transform of the input followed by
    a fixed per-block color remap."""
    try:
        r = _ratio(train)
        if r is None:
            return None
        rh, rw = r
        # Only bother if the plain mosaic didn't already fit (keeps this from
        # shadowing the simpler, safer detector). We still return it; the engine
        # verifies and voting orders by detector priority.
        grid = _learn_block_recolor(train, rh, rw)
        if grid is None:
            return None

        def fn(g, grid=grid, rh=rh, rw=rw):
            rows = []
            for rr in range(rh):
                cols = []
                for cc in range(rw):
                    op, mapping = grid[rr][cc]
                    blk = np.asarray(op(g), dtype=int)
                    out = blk.copy()
                    for a, b in mapping.items():
                        out[blk == a] = b
                    cols.append(out)
                rows.append(np.hstack(cols))
            return np.vstack(rows)
        return fn
    except Exception:
        return None


DETECTORS = [geo_mosaic, geo_mosaic_recolor]
