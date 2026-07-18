"""Periodicity & pattern-completion detectors for ARC-AGI.

Family: detect a repeating 2-D tile (row period pr, col period pc) in the input,
then extend / complete / denoise the grid using that periodic pattern. Cells that
are "masked" (a hole color) get filled with the periodically-consistent value.

Each detector: det(train) -> transform_fn | None.  The engine verifies the
returned transform reproduces every training output exactly before use, so we can
afford to be liberal and let wrong guesses be auto-rejected.

Only numpy + stdlib.
"""
import numpy as np
from collections import Counter, defaultdict


# ---------------------------------------------------------------- helpers ----
def _period_ok(known, mask, pr, pc):
    """Are all KNOWN cells consistent under period (pr,pc)?

    known: 2D int array. mask: bool array, True where value is known.
    Returns the tile dict {(i%pr, j%pc): value} if consistent, else None.
    """
    H, W = known.shape
    tile = {}
    for r in range(H):
        row_mask = mask[r]
        rr = r % pr
        for c in range(W):
            if not row_mask[c]:
                continue
            key = (rr, c % pc)
            v = known[r, c]
            prev = tile.get(key)
            if prev is None:
                tile[key] = v
            elif prev != v:
                return None
    return tile


def _find_period(known, mask):
    """Find smallest (pr,pc) s.t. known cells are periodic AND the tile is fully
    determined (every residue class has at least one known cell).

    Returns (pr, pc, tile_dict) or None.
    """
    H, W = known.shape
    best = None
    for pr in range(1, H + 1):
        for pc in range(1, W + 1):
            if pr == H and pc == W:
                continue  # whole grid completes nothing
            area = pr * pc
            if best is not None and area >= best[0]:
                continue
            tile = _period_ok(known, mask, pr, pc)
            if tile is None:
                continue
            if len(tile) != pr * pc:  # tile must be complete
                continue
            best = (area, pr, pc, tile)
    if best is None:
        return None
    _, pr, pc, tile = best
    return pr, pc, tile


def _find_period_local(known, mask):
    """Find smallest (pr,pc) consistent with known cells, requiring only that
    every HOLE residue class is determined by at least one known cell."""
    H, W = known.shape
    holes = np.argwhere(~mask)
    if len(holes) == 0:
        return None
    best = None
    for pr in range(1, H + 1):
        for pc in range(1, W + 1):
            if pr == H and pc == W:
                continue
            area = pr * pc
            if best is not None and area >= best[0]:
                continue
            tile = _period_ok(known, mask, pr, pc)
            if tile is None:
                continue
            good = True
            for (hr, hc) in holes:
                if (hr % pr, hc % pc) not in tile:
                    good = False
                    break
            if not good:
                continue
            best = (area, pr, pc, tile)
    if best is None:
        return None
    _, pr, pc, tile = best
    return pr, pc, tile


def _apply_period(shape, tile, pr, pc):
    H, W = shape
    out = np.empty((H, W), dtype=int)
    for r in range(H):
        rr = r % pr
        for c in range(W):
            out[r, c] = tile[(rr, c % pc)]
    return out


def _hole_candidates(train):
    """Colors that plausibly act as the 'hole'/mask color: present in an input's
    changed region. Most-frequent-across-pairs first."""
    cand = Counter()
    npairs = 0
    for i, o in train:
        if i.shape != o.shape:
            continue
        diff = i != o
        if not diff.any():
            continue
        npairs += 1
        for v in np.unique(i[diff]).tolist():
            cand[v] += 1
    # prefer colors present in every differing pair
    return [c for c, _ in cand.most_common()]


# --------------------------------------- 1) same-shape full periodic fill ----
def periodic_fill(train):
    """Input and output same shape. A single 'hole' color masks a region; the
    rest of the grid is 2-D periodic. Fill the whole grid from the period."""
    if any(i.shape != o.shape for i, o in train):
        return None
    holes = _hole_candidates(train)
    if not holes:
        return None

    for hole in holes:
        ok = True
        for i, o in train:
            mask = i != hole
            if mask.all():
                if not np.array_equal(i, o):
                    ok = False
                    break
                continue
            res = _find_period(i, mask)
            if res is None:
                ok = False
                break
            pr, pc, tile = res
            if not np.array_equal(_apply_period(i.shape, tile, pr, pc), o):
                ok = False
                break
        if not ok:
            continue

        def fn(g, hole=hole):
            mask = g != hole
            if mask.all():
                return g.copy()
            res = _find_period(g, mask)
            if res is None:
                return None
            pr, pc, tile = res
            return _apply_period(g.shape, tile, pr, pc)

        return fn
    return None


# --------------------------- 2) same-shape, overwrite ONLY the hole cells ----
def periodic_fill_local(train):
    """Keep all non-hole cells unchanged; fill hole cells from the period implied
    by surrounding cells. Robust when the grid is only locally periodic."""
    if any(i.shape != o.shape for i, o in train):
        return None
    holes = _hole_candidates(train)
    if not holes:
        return None

    def complete(g, hole):
        mask = g != hole
        if mask.all():
            return g.copy()
        res = _find_period_local(g, mask)
        if res is None:
            return None
        pr, pc, tile = res
        out = g.copy()
        H, W = g.shape
        for r in range(H):
            rr = r % pr
            for c in range(W):
                if not mask[r, c]:
                    key = (rr, c % pc)
                    if key not in tile:
                        return None
                    out[r, c] = tile[key]
        return out

    for hole in holes:
        ok = True
        for i, o in train:
            got = complete(i, hole)
            if got is None or not np.array_equal(got, o):
                ok = False
                break
        if ok:
            return (lambda g, hole=hole: complete(g, hole))
    return None


# ------------------------------------- 3) extract the fundamental period tile
def extract_period_tile(train):
    """Output is the minimal repeating tile of the input (optionally ignoring a
    hole color). Output shape == (pr, pc)."""
    hole_opts = _hole_candidates(train) + [None]

    def tile_of(g, hole):
        mask = (g != hole) if hole is not None else np.ones_like(g, dtype=bool)
        res = _find_period(g, mask)
        if res is None:
            return None
        pr, pc, tile = res
        out = np.empty((pr, pc), dtype=int)
        for (r, c), v in tile.items():
            out[r, c] = v
        return out

    for hole in hole_opts:
        ok = True
        for i, o in train:
            got = tile_of(i, hole)
            if got is None or got.shape != o.shape or not np.array_equal(got, o):
                ok = False
                break
        if ok:
            return (lambda g, hole=hole: tile_of(g, hole))
    return None


# ------------------------ 4) extend a periodic grid to an integer-scaled size
def periodic_extend(train):
    """Output larger than input by a fixed integer factor and is the periodic
    continuation of the input (input == top-left window of the pattern)."""
    factors = set()
    for i, o in train:
        if i.shape[0] == 0 or i.shape[1] == 0:
            return None
        if o.shape[0] % i.shape[0] or o.shape[1] % i.shape[1]:
            return None
        factors.add((o.shape[0] // i.shape[0], o.shape[1] // i.shape[1]))
    if len(factors) != 1:
        return None
    fh, fw = factors.pop()
    if fh * fw <= 1 or fh > 12 or fw > 12:
        return None

    hole_opts = _hole_candidates(train) + [None]

    def build(g, hole):
        mask = (g != hole) if hole is not None else np.ones_like(g, dtype=bool)
        res = _find_period(g, mask)
        if res is None:
            return None
        pr, pc, tile = res
        return _apply_period((g.shape[0] * fh, g.shape[1] * fw), tile, pr, pc)

    for hole in hole_opts:
        ok = True
        for i, o in train:
            got = build(i, hole)
            if got is None or got.shape != o.shape or not np.array_equal(got, o):
                ok = False
                break
        if ok:
            return (lambda g, hole=hole: build(g, hole))
    return None


DETECTORS = [
    periodic_fill,
    periodic_fill_local,
    extract_period_tile,
    periodic_extend,
]
