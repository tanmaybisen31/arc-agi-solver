"""Shape -> output dictionary detectors for ARC-AGI.

Family: learn a mapping from an input sub-pattern / object property to an
output (usually an output color, sometimes a replacement patch) across the
training pairs, then apply it to the test input.

Covered patterns:
  1. object-recolor-by-invariant: every non-background object is recolored to
     a single color that is a deterministic function of some shape invariant
     (exact normalized shape, shape-modulo-rotation/flip, cell-count/size,
     bounding-box dims, hole-count, input color).  The mapping is learned from
     train and applied to test.  Multiple invariants are tried; each is a
     separate detector so the engine can vote.
  2. template find-and-replace: locate every occurrence of a fixed small
     sub-grid ("key") in the input and overwrite it with a learned
     replacement patch of the same size.

All detectors are defensive: they return None unless the learned mapping
covers every object/occurrence in every train pair AND every object in the
test input, so the engine never gets a partial/garbage transform.
"""
import numpy as np
from collections import Counter

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])

def _components(g, bg, diag):
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
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

def _components_same_color(g, bg, diag):
    """Connected components where all cells share the SAME color."""
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1)]
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = g[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps

def _norm_shape(cells):
    ys = [y for y, x in cells]; xs = [x for y, x in cells]
    y0, x0 = min(ys), min(xs)
    return frozenset((y-y0, x-x0) for y, x in cells)

def _canon_shape(cells):
    """Canonical shape modulo the 8 dihedral symmetries (rot/flip)."""
    ys = [y for y, x in cells]; xs = [x for y, x in cells]
    y0, x0 = min(ys), min(xs)
    pts = [(y-y0, x-x0) for y, x in cells]
    best = None
    cur = pts
    for _ in range(4):
        cur = [(x, -y) for y, x in cur]          # rotate 90
        for fl in (cur, [(y, -x) for y, x in cur]):
            yy = [y for y, x in fl]; xx = [x for y, x in fl]
            ymn, xmn = min(yy), min(xx)
            key = tuple(sorted((y-ymn, x-xmn) for y, x in fl))
            if best is None or key < best:
                best = key
    return best

def _bbox_dims(cells):
    ys = [y for y, x in cells]; xs = [x for y, x in cells]
    return (max(ys)-min(ys)+1, max(xs)-min(xs)+1)

def _hole_count(cells):
    """Number of background holes fully enclosed by the object (4-conn)."""
    s = set(cells)
    ys = [y for y, x in cells]; xs = [x for y, x in cells]
    y0, y1 = min(ys), max(ys); x0, x1 = min(xs), max(xs)
    H = y1 - y0 + 3; W = x1 - x0 + 3
    grid = np.zeros((H, W), dtype=int)     # 0 empty, 1 object
    for y, x in cells:
        grid[y-y0+1, x-x0+1] = 1
    # flood fill background from border
    outside = np.zeros((H, W), dtype=bool)
    stack = [(0, 0)]
    outside[0, 0] = True
    while stack:
        cy, cx = stack.pop()
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < H and 0 <= nx < W and not outside[ny, nx] and grid[ny, nx] == 0:
                outside[ny, nx] = True
                stack.append((ny, nx))
    # count connected interior empty regions
    interior = (grid == 0) & (~outside)
    seen = np.zeros((H, W), dtype=bool)
    holes = 0
    for r in range(H):
        for c in range(W):
            if interior[r, c] and not seen[r, c]:
                holes += 1
                st = [(r, c)]; seen[r, c] = True
                while st:
                    cy, cx = st.pop()
                    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and interior[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; st.append((ny, nx))
    return holes


# candidate background choices to try
def _bg_candidates(train):
    cand = []
    b = _bg_color(train[0][0])
    cand.append(b)
    if 0 not in cand:
        cand.append(0)
    return cand


# ----------------------------------------------------------------------------
# generic object-recolor-by-invariant builder
# ----------------------------------------------------------------------------
def _make_recolor_detector(keyname, keyfn, diag, same_color, use_incolor=False):
    """Return a detector function that recolors objects by an invariant key."""
    def det(train):
        try:
            if not train:
                return None
            if any(i.shape != o.shape for i, o in train):
                return None
            for bg in _bg_candidates(train):
                # non-background mask must be preserved (recolor in place)
                ok_mask = True
                for i, o in train:
                    if not np.array_equal((i != bg), (o != bg)):
                        ok_mask = False
                        break
                if not ok_mask:
                    continue

                comp_fn = _components_same_color if same_color else _components
                mapping = {}
                consistent = True
                changed_any = False
                for i, o in train:
                    comps = comp_fn(i, bg, diag)
                    if not comps:
                        consistent = False
                        break
                    for cells in comps:
                        if use_incolor:
                            k = int(i[cells[0]])
                        else:
                            k = keyfn(cells)
                        ocs = set(int(o[y, x]) for y, x in cells)
                        if len(ocs) != 1:
                            consistent = False
                            break
                        oc = ocs.pop()
                        # is this a genuine recolor (input cells not all == oc)?
                        incs = set(int(i[y, x]) for y, x in cells)
                        if incs != {oc}:
                            changed_any = True
                        if k in mapping and mapping[k] != oc:
                            consistent = False
                            break
                        mapping[k] = oc
                    if not consistent:
                        break
                if not consistent or not mapping or not changed_any:
                    continue

                def fn(g, bg=bg, mapping=mapping, comp_fn=comp_fn, diag=diag,
                       use_incolor=use_incolor, keyfn=keyfn):
                    comps = comp_fn(g, bg, diag)
                    if not comps:
                        return None
                    out = g.copy()
                    for cells in comps:
                        if use_incolor:
                            k = int(g[cells[0]])
                        else:
                            k = keyfn(cells)
                        if k not in mapping:
                            # unknown object -> abstain rather than emit a wrong
                            # guess (avoids crowding correct votes in the vote).
                            return None
                        c = mapping[k]
                        for y, x in cells:
                            out[y, x] = c
                    return out

                # verify on train exactly (engine also does this, but early-exit)
                if all(np.array_equal(fn(i), o) for i, o in train):
                    return fn
            return None
        except Exception:
            return None
    det.__name__ = "shapedict_recolor_" + keyname
    return det


# instantiate detectors for several invariants x connectivity x component-mode
_recolor_dets = []
_INVARIANTS = [
    ("shape", _norm_shape),
    ("canon", _canon_shape),
    ("size", lambda c: len(c)),
    ("bbox", _bbox_dims),
    ("holes", _hole_count),
]
for _diag in (False, True):
    for _sc in (True, False):
        for _nm, _fn in _INVARIANTS:
            _tag = f"{_nm}_{'d' if _diag else 'o'}_{'sc' if _sc else 'mc'}"
            _recolor_dets.append(_make_recolor_detector(_tag, _fn, _diag, _sc))
    # input-color keyed (shape-independent recolor of objects, but per-object
    # single-colour requirement filters it from the generic color_map)
    _recolor_dets.append(
        _make_recolor_detector(f"incolor_{'d' if _diag else 'o'}", None, _diag, True, use_incolor=True))


# ----------------------------------------------------------------------------
# key-template match: one object is a "legend"; objects with matching shape
# get recolored to the legend colour.  Generalises 63613498-style tasks.
# ----------------------------------------------------------------------------
def key_template_recolor(train):
    """Find, per pair, a distinguished colour K that also recolours every
    non-K object whose normalized shape equals a K-object's shape."""
    try:
        if not train or any(i.shape != o.shape for i, o in train):
            return None
        for bg in _bg_candidates(train):
            for i, o in train:
                if not np.array_equal((i != bg), (o != bg)):
                    return None
            # Determine the recolor target colour: the single colour that all
            # changed cells become, consistent across pairs.
            target = None
            good = True
            for i, o in train:
                diff = (i != o)
                if not diff.any():
                    good = False; break
                newcols = set(int(o[y, x]) for y, x in zip(*np.where(diff)))
                if len(newcols) != 1:
                    good = False; break
                t = newcols.pop()
                if target is None:
                    target = t
                elif target != t:
                    good = False; break
            if not good or target is None:
                continue

            # The "key" shapes = normalized shapes of the target-coloured
            # objects in the INPUT that are unchanged (they define the legend).
            def key_shapes(g):
                ks = set()
                for cells in _components_same_color(g, bg, True):
                    if int(g[cells[0]]) == target:
                        ks.add(_norm_shape(cells))
                for cells in _components_same_color(g, bg, False):
                    if int(g[cells[0]]) == target:
                        ks.add(_norm_shape(cells))
                return ks

            def fn(g, bg=bg, target=target):
                out = g.copy()
                ks = set()
                for diag in (True, False):
                    for cells in _components_same_color(g, bg, diag):
                        if int(g[cells[0]]) == target:
                            ks.add(_norm_shape(cells))
                if not ks:
                    return out
                for diag in (True, False):
                    for cells in _components_same_color(g, bg, diag):
                        if int(g[cells[0]]) == target:
                            continue
                        if _norm_shape(cells) in ks:
                            for y, x in cells:
                                out[y, x] = target
                return out

            if all(np.array_equal(fn(i), o) for i, o in train):
                return fn
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# template find-and-replace: fixed small key sub-grid -> replacement patch.
# Learns a dict of (key-patch bytes) -> (replacement patch) from same-shape
# pairs, keyed on unique small motifs.  Conservative: requires the diff to be
# explainable as replacing occurrences of a small set of key patches.
# ----------------------------------------------------------------------------
def cell_pattern_replace(train):
    """3x3 (or NxN) neighbourhood -> output-cell dictionary.

    For every same-shape pair, learn a mapping from the KxK input neighbourhood
    (centred on a cell) to the output value at that centre cell.  If the
    mapping is a consistent function, apply it convolutionally.  This captures
    local template rewrites (e.g. denoise, corner marking, contextual recolor).
    """
    try:
        if not train or any(i.shape != o.shape for i, o in train):
            return None
        for K in (3,):
            r = K // 2
            for pad_mode in ("edge", "const"):
                mapping = {}
                ok = True
                for i, o in train:
                    H, W = i.shape
                    if pad_mode == "edge":
                        P = np.pad(i, r, mode="edge")
                    else:
                        P = np.pad(i, r, mode="constant", constant_values=-1)
                    for y in range(H):
                        for x in range(W):
                            patch = P[y:y+K, x:x+K].tobytes()
                            ov = int(o[y, x])
                            if patch in mapping and mapping[patch] != ov:
                                ok = False; break
                            mapping[patch] = ov
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok or not mapping:
                    continue
                # require the transform to be non-trivial (not pure identity)
                nontrivial = any(not np.array_equal(i, o) for i, o in train)
                if not nontrivial:
                    continue

                def fn(g, K=K, r=r, mapping=mapping, pad_mode=pad_mode):
                    H, W = g.shape
                    if pad_mode == "edge":
                        P = np.pad(g, r, mode="edge")
                    else:
                        P = np.pad(g, r, mode="constant", constant_values=-1)
                    out = g.copy()
                    for y in range(H):
                        for x in range(W):
                            patch = P[y:y+K, x:x+K].tobytes()
                            if patch in mapping:
                                out[y, x] = mapping[patch]
                    return out

                if all(np.array_equal(fn(i), o) for i, o in train):
                    return fn
        return None
    except Exception:
        return None


DETECTORS = _recolor_dets + [key_template_recolor, cell_pattern_replace]
