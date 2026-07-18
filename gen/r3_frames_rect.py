"""Detectors for the FRAMES & RECTANGLES family.

Patterns covered (all defensive; the engine re-verifies exact reproduction):

  * fill_hollow_rect        -- fill the interior of every hollow rectangle. The
                               fill colour may be a single learned colour or a
                               colour that depends on the interior's size
                               (e.g. 1x1 -> A, 2x2 -> B, ...).
  * outline_bbox            -- draw the bounding-box *outline* of every object in
                               a learned colour (object left intact).
  * fill_bbox               -- fill every object's bounding box solid with the
                               object's colour (or a learned colour).
  * halo_objects            -- draw a 1-cell frame of a learned colour hugging
                               each object's bounding box (outside it).
  * complete_rectangle      -- complete a partial / broken rectangle outline so
                               it becomes a full rectangle of its own colour.

Only numpy + stdlib are used.
"""
import numpy as np
from collections import Counter


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag=False):
    """Connected components of non-bg cells (ignores colour)."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    comps = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            comps.append(cells)
    return comps


def _components_color(g, bg):
    """Same-colour 4-connected components."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = int(g[r, c])
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            out.append((col, cells))
    return out


def _bbox(cells):
    ys = [y for y, x in cells]
    xs = [x for y, x in cells]
    return min(ys), max(ys), min(xs), max(xs)


# ----------------------------------------------------------------------------
# Fill hollow rectangle interiors (single colour OR size-dependent colour)
# ----------------------------------------------------------------------------
def _hollow_rects(g, bg):
    """Yield (col, r0, r1, c0, c1) for each same-colour component that is exactly
    the perimeter (outline) of a rectangle with a non-empty interior, whose
    interior is entirely bg."""
    H, W = g.shape
    for col, cells in _components_color(g, bg):
        r0, r1, c0, c1 = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        perim = 2 * h + 2 * w - 4
        if len(cells) != perim:
            continue
        # verify it really is the full outline and interior is bg
        ok = True
        for c in range(c0, c1 + 1):
            if g[r0, c] != col or g[r1, c] != col:
                ok = False
                break
        if ok:
            for r in range(r0, r1 + 1):
                if g[r, c0] != col or g[r, c1] != col:
                    ok = False
                    break
        if not ok:
            continue
        interior = g[r0 + 1:r1, c0 + 1:c1]
        if interior.size == 0 or np.any(interior != bg):
            continue
        yield col, r0, r1, c0, c1


def _apply_hollow_fill(g, size_map, uniform):
    bg = _bg(g)
    out = g.copy()
    for col, r0, r1, c0, c1 in _hollow_rects(g, bg):
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        if uniform is not None:
            fill = uniform
        else:
            key = (ih, iw)
            if key in size_map:
                fill = size_map[key]
            elif (iw, ih) in size_map:
                fill = size_map[(iw, ih)]
            elif min(ih, iw) in size_map:  # square keyed by side
                fill = size_map[min(ih, iw)]
            else:
                continue
        out[r0 + 1:r1, c0 + 1:c1] = fill
    return out


def fill_hollow_rect(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        if not any((i != o).any() for i, o in train):
            return None
        # gather (interior-size -> fill colour) evidence
        size_map = {}     # (ih,iw) -> colour
        side_map = {}     # side -> colour   (for squares)
        uniform_set = set()
        ok = True
        for i, o in train:
            bg = _bg(i)
            rects = list(_hollow_rects(i, bg))
            if not rects:
                # any diff without a hollow rect present is disqualifying
                if (i != o).any():
                    return None
                continue
            covered = np.zeros(i.shape, bool)
            for col, r0, r1, c0, c1 in rects:
                interior_out = o[r0 + 1:r1, c0 + 1:c1]
                vals = set(int(v) for v in np.unique(interior_out))
                if len(vals) != 1:
                    ok = False
                    break
                fill = next(iter(vals))
                ih, iw = r1 - r0 - 1, c1 - c0 - 1
                # record
                if (ih, iw) in size_map and size_map[(ih, iw)] != fill:
                    size_map[(ih, iw)] = None  # conflict
                elif (ih, iw) not in size_map:
                    size_map[(ih, iw)] = fill
                if ih == iw:
                    if ih in side_map and side_map[ih] != fill:
                        side_map[ih] = None
                    elif ih not in side_map:
                        side_map[ih] = fill
                uniform_set.add(fill)
                covered[r0:r1 + 1, c0:c1 + 1] = True
            if not ok:
                break
            # every diff must be inside a rect interior
            diff = (i != o)
            if np.any(diff & (~covered)):
                return None
        if not ok:
            return None
        # candidate 1: uniform fill
        cands = []
        if len(uniform_set) == 1:
            cands.append((None, next(iter(uniform_set))))
        # candidate 2: size-dependent (by interior h,w)
        sm = {k: v for k, v in size_map.items() if v is not None}
        if sm and all(v is not None for v in size_map.values()):
            cands.append((sm, None))
        # candidate 3: side-dependent (for squares)
        sd = {k: v for k, v in side_map.items() if v is not None}
        if sd and all(v is not None for v in side_map.values()):
            cands.append((sd, None))
        for size_arg, uni in cands:
            fn = (lambda g, sa=size_arg, u=uni: _apply_hollow_fill(g, sa if sa else {}, u))
            if all(np.array_equal(fn(i), o) for i, o in train):
                return fn
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Draw the bounding-box outline of every object in a learned colour
# ----------------------------------------------------------------------------
def _outline_bbox(g, lc, diag):
    bg = _bg(g)
    out = g.copy()
    H, W = g.shape
    for cells in _components(g, bg, diag=diag):
        r0, r1, c0, c1 = _bbox(cells)
        if r1 - r0 < 1 and c1 - c0 < 1:
            continue
        col = lc
        for c in range(c0, c1 + 1):
            if out[r0, c] == bg:
                out[r0, c] = col
            if out[r1, c] == bg:
                out[r1, c] = col
        for r in range(r0, r1 + 1):
            if out[r, c0] == bg:
                out[r, c0] = col
            if out[r, c1] == bg:
                out[r, c1] = col
    return out


def outline_bbox(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        if not any((i != o).any() for i, o in train):
            return None
        lcs = set()
        for i, o in train:
            diff = (i != o)
            if diff.any():
                lcs |= set(int(v) for v in o[diff])
        if not lcs or len(lcs) > 4:
            return None
        for diag in (False, True):
            for lc in sorted(lcs):
                fn = (lambda g, lc=lc, d=diag: _outline_bbox(g, lc, d))
                if all(np.array_equal(fn(i), o) for i, o in train):
                    return fn
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Fill every object's bounding box solid (own colour or learned colour)
# ----------------------------------------------------------------------------
def _fill_bbox(g, lc, diag):
    bg = _bg(g)
    out = g.copy()
    for cells in _components(g, bg, diag=diag):
        r0, r1, c0, c1 = _bbox(cells)
        if lc is None:
            cols = [int(g[y, x]) for y, x in cells]
            col = Counter(cols).most_common(1)[0][0]
        else:
            col = lc
        out[r0:r1 + 1, c0:c1 + 1] = col
    return out


def fill_bbox(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        if not any((i != o).any() for i, o in train):
            return None
        lcs = set()
        for i, o in train:
            diff = (i != o)
            if diff.any():
                lcs |= set(int(v) for v in o[diff])
        cand = [None] + sorted(lcs)
        for diag in (False, True):
            for lc in cand:
                fn = (lambda g, lc=lc, d=diag: _fill_bbox(g, lc, d))
                if all(np.array_equal(fn(i), o) for i, o in train):
                    return fn
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Draw a 1-cell halo frame of a learned colour around each object
# ----------------------------------------------------------------------------
def _halo(g, lc, diag):
    bg = _bg(g)
    out = g.copy()
    H, W = g.shape
    for cells in _components(g, bg, diag=diag):
        r0, r1, c0, c1 = _bbox(cells)
        rr0, rr1 = r0 - 1, r1 + 1
        cc0, cc1 = c0 - 1, c1 + 1
        for c in range(cc0, cc1 + 1):
            for r in (rr0, rr1):
                if 0 <= r < H and 0 <= c < W and out[r, c] == bg:
                    out[r, c] = lc
        for r in range(rr0, rr1 + 1):
            for c in (cc0, cc1):
                if 0 <= r < H and 0 <= c < W and out[r, c] == bg:
                    out[r, c] = lc
    return out


def halo_objects(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        if not any((i != o).any() for i, o in train):
            return None
        lcs = set()
        for i, o in train:
            diff = (i != o)
            if diff.any():
                lcs |= set(int(v) for v in o[diff])
        if not lcs or len(lcs) > 4:
            return None
        for diag in (False, True):
            for lc in sorted(lcs):
                fn = (lambda g, lc=lc, d=diag: _halo(g, lc, d))
                if all(np.array_equal(fn(i), o) for i, o in train):
                    return fn
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Complete a partial / broken rectangle outline into a full rectangle
# ----------------------------------------------------------------------------
def _complete_rects(g):
    """For each same-colour component whose cells all lie on the perimeter of its
    own bounding box, draw the full perimeter in that colour."""
    bg = _bg(g)
    out = g.copy()
    for col, cells in _components_color(g, bg):
        r0, r1, c0, c1 = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 2 or w < 2:
            continue
        cellset = set(cells)
        # all existing cells must lie on the bbox perimeter
        on_perim = all(y in (r0, r1) or x in (c0, c1) for y, x in cellset)
        if not on_perim:
            continue
        # interior must be empty (bg) so we don't overwrite content
        interior = g[r0 + 1:r1, c0 + 1:c1]
        if interior.size and np.any(interior != bg):
            continue
        for c in range(c0, c1 + 1):
            out[r0, c] = col
            out[r1, c] = col
        for r in range(r0, r1 + 1):
            out[r, c0] = col
            out[r, c1] = col
    return out


def complete_rectangle(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        if not any((i != o).any() for i, o in train):
            return None
        fn = _complete_rects
        if all(np.array_equal(fn(i), o) for i, o in train):
            return fn
        return None
    except Exception:
        return None


DETECTORS = [
    fill_hollow_rect,
    outline_bbox,
    fill_bbox,
    halo_objects,
    complete_rectangle,
]
