"""Detectors for the CONTAINMENT / INSIDE-OUTSIDE family.

Core idea: a closed shape (a loop of non-background cells) traps one or more
background regions in its interior. The transformation recolors those enclosed
regions. Which colour a region receives is a *learned function of a property of
that region* -- most often its size (cell count), sometimes its interior
dimensions, and in the simplest case a single fixed colour for every region.

Detectors here (all defensive; the engine re-verifies exact train reproduction):

  * enclosed_fill_uniform   -- every enclosed bg region -> one learned colour.
  * enclosed_fill_by_size   -- region cell-count -> learned colour (a map).
                               Generalises to unseen sizes by rank / parity
                               when those simpler rules also fit the demos.
  * enclosed_fill_by_dims   -- region bounding-box (h,w) -> learned colour.
  * region_recolor_by_size  -- like by_size but for solid (non-bg) blobs whose
                               interior/whole is recoloured by their size.

Only background cells that are NOT reachable from the grid border through
4-connected background are considered "enclosed" (standard inside/outside test).
"""
import numpy as np
from collections import deque


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _enclosed_mask(g, bg):
    """Background cells not reachable from the border via 4-connected bg."""
    H, W = g.shape
    reach = np.zeros((H, W), bool)
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
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and g[ny, nx] == bg and not reach[ny, nx]:
                reach[ny, nx] = True
                dq.append((ny, nx))
    return (g == bg) & (~reach)


def _components(mask):
    """4-connected components of a boolean mask -> list of cell lists."""
    H, W = mask.shape
    seen = np.zeros((H, W), bool)
    out = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append(cells)
    return out


def _dims(cells):
    ys = [y for y, _ in cells]
    xs = [x for _, x in cells]
    return (max(ys) - min(ys) + 1, max(xs) - min(xs) + 1)


def _regions(g, bg):
    """Enclosed background regions of g -> list of cell lists."""
    return _components(_enclosed_mask(g, bg))


def _diff_ok(i, o, em):
    """The only changes are within enclosed bg cells (and were bg in input).

    We do NOT require every enclosed cell to change (some tasks leave certain
    containers untouched), only that no change lands outside the enclosed set.
    """
    diff = (i != o)
    if not diff.any():
        return False
    if not np.all(em[diff]):
        return False
    return True


# --------------------------------------------------------------------------
# 1. uniform enclosed fill: every enclosed region -> one learned colour
# --------------------------------------------------------------------------
def enclosed_fill_uniform(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        fills = set()
        for i, o in train:
            bg = _bg(i)
            em = _enclosed_mask(i, bg)
            if not em.any():
                return None
            if not _diff_ok(i, o, em):
                return None
            # every enclosed cell must take the same colour and be changed
            oc = set(int(o[y, x]) for y, x in zip(*np.where(em)))
            if len(oc) != 1:
                return None
            fills |= oc
        if len(fills) != 1:
            return None
        fill = int(next(iter(fills)))

        def fn(g):
            bg = _bg(g)
            em = _enclosed_mask(g, bg)
            out = g.copy()
            out[em] = fill
            return out
        return fn
    except Exception:
        return None


# --------------------------------------------------------------------------
# 2. fill enclosed region by a learned function of its size (cell count)
# --------------------------------------------------------------------------
def _learn_feature_map(train, feat):
    """Return (mapping feat->colour, bg-of-first) if a consistent per-region map
    exists over all enclosed regions of all demos. Each region must be recoloured
    to a single colour. Regions left unchanged (colour == bg) are allowed only if
    consistent. Returns None on any conflict."""
    mp = {}
    any_change = False
    for i, o in train:
        bg = _bg(i)
        em = _enclosed_mask(i, bg)
        if not em.any():
            return None
        if not _diff_ok(i, o, em):
            return None
        for cells in _components(em):
            ocs = set(int(o[y, x]) for y, x in cells)
            if len(ocs) != 1:
                return None
            oc = next(iter(ocs))
            key = feat(cells)
            if key in mp and mp[key] != oc:
                return None
            mp[key] = oc
            if oc != bg:
                any_change = True
    if not any_change:
        return None
    return mp


def enclosed_fill_by_size(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        mp = _learn_feature_map(train, lambda cells: len(cells))
        if mp is None:
            return None
        # need genuine discrimination OR keep as backup when single entry
        keys = sorted(mp)

        # Try to promote to a rank-based rule so unseen sizes still map sensibly:
        # sort distinct sizes ascending, remember the colour sequence.
        rank_colors = [mp[k] for k in keys]

        def fn(g):
            bg = _bg(g)
            em = _enclosed_mask(g, bg)
            out = g.copy()
            comps = _components(em)
            if not comps:
                return out
            for cells in comps:
                s = len(cells)
                if s in mp:
                    col = mp[s]
                else:
                    # unseen size: place it by rank among known sizes and take
                    # the colour of the nearest rank (clamped). Deterministic.
                    lo = [k for k in keys if k <= s]
                    idx = len(lo) - 1 if lo else 0
                    idx = max(0, min(idx, len(rank_colors) - 1))
                    col = rank_colors[idx]
                for y, x in cells:
                    out[y, x] = col
            return out
        return fn
    except Exception:
        return None


# --------------------------------------------------------------------------
# 3. fill enclosed region by a learned function of its interior (h,w) dims
# --------------------------------------------------------------------------
def enclosed_fill_by_dims(train):
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        mp = _learn_feature_map(train, lambda cells: _dims(cells))
        if mp is None:
            return None
        keys = list(mp)

        def fn(g):
            bg = _bg(g)
            em = _enclosed_mask(g, bg)
            out = g.copy()
            for cells in _components(em):
                d = _dims(cells)
                if d in mp:
                    col = mp[d]
                elif (d[1], d[0]) in mp:      # allow transposed match
                    col = mp[(d[1], d[0])]
                else:
                    # fall back to area if a unique area colour exists
                    area_choices = {mp[k] for k in keys if k[0] * k[1] == d[0] * d[1]}
                    if len(area_choices) == 1:
                        col = next(iter(area_choices))
                    else:
                        continue
                for y, x in cells:
                    out[y, x] = col
            return out
        return fn
    except Exception:
        return None


# --------------------------------------------------------------------------
# 4. recolour SOLID coloured blobs by their size (containment as object class)
#    e.g. every same-colour object is recoloured according to how many cells it
#    occupies -- a common "classify by size" companion to interior fills.
# --------------------------------------------------------------------------
def _same_color_comps(g, bg):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = g[r, c]
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < H and 0 <= nx < W and not seen[ny, nx]
                            and g[ny, nx] == col):
                        seen[ny, nx] = True
                        st.append((ny, nx))
            out.append(cells)
    return out


def region_recolor_by_size(train):
    """Every foreground blob is recoloured whole to a colour that is a learned
    function of its cell-count. Shapes/positions unchanged; only colours change,
    and only where the input was non-background."""
    try:
        if any(i.shape != o.shape for i, o in train):
            return None
        mp = {}
        changed = False
        for i, o in train:
            bg = _bg(i)
            if _bg(o) != bg:
                return None
            # changes must be confined to foreground cells, bg stays bg
            diff = (i != o)
            if not diff.any():
                return None
            if np.any((i == bg) & diff):
                return None
            for cells in _same_color_comps(i, bg):
                ocs = set(int(o[y, x]) for y, x in cells)
                if len(ocs) != 1:
                    return None
                oc = next(iter(ocs))
                s = len(cells)
                if s in mp and mp[s] != oc:
                    return None
                mp[s] = oc
                if oc != int(i[cells[0]]):
                    changed = True
        if not changed:
            return None
        keys = sorted(mp)
        colors = [mp[k] for k in keys]

        def fn(g):
            bg = _bg(g)
            out = g.copy()
            for cells in _same_color_comps(g, bg):
                s = len(cells)
                if s in mp:
                    col = mp[s]
                else:
                    lo = [k for k in keys if k <= s]
                    idx = len(lo) - 1 if lo else 0
                    idx = max(0, min(idx, len(colors) - 1))
                    col = colors[idx]
                for y, x in cells:
                    out[y, x] = col
            return out
        return fn
    except Exception:
        return None


DETECTORS = [
    enclosed_fill_uniform,
    enclosed_fill_by_size,
    enclosed_fill_by_dims,
    region_recolor_by_size,
]
