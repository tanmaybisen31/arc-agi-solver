"""Denoising & cleanup detectors for ARC-AGI.

Family: same-shape input->output where the output is a cleaned version of the
input. Sub-patterns:

  1. remove_small_components  - delete connected components of non-background
     color whose size is <= a learned threshold (isolated pixels / tiny specks),
     recolored to background. Learns connectivity (4/8) and threshold from train.

  2. remove_isolated_per_color - like (1) but connectivity is computed per color
     (components of a single color); removes size<=T specks. Handles overlapping
     multi-color scenes where one color is the noise.

  3. recolor_isolated_to_neighbor - isolated single pixels of a "noise" color are
     recolored to the majority color of their surrounding (non-bg) neighbours,
     instead of being erased.

  4. keep_largest_component - erase everything except the single biggest object.

  5. fill_holes - fill background pockets fully enclosed by a single object with
     that object's color (or a learned fill color).

  6. remove_noise_color - delete every pixel of a single color that appears in
     the input but never in any output (a pure "noise colour" removal).

Every detector is defensive and returns None when the rule doesn't fit; the
engine verifies each transform reproduces all training outputs exactly before it
is ever used, so a mis-fit is harmless.
"""
import numpy as np
from collections import Counter, deque

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _components(mask, diag):
    """Connected components of a boolean mask. Returns list of cell lists."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    nbrs = _N8 if diag else _N4
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and mask[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append(cells)
    return comps


def _all_same_shape(train):
    return all(i.shape == o.shape for i, o in train)


def _verify(fn, train):
    try:
        for i, o in train:
            r = fn(i)
            if r is None or r.shape != o.shape or not np.array_equal(r, o):
                return False
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# 1 + 2. remove small components (all-color mask, and per-color mask)
# ----------------------------------------------------------------------------
def _make_remove_small(diag, thresh, per_color):
    def fn(g):
        try:
            bg = _bg_color(g)
            out = g.copy()
            if per_color:
                for col in np.unique(g):
                    col = int(col)
                    if col == bg:
                        continue
                    mask = (g == col)
                    for cells in _components(mask, diag):
                        if len(cells) <= thresh:
                            for (r, c) in cells:
                                out[r, c] = bg
            else:
                mask = (g != bg)
                for cells in _components(mask, diag):
                    if len(cells) <= thresh:
                        for (r, c) in cells:
                            out[r, c] = bg
            return out
        except Exception:
            return None
    return fn


def remove_small_components(train):
    """Erase non-background components of size <= T to background.

    Learns connectivity (4/8), per-color vs all-color grouping, and the largest
    threshold T (1..limit) that still reproduces every training output. We pick
    the SMALLEST T that works to stay conservative and generalize well, then also
    the diag/per-color combo whose kept/removed split is unambiguous.
    """
    if not _all_same_shape(train):
        return None
    # every change must be color->bg
    for i, o in train:
        bg = _bg_color(i)
        diff = (i != o)
        if not diff.any():
            return None
        if not np.all(o[diff] == bg):
            return None

    best = None
    for per_color in (False, True):
        for diag in (False, True):
            # collect the sizes of removed vs kept components across train to
            # find whether a threshold cleanly separates them.
            removed_sizes = []
            kept_sizes = []
            consistent = True
            for i, o in train:
                bg = _bg_color(i)
                if per_color:
                    groups = []
                    for col in np.unique(i):
                        col = int(col)
                        if col == bg:
                            continue
                        for cells in _components(i == col, diag):
                            groups.append(cells)
                else:
                    groups = _components(i != bg, diag)
                for cells in groups:
                    rem = sum(1 for (r, c) in cells if o[r, c] == bg)
                    if rem == len(cells):
                        removed_sizes.append(len(cells))
                    elif rem == 0:
                        kept_sizes.append(len(cells))
                    else:
                        consistent = False
                        break
                if not consistent:
                    break
            if not consistent or not removed_sizes:
                continue
            max_rem = max(removed_sizes)
            min_kept = min(kept_sizes) if kept_sizes else max_rem + 1
            # need a clean size separation: everything removed strictly smaller
            # than everything kept.
            if max_rem >= min_kept:
                continue
            thresh = max_rem  # remove up to the largest removed size
            fn = _make_remove_small(diag, thresh, per_color)
            if _verify(fn, train):
                # prefer smaller thresh, then 4-conn, then all-color (simpler)
                score = (thresh, diag, per_color)
                if best is None or score < best[0]:
                    best = (score, fn)
    return best[1] if best else None


# ----------------------------------------------------------------------------
# 3. recolor isolated pixels to the majority surrounding colour
# ----------------------------------------------------------------------------
def _neighbor_majority(g, r, c, bg, diag):
    H, W = g.shape
    nbrs = _N8 if diag else _N4
    cnt = Counter()
    for dy, dx in nbrs:
        ny, nx = r + dy, c + dx
        if 0 <= ny < H and 0 <= nx < W:
            v = int(g[ny, nx])
            if v != bg:
                cnt[v] += 1
    if not cnt:
        return None
    return cnt.most_common(1)[0][0]


def recolor_isolated_to_neighbor(train):
    """Single-cell components (isolated pixels) get recolored to the majority
    colour among their surrounding non-background neighbours. Output shape ==
    input shape; only isolated pixels change and they never become bg."""
    if not _all_same_shape(train):
        return None
    # changes must be non-bg -> non-bg
    changed_any = False
    for i, o in train:
        bg = _bg_color(i)
        diff = (i != o)
        if diff.any():
            changed_any = True
        if np.any(o[diff] == bg) or np.any(i[diff] == bg):
            return None
    if not changed_any:
        return None

    for diag in (True, False):
        def fn(g, diag=diag):
            try:
                bg = _bg_color(g)
                out = g.copy()
                mask = (g != bg)
                for cells in _components(mask, diag):
                    if len(cells) == 1:
                        (r, c) = cells[0]
                        m = _neighbor_majority(g, r, c, bg, diag)
                        if m is not None:
                            out[r, c] = m
                return out
            except Exception:
                return None
        if _verify(fn, train):
            return fn
    return None


# ----------------------------------------------------------------------------
# 4. keep largest component only
# ----------------------------------------------------------------------------
def keep_largest_component(train):
    """Everything except the single largest non-bg component becomes background.
    Tries 4- and 8-connectivity, and both color-blind and per-color grouping."""
    if not _all_same_shape(train):
        return None
    for i, o in train:
        bg = _bg_color(i)
        diff = (i != o)
        if not diff.any():
            return None
        if not np.all(o[diff] == bg):
            return None

    def make(diag, per_color):
        def fn(g):
            try:
                bg = _bg_color(g)
                if per_color:
                    groups = []
                    for col in np.unique(g):
                        col = int(col)
                        if col == bg:
                            continue
                        for cells in _components(g == col, diag):
                            groups.append(cells)
                else:
                    groups = _components(g != bg, diag)
                if not groups:
                    return g.copy()
                big = max(groups, key=len)
                out = np.full_like(g, bg)
                for (r, c) in big:
                    out[r, c] = g[r, c]
                return out
            except Exception:
                return None
        return fn

    for diag in (False, True):
        for per_color in (False, True):
            fn = make(diag, per_color)
            if _verify(fn, train):
                return fn
    return None


# ----------------------------------------------------------------------------
# 5. fill interior holes of objects
# ----------------------------------------------------------------------------
def _background_touching_exterior(g, bg, diag):
    """Return boolean mask of bg cells connected to the grid border (exterior)."""
    H, W = g.shape
    ext = np.zeros((H, W), dtype=bool)
    nbrs = _N8 if diag else _N4
    q = deque()
    for r in range(H):
        for c in (0, W - 1):
            if g[r, c] == bg and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if g[r, c] == bg and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    while q:
        y, x = q.popleft()
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not ext[ny, nx] and g[ny, nx] == bg:
                ext[ny, nx] = True
                q.append((ny, nx))
    return ext


def fill_holes(train):
    """Background pockets NOT connected to the border (enclosed holes) get
    filled. Fill colour is learned: either the enclosing object's colour
    (per-hole) or a single constant colour used across all holes."""
    if not _all_same_shape(train):
        return None
    # changes must be bg -> non-bg only (adding pixels)
    for i, o in train:
        bg = _bg_color(i)
        diff = (i != o)
        if not diff.any():
            return None
        if not np.all(i[diff] == bg):
            return None

    # exterior bg uses 4-conn (holes are enclosed if the pocket cannot reach
    # the border by orthogonal steps -> standard hole definition)
    def enclosing_color(g, cells, bg):
        # find the color surrounding this pocket; if unique, use it
        H, W = g.shape
        border = Counter()
        for (r, c) in cells:
            for dy, dx in _N4:
                ny, nx = r + dy, c + dx
                if 0 <= ny < H and 0 <= nx < W and g[ny, nx] != bg:
                    border[int(g[ny, nx])] += 1
        if not border:
            return None
        return border.most_common(1)[0][0]

    def make(fill_mode, const_color):
        def fn(g):
            try:
                bg = _bg_color(g)
                ext = _background_touching_exterior(g, bg, diag=False)
                out = g.copy()
                # interior bg cells = bg and not exterior
                interior = (g == bg) & (~ext)
                for cells in _components(interior, diag=False):
                    if fill_mode == "const":
                        fill = const_color
                    else:
                        fill = enclosing_color(g, cells, bg)
                    if fill is None:
                        continue
                    for (r, c) in cells:
                        out[r, c] = fill
                return out
            except Exception:
                return None
        return fn

    # learn a candidate constant fill color from training changes
    const_colors = set()
    for i, o in train:
        diff = (i != o)
        for v in np.unique(o[diff]):
            const_colors.add(int(v))

    # try enclosing-color mode first (more general), then each constant
    fn = make("enclose", None)
    if _verify(fn, train):
        return fn
    for cc in sorted(const_colors):
        fn = make("const", cc)
        if _verify(fn, train):
            return fn
    return None


# ----------------------------------------------------------------------------
# 6. remove a whole noise colour (color present in inputs, absent in outputs)
# ----------------------------------------------------------------------------
def remove_noise_color(train):
    """A single colour appears in inputs but never in outputs; erase it to bg
    everywhere. Also handles the case where several such colours exist."""
    if not _all_same_shape(train):
        return None
    in_colors = set()
    out_colors = set()
    for i, o in train:
        in_colors |= set(int(v) for v in np.unique(i))
        out_colors |= set(int(v) for v in np.unique(o))
    noise = in_colors - out_colors
    if not noise:
        return None
    # every change must be one of the noise colors -> bg
    for i, o in train:
        bg = _bg_color(i)
        diff = (i != o)
        if not diff.any():
            return None
        if not np.all(o[diff] == bg):
            return None
        if not np.all(np.isin(i[diff], list(noise))):
            return None

    noise_list = sorted(noise)

    def fn(g):
        try:
            bg = _bg_color(g)
            out = g.copy()
            for nc in noise_list:
                out[g == nc] = bg
            return out
        except Exception:
            return None
    if _verify(fn, train):
        return fn
    return None


# ----------------------------------------------------------------------------
# 7. remove "impure" objects (too many pixels of a secondary/impurity colour)
# ----------------------------------------------------------------------------
def remove_impure_objects(train):
    """Objects (non-bg connected components, color-blind) are kept or erased
    based on how many pixels of a learned "impurity" colour M they contain.

    Cleanup framing: a scene has objects built mostly of a main colour, but some
    are contaminated with pixels of a second colour. The output keeps the clean
    objects and erases the contaminated ones (or vice-versa). We learn:
      - connectivity (4/8),
      - the impurity colour M,
      - a count threshold T and direction (keep if count<=T, or keep if count>T),
    picking whichever cleanly separates removed from kept components on train.
    """
    if not _all_same_shape(train):
        return None
    # every change must be color -> bg (objects are erased)
    for i, o in train:
        bg = _bg_color(i)
        diff = (i != o)
        if not diff.any():
            return None
        if not np.all(o[diff] == bg):
            return None

    colors = set()
    for i, _ in train:
        colors |= set(int(v) for v in np.unique(i))

    best = None
    for diag in (True, False):
        for M in sorted(colors):
            data = []          # (count_of_M, removed_bool)
            consistent = True
            for i, o in train:
                bg = _bg_color(i)
                if M == bg:
                    consistent = False
                    break
                for cells in _components(i != bg, diag):
                    cm = sum(1 for (r, c) in cells if i[r, c] == M)
                    rem = sum(1 for (r, c) in cells if o[r, c] == bg)
                    if rem == len(cells):
                        data.append((cm, True))
                    elif rem == 0:
                        data.append((cm, False))
                    else:
                        consistent = False
                        break
                if not consistent:
                    break
            if not consistent or not data:
                continue
            rem_counts = [cm for cm, r in data if r]
            keep_counts = [cm for cm, r in data if not r]
            if not rem_counts or not keep_counts:
                continue  # need both classes present to define a boundary
            # find a clean count-based split between kept and removed
            if max(keep_counts) < min(rem_counts):
                T, mode = max(keep_counts), "keep_le"    # keep if count <= T
            elif max(rem_counts) < min(keep_counts):
                T, mode = max(rem_counts), "keep_gt"      # keep if count > T
            else:
                continue

            def fn(g, M=M, T=T, mode=mode, diag=diag):
                try:
                    bg = _bg_color(g)
                    out = g.copy()
                    for cells in _components(g != bg, diag):
                        cm = sum(1 for (r, c) in cells if g[r, c] == M)
                        if mode == "keep_le":
                            remove = cm > T
                        else:
                            remove = cm <= T
                        if remove:
                            for (r, c) in cells:
                                out[r, c] = bg
                    return out
                except Exception:
                    return None
            if _verify(fn, train):
                # prefer the smallest boundary / 8-conn as it matched real tasks
                score = (0 if diag else 1, T, M)
                if best is None or score < best[0]:
                    best = (score, fn)
    return best[1] if best else None


DETECTORS = [
    remove_small_components,
    recolor_isolated_to_neighbor,
    keep_largest_component,
    remove_impure_objects,
    fill_holes,
    remove_noise_color,
]
