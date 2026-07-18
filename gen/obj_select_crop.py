"""Object selection + crop detectors for ARC-AGI.

Family: the output is the bounding-box crop of ONE object selected from the
input by some criterion:
  - largest / smallest object (by cell count or bbox area)
  - the unique-colored / unique-shaped / unique-sized object (odd one out)
  - the object with the most colors
  - the (uniquely) symmetric or (uniquely) asymmetric object
  - the object whose shape is the most common among all objects

Design notes
------------
Each detector below realises ONE selection criterion. For a given criterion it
searches over object-extraction settings (connectivity 4/8, same-colour vs
any-colour components, whether to drop size-1 speckle noise) and keeps the
first setting under which the criterion reproduces EVERY training output
exactly. The harness re-verifies the returned transform, so a mis-fit is
harmless; the real job is to generalise to the test input.

To avoid trivially-fitting-but-wrong rules (e.g. a single-object task where
every criterion "fits"), a criterion is only accepted if it is actually
*discriminating*: at least one training pair had >=2 candidate objects and the
criterion made a genuine, unique choice there. Single-object crops are already
handled by the base `crop_bbox` detector, so we deliberately abstain on them.

Only numpy + stdlib are imported.
"""
import numpy as np
from collections import Counter


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def _bg_color(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag, samecolor):
    """Connected components of non-bg cells.

    diag=True -> 8-connectivity, else 4-connectivity.
    samecolor=True -> a component may only grow into cells of the same colour.
    Returns list of lists of (row, col).
    """
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
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
            col = g[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        if samecolor and g[ny, nx] != col:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _features(g, cells):
    ys = [y for y, x in cells]
    xs = [x for y, x in cells]
    r0, r1 = min(ys), max(ys)
    c0, c1 = min(xs), max(xs)
    crop = g[r0:r1 + 1, c0:c1 + 1]
    colors = frozenset(int(g[y, x]) for y, x in cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=np.int8)
    for y, x in cells:
        mask[y - r0, x - c0] = 1
    symh = bool(np.array_equal(crop, np.fliplr(crop)))
    symv = bool(np.array_equal(crop, np.flipud(crop)))
    return {
        "size": len(cells),
        "colors": colors,
        "ncolors": len(colors),
        "bbox": (h, w),
        "area": h * w,
        "shape": mask.tobytes() + bytes([h % 256, w % 256]),
        "crop": crop.copy(),
        "sym": symh or symv,
    }


def _objects(g, diag, samecolor, drop_singletons):
    bg = _bg_color(g)
    comps = _components(g, bg, diag, samecolor)
    objs = [_features(g, c) for c in comps]
    if drop_singletons:
        big = [o for o in objs if o["size"] > 1]
        if big:                       # never leave zero objects behind
            objs = big
    return objs


def _target_index(objs, out):
    """Index of the object whose bbox crop equals `out`, else None."""
    for k, o in enumerate(objs):
        cr = o["crop"]
        if cr.shape == out.shape and np.array_equal(cr, out):
            return k
    return None


# --------------------------------------------------------------------------- #
# selection criteria  (objs -> index | None)
# --------------------------------------------------------------------------- #
def _sel_max(objs, keyfn):
    vals = [keyfn(o) for o in objs]
    m = max(vals)
    idxs = [k for k, v in enumerate(vals) if v == m]
    return idxs[0] if len(idxs) == 1 else None


def _sel_min(objs, keyfn):
    vals = [keyfn(o) for o in objs]
    m = min(vals)
    idxs = [k for k, v in enumerate(vals) if v == m]
    return idxs[0] if len(idxs) == 1 else None


def _sel_unique(objs, keyfn):
    """The single object whose feature value is unique (appears exactly once)."""
    cnt = Counter(keyfn(o) for o in objs)
    uni = [k for k, o in enumerate(objs) if cnt[keyfn(o)] == 1]
    return uni[0] if len(uni) == 1 else None


def _sel_odd(objs, keyfn):
    """The object whose feature value is the rarest, uniquely so."""
    cnt = Counter(keyfn(o) for o in objs)
    rarest = min(cnt.values())
    idxs = [k for k, o in enumerate(objs) if cnt[keyfn(o)] == rarest]
    return idxs[0] if len(idxs) == 1 else None


def _sel_majority_shape(objs):
    """The object whose shape is the most common (first such, ties -> most common)."""
    cnt = Counter(o["shape"] for o in objs)
    top, topc = cnt.most_common(1)[0]
    # require a genuine majority winner (strictly more common than any other)
    ordered = cnt.most_common()
    if len(ordered) > 1 and ordered[1][1] == topc:
        return None
    idxs = [k for k, o in enumerate(objs) if o["shape"] == top]
    return idxs[0] if idxs else None


def _sel_unique_symmetric(objs):
    idxs = [k for k, o in enumerate(objs) if o["sym"]]
    return idxs[0] if len(idxs) == 1 else None


def _sel_unique_asymmetric(objs):
    idxs = [k for k, o in enumerate(objs) if not o["sym"]]
    return idxs[0] if len(idxs) == 1 else None


# name -> selector callable
_CRITERIA = {
    "largest":        lambda o: _sel_max(o, lambda x: x["size"]),
    "smallest":       lambda o: _sel_min(o, lambda x: x["size"]),
    "largest_area":   lambda o: _sel_max(o, lambda x: x["area"]),
    "smallest_area":  lambda o: _sel_min(o, lambda x: x["area"]),
    "most_colors":    lambda o: _sel_max(o, lambda x: x["ncolors"]),
    "unique_color":   lambda o: _sel_unique(o, lambda x: x["colors"]),
    "unique_shape":   lambda o: _sel_unique(o, lambda x: x["shape"]),
    "unique_size":    lambda o: _sel_unique(o, lambda x: x["size"]),
    "odd_size":       lambda o: _sel_odd(o, lambda x: x["size"]),
    "majority_shape": _sel_majority_shape,
    "unique_symmetric":  _sel_unique_symmetric,
    "unique_asymmetric": _sel_unique_asymmetric,
}

# extraction settings, tried in priority order: prefer 4-conn any-colour first,
# fall back to 8-conn and same-colour and speckle-dropping variants.
_EXTRACTIONS = [
    (False, False, False),   # 4-conn, any-colour, keep all
    (False, False, True),    # 4-conn, any-colour, drop singletons
    (True, False, False),    # 8-conn, any-colour
    (True, False, True),
    (False, True, False),    # 4-conn, same-colour
    (False, True, True),
    (True, True, False),     # 8-conn, same-colour
    (True, True, True),
]


def _make_detector(crit_name):
    selector = _CRITERIA[crit_name]

    def det(train):
        try:
            if not train:
                return None
            # A crop task: every output must be no larger than its input, and
            # at least one strictly smaller (otherwise it's not a crop).
            if any(o.shape[0] > i.shape[0] or o.shape[1] > i.shape[1]
                   for i, o in train):
                return None
            if all(o.shape == i.shape for i, o in train):
                return None

            for diag, samecolor, drop in _EXTRACTIONS:
                ok = True
                discriminating = False
                for i, o in train:
                    objs = _objects(i, diag, samecolor, drop)
                    if not objs:
                        ok = False
                        break
                    tgt = _target_index(objs, o)
                    if tgt is None:
                        ok = False
                        break
                    sel = selector(objs)
                    if sel != tgt:
                        ok = False
                        break
                    if len(objs) >= 2:
                        discriminating = True
                if not ok or not discriminating:
                    continue

                def fn(g, diag=diag, samecolor=samecolor, drop=drop):
                    objs = _objects(g, diag, samecolor, drop)
                    if not objs:
                        return None
                    sel = selector(objs)
                    if sel is None:
                        return None
                    return objs[sel]["crop"].copy()

                return fn
            return None
        except Exception:
            return None

    det.__name__ = "obj_crop_" + crit_name
    return det


# --------------------------------------------------------------------------- #
# public detector list
# --------------------------------------------------------------------------- #
# Ordered by rough specificity so the more distinctive criteria win voting ties.
_ORDER = [
    "unique_color", "unique_shape", "unique_size", "odd_size",
    "unique_symmetric", "unique_asymmetric",
    "majority_shape", "most_colors",
    "largest", "smallest", "largest_area", "smallest_area",
]

DETECTORS = [_make_detector(name) for name in _ORDER]
