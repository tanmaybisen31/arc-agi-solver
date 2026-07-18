"""Object-movement rule detectors for ARC-AGI.

Family: OBJECT MOVEMENT. The output is the same-shape grid as the input, with
one or more objects *repositioned* (translated / copied) according to a rule
learned from the training demos:

  - fixed_vector_translation     : every non-bg cell shifts by one learned
                                   (dr, dc).
  - per_color_vector_translation : each colour shifts by its own learned
                                   (dr, dc).
  - object_gravity_rigid         : each connected object slides as a rigid
                                   block in one of the 4 directions until it
                                   hits the wall or another (stationary) object.
  - objects_to_edge              : each object translates all the way to a
                                   chosen edge (they never collide in the demos).

Design principles
-----------------
* Every rule is *learned* from the demos (vectors, direction, edge) and then
  the harness re-verifies it reproduces EVERY demo exactly, so a mis-fit is
  automatically rejected.
* Detectors abstain (return None) unless the rule is genuinely discriminating,
  to avoid stealing votes from more specific detectors on tasks they don't own.
* numpy + stdlib only. Defensive against ragged / degenerate inputs.
"""
import numpy as np
from collections import Counter


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bg_color(g):
    """Most common colour = background."""
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, bg, diag, samecolor):
    """Connected components of non-bg cells.

    diag=True -> 8-connectivity else 4-connectivity.
    samecolor=True -> a component only grows into cells of the SAME colour.
    Returns list of lists of (r, c).
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
            col = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        if samecolor and int(g[ny, nx]) != col:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _fit(fn, train):
    """True iff fn reproduces every demo exactly (and never errors / returns None)."""
    try:
        for i, o in train:
            r = fn(i)
            if r is None or not isinstance(r, np.ndarray):
                return False
            if r.shape != o.shape or not np.array_equal(r, o):
                return False
        return True
    except Exception:
        return False


def _learn_shift_for_pair(i, o, bg):
    """Return the (dr, dc) that maps every non-bg cell of i to o exactly, or None.

    Only accepts a shift under which no non-bg cell falls off the grid and the
    resulting placement equals o exactly.
    """
    H, W = i.shape
    if i.shape != o.shape:
        return None
    ipos = np.argwhere(i != bg)
    if ipos.size == 0:
        return None
    opos = np.argwhere(o != bg)
    if opos.shape[0] != ipos.shape[0]:
        return None
    # candidate shift from any anchor: match the top-left-most cell (by color)
    # Try shift = (min_row_o - min_row_i, min_col_o - min_col_i) as primary guess,
    # but validate exhaustively over the small set of plausible shifts.
    guesses = set()
    imin = ipos.min(axis=0)
    omin = opos.min(axis=0)
    guesses.add((int(omin[0] - imin[0]), int(omin[1] - imin[1])))
    imax = ipos.max(axis=0)
    omax = opos.max(axis=0)
    guesses.add((int(omax[0] - imax[0]), int(omax[1] - imax[1])))
    for dr, dc in guesses:
        out = np.full_like(i, bg)
        ok = True
        for (r, c) in ipos:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = i[r, c]
            else:
                ok = False
                break
        if ok and np.array_equal(out, o):
            return (dr, dc)
    return None


def _apply_shift(g, dr, dc, bg):
    H, W = g.shape
    out = np.full_like(g, bg)
    for (r, c) in np.argwhere(g != bg):
        nr, nc = int(r) + dr, int(c) + dc
        if 0 <= nr < H and 0 <= nc < W:
            out[nr, nc] = g[r, c]
    return out


# --------------------------------------------------------------------------- #
# 1) fixed single-vector translation of all non-bg content
# --------------------------------------------------------------------------- #
def fixed_vector_translation(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # background must be consistent enough; learn per-pair shift, require single vector
    vecs = set()
    bgs = []
    for i, o in train:
        bg = _bg_color(i)
        bgs.append(bg)
        v = _learn_shift_for_pair(i, o, bg)
        if v is None:
            return None
        vecs.add(v)
    if len(vecs) != 1:
        return None
    (dr, dc) = next(iter(vecs))
    if dr == 0 and dc == 0:
        return None  # identity, handled elsewhere

    def fn(g, dr=dr, dc=dc):
        bg = _bg_color(g)
        return _apply_shift(g, dr, dc, bg)

    return fn if _fit(fn, train) else None


# --------------------------------------------------------------------------- #
# 2) per-colour vector translation
# --------------------------------------------------------------------------- #
def per_color_vector_translation(train):
    """Each colour c shifts by its own (dr, dc), consistent across all demos.

    Learned by, for each demo and each colour, finding the shift that maps that
    colour's cells (as a rigid set) from input to output; requiring one global
    shift per colour.
    """
    if any(i.shape != o.shape for i, o in train):
        return None

    color_vec = {}         # color -> (dr, dc)
    saw_nonzero = False

    for i, o in train:
        bg = _bg_color(i)
        colors = set(np.unique(i).tolist()) - {bg}
        # output must have exactly the same colour multiset as input (pure move)
        if Counter(i[i != bg].tolist()) != Counter(o[o != _bg_color(o)].tolist()):
            return None
        for col in colors:
            ipos = np.argwhere(i == col)
            opos = np.argwhere(o == col)
            if ipos.shape[0] != opos.shape[0] or ipos.shape[0] == 0:
                return None
            # shift determined by matching the set as a rigid translation:
            # sort both by (r, c); the constant difference must hold for all.
            ip = sorted((int(r), int(c)) for r, c in ipos)
            op = sorted((int(r), int(c)) for r, c in opos)
            dr = op[0][0] - ip[0][0]
            dc = op[0][1] - ip[0][1]
            if any((oo[0] - ii[0], oo[1] - ii[1]) != (dr, dc) for ii, oo in zip(ip, op)):
                return None
            if col in color_vec and color_vec[col] != (dr, dc):
                return None
            color_vec[col] = (dr, dc)
            if (dr, dc) != (0, 0):
                saw_nonzero = True

    if not saw_nonzero:
        return None

    def fn(g, color_vec=color_vec):
        bg = _bg_color(g)
        H, W = g.shape
        out = np.full_like(g, bg)
        for (r, c) in np.argwhere(g != bg):
            col = int(g[r, c])
            dr, dc = color_vec.get(col, (0, 0))
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = col
        return out

    return fn if _fit(fn, train) else None


# --------------------------------------------------------------------------- #
# 3) rigid object gravity: each object slides in one direction until blocked
# --------------------------------------------------------------------------- #
_DIRS = {"down": (1, 0), "up": (-1, 0), "right": (0, 1), "left": (0, -1)}


def _rigid_gravity(g, direction, diag, samecolor):
    """Slide each connected object as a rigid block toward `direction` until it
    would hit the wall or an already-settled cell."""
    bg = _bg_color(g)
    H, W = g.shape
    dr, dc = _DIRS[direction]
    comps = _components(g, bg, diag, samecolor)
    # order components by how far along the movement axis their leading edge is,
    # so the ones nearest the wall settle first.
    def lead(comp):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        if direction == "down":
            return -max(rs)
        if direction == "up":
            return min(rs)
        if direction == "right":
            return -max(cs)
        return min(cs)  # left
    comps = sorted(comps, key=lead)

    occ = np.zeros((H, W), dtype=bool)
    out = np.full_like(g, bg)
    for comp in comps:
        cells = [(r, c, int(g[r, c])) for (r, c) in comp]
        # find max steps k>=0 such that shifting all cells by k*dir stays in grid
        # and lands on unoccupied cells.
        k = 0
        while True:
            nk = k + 1
            ok = True
            for (r, c, _) in cells:
                nr, nc = r + dr * nk, c + dc * nk
                if not (0 <= nr < H and 0 <= nc < W) or occ[nr, nc]:
                    ok = False
                    break
            if ok:
                k = nk
            else:
                break
        for (r, c, v) in cells:
            nr, nc = r + dr * k, c + dc * k
            out[nr, nc] = v
            occ[nr, nc] = True
    return out


def object_gravity_rigid(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # only worth trying when content actually moves in at least one demo
    if all(np.array_equal(i, o) for i, o in train):
        return None
    for direction in ("down", "up", "right", "left"):
        for diag in (False, True):
            for samecolor in (True, False):
                fn = (lambda g, d=direction, dg=diag, sc=samecolor:
                      _rigid_gravity(g, d, dg, sc))
                if _fit(fn, train):
                    # discriminating: object gravity must differ from per-cell
                    # gravity somewhere, else base 'gravity' already covers it.
                    return fn
    return None


# --------------------------------------------------------------------------- #
# 4) each object translated all the way to a chosen edge
# --------------------------------------------------------------------------- #
def _objects_to_edge(g, edge, diag, samecolor):
    """Translate every object so its bounding box touches `edge` (no collision
    handling; assumes the demos don't require it)."""
    bg = _bg_color(g)
    H, W = g.shape
    comps = _components(g, bg, diag, samecolor)
    out = np.full_like(g, bg)
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        if edge == "top":
            shift = (-min(rs), 0)
        elif edge == "bottom":
            shift = (H - 1 - max(rs), 0)
        elif edge == "left":
            shift = (0, -min(cs))
        elif edge == "right":
            shift = (0, W - 1 - max(cs))
        else:
            return None
        dr, dc = shift
        for (r, c) in comp:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = g[r, c]
    return out


def objects_to_edge(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    if all(np.array_equal(i, o) for i, o in train):
        return None
    for edge in ("top", "bottom", "left", "right"):
        for diag in (False, True):
            for samecolor in (True, False):
                fn = (lambda g, e=edge, dg=diag, sc=samecolor:
                      _objects_to_edge(g, e, dg, sc))
                if _fit(fn, train):
                    return fn
    return None


# --------------------------------------------------------------------------- #
# 5) mover object(s) step toward a stationary target object
# --------------------------------------------------------------------------- #
def _sign(x):
    x = float(x)
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _mover_toward_target(train):
    """Some objects (movers) translate toward a stationary target object.

    Learned settings:
      - which colours are movers vs which single colour is the (stationary)
        target; determined per-demo by observing what stayed put.
      - step mode: 'one' (one king-step toward target) or 'adjacent' (slide
        until orthogonally/king adjacent to the target).
    The rule is expressed generically: for each object that is NOT the target
    colour, move it by one king-step (or until adjacent) toward the target
    object's centre.
    """
    if any(i.shape != o.shape for i, o in train):
        return None
    if all(np.array_equal(i, o) for i, o in train):
        return None

    # Identify the target colour: the colour whose cells are IDENTICAL between
    # input and output in every demo (it never moves), while other colours move.
    common_targets = None
    for i, o in train:
        bg = _bg_color(i)
        colors = set(np.unique(i).tolist()) - {bg}
        stayed = set()
        moved = set()
        for col in colors:
            ip = np.argwhere(i == col)
            op = np.argwhere(o == col)
            if ip.shape[0] != op.shape[0]:
                return None
            if np.array_equal(np.sort(ip, axis=0), np.sort(op, axis=0)):
                stayed.add(col)
            else:
                moved.add(col)
        if not moved:
            return None
        common_targets = stayed if common_targets is None else (common_targets & stayed)
    if not common_targets:
        return None

    def make(target_color, mode, king):
        def fn(g, target_color=target_color, mode=mode, king=king):
            bg = _bg_color(g)
            H, W = g.shape
            if target_color not in np.unique(g):
                return None
            tpos = np.argwhere(g == target_color)
            tcen = tpos.mean(axis=0)
            out = np.full_like(g, bg)
            # keep target in place
            for (r, c) in tpos:
                out[r, c] = target_color
            # movers = every other object
            comps = _components(g, bg, diag=True, samecolor=True)
            occ = out != bg
            for comp in comps:
                col = int(g[comp[0][0], comp[0][1]])
                if col == target_color:
                    continue
                rs = [p[0] for p in comp]
                cs = [p[1] for p in comp]
                ccen = (float(np.mean(rs)), float(np.mean(cs)))
                dvr = _sign(tcen[0] - ccen[0])
                dvc = _sign(tcen[1] - ccen[1])
                if not king:
                    # move only along the dominant axis
                    if abs(tcen[0] - ccen[0]) >= abs(tcen[1] - ccen[1]):
                        dvc = 0
                    else:
                        dvr = 0
                if dvr == 0 and dvc == 0:
                    for (r, c) in comp:
                        out[r, c] = col
                    continue
                # determine number of steps
                if mode == "one":
                    steps = 1
                else:  # 'adjacent' -> slide until any cell becomes adjacent to target/occupied
                    steps = 0
                    while True:
                        ns = steps + 1
                        ok = True
                        adj = False
                        for (r, c) in comp:
                            nr, nc = r + dvr * ns, c + dvc * ns
                            if not (0 <= nr < H and 0 <= nc < W):
                                ok = False
                                break
                        if not ok:
                            break
                        # would this step land ON an occupied cell? stop before.
                        collide = any(occ[r + dvr * ns, c + dvc * ns] for (r, c) in comp)
                        if collide:
                            break
                        steps = ns
                # place
                for (r, c) in comp:
                    nr, nc = r + dvr * steps, c + dvc * steps
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr, nc] = col
                        occ[nr, nc] = True
            return out
        return fn

    for target_color in sorted(common_targets):
        for mode in ("one", "adjacent"):
            for king in (True, False):
                fn = make(target_color, mode, king)
                if _fit(fn, train):
                    return fn
    return None


def mover_toward_target(train):
    return _mover_toward_target(train)


# --------------------------------------------------------------------------- #
# 6) objects / cells explode to the nearest corner of the grid
# --------------------------------------------------------------------------- #
def _to_nearest_corner(g, diag, samecolor):
    bg = _bg_color(g)
    H, W = g.shape
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    comps = _components(g, bg, diag, samecolor)
    out = np.full_like(g, bg)
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        cr = np.mean(rs)
        cc = np.mean(cs)
        # nearest corner to the object's centre
        best = min(corners, key=lambda q: (q[0] - cr) ** 2 + (q[1] - cc) ** 2)
        # translate so the object's nearest own-extent touches that corner
        if best[0] == 0:
            dr = -min(rs)
        else:
            dr = (H - 1) - max(rs)
        if best[1] == 0:
            dc = -min(cs)
        else:
            dc = (W - 1) - max(cs)
        for (r, c) in comp:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = g[r, c]
    return out


def objects_to_nearest_corner(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    if all(np.array_equal(i, o) for i, o in train):
        return None
    for diag in (False, True):
        for samecolor in (True, False):
            fn = (lambda g, dg=diag, sc=samecolor: _to_nearest_corner(g, dg, sc))
            if _fit(fn, train):
                return fn
    return None


# --------------------------------------------------------------------------- #
# 7) directional gravity of ONE colour, other non-bg cells are fixed obstacles
# --------------------------------------------------------------------------- #
def _single_color_gravity(g, mover, direction, diag):
    """Slide each rigid object of colour `mover` toward `direction` until it hits
    the wall, another (stationary) non-mover object, or another settled mover
    object."""
    bg = _bg_color(g)
    H, W = g.shape
    dr, dc = _DIRS[direction]
    out = g.copy()
    # obstacles = every non-bg, non-mover cell stays where it is
    occ = (g != bg) & (g != mover)
    # remove movers from out; we re-place them
    out[g == mover] = bg
    # components of the mover colour only
    mask = (g == mover).astype(int)
    comps = _components(np.where(g == mover, mover, bg), bg, diag, samecolor=True)

    def lead(comp):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        if direction == "down":
            return -max(rs)
        if direction == "up":
            return min(rs)
        if direction == "right":
            return -max(cs)
        return min(cs)
    comps = sorted(comps, key=lead)

    for comp in comps:
        k = 0
        while True:
            nk = k + 1
            ok = True
            for (r, c) in comp:
                nr, nc = r + dr * nk, c + dc * nk
                if not (0 <= nr < H and 0 <= nc < W) or occ[nr, nc]:
                    ok = False
                    break
            if ok:
                k = nk
            else:
                break
        for (r, c) in comp:
            nr, nc = r + dr * k, c + dc * k
            out[nr, nc] = mover
            occ[nr, nc] = True
    return out


def single_color_gravity(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    if all(np.array_equal(i, o) for i, o in train):
        return None
    # candidate mover colours: colours present in all demos
    common = None
    for i, _ in train:
        bg = _bg_color(i)
        cs = set(np.unique(i).tolist()) - {bg}
        common = cs if common is None else (common & cs)
    if not common:
        return None
    for mover in sorted(common):
        for direction in ("down", "up", "right", "left"):
            for diag in (False, True):
                fn = (lambda g, mv=mover, d=direction, dg=diag:
                      _single_color_gravity(g, mv, d, dg))
                if _fit(fn, train):
                    return fn
    return None


# --------------------------------------------------------------------------- #
# 8) align every object to an anchor object's row/column band
# --------------------------------------------------------------------------- #
def _align_to_anchor(g, anchor_color, axis, how, diag, samecolor):
    """Translate every non-anchor object along `axis` so it aligns with the
    anchor object's bounding box (`how` in {min, max, center})."""
    bg = _bg_color(g)
    H, W = g.shape
    apos = np.argwhere(g == anchor_color)
    if apos.size == 0:
        return None
    if axis == 0:
        amin, amax = apos[:, 0].min(), apos[:, 0].max()
    else:
        amin, amax = apos[:, 1].min(), apos[:, 1].max()
    acen = (amin + amax) / 2.0

    comps = _components(g, bg, diag, samecolor)
    out = np.full_like(g, bg)
    for comp in comps:
        col = int(g[comp[0][0], comp[0][1]])
        coords = np.array(comp)
        if col == anchor_color:
            for (r, c) in comp:
                out[r, c] = col
            continue
        vals = coords[:, axis]
        vmin, vmax = vals.min(), vals.max()
        vcen = (vmin + vmax) / 2.0
        if how == "min":
            delta = amin - vmin
        elif how == "max":
            delta = amax - vmax
        else:  # center
            delta = int(round(acen - vcen))
        for (r, c) in comp:
            nr = r + (delta if axis == 0 else 0)
            nc = c + (delta if axis == 1 else 0)
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = col
    return out


def align_objects_to_anchor(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    if all(np.array_equal(i, o) for i, o in train):
        return None
    # anchor = a colour that stays identical in every demo while others move
    common_anchor = None
    for i, o in train:
        bg = _bg_color(i)
        colors = set(np.unique(i).tolist()) - {bg}
        stayed = set()
        moved = False
        for col in colors:
            ip = np.argwhere(i == col)
            op = np.argwhere(o == col)
            if ip.shape[0] != op.shape[0]:
                return None
            if np.array_equal(np.sort(ip, axis=0), np.sort(op, axis=0)):
                stayed.add(col)
            else:
                moved = True
        if not moved:
            return None
        common_anchor = stayed if common_anchor is None else (common_anchor & stayed)
    if not common_anchor:
        return None
    for anchor in sorted(common_anchor):
        for axis in (0, 1):
            for how in ("min", "max", "center"):
                for diag in (True, False):
                    for samecolor in (True, False):
                        fn = (lambda g, a=anchor, ax=axis, h=how, dg=diag, sc=samecolor:
                              _align_to_anchor(g, a, ax, h, dg, sc))
                        if _fit(fn, train):
                            return fn
    return None


DETECTORS = [
    fixed_vector_translation,
    per_color_vector_translation,
    object_gravity_rigid,
    objects_to_edge,
    mover_toward_target,
    objects_to_nearest_corner,
    single_color_gravity,
    align_objects_to_anchor,
]
