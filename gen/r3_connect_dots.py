"""Detectors for the CONNECT / PROJECT family.

Same-shape input->output tasks where the transform *adds* line/ray cells on top
of the input (markers are preserved, only background cells get painted):

  * connect_pairs_orth   -- join collinear same-colour markers (row/col) with a
                            straight segment; line colour learned (own or fixed)
  * connect_pairs_diag   -- same idea along the two diagonals
  * cast_rays_orth       -- shoot straight rays from every marker in learned
                            direction(s) until the edge or a non-background cell;
                            ray colour learned (own marker colour or a fixed one)
  * cast_rays_diag       -- diagonal version of the above
  * l_connect_two        -- exactly two distinct-colour markers joined by an
                            L-shaped path (one leg horizontal, one leg vertical)

Every detector is defensive and self-verifying at fit time; the engine re-checks
that the returned transform reproduces every training pair exactly before it is
ever used, so a detector can only ever *add* solves.
"""
import numpy as np
from itertools import combinations


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _added_colors(train):
    """Colours that appear on changed cells across all demos (the palette the
    line/ray colour is drawn from)."""
    cols = set()
    for i, o in train:
        d = (i != o)
        if d.any():
            cols |= set(int(x) for x in o[d].tolist())
    return cols


def _pure_addition(train):
    """True iff every changed cell was background in the input (markers kept)."""
    for i, o in train:
        if i.shape != o.shape:
            return False
        bg = _bg(i)
        d = (i != o)
        if d.any() and not np.all(i[d] == bg):
            return False
    return True


# ---------------------------------------------------------------------------
# Connect collinear same-colour marker pairs (orthogonal)
# ---------------------------------------------------------------------------
def _connect_orth(g, bg, lc):
    """Join adjacent same-colour markers that are collinear (same row or column)
    with only background between them.  lc=None -> draw in the pair's own colour,
    otherwise use the fixed colour lc."""
    out = g.copy()
    H, W = g.shape
    for r in range(H):
        cols = [c for c in range(W) if g[r, c] != bg]
        for a, b in zip(cols, cols[1:]):
            if g[r, a] == g[r, b] and all(g[r, c] == bg for c in range(a + 1, b)):
                col = int(g[r, a]) if lc is None else lc
                for c in range(a + 1, b):
                    out[r, c] = col
    for c in range(W):
        rows = [r for r in range(H) if g[r, c] != bg]
        for a, b in zip(rows, rows[1:]):
            if g[a, c] == g[b, c] and all(g[r, c] == bg for r in range(a + 1, b)):
                col = int(g[a, c]) if lc is None else lc
                for r in range(a + 1, b):
                    out[r, c] = col
    return out


def connect_pairs_orth(train):
    try:
        if not _pure_addition(train):
            return None
        if not any((i != o).any() for i, o in train):
            return None
        cands = [int(x) for x in sorted(_added_colors(train))] + [None]
        for lc in cands:
            if all(np.array_equal(_connect_orth(i, _bg(i), lc), o) for i, o in train):
                return (lambda g, lc=lc: _connect_orth(g, _bg(g), lc))
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Connect collinear same-colour marker pairs (diagonal)
# ---------------------------------------------------------------------------
def _connect_diag(g, bg, lc):
    out = g.copy()
    H, W = g.shape
    pts = [(r, c) for r in range(H) for c in range(W) if g[r, c] != bg]
    for idx, (r1, c1) in enumerate(pts):
        for (r2, c2) in pts[idx + 1:]:
            if r1 == r2 or c1 == c2:
                continue
            if abs(r2 - r1) != abs(c2 - c1):
                continue
            if g[r1, c1] != g[r2, c2]:
                continue
            dr = 1 if r2 > r1 else -1
            dc = 1 if c2 > c1 else -1
            # background all along the diagonal between them?
            ok = True
            y, x = r1 + dr, c1 + dc
            while (y, x) != (r2, c2):
                if g[y, x] != bg:
                    ok = False
                    break
                y += dr
                x += dc
            if not ok:
                continue
            col = int(g[r1, c1]) if lc is None else lc
            y, x = r1 + dr, c1 + dc
            while (y, x) != (r2, c2):
                out[y, x] = col
                y += dr
                x += dc
    return out


def connect_pairs_diag(train):
    try:
        if not _pure_addition(train):
            return None
        if not any((i != o).any() for i, o in train):
            return None
        cands = [int(x) for x in sorted(_added_colors(train))] + [None]
        for lc in cands:
            if all(np.array_equal(_connect_diag(i, _bg(i), lc), o) for i, o in train):
                return (lambda g, lc=lc: _connect_diag(g, _bg(g), lc))
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cast straight rays from markers until the edge or a non-background cell
# ---------------------------------------------------------------------------
_ORTH = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
_DIAG = {'UL': (-1, -1), 'UR': (-1, 1), 'DL': (1, -1), 'DR': (1, 1)}


def _rays(g, bg, dirs, dirmap, fixed):
    out = g.copy()
    H, W = g.shape
    src = [(r, c) for r in range(H) for c in range(W) if g[r, c] != bg]
    for (r, c) in src:
        col = int(g[r, c]) if fixed is None else fixed
        for d in dirs:
            dy, dx = dirmap[d]
            y, x = r + dy, c + dx
            while 0 <= y < H and 0 <= x < W and g[y, x] == bg:
                out[y, x] = col
                y += dy
                x += dx
    return out


def _cast_rays_family(train, dirmap):
    if not _pure_addition(train):
        return None
    if not any((i != o).any() for i, o in train):
        return None
    fixedcands = [None] + [int(x) for x in sorted(_added_colors(train))]
    alld = list(dirmap)
    for k in range(1, len(alld) + 1):
        for combo in combinations(alld, k):
            for fixed in fixedcands:
                if all(np.array_equal(_rays(i, _bg(i), combo, dirmap, fixed), o)
                       for i, o in train):
                    return (lambda g, c=combo, f=fixed, dm=dirmap:
                            _rays(g, _bg(g), c, dm, f))
    return None


def cast_rays_orth(train):
    try:
        return _cast_rays_family(train, _ORTH)
    except Exception:
        return None


def cast_rays_diag(train):
    try:
        return _cast_rays_family(train, _DIAG)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# L-connect: exactly two distinct-colour markers joined by an L-shaped path
# ---------------------------------------------------------------------------
def _markers(g, bg):
    return [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] != bg]


def _l_connect(g, bg, lc, pick_row):
    """Draw an L between the two markers.  pick_row selects which marker anchors
    the horizontal leg (its row holds the horizontal segment); the other marker
    anchors the vertical leg (its column holds the vertical segment).  The corner
    sits at (row-anchor's row, col-anchor's column).

    pick_row(pts) -> 0 or 1, the index of the row-anchor marker.  The line colour
    lc is painted on bg cells only; both markers are preserved."""
    pts = _markers(g, bg)
    if len(pts) != 2:
        return g.copy()
    ri = pick_row(g, pts)
    if ri is None:
        return g.copy()
    (rr, rc) = pts[ri]          # row anchor
    (cr, cc) = pts[1 - ri]      # col anchor
    out = g.copy()
    # horizontal leg along the row-anchor's row, from its column to the corner
    for c in range(min(rc, cc), max(rc, cc) + 1):
        if out[rr, c] == bg:
            out[rr, c] = lc
    # vertical leg along the col-anchor's column, from the corner to it
    for r in range(min(rr, cr), max(rr, cr) + 1):
        if out[r, cc] == bg:
            out[r, cc] = lc
    return out


def _pick_by_color(color):
    def pick(g, pts):
        for k, (r, c) in enumerate(pts):
            if int(g[r, c]) == color:
                return k
        return None
    return pick


def l_connect_two(train):
    try:
        if not _pure_addition(train):
            return None
        if not any((i != o).any() for i, o in train):
            return None
        colors = set()
        for i, o in train:
            ms = _markers(i, _bg(i))
            if len(ms) != 2:
                return None
            for (r, c) in ms:
                colors.add(int(i[r, c]))
        # picking strategies for the row-anchor marker
        picks = [("first", lambda g, pts: 0),
                 ("last", lambda g, pts: 1)]
        for col in sorted(colors):
            picks.append((f"color{col}", _pick_by_color(col)))
        cands = [int(x) for x in sorted(_added_colors(train))]
        for lc in cands:
            for _, pick in picks:
                if all(np.array_equal(_l_connect(i, _bg(i), lc, pick), o)
                       for i, o in train):
                    return (lambda g, lc=lc, pk=pick: _l_connect(g, _bg(g), lc, pk))
        return None
    except Exception:
        return None


DETECTORS = [
    connect_pairs_orth,
    connect_pairs_diag,
    cast_rays_orth,
    cast_rays_diag,
    l_connect_two,
]
