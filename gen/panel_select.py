"""Panel-selection detectors for ARC-AGI.

The input is divided into equal (or separator-delimited) panels; the output is
exactly ONE panel chosen by a learnable criterion:
  * the odd-one-out panel (unique content, unique shape-mask, or unique color set),
  * the panel with the most / fewest non-background cells,
  * the panel with the most / fewest distinct colors,
  * the panel with the most / fewest connected components (objects),
  * the panel that occurs most / fewest times,
  * the panel that is uniquely (a)symmetric,
  * the panel that carries a special (non-background) color.

Approach: enumerate deterministic *panelization strategies* (each maps ANY grid
to an ordered list of panels), then try a battery of scalar scorers with
argmax / argmin / unique tie-break rules, plus explicit odd-one-out rules.  A
(strategy, rule) pair is accepted only if it reproduces EVERY training output
exactly; the same pair is then applied to the test input.  The engine also
re-verifies every fit, so mistaken combinations are harmless.
"""
import numpy as np
from collections import Counter

MAXPAN = 64  # guard against absurd panel counts


# ---------------------------------------------------------------- helpers
def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _global_bg(train):
    c = Counter()
    for i, _ in train:
        vals, cnts = np.unique(i, return_counts=True)
        for v, n in zip(vals, cnts):
            c[int(v)] += int(n)
    return c.most_common(1)[0][0] if c else 0


def _components(mask):
    """4-connected component count of a boolean mask."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    n = 0
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                n += 1
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return n


def _segments(is_sep, n):
    segs = []
    start = None
    for k in range(n):
        if not is_sep[k]:
            if start is None:
                start = k
        else:
            if start is not None:
                segs.append((start, k))
                start = None
    if start is not None:
        segs.append((start, n))
    return segs


# ---------------------------------------------------------------- strategies
# A strategy is a callable grid -> (list_of_panels) | None.  It must be
# deterministic and computable on any grid so the learned rule transfers to test.

def _split_by_sep_color(g, sc):
    H, W = g.shape
    rsep = np.array([bool(np.all(g[r, :] == sc)) for r in range(H)])
    csep = np.array([bool(np.all(g[:, c] == sc)) for c in range(W)])
    if not rsep.any() and not csep.any():
        return None
    if rsep.all() or csep.all():
        return None
    rs = _segments(rsep, H)
    cs = _segments(csep, W)
    if not rs or not cs:
        return None
    if len(rs) * len(cs) < 2 or len(rs) * len(cs) > MAXPAN:
        return None
    panels = []
    for a, b in rs:
        for c, d in cs:
            panels.append(g[a:b, c:d])
    return panels


def _auto_sep_color(g):
    """Choose the most plausible separator color: the color that forms >=1 full
    line AND partitions the grid into >=2 equal-shaped panels; prefer the one
    giving the cleanest (uniform-shape) partition, tie-broken by fewest panels."""
    H, W = g.shape
    best = None
    best_key = None
    for sc in np.unique(g):
        sc = int(sc)
        panels = _split_by_sep_color(g, sc)
        if panels is None:
            continue
        shapes = set(p.shape for p in panels)
        uniform = len(shapes) == 1
        # key: prefer uniform panels, then more panels lines removed (bigger split)
        key = (1 if uniform else 0, len(panels))
        if best_key is None or key > best_key:
            best_key = key
            best = sc
    return best


def strat_sep_auto(g):
    sc = _auto_sep_color(g)
    if sc is None:
        return None
    return _split_by_sep_color(g, sc)


def _make_strat_sep_fixed(sc):
    def s(g, sc=sc):
        return _split_by_sep_color(g, sc)
    return s


def _make_strat_even(nr, nc):
    def s(g, nr=nr, nc=nc):
        H, W = g.shape
        if H % nr or W % nc:
            return None
        if nr * nc < 2 or nr * nc > MAXPAN:
            return None
        ph, pw = H // nr, W // nc
        return [g[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
                for r in range(nr) for c in range(nc)]
    return s


def _strat_even_square(g):
    """Split a 1xN or Nx1 strip into square panels (panel side = short side)."""
    H, W = g.shape
    if H == W:
        return None
    if H < W:
        if W % H:
            return None
        n = W // H
        if n < 2 or n > MAXPAN:
            return None
        return [g[:, k * H:(k + 1) * H] for k in range(n)]
    else:
        if H % W:
            return None
        n = H // W
        if n < 2 or n > MAXPAN:
            return None
        return [g[k * W:(k + 1) * W, :] for k in range(n)]


def _candidate_strategies(train):
    """Return list of (name, strategy_fn).  Includes auto-separator, square
    strip, and a few even layouts inferred from output shapes."""
    strats = [("sep_auto", strat_sep_auto), ("even_square", _strat_even_square)]

    # even layouts (nr,nc) that are consistent with every pair: for each pair,
    # H/oh and W/ow give the layout when the output is a panel.
    layouts = None
    ok_layout = True
    for i, o in train:
        H, W = i.shape
        oh, ow = o.shape
        if oh == 0 or ow == 0 or H % oh or W % ow:
            ok_layout = False
            break
        lay = (H // oh, W // ow)
        if lay[0] * lay[1] < 2 or lay[0] * lay[1] > MAXPAN:
            ok_layout = False
            break
        if layouts is None:
            layouts = lay
        elif layouts != lay:
            layouts = None  # not a fixed layout; rely on even_square / sep
            break
    if ok_layout and layouts is not None:
        strats.append((f"even_{layouts[0]}x{layouts[1]}",
                       _make_strat_even(layouts[0], layouts[1])))

    # also a couple of generic even splits that recur (2x2, 1x3, 3x1, 3x3, 1x2, 2x1)
    for nr, nc in [(2, 2), (1, 2), (2, 1), (1, 3), (3, 1), (3, 3), (1, 4), (4, 1)]:
        strats.append((f"evenfix_{nr}x{nc}", _make_strat_even(nr, nc)))

    return strats


# ---------------------------------------------------------------- scorers
def _score_nonbg(panels, gbg):
    return [int((p != _bg(p)).sum()) for p in panels]


def _score_nonglobalbg(panels, gbg):
    return [int((p != gbg).sum()) for p in panels]


def _score_ndistinct(panels, gbg):
    return [len(np.unique(p)) for p in panels]


def _score_ndistinct_nonbg(panels, gbg):
    out = []
    for p in panels:
        u = set(np.unique(p).tolist())
        u.discard(_bg(p))
        out.append(len(u))
    return out


def _score_ncomp(panels, gbg):
    return [_components(p != _bg(p)) for p in panels]


def _score_ncomp_global(panels, gbg):
    return [_components(p != gbg) for p in panels]


def _score_freq(panels, gbg):
    keys = [p.tobytes() + bytes(p.shape) for p in panels]
    c = Counter(keys)
    return [c[k] for k in keys]


def _score_maskfreq(panels, gbg):
    keys = [(p != _bg(p)).tobytes() + bytes(p.shape) for p in panels]
    c = Counter(keys)
    return [c[k] for k in keys]


def _sym_count(p):
    n = 0
    if np.array_equal(p, np.fliplr(p)):
        n += 1
    if np.array_equal(p, np.flipud(p)):
        n += 1
    if p.shape[0] == p.shape[1]:
        if np.array_equal(p, p.T):
            n += 1
        if np.array_equal(p, np.rot90(p, 2).T):
            n += 1
    return n


def _score_sym(panels, gbg):
    return [_sym_count(p) for p in panels]


_SCORERS = [
    _score_nonbg, _score_nonglobalbg, _score_ndistinct, _score_ndistinct_nonbg,
    _score_ncomp, _score_ncomp_global, _score_freq, _score_maskfreq, _score_sym,
]


# ---------------------------------------------------------------- selection rules
def _pick_extreme(panels, scores, want_max):
    if not scores:
        return None
    best = max(scores) if want_max else min(scores)
    idxs = [k for k, s in enumerate(scores) if s == best]
    return idxs[0]  # reading-order tie-break; verified against train


def _pick_unique(panels, scores):
    c = Counter(scores)
    uniq = [k for k, s in enumerate(scores) if c[s] == 1]
    if len(uniq) == 1:
        return uniq[0]
    return None


def _odd_one_out(panels, mode, gbg):
    if len(panels) < 3:
        return None
    if mode == "exact":
        keys = [p.tobytes() + bytes(p.shape) for p in panels]
    elif mode == "mask":
        keys = [(p != _bg(p)).tobytes() + bytes(p.shape) for p in panels]
    elif mode == "colorset":
        keys = [bytes(sorted(np.unique(p).tolist())) for p in panels]
    else:
        return None
    c = Counter(keys)
    odd = [k for k in range(len(panels)) if c[keys[k]] == 1]
    if len(odd) == 1:
        return odd[0]
    return None


def _special_color_panel(panels, gbg):
    painted = [k for k, p in enumerate(panels) if np.any(p != gbg)]
    if len(painted) == 1:
        return painted[0]
    return None


def _rule_builders():
    rb = []
    for mode in ("exact", "mask", "colorset"):
        rb.append((f"odd_{mode}", (lambda ps, gbg, m=mode: _odd_one_out(ps, m, gbg))))
    rb.append(("special_color", (lambda ps, gbg: _special_color_panel(ps, gbg))))
    for sc in _SCORERS:
        rb.append((f"max_{sc.__name__}",
                   (lambda ps, gbg, sc=sc: _pick_extreme(ps, sc(ps, gbg), True))))
        rb.append((f"min_{sc.__name__}",
                   (lambda ps, gbg, sc=sc: _pick_extreme(ps, sc(ps, gbg), False))))
        rb.append((f"uniq_{sc.__name__}",
                   (lambda ps, gbg, sc=sc: _pick_unique(ps, sc(ps, gbg)))))
    return rb


# ---------------------------------------------------------------- detector core
def _learn(train):
    if len(train) < 2:
        return None
    gbg = _global_bg(train)
    strategies = _candidate_strategies(train)
    rules = _rule_builders()

    # A strategy is viable only if, for the first pair, the output equals one of
    # its panels (so selection is even possible).
    i0, o0 = train[0]
    for sname, strat in strategies:
        try:
            p0 = strat(i0)
        except Exception:
            p0 = None
        if not p0:
            continue
        if not any(p.shape == o0.shape and np.array_equal(p, o0) for p in p0):
            continue
        for rname, rule in rules:
            def fn(g, strat=strat, rule=rule, gbg=gbg):
                try:
                    panels = strat(g)
                    if not panels:
                        return None
                    idx = rule(panels, gbg)
                    if idx is None or idx < 0 or idx >= len(panels):
                        return None
                    return np.asarray(panels[idx], dtype=int).copy()
                except Exception:
                    return None
            ok = True
            for i, o in train:
                try:
                    r = fn(i)
                except Exception:
                    r = None
                if r is None or r.shape != o.shape or not np.array_equal(r, o):
                    ok = False
                    break
            if ok:
                return fn
    return None


def panel_select(train):
    try:
        return _learn(train)
    except Exception:
        return None


# ---------------------------------------------------------------- de-tile
def _min_period(g):
    """Return (ph, pw) of the smallest tile that, repeated, reconstructs g
    exactly (an integer number of times in each axis).  None if the whole grid
    is the only period (no repetition)."""
    H, W = g.shape
    ph = None
    for p in range(1, H + 1):
        if H % p:
            continue
        block = g[:p, :]
        if all(np.array_equal(g[r:r + p, :], block) for r in range(0, H, p)):
            ph = p
            break
    pw = None
    for p in range(1, W + 1):
        if W % p:
            continue
        block = g[:, :p]
        if all(np.array_equal(g[:, c:c + p], block) for c in range(0, W, p)):
            pw = p
            break
    if ph is None or pw is None:
        return None
    if ph == H and pw == W:
        return None
    return ph, pw


def detile(train):
    """Input is a single tile repeated an integer number of times along one or
    both axes; output is exactly one copy of the tile."""
    if len(train) < 2:
        return None
    try:
        # verify: for each pair, the minimal period equals the output.
        for i, o in train:
            per = _min_period(i)
            if per is None:
                return None
            ph, pw = per
            if (ph, pw) != o.shape:
                return None
            if not np.array_equal(i[:ph, :pw], o):
                return None

        def fn(g):
            per = _min_period(g)
            if per is None:
                return None
            ph, pw = per
            return g[:ph, :pw].copy()
        return fn
    except Exception:
        return None


DETECTORS = [panel_select, detile]
