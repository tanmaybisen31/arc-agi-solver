"""Panel overlay / merge detectors for ARC-AGI.

Family: the input splits into N>=2 equal panels (side-by-side or stacked,
optionally separated by uniform-color divider lines). The output is a single
panel obtained by combining the panels cell-by-cell. Supported combination
rules learned from train:

  * boolean / count over "is-foreground" masks (AND, OR, XOR, NOR, exactly-k,
    at-least-k, ...) -> paint a fixed color where the rule holds, another color
    (usually background) elsewhere.
  * first-non-background wins (union overlay), optionally with a fixed
    geometric transform applied to each panel before overlaying.
  * per-cell majority vote of the raw panel values.
  * collapse of identical panels (all panels equal -> return one panel).

Everything is verified by the engine against every training pair before use,
so the detectors only need to *propose* a rule; wrong proposals are rejected.
"""
import numpy as np
from itertools import product

# ---------------------------------------------------------------- helpers

def _bg_global(train):
    """Guess the background color for the whole task (most common over inputs),
    with 0 strongly preferred when present."""
    from collections import Counter
    c = Counter()
    for i, _ in train:
        for v, n in zip(*np.unique(i, return_counts=True)):
            c[int(v)] += int(n)
    if not c:
        return 0
    if 0 in c:
        # prefer 0 unless another color is overwhelmingly more common AND 0 rare
        return 0
    return c.most_common(1)[0][0]


def _uniform_line(vec):
    return len(set(vec.tolist())) == 1


def _panel_splits(g, min_panels=2, max_panels=6):
    """Yield (panels, kind, meta) candidate decompositions of g into >=2 equal
    panels. kind in {'v','h'}; meta describes separators.

    Handles:
      - exact k-way division with no separator
      - k panels of equal width/height separated by single uniform divider lines
    """
    H, W = g.shape
    out = []

    # vertical (columns) splits
    for k in range(min_panels, max_panels + 1):
        # no separators
        if W % k == 0 and W // k >= 1:
            pw = W // k
            panels = [g[:, j * pw:(j + 1) * pw] for j in range(k)]
            out.append((panels, 'v', ('nosep', k, pw)))
        # single-column separators between panels: k*pw + (k-1) = W
        if (W - (k - 1)) > 0 and (W - (k - 1)) % k == 0:
            pw = (W - (k - 1)) // k
            if pw >= 1:
                panels = []
                seps = []
                idx = 0
                ok = True
                for j in range(k):
                    panels.append(g[:, idx:idx + pw])
                    idx += pw
                    if j < k - 1:
                        if idx >= W:
                            ok = False
                            break
                        seps.append(g[:, idx])
                        idx += 1
                if ok and all(_uniform_line(s) for s in seps):
                    # all separator lines same single color
                    sepcols = {int(s[0]) for s in seps}
                    if len(sepcols) == 1:
                        out.append((panels, 'v', ('sep', k, pw, list(sepcols)[0])))

    # horizontal (rows) splits
    for k in range(min_panels, max_panels + 1):
        if H % k == 0 and H // k >= 1:
            ph = H // k
            panels = [g[j * ph:(j + 1) * ph, :] for j in range(k)]
            out.append((panels, 'h', ('nosep', k, ph)))
        if (H - (k - 1)) > 0 and (H - (k - 1)) % k == 0:
            ph = (H - (k - 1)) // k
            if ph >= 1:
                panels = []
                seps = []
                idx = 0
                ok = True
                for j in range(k):
                    panels.append(g[idx:idx + ph, :])
                    idx += ph
                    if j < k - 1:
                        if idx >= H:
                            ok = False
                            break
                        seps.append(g[idx, :])
                        idx += 1
                if ok and all(_uniform_line(s) for s in seps):
                    sepcols = {int(s[0]) for s in seps}
                    if len(sepcols) == 1:
                        out.append((panels, 'h', ('sep', k, ph, list(sepcols)[0])))
    return out


def _make_splitter(kind, meta):
    """Return a function g -> list[panels] | None reproducing a specific split
    described by (kind, meta), so it generalizes to the test input."""
    def splitter(g):
        H, W = g.shape
        if kind == 'v':
            if meta[0] == 'nosep':
                _, k, pw = meta
                if W != k * pw:
                    # allow proportional generalization only if divisible
                    if W % k != 0:
                        return None
                    pw2 = W // k
                    return [g[:, j * pw2:(j + 1) * pw2] for j in range(k)]
                return [g[:, j * pw:(j + 1) * pw] for j in range(k)]
            else:
                _, k, pw, sepcol = meta
                if W != k * pw + (k - 1):
                    if (W - (k - 1)) % k != 0:
                        return None
                    pw = (W - (k - 1)) // k
                panels = []
                idx = 0
                for j in range(k):
                    panels.append(g[:, idx:idx + pw])
                    idx += pw
                    if j < k - 1:
                        idx += 1
                return panels
        else:
            if meta[0] == 'nosep':
                _, k, ph = meta
                if H != k * ph:
                    if H % k != 0:
                        return None
                    ph2 = H // k
                    return [g[j * ph2:(j + 1) * ph2, :] for j in range(k)]
                return [g[j * ph:(j + 1) * ph, :] for j in range(k)]
            else:
                _, k, ph, sepcol = meta
                if H != k * ph + (k - 1):
                    if (H - (k - 1)) % k != 0:
                        return None
                    ph = (H - (k - 1)) // k
                panels = []
                idx = 0
                for j in range(k):
                    panels.append(g[idx:idx + ph, :])
                    idx += ph
                    if j < k - 1:
                        idx += 1
                return panels
    return splitter


def _candidate_splitters(train):
    """Find split descriptors (kind, meta) that work for EVERY train pair such
    that each panel has the same shape as the corresponding output."""
    i0, o0 = train[0]
    descs = []
    for panels, kind, meta in _panel_splits(i0):
        if len(panels) < 2:
            continue
        if panels[0].shape != o0.shape:
            continue
        if any(p.shape != panels[0].shape for p in panels):
            continue
        splitter = _make_splitter(kind, meta)
        ok = True
        for i, o in train:
            ps = None
            try:
                ps = splitter(i)
            except Exception:
                ps = None
            if not ps or len(ps) < 2:
                ok = False
                break
            if any(p.shape != o.shape for p in ps):
                ok = False
                break
        if ok:
            descs.append((kind, meta, splitter))
    return descs


_GEO_OPS = {
    'id': lambda a: a,
    'fliplr': np.fliplr,
    'flipud': np.flipud,
    'rot180': lambda a: np.rot90(a, 2),
    'transpose': lambda a: a.T,
    'rot90': lambda a: np.rot90(a, 1),
    'rot270': lambda a: np.rot90(a, 3),
}


# ---------------------------------------------------------------- detectors

def panel_boolean_combine(train):
    """Cellwise boolean / count rule over per-panel foreground masks.

    A cell in the output is painted `on` if the number of panels that are
    foreground (non-bg) at that cell satisfies a learned predicate, else `off`.
    Predicates cover the classic logical ops for any panel count.
    """
    try:
        descs = _candidate_splitters(train)
        if not descs:
            return None
        bg = _bg_global(train)

        # collect the set of output colors to pick on/off from
        out_colors = set()
        for _, o in train:
            out_colors |= set(int(x) for x in np.unique(o))
        out_colors = sorted(out_colors)
        if not out_colors:
            return None

        for kind, meta, splitter in descs:
            k = meta[1]
            # candidate count-predicates over number of foreground panels (0..k)
            preds = {}
            preds['all'] = lambda c, k=k: c == k              # AND
            preds['any'] = lambda c, k=k: c >= 1              # OR
            preds['none'] = lambda c, k=k: c == 0             # NOR
            preds['notall'] = lambda c, k=k: c < k            # NAND
            preds['odd'] = lambda c, k=k: (c % 2) == 1        # XOR (generalized)
            preds['even_ge1'] = lambda c, k=k: (c % 2) == 0 and c >= 1
            for t in range(0, k + 1):
                preds[f'eq{t}'] = (lambda c, t=t: c == t)
                preds[f'ge{t}'] = (lambda c, t=t: c >= t)
                preds[f'le{t}'] = (lambda c, t=t: c <= t)

            def counts_for(g):
                ps = splitter(g)
                if not ps:
                    return None
                masks = [(p != bg) for p in ps]
                cnt = np.zeros(masks[0].shape, dtype=int)
                for m in masks:
                    cnt = cnt + m.astype(int)
                return cnt

            # precompute counts for all train
            train_counts = []
            good = True
            for i, o in train:
                c = counts_for(i)
                if c is None or c.shape != o.shape:
                    good = False
                    break
                train_counts.append(c)
            if not good:
                continue

            for pname, pred in preds.items():
                for on in out_colors:
                    # off default = bg, but also try each out color
                    off_candidates = [bg] + [c for c in out_colors if c != on]
                    seen_off = set()
                    for off in off_candidates:
                        if off in seen_off:
                            continue
                        seen_off.add(off)
                        if on == off:
                            continue
                        ok = True
                        for (i, o), c in zip(train, train_counts):
                            res = pred(c)
                            pred_out = np.where(res, on, off).astype(int)
                            if not np.array_equal(pred_out, o):
                                ok = False
                                break
                        if ok:
                            def fn(g, splitter=splitter, bg=bg, pred=pred,
                                   on=on, off=off):
                                ps = splitter(g)
                                if not ps:
                                    return None
                                masks = [(p != bg) for p in ps]
                                cnt = np.zeros(masks[0].shape, dtype=int)
                                for m in masks:
                                    cnt = cnt + m.astype(int)
                                res = pred(cnt)
                                return np.where(res, on, off).astype(int)
                            return fn
        return None
    except Exception:
        return None


def panel_overlay_first(train):
    """Overlay panels: first (in panel order) non-background value wins per cell.

    Also tries applying a single geometric op to every panel, and reversing
    panel order, since some tasks mirror one half onto the other.
    """
    try:
        descs = _candidate_splitters(train)
        if not descs:
            return None
        bg = _bg_global(train)

        for kind, meta, splitter in descs:
            # panel-order variants and geometric variants
            order_variants = [False, True]  # reverse order?
            for rev in order_variants:
                for gname, gop in _GEO_OPS.items():
                    def build(splitter=splitter, bg=bg, rev=rev, gop=gop):
                        def fn(g):
                            ps = splitter(g)
                            if not ps:
                                return None
                            ps = [np.asarray(gop(p)) for p in ps]
                            base_shape = ps[0].shape
                            if any(p.shape != base_shape for p in ps):
                                return None
                            order = list(range(len(ps)))
                            if rev:
                                order = order[::-1]
                            out = np.full(base_shape, bg, dtype=int)
                            filled = np.zeros(base_shape, dtype=bool)
                            for idx in order:
                                p = ps[idx]
                                m = (p != bg) & (~filled)
                                out[m] = p[m]
                                filled |= (p != bg)
                            return out
                        return fn
                    fn = build()
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
    except Exception:
        return None


def panel_overlay_last(train):
    """Overlay panels: last non-background value wins per cell (later panel
    overwrites earlier). Geometric variants included."""
    try:
        descs = _candidate_splitters(train)
        if not descs:
            return None
        bg = _bg_global(train)
        for kind, meta, splitter in descs:
            for gname, gop in _GEO_OPS.items():
                def build(splitter=splitter, bg=bg, gop=gop):
                    def fn(g):
                        ps = splitter(g)
                        if not ps:
                            return None
                        ps = [np.asarray(gop(p)) for p in ps]
                        base_shape = ps[0].shape
                        if any(p.shape != base_shape for p in ps):
                            return None
                        out = np.full(base_shape, bg, dtype=int)
                        for p in ps:
                            m = (p != bg)
                            out[m] = p[m]
                        return out
                    return fn
                fn = build()
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
    except Exception:
        return None


def panel_majority(train):
    """Per-cell majority vote of the raw panel values (ties -> lowest color).
    Also handles the special case where all panels are identical."""
    try:
        descs = _candidate_splitters(train)
        if not descs:
            return None
        for kind, meta, splitter in descs:
            def fn(g, splitter=splitter):
                ps = splitter(g)
                if not ps:
                    return None
                base_shape = ps[0].shape
                if any(p.shape != base_shape for p in ps):
                    return None
                stack = np.stack(ps, axis=0)  # (k, H, W)
                kk, H, W = stack.shape
                out = np.zeros((H, W), dtype=int)
                for r in range(H):
                    for c in range(W):
                        vals = stack[:, r, c]
                        u, cnt = np.unique(vals, return_counts=True)
                        mx = cnt.max()
                        # tie -> smallest color among the winners
                        winners = u[cnt == mx]
                        out[r, c] = int(winners.min())
                return out
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
    except Exception:
        return None


def panel_pick_by_content(train):
    """Two-color mask combine where the OUTPUT color at a cell is taken from a
    fixed 'palette' panel: i.e. where the reference (mask) panels agree on
    foreground, paint with the color from a chosen panel at that cell.

    Concretely: try, for each panel index p_color as the color source and a
    boolean predicate over the *other* panels' masks, output = color_source
    value where predicate holds else bg. This captures tasks like 'keep the
    colored panel's value only where all panels are lit'."""
    try:
        descs = _candidate_splitters(train)
        if not descs:
            return None
        bg = _bg_global(train)
        for kind, meta, splitter in descs:
            k = meta[1]
            if k < 2:
                continue
            for src in range(k):
                # predicates over count of foreground among ALL panels
                preds = {
                    'all': (lambda c, k=k: c == k),
                    'any': (lambda c, k=k: c >= 1),
                }
                for pname, pred in preds.items():
                    def fn(g, splitter=splitter, bg=bg, src=src, pred=pred):
                        ps = splitter(g)
                        if not ps:
                            return None
                        base_shape = ps[0].shape
                        if any(p.shape != base_shape for p in ps):
                            return None
                        masks = [(p != bg) for p in ps]
                        cnt = np.zeros(base_shape, dtype=int)
                        for m in masks:
                            cnt += m.astype(int)
                        res = pred(cnt)
                        out = np.full(base_shape, bg, dtype=int)
                        out[res] = ps[src][res]
                        return out
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
    except Exception:
        return None


def panel_overlay_priority(train):
    """Overlay panels where, per cell, the output value is the non-background
    panel value with the highest priority in a learned total color order.

    Learns the order from constraints: at each cell, the chosen output color
    must outrank every other non-bg color present at that cell. Solved via a
    topological sort of the resulting precedence graph, then verified.
    """
    try:
        descs = _candidate_splitters(train)
        if not descs:
            return None
        bg = _bg_global(train)

        for kind, meta, splitter in descs:
            # gather precedence constraints: winner -> beats set of losers
            # and validate that output is always among present colors (or bg).
            beats = {}   # color -> set(colors it must outrank)
            colors = set()
            feasible = True
            for i, o in train:
                ps = splitter(i)
                if not ps:
                    feasible = False
                    break
                base_shape = ps[0].shape
                if base_shape != o.shape or any(p.shape != base_shape for p in ps):
                    feasible = False
                    break
                stack = np.stack(ps, axis=0)
                H, W = base_shape
                for r in range(H):
                    for c in range(W):
                        vals = [int(v) for v in stack[:, r, c] if int(v) != bg]
                        ov = int(o[r, c])
                        present = set(vals)
                        if ov == bg:
                            # all panels must be bg here
                            if present:
                                feasible = False
                                break
                            continue
                        if ov not in present:
                            feasible = False
                            break
                        colors |= present
                        beats.setdefault(ov, set())
                        for other in present:
                            if other != ov:
                                beats[ov].add(other)
                    if not feasible:
                        break
                if not feasible:
                    break
            if not feasible or not colors:
                continue

            # build a strict order via repeated selection: a color can be chosen
            # as current-highest if no *other still-unplaced* color must beat it.
            # i.e. we place winners first. color X is placed when nothing
            # unplaced needs to beat X... Instead: X outranks Y means X before Y.
            order = []
            remaining = set(colors)
            # adjacency: winner precedes loser
            precedes = {c: set() for c in colors}
            indeg = {c: 0 for c in colors}
            for w, losers in beats.items():
                for l in losers:
                    if l in colors and w in colors and l not in precedes[w]:
                        precedes[w].add(l)
            for w in colors:
                for l in precedes[w]:
                    indeg[l] += 1
            # Kahn's algorithm; pick smallest color on ties for determinism
            import heapq
            heap = [c for c in colors if indeg[c] == 0]
            heapq.heapify(heap)
            ok_topo = True
            while heap:
                c = heapq.heappop(heap)
                order.append(c)
                for l in precedes[c]:
                    indeg[l] -= 1
                    if indeg[l] == 0:
                        heapq.heappush(heap, l)
            if len(order) != len(colors):
                # cycle -> inconsistent priority
                continue
            rank = {c: idx for idx, c in enumerate(order)}  # lower = higher prio

            def fn(g, splitter=splitter, bg=bg, rank=dict(rank)):
                ps = splitter(g)
                if not ps:
                    return None
                base_shape = ps[0].shape
                if any(p.shape != base_shape for p in ps):
                    return None
                stack = np.stack(ps, axis=0)
                H, W = base_shape
                out = np.full(base_shape, bg, dtype=int)
                big = len(rank) + 10
                for r in range(H):
                    for c in range(W):
                        best = None
                        best_rank = big
                        for v in stack[:, r, c]:
                            v = int(v)
                            if v == bg:
                                continue
                            rk = rank.get(v, big - 1)
                            if rk < best_rank:
                                best_rank = rk
                                best = v
                        if best is not None:
                            out[r, c] = best
                return out

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
    except Exception:
        return None


def panel_overlay_geo_pair(train):
    """Two-panel overlay where each panel may first be transformed by its own
    geometric op (e.g. fold the right half onto the left). First non-background
    wins; both fill orders are tried."""
    try:
        descs = _candidate_splitters(train)
        if not descs:
            return None
        bg = _bg_global(train)
        for kind, meta, splitter in descs:
            if meta[1] != 2:
                continue

            def valid(fn):
                for i, o in train:
                    try:
                        r = fn(i)
                    except Exception:
                        r = None
                    if r is None or r.shape != o.shape or not np.array_equal(r, o):
                        return False
                return True

            for ganame, ga in _GEO_OPS.items():
                for gbname, gb in _GEO_OPS.items():
                    for a_first in (True, False):
                        def build(splitter=splitter, bg=bg, ga=ga, gb=gb,
                                  a_first=a_first):
                            def fn(g):
                                ps = splitter(g)
                                if not ps or len(ps) != 2:
                                    return None
                                A = np.asarray(ga(ps[0]))
                                B = np.asarray(gb(ps[1]))
                                if A.shape != B.shape:
                                    return None
                                out = np.full(A.shape, bg, dtype=int)
                                first, second = (A, B) if a_first else (B, A)
                                out[second != bg] = second[second != bg]
                                out[first != bg] = first[first != bg]
                                return out
                            return fn
                        fn = build()
                        if valid(fn):
                            return fn
        return None
    except Exception:
        return None


DETECTORS = [
    panel_boolean_combine,
    panel_overlay_first,
    panel_overlay_last,
    panel_overlay_priority,
    panel_overlay_geo_pair,
    panel_pick_by_content,
    panel_majority,
]
