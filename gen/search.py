"""Compositional program search.

Instead of one detector per task-family, search over SEQUENCES of parameter-free
primitives (depth <=3) that map every train input to its output, with an optional
learned color-remap at the end. This is the generalization lever: it composes
primitives to hit multi-step tasks no single family covers. Iterative-deepening
DFS with a per-task node budget and state-dedup; engine still verifies.
"""
import numpy as np

def _bg(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[np.argmax(c)])

def _crop(g):
    bg = _bg(g); m = g != bg
    if not m.any(): return None
    r, c = np.where(m)
    return g[r.min():r.max()+1, c.min():c.max()+1].copy()

def _largest_obj_crop(g):
    bg = _bg(g); H, W = g.shape
    seen = np.zeros_like(g, bool); best = None; bestn = 0
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg: continue
            st = [(r, c)]; seen[r, c] = True; cells = []
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True; st.append((ny, nx))
            if len(cells) > bestn:
                bestn = len(cells); best = cells
    if not best: return None
    ys = [y for y, x in best]; xs = [x for y, x in best]
    return g[min(ys):max(ys)+1, min(xs):max(xs)+1].copy()

def _half(which):
    def f(g):
        H, W = g.shape
        if which == "L": return g[:, :W//2].copy() if W >= 2 else None
        if which == "R": return g[:, (W+1)//2:].copy() if W >= 2 else None
        if which == "T": return g[:H//2, :].copy() if H >= 2 else None
        if which == "B": return g[(H+1)//2:, :].copy() if H >= 2 else None
    return f

def _strip_border(g):
    if g.shape[0] < 3 or g.shape[1] < 3: return None
    return g[1:-1, 1:-1].copy()

def _tile2h(g): return np.hstack([g, g])
def _tile2v(g): return np.vstack([g, g])
def _scale2(g): return np.kron(g, np.ones((2, 2), dtype=int))

PRIMS = {
    "fliplr": np.fliplr, "flipud": np.flipud,
    "rot90": lambda g: np.rot90(g, 1), "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3), "transpose": lambda g: g.T.copy(),
    "crop": _crop, "lobj": _largest_obj_crop,
    "L": _half("L"), "R": _half("R"), "T": _half("T"), "B": _half("B"),
    "strip": _strip_border, "tile2h": _tile2h, "tile2v": _tile2v, "scale2": _scale2,
}

def _valid(g):
    return g is not None and g.ndim == 2 and g.size > 0 and g.shape[0] <= 60 and g.shape[1] <= 60

def _learn_cmap(pairs):
    mp = {}
    for a, b in pairs:
        if a.shape != b.shape: return None
        for x, y in zip(a.flatten(), b.flatten()):
            x, y = int(x), int(y)
            if x in mp and mp[x] != y: return None
            mp[x] = y
    return mp

def compose_search(train, max_depth=3, node_budget=4000):
    inputs = [i for i, _ in train]
    outputs = [o for _, o in train]

    def leaf_ok(state):
        if any(s.shape != o.shape for s, o in zip(state, outputs)):
            return None
        mp = _learn_cmap(list(zip(state, outputs)))
        if mp is None:
            return None
        return mp

    # depth-0 (pure colormap / identity) is covered by base detectors; start depth>=1
    best = None
    nodes = [0]
    seen = set()

    def sig(state):
        return tuple((s.shape, s.tobytes()) for s in state)

    def dfs(state, seq, depth):
        if best is not None or nodes[0] > node_budget:
            return
        nodes[0] += 1
        mp = leaf_ok(state)
        if mp is not None:
            _finish(seq, mp)
            return
        if depth >= max_depth:
            return
        for name, fn in PRIMS.items():
            try:
                ns = [fn(s) for s in state]
            except Exception:
                continue
            if any(not _valid(s) for s in ns):
                continue
            k = sig(tuple(ns))
            if k in seen:
                continue
            seen.add(k)
            dfs(tuple(ns), seq + [name], depth + 1)
            if best is not None:
                return

    def _finish(seq, mp):
        nonlocal best
        def fn(g, seq=list(seq), mp=dict(mp)):
            cur = g
            for name in seq:
                cur = PRIMS[name](cur)
                if not _valid(cur):
                    return None
            out = cur.copy()
            for a, b in mp.items():
                out[cur == a] = b
            return out
        # verify on train
        try:
            if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
                best = fn
        except Exception:
            pass

    dfs(tuple(inputs), [], 0)
    return best

DETECTORS = [compose_search]
