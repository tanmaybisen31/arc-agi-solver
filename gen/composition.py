"""Composition / 2-step search detectors.

Many ARC tasks are a shape/structure op composed with a color op. A single
primitive can't fit them, but a short composition can. The engine still verifies
exact reproduction of all train pairs, so wrong compositions auto-reject.
"""
import numpy as np

def _bg(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[np.argmax(c)])

def _crop(g):
    bg = _bg(g)
    m = g != bg
    if not m.any():
        return g.copy()
    rs, cs = np.where(m)
    return g[rs.min():rs.max()+1, cs.min():cs.max()+1].copy()

SHAPE_OPS = {
    "id": lambda g: g.copy(),
    "fliplr": np.fliplr, "flipud": np.flipud,
    "rot90": lambda g: np.rot90(g, 1), "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "transpose": lambda g: g.T.copy(),
    "antitranspose": lambda g: np.rot90(g, 2).T.copy(),
    "crop": _crop,
}

def _learn_colormap(pairs):
    """pairs = [(gridA, gridB)] same shape; return consistent color map or None."""
    mp = {}
    for a, b in pairs:
        if a.shape != b.shape:
            return None
        for x, y in zip(a.flatten(), b.flatten()):
            x, y = int(x), int(y)
            if x in mp and mp[x] != y:
                return None
            mp[x] = y
    return mp

def shape_then_recolor(train):
    for name, S in SHAPE_OPS.items():
        try:
            sins = [S(i) for i, _ in train]
        except Exception:
            continue
        if any(s.shape != o.shape for s, (_, o) in zip(sins, train)):
            continue
        mp = _learn_colormap([(s, o) for s, (_, o) in zip(sins, train)])
        if mp is None:
            continue
        def fn(g, S=S, mp=dict(mp)):
            s = S(g)
            out = s.copy()
            for a, b in mp.items():
                out[s == a] = b
            return out
        return fn
    return None

def recolor_then_shape(train):
    # learn colormap on same-shape inputs first (only if input shape == input shape trivially),
    # then a shape op. Learn map from input to a recolored input that, after S, equals output.
    for name, S in SHAPE_OPS.items():
        # need S(input).shape == output.shape
        try:
            if any(S(i).shape != o.shape for i, o in train):
                continue
        except Exception:
            continue
        # invert: we want map M s.t. S(M(input)) == output  => M(input) == Sinv(output).
        # Only handle shape ops that are their own easy inverse via applying S to output.
        # Simpler: learn map on (input, S^{-1}(output)) using S applied to output when S is an involution/rotation.
        inv = {"id": lambda g: g, "fliplr": np.fliplr, "flipud": np.flipud,
               "rot90": lambda g: np.rot90(g, 3), "rot180": lambda g: np.rot90(g, 2),
               "rot270": lambda g: np.rot90(g, 1), "transpose": lambda g: g.T.copy(),
               "antitranspose": lambda g: np.rot90(g, 2).T.copy()}.get(name)
        if inv is None:  # crop not invertible
            continue
        try:
            pairs = [(i, inv(o)) for i, o in train]
        except Exception:
            continue
        mp = _learn_colormap(pairs)
        if mp is None:
            continue
        def fn(g, S=S, mp=dict(mp)):
            out = g.copy()
            for a, b in mp.items():
                out[g == a] = b
            return S(out)
        return fn
    return None

def crop_then_geo(train):
    geos = {"id": lambda g: g.copy(), "fliplr": np.fliplr, "flipud": np.flipud,
            "rot90": lambda g: np.rot90(g, 1), "rot180": lambda g: np.rot90(g, 2),
            "rot270": lambda g: np.rot90(g, 3), "transpose": lambda g: g.T.copy()}
    for gname, G in geos.items():
        def fn(g, G=G):
            return G(_crop(g))
        try:
            if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
                return fn
        except Exception:
            continue
    return None

DETECTORS = [shape_then_recolor, recolor_then_shape, crop_then_geo]
