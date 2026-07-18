"""Round-2 batch-1 detectors for ARC-AGI eval tasks.

Each detector: det(train) -> transform_fn | None. The engine verifies a returned
transform reproduces every train pair exactly before use, so detectors aim for a
*general* rule inferred from the demos rather than a hardcoded answer.

Import only numpy + stdlib. Every detector is defensive (guards + try/except).
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations


def _bg(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[np.argmax(c)])


# ------------------------------------------------------------------ a59b95c0
# Tile the grid k x k where k = number of distinct colors present.
def tile_by_distinct(train):
    def fn(g):
        k = len(np.unique(g))
        if k < 1 or k > 8:
            return None
        return np.tile(g, (k, k))
    # quick shape sanity
    try:
        for i, o in train:
            k = len(np.unique(i))
            if o.shape != (i.shape[0] * k, i.shape[1] * k):
                return None
    except Exception:
        return None
    return fn


# ------------------------------------------------------------------ 5af49b42
# One or more "key" runs (contiguous distinct nonzero cells, len>=2) sit on grid
# lines. Each isolated seed pixel carries a value belonging to exactly one key;
# stamp that whole key aligned so the value's index lands on the seed cell.
def _find_key_runs(g):
    H, W = g.shape
    keys = []
    for r in range(H):
        c = 0
        while c < W:
            if g[r, c] != 0:
                c2 = c
                while c2 < W and g[r, c2] != 0:
                    c2 += 1
                seq = g[r, c:c2].tolist()
                if len(seq) >= 2 and len(set(seq)) == len(seq):
                    keys.append((seq, 'h', r, c))
                c = c2
            else:
                c += 1
    for c in range(W):
        r = 0
        while r < H:
            if g[r, c] != 0:
                r2 = r
                while r2 < H and g[r2, c] != 0:
                    r2 += 1
                seq = g[r:r2, c].tolist()
                if len(seq) >= 2 and len(set(seq)) == len(seq):
                    keys.append((seq, 'v', c, r))
                r = r2
            else:
                r += 1
    return keys


def stamp_key_from_seed(train):
    def fn(g):
        keys = _find_key_runs(g)
        if not keys:
            return None
        val2keys = defaultdict(list)
        keycells = set()
        for seq, orient, line, start in keys:
            for v in set(seq):
                val2keys[v].append((seq, orient))
            for k in range(len(seq)):
                if orient == 'h':
                    keycells.add((line, start + k))
                else:
                    keycells.add((start + k, line))
        out = g.copy()
        H, W = g.shape
        for r in range(H):
            for c in range(W):
                v = int(g[r, c])
                if v == 0 or (r, c) in keycells:
                    continue
                cand = val2keys.get(v)
                if not cand or len(cand) != 1:
                    return None
                seq, orient = cand[0]
                idx = seq.index(v)
                for k, kv in enumerate(seq):
                    if orient == 'h':
                        cc = c - idx + k
                        if 0 <= cc < W:
                            out[r, cc] = kv
                    else:
                        rr = r - idx + k
                        if 0 <= rr < H:
                            out[rr, c] = kv
        return out
    return fn


# ------------------------------------------------------------------ 6f473927
# Single object (one non-zero color) hugging one vertical edge. Reflect+recolor
# (obj->0, bg 0->8) and place the mirror on the touched side, doubling width.
def mirror_complement_concat(train):
    def fn(g):
        cols = [c for c in np.unique(g) if c != 0]
        if len(cols) != 1:
            return None
        oc = int(cols[0])
        rec = g.copy()
        rec[g == 0] = 8
        rec[g == oc] = 0
        mir = np.fliplr(rec)
        lt = bool((g[:, 0] == oc).any())
        rt = bool((g[:, -1] == oc).any())
        if rt and not lt:
            return np.hstack([g, mir])
        if lt and not rt:
            return np.hstack([mir, g])
        return None
    return fn


# ------------------------------------------------------------------ 73182012
# A single symmetric object; output is the top-left quadrant of its bbox crop
# (output shape gives the quadrant size).
def crop_topleft_quadrant(train):
    try:
        oh, ow = train[0][1].shape
        for i, o in train:
            if o.shape != (oh, ow):
                return None
    except Exception:
        return None

    def fn(g):
        b = _bg(g)
        mask = g != b
        if not mask.any():
            return None
        rs, cs = np.where(mask)
        crop = g[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
        if crop.shape[0] < oh or crop.shape[1] < ow:
            return None
        return crop[:oh, :ow].copy()
    return fn


# ------------------------------------------------------------------ 695367ec
# Solid NxN color block -> 15x15 mesh: full lines + dots every (N+1) starting N.
def mesh_from_block(train):
    try:
        S0 = train[0][1].shape
        for i, o in train:
            if o.shape != S0 or o.shape[0] != o.shape[1]:
                return None
            if len(np.unique(i)) != 1 or i.shape[0] != i.shape[1]:
                return None
    except Exception:
        return None
    S = S0[0]

    def fn(g):
        vals = np.unique(g)
        if len(vals) != 1:
            return None
        C = int(vals[0])
        N = g.shape[0]
        if N < 1 or N + 1 > S:
            return None
        out = np.zeros((S, S), dtype=int)
        lines = list(range(N, S, N + 1))
        ls = set(lines)
        for r in range(S):
            if r in ls:
                out[r, :] = C
            else:
                for c in lines:
                    out[r, c] = C
        return out
    return fn


# ------------------------------------------------------------------ c074846d
# A pivot cell (5) with a straight arm of color 2. Recolor the arm to 3 and add
# a new arm of 2s of equal length rotated 90 deg clockwise about the pivot.
def rotate_arm_cw(train):
    def fn(g):
        g = g.copy()
        H, W = g.shape
        fives = np.argwhere(g == 5)
        if len(fives) != 1:
            return None
        pr, pc = int(fives[0][0]), int(fives[0][1])
        twos = np.argwhere(g == 2)
        if len(twos) == 0:
            return None
        rs = set(twos[:, 0].tolist())
        cs = set(twos[:, 1].tolist())
        if rs == {pr}:
            if all(c < pc for c in twos[:, 1]):
                d = (0, -1)
            elif all(c > pc for c in twos[:, 1]):
                d = (0, 1)
            else:
                return None
        elif cs == {pc}:
            if all(r < pr for r in twos[:, 0]):
                d = (-1, 0)
            elif all(r > pr for r in twos[:, 0]):
                d = (1, 0)
            else:
                return None
        else:
            return None
        L = len(twos)
        out = g.copy()
        for r, c in twos:
            out[r, c] = 3
        ndr, ndc = d[1], -d[0]
        for k in range(1, L + 1):
            rr, cc = pr + ndr * k, pc + ndc * k
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = 2
            else:
                return None
        return out
    return fn


# ------------------------------------------------------------------ 60a26a3e
# Diamonds = hollow plus-shapes of 2s. Connect aligned adjacent diamond pairs
# (same row or column, nothing between) with a line of 1s spanning their gap.
def connect_diamonds(train):
    def fn(g):
        H, W = g.shape
        centers = []
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if (g[r - 1, c] == 2 and g[r + 1, c] == 2 and
                        g[r, c - 1] == 2 and g[r, c + 1] == 2 and g[r, c] == 0):
                    centers.append((r, c))
        if len(centers) < 2:
            return g.copy()
        cset = set(centers)
        out = g.copy()
        for (r1, c1), (r2, c2) in combinations(centers, 2):
            if r1 == r2:
                a, b = sorted([c1, c2])
                if any((r1, c) in cset for c in range(a + 1, b)):
                    continue
                for c in range(a + 2, b - 1):
                    if out[r1, c] == 0:
                        out[r1, c] = 1
            elif c1 == c2:
                a, b = sorted([r1, r2])
                if any((r, c1) in cset for r in range(a + 1, b)):
                    continue
                for r in range(a + 2, b - 1):
                    if out[r, c1] == 0:
                        out[r, c1] = 1
        return out
    return fn


# ------------------------------------------------------------------ d47aa2ff
# Two equal panels split by a uniform separator line. Same markers, each shifted
# between panels. Overlay: same-in-both -> keep; first-panel-only -> 2;
# second-panel-only -> 1.
def two_panel_shift_overlay(train):
    def split(g):
        H, W = g.shape
        vseps = [c for c in range(W)
                 if len(set(g[:, c].tolist())) == 1 and g[0, c] != 0]
        hseps = [r for r in range(H)
                 if len(set(g[r, :].tolist())) == 1 and g[r, 0] != 0]
        if len(vseps) == 1:
            s = vseps[0]
            return g[:, :s], g[:, s + 1:]
        if len(hseps) == 1:
            s = hseps[0]
            return g[:s, :], g[s + 1:, :]
        return None

    def fn(g):
        sp = split(g)
        if sp is None:
            return None
        A, B = sp
        if A.shape != B.shape or A.size == 0:
            return None
        out = np.zeros_like(A)
        for r in range(A.shape[0]):
            for c in range(A.shape[1]):
                a, b = int(A[r, c]), int(B[r, c])
                if a != 0 and b != 0:
                    out[r, c] = a
                elif a != 0:
                    out[r, c] = 2
                elif b != 0:
                    out[r, c] = 1
        return out
    return fn


# ------------------------------------------------------------------ ca8de6ea
# 5x5 X-pattern (two diagonals) folded into a 3x3 diamond via fixed sampling.
_CA_SRC = [[(0, 0), (1, 1), (0, 4)],
           [(1, 3), (2, 2), (3, 1)],
           [(4, 0), (3, 3), (4, 4)]]


def fold_x_to_diamond(train):
    try:
        for i, o in train:
            if i.shape != (5, 5) or o.shape != (3, 3):
                return None
    except Exception:
        return None

    def fn(g):
        if g.shape != (5, 5):
            return None
        out = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                r, c = _CA_SRC[i][j]
                out[i, j] = g[r, c]
        return out
    return fn


DETECTORS = [
    tile_by_distinct,
    stamp_key_from_seed,
    mirror_complement_concat,
    crop_topleft_quadrant,
    mesh_from_block,
    rotate_arm_cw,
    connect_diamonds,
    two_panel_shift_overlay,
    fold_x_to_diamond,
]
