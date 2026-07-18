"""Round-2 batch-3 detectors for ARC-AGI evaluation tasks.

Each detector: def det(train) -> transform_fn | None
The engine verifies the returned transform reproduces every train pair exactly
before it is used, so detectors infer general rules and stay defensive.
"""
import numpy as np
from collections import Counter


# ---------------- helpers ----------------
def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(g, col, diag=False):
    """Connected cells equal to `col`."""
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] != col:
                continue
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            out.append(cells)
    return out


def _colors(g, bg):
    return [int(c) for c in np.unique(g) if int(c) != bg]


# ---------------- inverted fractal (8e2edd66) ----------------
def inverted_fractal(train):
    """Output is (H*H, W*W). Where input cell == bg, stamp the *inverted*
    pattern (bg <-> fg swapped) into that block; elsewhere leave blank.
    bg is whichever of the two colors makes the rule reproduce train."""
    i0, o0 = train[0]
    H, W = i0.shape
    if o0.shape != (H * H, W * W):
        return None
    if H > 6 or W > 6:
        return None
    if len(np.unique(i0)) != 2:
        return None

    def build(g, bg):
        vals = [int(c) for c in np.unique(g) if int(c) != bg]
        if len(vals) != 1:
            return None
        fg = vals[0]
        h, w = g.shape
        inv = np.where(g == bg, fg, bg).astype(int)
        out = np.full((h * h, w * w), bg, dtype=int)
        for r in range(h):
            for c in range(w):
                if g[r, c] == bg:
                    out[r * h:(r + 1) * h, c * w:(c + 1) * w] = inv
        return out

    # pick a fixed bg convention from the demos: prefer 0
    for bg_pref in ([0] if (i0 == 0).any() else []) + [None]:
        def fn(g, bg_pref=bg_pref):
            if bg_pref is not None:
                bg = bg_pref
            else:
                bg = _bg(g)
            return build(g, bg)
        try:
            if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
                return fn
        except Exception:
            continue
    return None


# ---------------- scale by distinct-color count (d4b1c2b1) ----------------
def scale_by_ncolors(train):
    """Output = kron(input, ones(k,k)) where k = number of distinct colors."""
    for i, o in train:
        k = len(np.unique(i))
        if o.shape != (i.shape[0] * k, i.shape[1] * k):
            return None

    def fn(g):
        k = len(np.unique(g))
        if k < 1 or k > 12:
            return None
        return np.kron(g, np.ones((k, k), dtype=int))
    return fn


# ---------------- extract shape, tile by marker count (4852f2fa) ----------------
def shape_tile_by_markers(train):
    """One color forms a compact blob (shape); another appears as isolated
    single cells (markers). Pad the shape bbox to a fixed HxW anchored
    bottom-left, then tile it horizontally `#markers` times."""
    outs_h = set(o.shape[0] for _, o in train)
    if len(outs_h) != 1:
        return None

    def analyze(g):
        bg = _bg(g)
        cols = _colors(g, bg)
        if len(cols) != 2:
            return None
        shape_col = mark_col = None
        for c in cols:
            comps = _components(g, c)
            if any(len(x) > 1 for x in comps):
                if shape_col is not None:
                    return None
                shape_col = c
            else:
                mark_col = c
        if shape_col is None or mark_col is None:
            return None
        return bg, shape_col, mark_col

    # learn target box shape from train (max shape bbox over demos)
    box_h = box_w = 0
    for i, o in train:
        a = analyze(i)
        if a is None:
            return None
        bg, sc, mc = a
        ys, xs = np.where(i == sc)
        box_h = max(box_h, ys.max() - ys.min() + 1)
        box_w = max(box_w, xs.max() - xs.min() + 1)
        # output width must equal box_w * (#marks)
        nm = int((i == mc).sum())
        if nm < 1 or o.shape[1] % box_w != 0:
            pass
    if box_h < 1 or box_w < 1:
        return None

    def fn(g):
        a = analyze(g)
        if a is None:
            return None
        bg, sc, mc = a
        ys, xs = np.where(g == sc)
        sub = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        h, w = sub.shape
        if h > box_h or w > box_w:
            return None
        padded = np.full((box_h, box_w), bg, dtype=int)
        padded[box_h - h:box_h, 0:w] = sub  # anchor bottom-left
        nm = int((g == mc).sum())
        if nm < 1:
            return None
        return np.hstack([padded] * nm)
    return fn


# ---------------- panel comparison -> solid block (8597cfd7) ----------------
def panel_diff_winner(train):
    """A separator (full row or col of one color) splits the grid into two
    panels A and B. For each other color, score = count(B) - count(A).
    Output is a solid block (of the train output size) in the winning color."""
    out_shape = None
    for _, o in train:
        if out_shape is None:
            out_shape = o.shape
        elif out_shape != o.shape:
            out_shape = None
            break
    # collect output must be solid single color
    for _, o in train:
        if len(np.unique(o)) != 1:
            return None

    def split(g):
        H, W = g.shape
        bg = _bg(g)
        # look for a full row that is a single NON-bg color -> horizontal split
        for r in range(H):
            if len(set(g[r].tolist())) == 1 and int(g[r, 0]) != bg:
                sep = int(g[r, 0])
                A = g[:r]
                B = g[r + 1:]
                if A.size and B.size:
                    return A, B, sep
        for c in range(W):
            if len(set(g[:, c].tolist())) == 1 and int(g[0, c]) != bg:
                sep = int(g[0, c])
                A = g[:, :c]
                B = g[:, c + 1:]
                if A.size and B.size:
                    return A, B, sep
        return None

    def winner(g):
        s = split(g)
        if s is None:
            return None
        A, B, sep = s
        bg = _bg(g)
        cand = set(np.unique(g).tolist()) - {bg, sep}
        if not cand:
            return None
        best = None
        best_score = None
        for col in sorted(cand):
            score = int((B == col).sum()) - int((A == col).sum())
            if best_score is None or score > best_score:
                best_score = score
                best = col
        return best

    def fn(g):
        w = winner(g)
        if w is None:
            return None
        if out_shape is not None:
            return np.full(out_shape, w, dtype=int)
        return np.array([[w]], dtype=int)
    return fn


# ---------------- 2x2 block to corners (66e6c45b) ----------------
def block_to_corners(train):
    """The non-bg content forms a 2x2 block; each of its 4 cells moves to the
    nearest corner of the grid; everything else becomes bg."""
    for i, o in train:
        if i.shape != o.shape:
            return None

    def fn(g):
        bg = _bg(g)
        mask = g != bg
        if not mask.any():
            return g.copy()
        ys, xs = np.where(mask)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        if r1 - r0 != 1 or c1 - c0 != 1:
            return None
        H, W = g.shape
        out = np.full_like(g, bg)
        # 4 cells of the 2x2 -> 4 corners
        out[0, 0] = g[r0, c0]
        out[0, W - 1] = g[r0, c1]
        out[H - 1, 0] = g[r1, c0]
        out[H - 1, W - 1] = g[r1, c1]
        return out
    return fn


# ---------------- L-ray from each dot to nearest corner (705a3229) ----------------
def corner_l_rays(train):
    """Each isolated colored dot draws two rays: vertical toward the nearer
    horizontal edge (up if in top half else down) and horizontal toward the
    nearer vertical edge (left if in left half else right)."""
    for i, o in train:
        if i.shape != o.shape:
            return None

    def fn(g):
        bg = _bg(g)
        H, W = g.shape
        out = g.copy()
        ys, xs = np.where(g != bg)
        if len(ys) == 0:
            return None
        for y, x in zip(ys, xs):
            col = int(g[y, x])
            if y < H / 2.0:
                out[0:y + 1, x] = col
            else:
                out[y:H, x] = col
            if x < W / 2.0:
                out[y, 0:x + 1] = col
            else:
                out[y, x:W] = col
        return out
    return fn


# ---------------- vertical code writer (e7b06bea) ----------------
def code_bar_writer(train):
    """A vertical bar of one 'ruler' color (length L) sits at one edge. Several
    single-color columns encode a sequence of colors (read left->right). Write
    that sequence down a column just left of the code, each color repeated L
    times, cycling to fill all rows. Preserve the ruler bar."""
    def analyze(g):
        H, W = g.shape
        bg = _bg(g)
        # ruler color = a color forming a single vertical run at col 0 (or last)
        # find color appearing in exactly one column, contiguous, at an edge
        counts = Counter(g[g != bg].tolist())
        if not counts:
            return None
        # code columns: columns with exactly one non-bg color, contiguous full
        codecols = []
        colcolor = {}
        for c in range(W):
            vals = set(g[:, c].tolist()) - {bg}
            if len(vals) == 1:
                colcolor[c] = list(vals)[0]
        # ruler = the color that forms the bar; find via largest contiguous single-color column at an edge
        # detect ruler as color occupying col 0 fully-ish
        ruler_col = 0
        ruler_vals = set(g[:, 0].tolist()) - {bg}
        if len(ruler_vals) != 1:
            # try last column
            ruler_col = W - 1
            ruler_vals = set(g[:, W - 1].tolist()) - {bg}
            if len(ruler_vals) != 1:
                return None
        ruler = list(ruler_vals)[0]
        L = int((g[:, ruler_col] == ruler).sum())
        if L < 1:
            return None
        # code columns = single-color columns whose color != ruler and != bg
        code = []
        for c in sorted(colcolor):
            if c == ruler_col:
                continue
            if colcolor[c] == ruler:
                continue
            code.append((c, colcolor[c]))
        if not code:
            return None
        seq = []
        for _, v in code:
            seq += [v] * L
        if not seq:
            return None
        if ruler_col == 0:
            ocol = min(c for c, _ in code) - 1
        else:
            ocol = max(c for c, _ in code) + 1
        if not (0 <= ocol < W):
            return None
        return bg, ruler, ruler_col, L, seq, ocol

    def fn(g):
        a = analyze(g)
        if a is None:
            return None
        bg, ruler, ruler_col, L, seq, ocol = a
        H, W = g.shape
        out = np.full_like(g, bg)
        out[:, ruler_col] = np.where(g[:, ruler_col] == ruler, ruler, bg)
        for r in range(H):
            out[r, ocol] = seq[r % len(seq)]
        return out
    return fn


# ---------------- ruler extrude (9c1e755f) ----------------
def _components_nonbg(g, bg):
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            out.append(cells)
    return out


def ruler_extrude(train):
    """Each L-shaped object has a straight 1-wide 'ruler' arm of one color plus
    a small perpendicular 'pattern' block at one end. Tile the pattern along
    the ruler's extent, filling the rectangle spanned by the ruler length."""
    for i, o in train:
        if i.shape != o.shape:
            return None

    def fn(g):
        bg = _bg(g)
        H, W = g.shape
        out = g.copy()
        touched = False
        for cells in _components_nonbg(g, bg):
            if len(cells) < 2:
                continue
            # find a color whose cells are collinear (all same col or same row)
            bycolor = {}
            for y, x in cells:
                bycolor.setdefault(int(g[y, x]), []).append((y, x))
            ruler_col_color = None
            for col, pts in bycolor.items():
                xs = set(x for _, x in pts)
                ys = set(y for y, _ in pts)
                if len(pts) >= 2 and (len(xs) == 1 or len(ys) == 1):
                    # prefer the longest straight line as ruler
                    if ruler_col_color is None or len(pts) > len(bycolor[ruler_col_color]):
                        ruler_col_color = col
            if ruler_col_color is None:
                continue
            rpts = bycolor[ruler_col_color]
            rxs = set(x for _, x in rpts)
            rys = set(y for y, _ in rpts)
            pat = [(y, x) for y, x in cells if int(g[y, x]) != ruler_col_color]
            if not pat:
                continue
            if len(rxs) == 1:  # vertical ruler
                col5 = next(iter(rxs))
                r0 = min(y for y, _ in rpts)
                r1 = max(y for y, _ in rpts)
                pc = [x for _, x in pat]
                pr = [y for y, _ in pat]
                c0, c1 = min(pc), max(pc)
                pr0, pr1 = min(pr), max(pr)
                block = g[pr0:pr1 + 1, c0:c1 + 1]
                bh = pr1 - pr0 + 1
                if bh < 1:
                    continue
                for idx, r in enumerate(range(r0, r1 + 1)):
                    out[r, c0:c1 + 1] = block[idx % bh]
                touched = True
            elif len(rys) == 1:  # horizontal ruler
                row5 = next(iter(rys))
                c0 = min(x for _, x in rpts)
                c1 = max(x for _, x in rpts)
                pc = [x for _, x in pat]
                pr = [y for y, _ in pat]
                r0, r1 = min(pr), max(pr)
                pc0, pc1 = min(pc), max(pc)
                block = g[r0:r1 + 1, pc0:pc1 + 1]
                bw = pc1 - pc0 + 1
                if bw < 1:
                    continue
                for idx, c in enumerate(range(c0, c1 + 1)):
                    out[r0:r1 + 1, c] = block[:, idx % bw]
                touched = True
        if not touched:
            return None
        return out
    return fn


DETECTORS = [
    inverted_fractal,
    scale_by_ncolors,
    shape_tile_by_markers,
    panel_diff_winner,
    block_to_corners,
    corner_l_rays,
    code_bar_writer,
    ruler_extrude,
]
