"""Round 4, batch 0 detectors for ARC-AGI.

Each detector: def det(train) -> transform_fn | None
  train = [(input_grid, output_grid), ...]  (numpy int arrays)
  transform_fn: grid -> grid   (engine verifies exact reproduction of all demos)

Keep detectors defensive & general. numpy + stdlib only.
"""
import numpy as np
from collections import Counter


def _bg(g):
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


# ---------------------------------------------------------------------------
# 1) Per-axis "stretch by duplicating rows/columns" (e.g. duplicate the first
#    and last row/col). We learn, from the first demo, a row-index sequence and
#    a col-index sequence that reproduce the output, then verify on all demos.
# ---------------------------------------------------------------------------
def _learn_index_seq(inp_len, out_len, inp_line_fn, out_line_fn):
    """Greedy: build a monotonic-ish index sequence of length out_len that maps
    input lines to output lines. Return list of input indices or None."""
    seq = []
    j = 0  # pointer into input lines
    for oi in range(out_len):
        oline = out_line_fn(oi)
        # try current input index, then advance
        matched = None
        # allow: same as current (duplicate), or next
        for cand in (j, j + 1):
            if 0 <= cand < inp_len and np.array_equal(inp_line_fn(cand), oline):
                matched = cand
                break
        if matched is None:
            return None
        seq.append(matched)
        j = matched
    if j != inp_len - 1:
        return None
    return seq


def stretch_dup_lines(train):
    i0, o0 = train[0]
    Hi, Wi = i0.shape
    Ho, Wo = o0.shape
    if Ho < Hi or Wo < Wi or Ho > 2 * Hi or Wo > 2 * Wi:
        return None
    row_seq = _learn_index_seq(Hi, Ho, lambda r: i0[r, :], lambda r: o0[r, :])
    # if row lines don't line up directly (because cols also change), fall back
    # to learning row_seq on a col-collapsed representation is unreliable; do a
    # combined learn: first learn col_seq from a row that we can match.
    # Simpler robust approach: learn both sequences purely from index-expansion
    # assuming out[r,c] = in[row_seq[r], col_seq[c]].
    def learn_axis(n_in, n_out, getcol_in, getcol_out):
        seq = []
        j = 0
        for oi in range(n_out):
            oc = getcol_out(oi)
            cand_found = None
            for cand in (j, j + 1):
                if 0 <= cand < n_in and np.array_equal(getcol_in(cand), oc):
                    cand_found = cand
                    break
            if cand_found is None:
                return None
            seq.append(cand_found)
            j = cand_found
        if j != n_in - 1:
            return None
        return seq

    # We need col_seq consistent; try learning col_seq using the full output
    # columns compared to input columns, but input columns have Hi rows while
    # output columns have Ho rows. So learn row_seq first (rows must match by
    # value across full width only if widths equal). Handle the general case:
    # learn row_seq by matching output rows to input rows AFTER we know col_seq,
    # and vice versa -> chicken/egg. Use this: if widths equal, learn row_seq
    # directly; if heights equal, learn col_seq directly. Then derive the other.
    row_seq = None
    col_seq = None
    if Wi == Wo:
        row_seq = learn_axis(Hi, Ho, lambda r: i0[r, :], lambda r: o0[r, :])
        col_seq = list(range(Wi))
    if Hi == Ho:
        col_seq = learn_axis(Wi, Wo, lambda c: i0[:, c], lambda c: o0[:, c])
        row_seq = list(range(Hi))
    if row_seq is None or col_seq is None:
        # both axes change: try to learn independently using single reference
        # line assumption (duplicate first & last on both axes).
        def dup_edges_seq(n_in, n_out):
            extra = n_out - n_in
            if extra < 0 or extra > 2:
                return None
            if extra == 0:
                return list(range(n_in))
            if extra == 1:
                # duplicate first OR last: two options handled by caller
                return None
            if extra == 2:
                return [0] + list(range(n_in)) + [n_in - 1]
        row_seq = dup_edges_seq(Hi, Ho)
        col_seq = dup_edges_seq(Wi, Wo)
        if row_seq is None or col_seq is None:
            return None

    def fn(g, row_seq=row_seq, col_seq=col_seq):
        H, W = g.shape
        rs = row_seq if len(row_seq) and max(row_seq) < H else list(range(H))
        cs = col_seq if len(col_seq) and max(col_seq) < W else list(range(W))
        # rebuild sequences relative to this grid: reproduce the "duplicate
        # first & last" pattern generically by using the learned offsets.
        return g[np.ix_(rs, cs)]

    try:
        if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 2) Symmetry completion: output has same shape as input; some background cells
#    are filled so the WHOLE grid (or a masked region) becomes symmetric under a
#    set of transforms. Occluded cells are recovered from their symmetric twins.
# ---------------------------------------------------------------------------
def _symmetry_fill(g, hole):
    """Fill cells equal to `hole` using self-symmetry (mirror/rotate/translate).
    Return filled grid (best effort). Uses only exact symmetries present in the
    non-hole cells."""
    H, W = g.shape
    out = g.copy()
    known = g != hole

    # candidate mirror symmetries about grid center
    syms = []
    if True:
        syms.append(lambda r, c: (r, W - 1 - c))          # fliplr
        syms.append(lambda r, c: (H - 1 - r, c))          # flipud
        syms.append(lambda r, c: (H - 1 - r, W - 1 - c))  # rot180
    if H == W:
        syms.append(lambda r, c: (c, r))                  # transpose
        syms.append(lambda r, c: (W - 1 - c, H - 1 - r))  # anti-transpose

    # keep only symmetries consistent with known cells
    good = []
    for s in syms:
        ok = True
        for r in range(H):
            for c in range(W):
                if not known[r, c]:
                    continue
                rr, cc = s(r, c)
                if 0 <= rr < H and 0 <= cc < W and known[rr, cc]:
                    if g[r, c] != g[rr, cc]:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            good.append(s)

    changed = True
    while changed:
        changed = False
        for r in range(H):
            for c in range(W):
                if known[r, c]:
                    continue
                for s in good:
                    rr, cc = s(r, c)
                    if 0 <= rr < H and 0 <= cc < W and known[rr, cc]:
                        out[r, c] = g[rr, cc]
                        known[r, c] = True
                        changed = True
                        break
    return out


def symmetry_fill_global(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # the hole color = a color present in input but whose cells become other
    # colors in output (added cells). Determine candidate hole colors.
    cand_holes = set()
    for i, o in train:
        diff = i != o
        if not diff.any():
            continue
        cand_holes |= set(int(i[r, c]) for r, c in np.argwhere(diff))
    if len(cand_holes) != 1:
        # multiple -> ambiguous; still try the most common changed-from color
        counter = Counter()
        for i, o in train:
            for r, c in np.argwhere(i != o):
                counter[int(i[r, c])] += 1
        if not counter:
            return None
        hole = counter.most_common(1)[0][0]
    else:
        hole = next(iter(cand_holes))

    def fn(g, hole=hole):
        return _symmetry_fill(g, hole)

    try:
        if all(np.array_equal(fn(i), o) for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 3) Symmetrize the interior of each rectangular frame. Frames are hollow
#    rectangles of a single "frame" color; inside, a "fill" pattern is made
#    symmetric (union with its mirror images) within the frame.
# ---------------------------------------------------------------------------
def _frame_boxes(g, frame_color):
    """Return bounding boxes (r0,r1,c0,c1) of 4-connected frame_color comps."""
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    boxes = []
    for r in range(H):
        for c in range(W):
            if g[r, c] == frame_color and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == frame_color:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                rs = [y for y, x in cells]
                xs = [x for y, x in cells]
                r0, r1, c0, c1 = min(rs), max(rs), min(xs), max(xs)
                if r1 - r0 >= 2 and c1 - c0 >= 2:
                    boxes.append((r0, r1, c0, c1))
    return boxes


def symmetrize_frame_interior(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # fill color = single changed-to color
    to_colors = set()
    for i, o in train:
        for r, c in np.argwhere(i != o):
            to_colors.add(int(o[r, c]))
    if len(to_colors) != 1:
        return None
    fill = next(iter(to_colors))
    # candidate frame colors: all colors except fill and bg(0)
    all_colors = set()
    for i, _ in train:
        all_colors |= set(int(v) for v in np.unique(i))
    cand_frames = [c for c in all_colors if c != fill]

    # symmetry subset options
    sym_variants = [
        ("both", lambda m: m | np.fliplr(m) | np.flipud(m) | np.fliplr(np.flipud(m))),
        ("lr", lambda m: m | np.fliplr(m)),
        ("ud", lambda m: m | np.flipud(m)),
        ("rot180", lambda m: m | np.fliplr(np.flipud(m))),
    ]

    def make(frame_color, symf):
        def fn(g):
            out = g.copy()
            for (r0, r1, c0, c1) in _frame_boxes(g, frame_color):
                ir0, ir1, ic0, ic1 = r0 + 1, r1 - 1, c0 + 1, c1 - 1
                if ir1 < ir0 or ic1 < ic0:
                    continue
                sub = out[ir0:ir1 + 1, ic0:ic1 + 1]
                mask = (sub == fill)
                if not mask.any():
                    continue
                sym = symf(mask)
                sub2 = sub.copy()
                sub2[sym] = fill
                out[ir0:ir1 + 1, ic0:ic1 + 1] = sub2
            return out
        return fn

    for fc in cand_frames:
        for _, symf in sym_variants:
            fn = make(fc, symf)
            try:
                if all(np.array_equal(fn(i), o) for i, o in train):
                    return fn
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# 4) Frame -> bar chart. Each rectangular frame encloses some "mark" cells;
#    emit one horizontal bar per frame (frame's color, length = #marks inside),
#    stacked & sorted by length, padded to the max length.
# ---------------------------------------------------------------------------
def _color_boxes(g, col):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    boxes = []
    for r in range(H):
        for c in range(W):
            if g[r, c] == col and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                rs = [y for y, x in cells]
                xs = [x for y, x in cells]
                boxes.append((min(rs), max(rs), min(xs), max(xs)))
    return boxes


def frame_bar_chart(train):
    # outputs must be 2d bars
    all_colors = set()
    for i, _ in train:
        all_colors |= set(int(v) for v in np.unique(i))

    def make(mark, ascending):
        def fn(g):
            gcolors = set(int(v) for v in np.unique(g)) - {0, mark}
            bars = []
            for col in gcolors:
                for (r0, r1, c0, c1) in _color_boxes(g, col):
                    if r1 - r0 < 2 or c1 - c0 < 2:
                        continue
                    interior = g[r0 + 1:r1, c0 + 1:c1]
                    cnt = int((interior == mark).sum())
                    if cnt > 0:
                        bars.append((cnt, col))
            if not bars:
                return None
            bars.sort(key=lambda b: b[0], reverse=not ascending)
            maxlen = max(b[0] for b in bars)
            out = np.zeros((len(bars), maxlen), int)
            for k, (cnt, col) in enumerate(bars):
                out[k, :cnt] = col
            return out
        return fn

    for mark in all_colors:
        if mark == 0:
            continue
        for asc in (True, False):
            fn = make(mark, asc)
            try:
                if all(fn(i) is not None and fn(i).shape == o.shape and np.array_equal(fn(i), o)
                       for i, o in train):
                    return fn
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# 5) Two panels: an "anchor" panel with a single marker and a "template" panel
#    holding a pattern that includes that same marker color. Output = the
#    template pattern shifted so its marker lands on the anchor's marker,
#    rendered in the anchor panel's frame.
# ---------------------------------------------------------------------------
def _full_rects(g, frame):
    H, W = g.shape
    nonf = (g != frame)
    seen = np.zeros((H, W), bool)
    rects = []
    for r in range(H):
        for c in range(W):
            if nonf[r, c] and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = [(r, c)]
                while st:
                    y, x = st.pop()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and nonf[ny, nx]:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                            cells.append((ny, nx))
                rs = [y for y, x in cells]
                xs = [x for y, x in cells]
                r0, r1, c0, c1 = min(rs), max(rs), min(xs), max(xs)
                if (r1 - r0 + 1) * (c1 - c0 + 1) == len(cells):
                    rects.append(g[r0:r1 + 1, c0:c1 + 1].copy())
    return rects


def stamp_template_on_anchor(train):
    def fn(g):
        frame = _bg(g)
        rects = _full_rects(g, frame)
        if len(rects) < 2:
            return None
        counts = [int((r != 0).sum()) for r in rects]
        # remap: treat frame as the "empty" cell too if frame != 0
        counts = [int(((r != 0) & (r != frame)).sum()) for r in rects]
        anchor_i = min(range(len(rects)), key=lambda k: counts[k])
        templ_i = max(range(len(rects)), key=lambda k: counts[k])
        if anchor_i == templ_i:
            return None
        anchor, templ = rects[anchor_i], rects[templ_i]
        empty = 0
        apos = np.argwhere((anchor != empty) & (anchor != frame))
        if len(apos) != 1:
            return None
        acolor = anchor[apos[0][0], apos[0][1]]
        ar, ac = apos[0]
        tpos = np.argwhere(templ == acolor)
        if len(tpos) == 0:
            return None
        tr, tc = tpos[0]
        out = np.full(anchor.shape, empty, dtype=int)
        H, W = anchor.shape
        for (yy, xx) in np.argwhere((templ != empty) & (templ != frame)):
            ny, nx = ar + (yy - tr), ac + (xx - tc)
            if 0 <= ny < H and 0 <= nx < W:
                out[ny, nx] = templ[yy, xx]
        return out
    try:
        if all(fn(i) is not None and fn(i).shape == o.shape and np.array_equal(fn(i), o)
               for i, o in train):
            return fn
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 6) Staircase chart: rank colors by some scalar (component-count, cell-count,
#    ...) and emit an NxN right- or left-aligned triangular staircase where the
#    k-th ranked color forms a bar of length N-k.
# ---------------------------------------------------------------------------
def _ncomp(g, col):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    n = 0
    for r in range(H):
        for c in range(W):
            if g[r, c] == col and not seen[r, c]:
                n += 1
                st = [(r, c)]
                seen[r, c] = True
                while st:
                    y, x = st.pop()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
    return n


def staircase_rank_chart(train):
    def score_fns():
        return {
            "ncomp": lambda g, col: _ncomp(g, col),
            "cells": lambda g, col: int((g == col).sum()),
        }

    def make(scoref, reverse, align, use_score_len):
        def fn(g):
            colors = [int(v) for v in np.unique(g) if int(v) != 0]
            if not colors:
                return None
            vals = {col: scoref(g, col) for col in colors}
            scored = sorted(colors, key=lambda col: vals[col], reverse=reverse)
            N = len(scored)
            if use_score_len:
                lengths = [vals[col] for col in scored]
            else:
                lengths = [N - k for k in range(N)]
            W = max(lengths) if lengths else 0
            if W <= 0:
                return None
            out = np.zeros((N, W), dtype=int)
            for k, col in enumerate(scored):
                length = lengths[k]
                if length <= 0:
                    continue
                if align == "right":
                    out[k, W - length:] = col
                else:
                    out[k, :length] = col
            return out
        return fn

    for _, scoref in score_fns().items():
        for reverse in (True, False):
            for align in ("right", "left"):
                for use_score_len in (True, False):
                    fn = make(scoref, reverse, align, use_score_len)
                    try:
                        if all(fn(i) is not None and fn(i).shape == o.shape and np.array_equal(fn(i), o)
                               for i, o in train):
                            return fn
                    except Exception:
                        continue
    return None


# ---------------------------------------------------------------------------
# 7) Region-keyed object gravity. The grid is partitioned into regions of
#    different "field" colors; small objects of a mobile color slide, within
#    their region, toward a region-specific wall (learned per field color).
# ---------------------------------------------------------------------------
def _objects(g, obj_color):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    objs = []
    for r in range(H):
        for c in range(W):
            if g[r, c] == obj_color and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = [(r, c)]
                while st:
                    y, x = st.pop()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == obj_color:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                            cells.append((ny, nx))
                objs.append(cells)
    return objs


def _region_color(g, cells, obj_color):
    H, W = g.shape
    cnt = Counter()
    cs = set(cells)
    for y, x in cells:
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in cs and g[ny, nx] != obj_color:
                cnt[int(g[ny, nx])] += 1
    return cnt.most_common(1)[0][0] if cnt else None


def region_object_gravity(train):
    if any(i.shape != o.shape for i, o in train):
        return None
    # obj_color candidates: colors appearing in input; the mobile one is usually
    # the least frequent forming compact blobs.
    all_colors = set()
    for i, _ in train:
        all_colors |= set(int(v) for v in np.unique(i))

    DIRS = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}

    def slide(g, cells, reg, dxy):
        H, W = g.shape
        dy, dx = dxy
        cur = list(cells)
        cset = set(cur)
        while True:
            nxt = [(y + dy, x + dx) for y, x in cur]
            ok = True
            nset = set(nxt)
            for (y, x) in nxt:
                if not (0 <= y < H and 0 <= x < W):
                    ok = False
                    break
                if (y, x) in cset:
                    continue
                if g[y, x] != reg:
                    ok = False
                    break
            if not ok:
                break
            cur = nxt
            cset = nset
        return cur

    def make(obj_color, dirmap):
        def fn(g):
            out = g.copy()
            for cells in _objects(g, obj_color):
                reg = _region_color(g, cells, obj_color)
                if reg not in dirmap:
                    return None
                d = DIRS[dirmap[reg]]
                for y, x in cells:
                    out[y, x] = reg
                final = slide(out, cells, reg, d)
                for y, x in final:
                    out[y, x] = obj_color
            return out
        return fn

    for obj_color in all_colors:
        if obj_color == 0:
            continue
        # find field colors present next to objects across all demos
        field_colors = set()
        ok_struct = True
        for i, _ in train:
            objs = _objects(i, obj_color)
            if not objs:
                ok_struct = False
                break
            for cells in objs:
                rc = _region_color(i, cells, obj_color)
                if rc is None:
                    ok_struct = False
                else:
                    field_colors.add(rc)
        if not ok_struct or not field_colors or len(field_colors) > 3:
            continue
        # brute force direction assignment per field color
        import itertools
        fcs = sorted(field_colors)
        for combo in itertools.product(DIRS.keys(), repeat=len(fcs)):
            dirmap = dict(zip(fcs, combo))
            fn = make(obj_color, dirmap)
            try:
                if all(fn(i) is not None and np.array_equal(fn(i), o) for i, o in train):
                    return fn
            except Exception:
                continue
    return None


DETECTORS = [
    stretch_dup_lines,
    symmetrize_frame_interior,
    frame_bar_chart,
    stamp_template_on_anchor,
    staircase_rank_chart,
    region_object_gravity,
    symmetry_fill_global,
]
