"""Render a hero image: one real ARC-AGI-1 evaluation task the solver gets right.
Run with an interpreter that has numpy + PIL (e.g. ../venv/bin/python).
Writes docs/solved_example.png. This is a repo-doc helper, safe to delete."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import harness, registry
from PIL import Image, ImageDraw, ImageFont

PAL = [(0,0,0),(0,116,217),(255,65,54),(46,204,64),(255,220,0),
       (170,170,170),(240,18,190),(255,133,27),(127,219,255),(135,12,37)]
BG=(12,15,23); CARD=(20,25,37); LINE=(38,48,74); INK=(231,236,245); DIM=(139,151,173); OK=(70,211,154)

def font(sz):
    for p in ("/System/Library/Fonts/SFNS.ttf","/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

def grid_img(g, cell=20, gap=1):
    g = np.asarray(g); H, W = g.shape
    im = Image.new("RGB", (W*cell+gap, H*cell+gap), LINE)
    d = ImageDraw.Draw(im)
    for r in range(H):
        for c in range(W):
            d.rectangle([c*cell+gap, r*cell+gap, (c+1)*cell, (r+1)*cell], fill=PAL[int(g[r,c]) % 10])
    return im

def arrow(h):
    im = Image.new("RGB", (36, h), CARD); d = ImageDraw.Draw(im)
    d.line([(6,h//2),(28,h//2)], fill=DIM, width=3)
    d.polygon([(28,h//2-6),(36,h//2),(28,h//2+6)], fill=DIM)
    return im

def pair_row(inp, outp, label, ok=False, cell=20):
    gi, go = grid_img(inp,cell), grid_img(outp,cell)
    h = max(gi.height, go.height); ar = arrow(h); pad, lblh = 10, 20
    W = pad + gi.width + ar.width + go.width + (44 if ok else 8)
    H = lblh + h + 8
    im = Image.new("RGB", (W, H), CARD); d = ImageDraw.Draw(im)
    d.text((pad, 3), label, fill=DIM, font=font(13))
    y = lblh
    im.paste(gi, (pad, y + (h-gi.height)//2))
    im.paste(ar, (pad+gi.width, y))
    im.paste(go, (pad+gi.width+ar.width, y + (h-go.height)//2))
    if ok:
        d.text((pad+gi.width+ar.width+go.width+10, y+h//2-12), "✓", fill=OK, font=font(22))
    return im

tasks = harness.load_tasks(os.path.join(HERE, "arc1", "data", "evaluation"))
dets = registry.load_all()
print(f"{len(dets)} detectors, scanning {len(tasks)} eval tasks for a clean solved one...")

pick = None
for name, task in tasks.items():
    if not (2 <= len(task["train"]) <= 3) or len(task["test"]) != 1:
        continue
    ti, gt = task["test"][0]
    if gt is None:
        continue
    if max(ti.shape) > 11 or max(gt.shape) > 11:
        continue
    if any(max(i.shape) > 11 or max(o.shape) > 11 for i, o in task["train"]):
        continue
    preds, _ = harness.solve_task(task, dets)
    cand = preds[0]
    if cand and any(np.array_equal(c, gt) for c in cand):
        pick = (name, task, cand[0]); break

if not pick:
    print("no clean small solved task found; loosening size limit")
    for name, task in tasks.items():
        if len(task["test"]) != 1:
            continue
        ti, gt = task["test"][0]
        if gt is None:
            continue
        preds, _ = harness.solve_task(task, dets)
        cand = preds[0]
        if cand and any(np.array_equal(c, gt) for c in cand):
            pick = (name, task, cand[0]); break

name, task, pred = pick
print("picked task", name)

rows = [pair_row(i, o, f"demo {k+1}") for k, (i, o) in enumerate(task["train"])]
rows.append(pair_row(task["test"][0][0], pred, "test  →  solver prediction", ok=True))

pad, title_h, gap, sep = 20, 52, 12, 22
W = max(r.width for r in rows) + pad*2
H = title_h + sum(r.height + gap for r in rows) + sep + pad
im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
d.text((pad, 14), "arc-agi-solver", fill=INK, font=font(22))
d.text((pad, 40), f"a solved ARC-AGI-1 evaluation task ({name}) — rule inferred from the demos, verified, then applied",
       fill=DIM, font=font(12))
y = title_h
for i, r in enumerate(rows):
    if i == len(rows) - 1:
        d.line([(pad, y+2), (W-pad, y+2)], fill=LINE, width=1); y += sep
    im.paste(r, (pad, y)); y += r.height + gap

os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
out = os.path.join(HERE, "docs", "solved_example.png")
im.save(out)
print("saved", out, im.size)
