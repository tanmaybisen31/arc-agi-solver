import sys, json
import numpy as np

def show(g):
    for row in g:
        print("".join(str(c) for c in row))

ids = sys.argv[1:]
for tid in ids:
    with open(f"/Users/tanmaybisen/Desktop/mywork/arc-agi/arc1/data/evaluation/{tid}.json") as f:
        t = json.load(f)
    print("="*60)
    print("TASK", tid)
    for i, p in enumerate(t["train"]):
        inp = np.array(p["input"]); out = np.array(p["output"])
        print(f"--- train {i}  in {inp.shape} -> out {out.shape}")
        print("IN:")
        show(inp)
        print("OUT:")
        show(out)
    for i, p in enumerate(t["test"]):
        inp = np.array(p["input"])
        print(f"--- test {i} in {inp.shape}")
        show(inp)
