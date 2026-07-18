"""In-context ARC baseline with an MLX small model (Mac-native).

Establishes the real neural floor on ARC-2 before we add test-time training.
Serializes a task's demos + test input as text, asks the model for the output
grid, parses it, and scores pass@2 (greedy + one sampled attempt).
"""
import sys, os, re, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

SETS = {
    "arc2-eval": "arc2/data/evaluation",
    "arc2-train": "arc2/data/training",
    "arc1-eval": "arc1/data/evaluation",
}
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def g2t(g):
    return "\n".join(" ".join(str(int(c)) for c in row) for row in g)

def parse_grid(text):
    rows = []
    for line in text.strip().splitlines():
        nums = re.findall(r"-?\d+", line)
        if not nums:
            continue
        row = [int(x) % 10 for x in nums]
        rows.append(row)
    if not rows:
        return None
    w = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == w] or rows
    w = len(rows[0])
    rows = [r for r in rows if len(r) == w]
    try:
        return np.array(rows, dtype=int)
    except Exception:
        return None

SYS = ("You solve ARC puzzles. Each shows input grids transformed to output grids "
       "by one hidden rule. Grids are rows of digits 0-9 separated by spaces. "
       "Infer the rule from the examples, then output ONLY the final output grid "
       "as rows of space-separated digits. No words, no explanation.")

def build_prompt(task, tok):
    parts = []
    for i, (gi, go) in enumerate(task["train"], 1):
        parts.append(f"Example {i} input:\n{g2t(gi)}\nExample {i} output:\n{g2t(go)}")
    ti = task["test"][0][0]
    parts.append(f"Test input:\n{g2t(ti)}\nTest output:")
    user = "\n\n".join(parts)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True)

def run(model_id, setname, n, max_tokens):
    tasks = harness.load_tasks(os.path.join(BASE, SETS[setname]))
    items = list(tasks.items())[:n]
    print(f"loading {model_id} ...")
    m, tok = load(model_id)
    greedy = make_sampler(temp=0.0)
    samped = make_sampler(temp=0.7)
    solved = 0; total = 0; t0 = time.time()
    for name, task in items:
        gt = task["test"][0][1]
        if gt is None:
            continue
        total += 1
        p = build_prompt(task, tok)
        cands = []
        for sampler in (greedy, samped):
            out = generate(m, tok, prompt=p, max_tokens=max_tokens, sampler=sampler, verbose=False)
            g = parse_grid(out)
            if g is not None:
                cands.append(g)
        ok = any(c.shape == gt.shape and np.array_equal(c, gt) for c in cands)
        solved += ok
        print(f"  {name}: {'SOLVED' if ok else 'miss'} ({solved}/{total})")
    dt = time.time() - t0
    print(f"\n[{model_id} in-context] {setname}: {solved}/{total} = {solved/max(1,total)*100:.2f}%  ({dt:.0f}s)")
    return solved, total

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--set", default="arc2-eval")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=900)
    a = ap.parse_args()
    run(a.model, a.set, a.n, a.max_tokens)
