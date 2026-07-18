"""ARC-AGI benchmark harness + solver engine.

Approach: an ensemble of *rule detectors*. Each detector inspects the training
input/output pairs and, if it can infer a transformation that reproduces EVERY
training output exactly, returns a callable transform. The engine then applies
every fitting detector to each test input, collects candidate outputs, votes,
and submits the top-2 (ARC allows 2 attempts per test output).

Scoring follows ARC Prize: a test output is solved if the ground truth appears
in the 2 submitted attempts. We report both per-output pass@2 and per-task
(all test outputs solved) accuracy.
"""
import os, json, glob
import numpy as np

# ---------------- data ----------------
def load_tasks(data_dir):
    tasks = {}
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(fp))[0]
        with open(fp) as f:
            t = json.load(f)
        train = [(np.array(p["input"], dtype=int), np.array(p["output"], dtype=int)) for p in t["train"]]
        test = [(np.array(p["input"], dtype=int),
                 (np.array(p["output"], dtype=int) if "output" in p else None)) for p in t["test"]]
        tasks[name] = {"train": train, "test": test}
    return tasks

def gkey(g):
    return (g.shape, g.tobytes())

# ---------------- engine ----------------
def fitting_transforms(train, detectors):
    """Return list of (name, transform) for detectors that reproduce all train outputs."""
    out = []
    for det in detectors:
        try:
            fn = det(train)
        except Exception:
            fn = None
        if fn is None:
            continue
        try:
            ok = all(np.array_equal(fn(ti), to) for ti, to in train)
        except Exception:
            ok = False
        if ok:
            out.append((det.__name__, fn))
    return out

def candidates_for(test_input, fits):
    """Apply every fitting transform to a test input, vote, return ranked unique grids."""
    votes = {}      # gkey -> [count, first_rank, grid]
    for rank, (name, fn) in enumerate(fits):
        try:
            g = fn(test_input)
        except Exception:
            continue
        if g is None or g.ndim != 2 or g.size == 0:
            continue
        g = np.asarray(g, dtype=int)
        k = gkey(g)
        if k not in votes:
            votes[k] = [0, rank, g]
        votes[k][0] += 1
    # sort by (votes desc, earliest detector rank asc)
    ranked = sorted(votes.values(), key=lambda v: (-v[0], v[1]))
    return [v[2] for v in ranked]

def solve_task(task, detectors):
    """Return, per test input, a list of up to 2 candidate grids."""
    fits = fitting_transforms(task["train"], detectors)
    preds = []
    for ti, _ in task["test"]:
        cands = candidates_for(ti, fits)
        preds.append(cands[:2])
    return preds, len(fits)

# ---------------- benchmark ----------------
def benchmark(tasks, detectors, verbose=False):
    n_out = n_out_solved = 0
    n_task = n_task_solved = 0
    attempted = 0            # tasks with >=1 fitting detector
    solved_names, unsolved_names = [], []
    for name, task in tasks.items():
        preds, nfit = solve_task(task, detectors)
        if nfit > 0:
            attempted += 1
        n_task += 1
        task_all = True
        has_gt = False
        for (ti, gt), cand in zip(task["test"], preds):
            if gt is None:
                task_all = False
                continue
            has_gt = True
            n_out += 1
            ok = any(np.array_equal(c, gt) for c in cand)
            if ok:
                n_out_solved += 1
            else:
                task_all = False
        if has_gt and task_all:
            n_task_solved += 1
            solved_names.append(name)
        elif has_gt:
            unsolved_names.append(name)
    return {
        "n_tasks": n_task,
        "tasks_solved": n_task_solved,
        "task_acc": n_task_solved / max(1, n_task),
        "n_outputs": n_out,
        "outputs_solved": n_out_solved,
        "output_acc": n_out_solved / max(1, n_out),
        "attempted": attempted,
        "solved_names": solved_names,
        "unsolved_names": unsolved_names,
    }

def print_report(tag, stats):
    print(f"[{tag}] tasks {stats['tasks_solved']}/{stats['n_tasks']} "
          f"= {stats['task_acc']*100:.2f}%  |  outputs pass@2 "
          f"{stats['outputs_solved']}/{stats['n_outputs']} = {stats['output_acc']*100:.2f}%  "
          f"|  had-a-fit on {stats['attempted']} tasks")
