"""Run the ARC-AGI benchmark and append results to RESULTS.md."""
import sys, time, json, os
from datetime import datetime, timezone
import harness
import detectors
import registry

BASE = os.path.dirname(os.path.abspath(__file__))
SETS = {
    "arc1-eval": os.path.join(BASE, "arc1", "data", "evaluation"),
    "arc1-train": os.path.join(BASE, "arc1", "data", "training"),
    "arc2-eval": os.path.join(BASE, "arc2", "data", "evaluation"),
    "arc2-train": os.path.join(BASE, "arc2", "data", "training"),
}

def run(setname, tag, save_unsolved=None):
    tasks = harness.load_tasks(SETS[setname])
    dets = registry.load_all()
    t0 = time.time()
    stats = harness.benchmark(tasks, dets)
    dt = time.time() - t0
    line = (f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} | "
            f"{tag} | {setname} | tasks {stats['tasks_solved']}/{stats['n_tasks']} "
            f"({stats['task_acc']*100:.2f}%) | outputs@2 {stats['outputs_solved']}/{stats['n_outputs']} "
            f"({stats['output_acc']*100:.2f}%) | dets {len(dets)} | {dt:.1f}s")
    print(line)
    with open(os.path.join(BASE, "RESULTS.md"), "a") as f:
        f.write(line + "\n")
    if save_unsolved:
        with open(os.path.join(BASE, save_unsolved), "w") as f:
            json.dump({"solved": stats["solved_names"], "unsolved": stats["unsolved_names"]}, f, indent=0)
    return stats

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    setname = sys.argv[2] if len(sys.argv) > 2 else "arc1-eval"
    save = sys.argv[3] if len(sys.argv) > 3 else None
    run(setname, tag, save)
