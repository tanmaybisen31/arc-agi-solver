"""Self-test one gen module in isolation: base detectors + this module only.

Usage: ./venv/bin/python selftest.py gen.symmetry
Prints base vs combined score on arc1 train+eval and how many NEW tasks the
module solves. Isolated (imports only your module) so parallel authors don't
interfere.
"""
import sys, importlib
import harness, detectors

def main(modname):
    m = importlib.import_module(modname)
    mine = list(getattr(m, "DETECTORS", []))
    base = detectors.DETECTORS
    combo = base + mine
    print(f"module {modname}: {len(mine)} detectors")
    for setname, path in [("train", "arc1/data/training"), ("eval", "arc1/data/evaluation")]:
        tasks = harness.load_tasks(path)
        b = harness.benchmark(tasks, base)
        c = harness.benchmark(tasks, combo)
        new = sorted(set(c["solved_names"]) - set(b["solved_names"]))
        lost = sorted(set(b["solved_names"]) - set(c["solved_names"]))
        print(f"  {setname}: base {b['tasks_solved']} -> +mine {c['tasks_solved']} "
              f"| NEW {len(new)} {new[:8]} | REGRESSED {len(lost)} {lost[:5]}")

if __name__ == "__main__":
    main(sys.argv[1])
