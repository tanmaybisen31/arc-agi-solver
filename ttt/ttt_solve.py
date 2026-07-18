"""Test-time training solver (Mac-native, MLX).

For each task: build an augmented training set from its demos, LoRA-fine-tune the
small model ON THAT TASK, then predict the test output using dihedral
augmented-inference + voting (pass@2). This is the ARChitects/MIT recipe at small
scale, to validate the pipeline on the Mac and establish a real TTT number.
"""
import os, sys, json, time, argparse, subprocess, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, HERE)
import harness
import augment as A
from incontext import parse_grid
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

SETS = {"arc2-eval": "arc2/data/evaluation", "arc2-train": "arc2/data/training",
        "arc1-eval": "arc1/data/evaluation", "arc1-train": "arc1/data/training"}

def write_data(examples, ddir):
    os.makedirs(ddir, exist_ok=True)
    n_val = max(1, len(examples) // 10)
    with open(os.path.join(ddir, "train.jsonl"), "w") as f:
        for e in examples[n_val:]:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(ddir, "valid.jsonl"), "w") as f:
        for e in examples[:n_val]:
            f.write(json.dumps(e) + "\n")

def train_lora(model_id, ddir, adir, iters, num_layers, lr, batch, max_seq):
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", model_id, "--train",
           "--data", ddir, "--fine-tune-type", "lora", "--num-layers", str(num_layers),
           "--batch-size", str(batch), "--iters", str(iters), "--learning-rate", str(lr),
           "--adapter-path", adir, "--mask-prompt", "--max-seq-length", str(max_seq),
           "--steps-per-eval", str(iters), "--val-batches", "1", "--steps-per-report", "50"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)[-500:]

def predict(model, tok, task, gt, n_aug):
    """Dihedral augmented inference + vote; return whether gt in top-2."""
    greedy = make_sampler(temp=0.0)
    votes = {}
    augs = A.DIHEDRAL[:n_aug]
    for dfwd, dinv in augs:
        demos = [(dfwd(np.asarray(i)), dfwd(np.asarray(o))) for i, o in task["train"]]
        ti = dfwd(np.asarray(task["test"][0][0]))
        msgs = [{"role": "system", "content": A.SYS},
                {"role": "user", "content": A._prompt_text(demos, ti)}]
        p = tok.apply_chat_template(msgs, add_generation_prompt=True)
        out = generate(model, tok, prompt=p, max_tokens=900, sampler=greedy, verbose=False)
        g = parse_grid(out)
        if g is None:
            continue
        try:
            g = dinv(g)  # back to original frame
        except Exception:
            continue
        k = (g.shape, g.tobytes())
        votes[k] = votes.get(k, [0, g]); votes[k][0] += 1
    ranked = sorted(votes.values(), key=lambda v: -v[0])
    top2 = [v[1] for v in ranked[:2]]
    return any(c.shape == gt.shape and np.array_equal(c, gt) for c in top2), len(top2)

def run(model_id, setname, n, iters, num_layers, lr, batch, n_aug, max_seq):
    tasks = list(harness.load_tasks(os.path.join(BASE, SETS[setname])).items())[:n]
    tmp = os.path.join(HERE, "_ttt_work")
    print(f"loading base tokenizer/model for building+inference: {model_id}")
    base_model, tok = load(model_id)
    solved = 0; total = 0; t0 = time.time()
    for name, task in tasks:
        gt = task["test"][0][1]
        if gt is None:
            continue
        total += 1
        ddir = os.path.join(tmp, name, "data"); adir = os.path.join(tmp, name, "adapter")
        exs = A.build_training_examples(task, tok)
        write_data(exs, ddir)
        ts = time.time()
        ok_train, log = train_lora(model_id, ddir, adir, iters, num_layers, lr, batch, max_seq)
        if not ok_train:
            print(f"  {name}: TRAIN FAILED | {log[-160:]}")
            continue
        model, _ = load(model_id, adapter_path=adir)   # base + task adapter
        ok, ncand = predict(model, tok, task, gt, n_aug)
        solved += ok
        print(f"  {name}: {'SOLVED' if ok else 'miss'} | {len(exs)} exs, train {time.time()-ts:.0f}s | {solved}/{total}")
        del model
    dt = time.time() - t0
    print(f"\n[TTT {model_id}] {setname}: {solved}/{total} = {solved/max(1,total)*100:.2f}% "
          f"(iters={iters},layers={num_layers},aug={n_aug}) {dt:.0f}s")
    try: shutil.rmtree(tmp)
    except Exception: pass
    return solved, total

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--set", default="arc2-eval")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n-aug", type=int, default=6)
    ap.add_argument("--max-seq", type=int, default=4096)
    a = ap.parse_args()
    run(a.model, a.set, a.n, a.iters, a.num_layers, a.lr, a.batch, a.n_aug, a.max_seq)
