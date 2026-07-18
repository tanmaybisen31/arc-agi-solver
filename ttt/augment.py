"""Task augmentation for test-time training.

Turns a task's ~3 demo pairs into many (input->output) training examples via the
dihedral group (8 orientations) x color permutations. This is the core TTT trick:
the model fine-tunes on the task's OWN rule under many surface variations, so it
learns the transformation rather than memorizing pixels.
"""
import numpy as np
import random

# dihedral group: (forward, inverse) grid transforms
def _dihedral():
    I = lambda g: g
    return [
        (I, I),
        (np.fliplr, np.fliplr),
        (np.flipud, np.flipud),
        (lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 3)),
        (lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 2)),
        (lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 1)),
        (lambda g: g.T.copy(), lambda g: g.T.copy()),
        (lambda g: np.rot90(g, 2).T.copy(), lambda g: np.rot90(np.asarray(g).T, 2).copy()),
    ]

DIHEDRAL = _dihedral()

def color_perm(seed):
    r = random.Random(seed)
    # permute colors 1..9, keep 0 (background) fixed
    cols = list(range(1, 10))
    r.shuffle(cols)
    mp = {0: 0}
    for a, b in zip(range(1, 10), cols):
        mp[a] = b
    lut = np.array([mp[i] for i in range(10)], dtype=int)
    inv = np.zeros(10, dtype=int)
    for k, v in mp.items():
        inv[v] = k
    return (lambda g: lut[g]), (lambda g: inv[g])

def g2t(g):
    return "\n".join(" ".join(str(int(c)) for c in row) for row in g)

SYS = ("You solve ARC puzzles. Each shows input grids transformed to output grids "
       "by one hidden rule. Grids are rows of digits 0-9 separated by spaces. "
       "Infer the rule from the examples, then output ONLY the final output grid "
       "as rows of space-separated digits. No words, no explanation.")

def _prompt_text(demos, query_in):
    parts = []
    for i, (gi, go) in enumerate(demos, 1):
        parts.append(f"Example {i} input:\n{g2t(gi)}\nExample {i} output:\n{g2t(go)}")
    parts.append(f"Test input:\n{g2t(query_in)}\nTest output:")
    return "\n\n".join(parts)

def build_training_examples(task, tok, n_color_perms=6, max_examples=220):
    """Leave-one-out over demos, under dihedral x color augmentations.
    Returns list of {"prompt":..., "completion":...} (chat-templated prompt)."""
    demos = task["train"]
    if len(demos) < 2:
        # single demo: still train to reproduce it under augmentation
        demos = demos + demos
    examples = []
    color_augs = [(lambda g: g, lambda g: g)] + [color_perm(s) for s in range(n_color_perms)]
    combos = [(df, cf) for df, _ in DIHEDRAL for cf, _ in color_augs]
    random.Random(0).shuffle(combos)
    for dfwd, cfwd in combos:
        aug = [(cfwd(dfwd(np.asarray(i))), cfwd(dfwd(np.asarray(o)))) for i, o in demos]
        for j in range(len(aug)):
            ctx = [aug[k] for k in range(len(aug)) if k != j]
            if not ctx:
                continue
            qi, qo = aug[j]
            prompt_txt = _prompt_text(ctx, qi)
            examples.append({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt_txt},
                {"role": "assistant", "content": g2t(qo)},
            ]})
            if len(examples) >= max_examples:
                return examples
    return examples

def inference_prompt(task, tok, dfwd, cfwd):
    """Build a chat-templated prompt for the real test input under an augmentation."""
    demos = [(cfwd(dfwd(np.asarray(i))), cfwd(dfwd(np.asarray(o)))) for i, o in task["train"]]
    ti = cfwd(dfwd(np.asarray(task["test"][0][0])))
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": _prompt_text(demos, ti)}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
