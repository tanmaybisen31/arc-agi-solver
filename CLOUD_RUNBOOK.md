# Phase 1 — Cloud Runbook (the real ARC-2 scores)

## Why cloud
The Mac (Metal/MLX) proved the *mechanism* but can't run the winning stacks:
NVARC/ARChitects/MindsAI/TRM-training all need **CUDA** (Unsloth, bitsandbytes
4-bit, triton, flash-attn) and 8B-scale TTT. Real ARC-2 scores (10%+) need a
CUDA GPU. Mac stays the **fast MLX experiment bench** for the paper track.

## The box (pick one)
- **RunPod** or **Lambda** — 1× **H100 80GB** (~$2-3/hr) or 1× **L40S 48GB**
  (~$1/hr). Start with L40S for dev, H100 for the big training runs.
- Storage: 200GB+ volume (models + synthetic datasets are large).
- Image: PyTorch 2.x + CUDA 12.x base.

## Setup (on the box)
```bash
git clone <your arc-agi repo>            # our harness + ttt/ pipeline + baselines/
cd arc-agi
pip install torch transformers accelerate peft datasets bitsandbytes trl unsloth
# ARC data already in arc1/ and arc2/ (or re-clone fchollet/ARC-AGI + arcprize/ARC-AGI-2)
```

## Two tracks (run in parallel)

### Track A — Reproduce the 2025 winner (leaderboard)
Fastest route to a real competitive number.
1. **MindsAI (smaller, easier first):** `baselines/MindsAI/` — CodeT5, clean
   TTFT pipeline. Follow its README (`prepare_data.py` → `train.py` →
   `predict.py`). Target: reproduce its ~15% ARC-2.
2. **NVARC (the 24% winner):** `baselines/NVARC/` — needs the Kaggle datasets:
   `kaggle datasets download -d sorokin/nvarc-synthetic-puzzles` (103k) and
   `-d sorokin/nvarc-augmented-puzzles` (3.2M). Then their ARChitects Qwen3-4B
   Unsloth-flash-LoRA fine-tune + TRM. Heavy (multi-GPU-days) — do this only
   after MindsAI works.

### Track B — Our own 8B TTT (portable, paper-ready)
Port our `ttt/` pipeline (already written, works on Mac) to CUDA + a bigger model:
- Swap model to `Qwen2.5-7B-Instruct` (or `-8B`), keep our `augment.py`
  (dihedral × color-perm) + per-task LoRA + augmented voting.
- Bump iters (300-1000), num-layers (-1 = all), batch (8-16) — trivial on H100.
- This is the substrate for the **paper insight** (internals-guided / efficient TTT).

## Kaggle submission (the actual prize gate)
- Runs **offline** on Kaggle: 4×L4, ~2hr wall-clock, no internet, no frontier API.
- Per-task TTT over 120 hidden tasks in 2hr is TIGHT → inference/TTT efficiency is
  make-or-break (your strength; also a paper angle).
- Model weights + adapters must be uploaded as a Kaggle dataset (offline).
- **Open-source (CC0/MIT-0) within 7 days** of close (Nov 2 → Nov 9) or you're
  removed from prize consideration.

## Cost estimate
- Dev/reproduction: ~$100-300 of L40S/H100 time.
- Full NVARC repro (synthetic data + multi-day training): $500-1500 if pursued.
- Track B (our 8B TTT): ~$100-400.

## Targets (from the plan)
- Paper prize (primary): a clean novel result — score-independent.
- Progress prize (stretch): top-8, ~12-20% on ARC-2.

## Verification (unchanged)
- Metric: held-out **arc2-eval (120)** — never train on it.
- Do a **Kaggle dry-run submission early** to confirm offline + budget compliance.
- Log every run in `RESULTS.md`.
