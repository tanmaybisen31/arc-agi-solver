2026-07-02 22:07:57Z | baseline | arc1-eval | tasks 2/400 (0.50%) | outputs@2 2/419 (0.48%) | dets 15 | 0.1s
2026-07-02 22:07:57Z | baseline | arc1-train | tasks 25/400 (6.25%) | outputs@2 25/416 (6.01%) | dets 15 | 0.1s
2026-07-02 22:28:23Z | round1 | arc1-eval | tasks 57/400 (14.25%) | outputs@2 61/419 (14.56%) | dets 129 | 41.5s
2026-07-02 22:28:57Z | round1 | arc1-train | tasks 108/400 (27.00%) | outputs@2 110/416 (26.44%) | dets 129 | 19.9s
2026-07-03 05:07:58Z | round2 | arc1-eval | tasks 106/400 (26.50%) | outputs@2 114/419 (27.21%) | dets 179 | 41.4s
2026-07-03 05:08:36Z | round2 | arc1-train | tasks 118/400 (29.50%) | outputs@2 120/416 (28.85%) | dets 179 | 21.6s
2026-07-03 05:09:10Z | round2 | arc2-eval | tasks 1/120 (0.83%) | outputs@2 1/167 (0.60%) | dets 179 | 15.3s
2026-07-03 05:10:14Z | round2 | arc2-train | tasks 210/1000 (21.00%) | outputs@2 220/1076 (20.45%) | dets 179 | 63.4s
2026-07-03 05:13:05Z | round3 | arc1-eval | tasks 106/400 (26.50%) | outputs@2 114/419 (27.21%) | dets 180 | 46.5s
2026-07-03 05:13:23Z | round3 | arc2-eval | tasks 1/120 (0.83%) | outputs@2 1/167 (0.60%) | dets 180 | 17.2s
2026-07-03 05:37:05Z | round3b | arc1-eval | tasks 110/400 (27.50%) | outputs@2 118/419 (28.16%) | dets 220 | 52.2s
2026-07-03 05:38:21Z | round3b | arc2-train | tasks 231/1000 (23.10%) | outputs@2 243/1076 (22.58%) | dets 220 | 75.7s
2026-07-03 05:38:40Z | round3b | arc2-eval | tasks 1/120 (0.83%) | outputs@2 1/167 (0.60%) | dets 220 | 18.7s
2026-07-03 06:25:23Z | round4 | arc1-eval | tasks 164/400 (41.00%) | outputs@2 175/419 (41.77%) | dets 276 | 90.3s
2026-07-03 06:27:26Z | round4 | arc2-train | tasks 291/1000 (29.10%) | outputs@2 306/1076 (28.44%) | dets 276 | 122.8s
2026-07-03 06:27:59Z | round4 | arc2-eval | tasks 2/120 (1.67%) | outputs@2 2/167 (1.20%) | dets 276 | 33.0s
2026-07-03 07:04:45Z | round5 | arc1-eval | tasks 207/400 (51.75%) | outputs@2 221/419 (52.74%) | dets 321 | 101.8s
2026-07-03 07:07:04Z | round5 | arc2-train | tasks 339/1000 (33.90%) | outputs@2 358/1076 (33.27%) | dets 321 | 139.4s
2026-07-03 07:07:43Z | round5 | arc2-eval | tasks 2/120 (1.67%) | outputs@2 2/167 (1.20%) | dets 321 | 38.8s

## Phase 0: Neural pivot (MLX, Mac-native)
2026-07-03 | env | MLX 0.29.3 + mlx-lm 0.29.1 installed; baselines cloned (TRM, NVARC, MindsAI); all confirmed CUDA-only (Unsloth/bitsandbytes/triton) => heavy repro goes to cloud.
2026-07-03 | in-context | Qwen2.5-1.5B-Instruct-4bit | arc2-eval (n=30): 0/30 = 0.00%. Confirms small-model prompting can't do ARC-2; TTT is required.
2026-07-03 | ttt-pipeline | per-task LoRA (aug: dihedral x color-perm, ~110-220 exs/task) + dihedral augmented-inference voting. Runs end-to-end on Mac; tuning + more compute (cloud) needed for real scores. Validation run in progress.
