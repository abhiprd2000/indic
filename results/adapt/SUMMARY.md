# Task 6: does the extended tokenizer work? (Qwen3-0.6B-Base, d K=8000, lowonly; 1x Tesla T4, fp16 autocast)
Ran `src/adapt_run.py` via Kaggle (`src/kaggle/`). Data 30 MB (4.09M tokens = 499 steps of 16x512, data-limited); LR trials 2e-4 vs 1e-3: 1e-3 won (mean dev BPB 1.116 vs 1.240; edge of grid, nothing higher tried).
Test BPB (A base / B untrained / C trained): bho 1.326/1.582/1.017, mai 1.391/1.653/1.076, mag 1.460/1.672/1.099, hin 0.756/1.587/0.939, ang 1.262/1.886/1.355, en 1.015/1.016/1.016.
Criteria (fixed in PLAN.md): MET. C < A on bho -23.3%, mai -22.7%, mag -24.7%; English +0.12% (limit +1%). No fallback run.
NOT GOOD: Hindi BPB is 24.2% worse in C than A (0.756 -> 0.939). Angika (unseen, no training text) is 7.4% worse (1.262 -> 1.355). B (no training) is worse than A on every language except English (+0.1%).
Hindi cause not tested (only 1.5 MB of Hindi in the training mix, 5%). More Hindi share is the obvious next run.
POS probe (train HDTB, 3 seeds, mean +- std): BHTB A 0.443+-0.001, B 0.517+-0.002, C 0.589+-0.007; MGTB A 0.407+-0.006, B 0.484+-0.007, C 0.583+-0.004. Higher with the extended tokenizer, but the arms use different tokenizers. Not comparable with Kumar et al. (1.7B model).
Speed (64 BHTB sentences, fp16 weights, T4): tokens/sentence A 80.8 vs C 25.0; ms/batch 1812 vs 622; bytes/s 7892 vs 22994. Peak memory 12.4 vs 7.0 GiB, mostly because padded length was 273 vs 81, not a fixed saving.
Trained: 8000 new rows only (old rows frozen), loss 8.98 -> 5.16 (nats/token). Run 1.49 h total; GPU training-related time 1.29 h (cap 2.5 h); throughput 4.16 s/step.
fp16 vs bf16 check on 200 bho sentences (arm A): BPB differs by -0.23%. T4 has no native bf16, so the plan's bf16 was replaced by fp16.
Caveats: single seed for training; BPB sets differ by language (English test is AngikaMT English, dev is FineWeb-edu). 6 Maithili sentences over 1500 bytes were dropped for all arms.
Files: bpb.csv, lr_trials.csv, train_log.csv, throughput.csv, pos_probe.csv, pos_summary.csv, speed.csv, dtype_check.csv, run_info.json. New rows: artifacts/adapt/new_rows_final.pt (gitignored).
