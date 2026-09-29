# Task 5 (CPU): sweep of variants a, b, d (mix lowonly, seed 0). Time per step: 5.2 109s (226s incl. training), 5.3 14s, 5.4 243s, 5.5 101s, 5.6 <1s.
5.1 d = b but only Devanagari marks join letters (the `_marks` class plus nukta U+093C, which `_marks` omitted). English pre-tokens identical on 2000/2000 sentences.
5.2 d K=8k fertility (base K=0): bho 1.29 (4.07), mag 1.33 (3.84), mai 1.65 (5.10), ang 1.77 (4.79), hin 1.45 (4.42), English 1.31 (1.31). d vs b: max diff 0.001. a (original regex) at 8k: 2.18/2.02/2.58/2.61/2.44.
Smallest K <=2.0 for d: bho 1000, mag 1000, hin 2000, mai 4000, ang 4000 (a: only mag, at 16k). <=1.5: bho 4000, mag 4000, hin 8000, mai 32000, ang 32000. One-token words at K=8k: bho 78.8% (base 6.7%).
5.2 NOT held: d at 8k is worse than Gemma3 on Hindi (1.45 vs 1.21) and Angika (1.77 vs 1.73), and English is unchanged (1.31 vs Gemma3 1.27).
5.3 Round trip decode(encode(x))==x: 0 failures on all test sets for base and a/b/d at K=8k. My first build had 145 failures: special-token ids collided with the last 26 new ids. Fixed in `build()`; fertility unchanged (max diff 0.0 over 186 rows).
5.3 Task 3 and 4 tokenizers were built with the old `build()` (wrong ids for last 26 merges); their fertility is unaffected, but they were not regenerated.
5.3 HELD: at K=0 d changes only Devanagari (hin 21.3% of sentences); b also changes Thai 100%, Bengali 31.9%, Arabic 28.6%, Urdu 3.0%.
5.3 NOT held at K=8k: d changes Bengali 87.6% and Tamil 99.1% of sentences (tokens -1.8%, -3.7%); a and b do the same. Latin, Arabic, Urdu, Thai, Japanese: 0%. Cause: 15 of 8000 new tokens are Bengali/Tamil, learned from stray non-Devanagari text in training data. Not fixed (filter that text).
5.4 d K=8k, natural bho+mai+mag: 0.25 MB gives bho 1.42, mag 1.47, mai 1.79, ang 1.91, hin 1.62; 1 MB gives 1.34/1.40/1.68/1.84/1.51; 5 MB matches 10 MB; 250 MB gives 1.30/1.37/1.63/1.78/1.45. Gains stop after about 2-5 MB.
5.5 Leave-one-out (other languages keep lowonly MB, total 21-34 MB): held-out vs all-language run: bho 1.47 vs 1.29, mai 1.96 vs 1.66, mag 1.37 vs 1.33. Angika, never trained: 1.77 (base 4.79). Unseen languages lose 0.04-0.31 but stay far below base.
5.6 H-Net chunks/word (value as given by you) 1.17 vs our tokens/word at K=8k: bho 1.29, mag 1.33. Not matched. Definitions differ, so not directly comparable.
Caveats: fertility only. No model was run (no GPU), so nothing here shows the model still works. K=8k adds 8000 tokens (5% of vocab). CIs: 95% bootstrap over test sentences, seed 0.
Files: fertility_vs_K.csv/.png, smallest_K.csv, safety.csv, script_leak.csv, data_efficiency.csv/.png, lolo.csv, granularity.csv, timing.csv.
