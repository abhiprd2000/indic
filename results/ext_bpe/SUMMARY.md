# Task 3: continued BPE on Qwen3 tokenizer
Ran `src/ext_bpe_train.py` then `src/ext_bpe_eval.py` (seed 0). Fertility = in context, whitespace words. Base -> K=32k (upsampled mix): bho 4.07->2.17, mag 3.84->1.99, mai 5.10->2.55, ang 4.79->2.59, hin 4.42->2.43.
Smallest K with fertility <=2.0: Magahi 16k only. Bhojpuri, Maithili, Angika, Hindi never reach 2 by 32k (curves flat after ~8k).
Cause: Qwen3 pre-tokenizer splits at every matra, so merges cannot cross it. 52% of tokens still end inside a grapheme cluster at 32k (base 53%).
Mix at K=8k (bho/mag/mai/ang/hin): hin_only 2.29/2.14/2.68/2.62/2.43; natural 2.18/2.02/2.58/2.61/2.44; upsampled 2.18/2.02/2.58/2.61/2.44; low-only 2.18/2.02/2.58/2.61/2.45.
Hindi data helps little for low-resource; the three other mixes are within 0.01 of each other (upsampling gives no gain).
MB of text per mix (all 40 MB effective): hin_only 40.0 unique; natural 39.99 unique; upsampled 39.76 unique (Magahi 3.76 MB repeated 1.12x); low-only 37.76 unique (Magahi repeated 1.6x). Per language in `mixes.csv`.
Regression (K=8k, upsampled): English 1.313 -> 1.313 (delta 0.000, 0% sentences changed); Hindi 4.421 -> 2.438 (delta -1.98). At 32k English delta -0.002 (3.55% of sentences changed).
Fertility can only fall when merges are appended, so this check cannot show harm to the model. Quality is not tested.
Angika is not in any training mix (as specified); it still gains via shared Devanagari merges.
K sweep uses the upsampled mix, chosen before seeing results. English test = 2k sentences from AngikaMT English column.
New ids are appended after Qwen's special tokens; embeddings for them are needed later.
Files: fertility.csv, regression.csv, smallest_K.csv, mixes.csv, fertility_vs_K.png, mix_bars.png. Tokenizers are in `artifacts/tok/` (gitignored, rebuild with the train script).
