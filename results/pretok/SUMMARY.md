# Task 4: pre-tokenizer
Ran `src/pretok_hypothesis.py` and `src/pretok_variants.py` (seed 0). K=8k, mix = lowonly (best by mean fertility on 4 target sets in Task 3; margin 0.001).
Qwen3 regex: `[^\r\n\p{L}\p{N}]?\p{L}+` for words. HYPOTHESIS CONFIRMED: 35.1% of BHTB Devanagari chars are Mn/Mc (not \p{L}). 74.5% of BHTB words split into >=2 pre-tokens (2.16 per word). Example: सोचलीं -> स | ोचल | ीं.
(b) `[\p{L}\p{M}]+`: 0.7% of BHTB words split (1.008 pre-tokens per word). (c) one akshara per pre-token: 2.19 pre-tokens per word, a hard floor on fertility.
Fertility a/b/c: bho 2.18/1.29/2.19, mag 2.02/1.33/2.15, mai 2.58/1.66/2.75, ang 2.61/1.77/2.30, hin 2.44/1.45/2.23.
% tokens ending inside a cluster a/b/c: bho 52.2/11.3/0.0, mag 46.8/10.0/0.0, mai 54.5/15.7/0.1, ang 57.9/29.5/1.5, hin 55.3/15.9/0.1.
Hindi fertility (base 4.42): a 2.44 (-1.98), b 1.45 (-2.97), c 2.23 (-2.19). English 1.313 in all (c: -0.0003).
Regex (b) identical to original on 2k English sentences: 2000/2000 pre-tokens and 2000/2000 token ids (also c).
Only (b) reaches <=2 on all four target languages. Caveat: at K=0, (b) changes tokenization of 5.0% of Hindi sentences, (c) of 99.3%. Pretrained embeddings see shifted inputs; needs training to check.
(b) also changes any script with combining marks (Latin accents, Thai, Arabic); only English tested. Fertility is not model quality.
transformers `AutoTokenizer` silently ignores a custom pre-tokenizer regex; load with `PreTrainedTokenizerFast(tokenizer_file=...)`. Task 3 used the original regex, so its numbers hold.
Files: hypothesis.csv, examples.csv, fertility.csv, regression.csv, english_check.csv, pretok_variants.png. Tokenizers in `artifacts/tok/pre_*` (gitignored; merges.json kept).
