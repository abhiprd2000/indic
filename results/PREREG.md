# Pre-registration (written before any Stage 1A run, 2026-10-02)

## Hypotheses
- H1: the scoped pre-tokenizer rule (variant d: only Devanagari combining marks join letters) plus K new merges gives lower fertility than the original rule (variant a) at equal K. Test: paired bootstrap over test sentences (1000 draws, seed 0) on Bhojpuri, Magahi, Maithili, Angika, Hindi; supported if the 95% CI of (a - d) is above 0 for K in {500, 1000, 2000, 4000, 8000}.
- H2: adapting Qwen3-0.6B-Base with the extended tokenizer beats a matched continued-pretraining control on bits per byte. Control: base tokenizer, same data, same steps and LR protocol, same number of trainable parameters (embedding rows of the 8000 most frequent existing tokens in the training mix, same size as the new rows). Supported if bits per byte is lower for the extended arm on bho, mai, mag with the 95% bootstrap CI of the difference below 0. Not run in Stage 1A.
- H3: the pre-tokenizer cut (marks outside \p{L}) affects other open tokenizers. Test: share of Devanagari whitespace words split into 2+ pre-tokens by each tokenizer's own pre-tokenizer (Qwen3, Llama-3.1, Gemma3, XLM-R, mBERT, IndicBERTv2). Not run in Stage 1A.

## Selection rules (dev only)
- K, mix and learning rate are chosen on dev sets only. Dev sets: bho, mai, mag, ang, hin, en (plus FLORES+ dev where available). Test sets are never read by a selection script; `FINAL_EVAL` is never set in one.
- K candidates for model runs: {500, 1000, 2000, 4000, 8000}. Mix: lowest mean dev fertility over bho, mai, mag, ang; if within 0.01, the smaller mix. LR: chosen on dev bits per byte.
- Stage 1A train filter (train text only; dev and test untouched): drop train sentences with near-duplicates in dev/test (5-gram word shingles, Jaccard >= 0.5) and sentences with letters from a script other than Devanagari or Latin. Merges are rebuilt at K=32000 for a and d on the filtered lowonly mix.
- `src/data_access.py` is the only way to read text. Audit scripts may read test text for leak, language-ID and script statistics only (`purpose="audit"`); every test read is logged in `results/audit/test_access.log`.

## Final evaluation
- Test sets (existing six, plus FLORES+ devtest bho/mag/mai/hin/eng as extra) are read only by final-eval scripts with `FINAL_EVAL=1`, on candidate tokenizers fixed by the rules above.

## Logging
All runs are logged, including failures; earlier results that used test sets for choices (Tasks 3-6: K, mix, LR) are not valid selections and are superseded by dev-based selection.

## Addendum before step 5 (written before any dev fertility was computed)
- Mix selection: candidate mixes hin_only, natural4, upsampled4, lowonly (all 40 MB budget, built on filtered train, variant d). Score = mean dev fertility over bho, mai, mag, ang and over K in {500, 1000, 2000, 4000, 8000}. If the best score is within 0.01 of another mix, prefer the one with fewer unique MB of text, then fewer languages.
- English dev = AngikaMT English dev column (997 sentences, disjoint from the English test sample, checked in code).
- Step 4 FLORES+ safety check and the round-trip check use FLORES+ dev and the dev/test sets with `purpose="audit"` (integrity only, no metric). FLORES+ devtest is read only in step 6 with FINAL_EVAL=1.
- New-token script check: complete characters in a new token must be Devanagari, Latin, digits, punctuation or spaces; fragments of a split UTF-8 character count as violations only if their known bytes already lie outside those ranges.
