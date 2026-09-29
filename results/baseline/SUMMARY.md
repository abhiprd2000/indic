# Task 2: baseline fertility
Ran `src/baseline_fertility.py` (table `fertility.csv`) and `src/baseline_plot.py` (`fertility.png`, bytes omitted, off scale). All 6 tokenizers loaded, none skipped. Words = whitespace words of the sentence text.
Test sets: BHTB, MGTB, HDTB (UD test, "# text"), Maithili web test, Angika MT test. Modes: alone, context.
Context fertility, Qwen3: bho 4.07, mag 3.84, hin 4.42, mai 5.10, ang 4.79. Gemma3: 1.50, 1.56, 1.21, 1.98, 1.73.
Qwen3 splits 93% of BHTB words in context; 53% of its tokens end inside a grapheme cluster (Gemma3: 6%).
SANITY CHECK FAILED for Qwen3, not resolved. Word-alone Qwen3: hin 3.93 (pub 4.11), bho 3.54 (4.43), mag 3.28 (4.90).
Gemma3 word-alone matches published: hin 1.61 (1.59), bho 1.76 (1.76), mag 1.74 (1.74).
Checked: plain `tokenize` gives the same numbers as my code; NFD no change; a leading space gives bho 4.11, mag 3.90, hin 4.45 (still off for mag).
Likely the published Qwen3 numbers use a different word list or setup. I do not know their source.
FLORES+ (`openlanguagedata/flores_plus`) is gated; token has no access, so `src/baseline_sanity.py` did not run.
Bytes row: mid-cluster counts a byte boundary inside a codepoint or before a mark.
