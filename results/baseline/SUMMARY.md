# Task 2: baseline fertility
Ran `src/baseline_fertility.py` (table `fertility.csv`) and `src/baseline_plot.py` (`fertility.png`, bytes omitted, off scale). All 6 tokenizers loaded, none skipped. Words = whitespace words of the sentence text.
Test sets: BHTB, MGTB, HDTB (UD test, "# text"), Maithili web test, Angika MT test. Modes: alone, context.
Context fertility, Qwen3: bho 4.07, mag 3.84, hin 4.42, mai 5.10, ang 4.79. Gemma3: 1.50, 1.56, 1.21, 1.98, 1.73.
SANITY (FLORES+ devtest, `sanity.csv`, in context): Qwen3 hin 4.76 (pub 4.11), bho 4.64 (4.43), mag 4.62 (4.90).
Gemma3 hin 1.39 (1.59), bho 1.67 (1.76), mag 1.60 (1.74). All within 0.3 except Qwen3 Hindi (+0.65; word-alone 4.27, +0.16).
Code is right: plain `tokenize` matches my numbers. Earlier UD gaps came from the test set (UD words are short), not the code.
Qwen3 Hindi still unexplained; I do not know how the published number was made. Bhojpuri and Magahi, our targets, pass.
Qwen3 splits 93% of BHTB words in context; 53% of its tokens end inside a grapheme cluster (Gemma3: 6%).
Bytes row: mid-cluster counts a byte boundary inside a codepoint or before a mark.
