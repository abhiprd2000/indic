# Task 1: data
Ran `src/fetch_data.py` (seed 0). NFC on all text. UD (POS kept) in `data/ud/`, splits in `data/<lang>/{tok_train,dev,test}.txt`.
Fetched all: FineWeb2 bho/mai/mag/hin, AngikaMT (train+dev csv, angika column), UD BHTB/MGTB/HDTB. `data/mine/` is empty.
Sentences (train/dev/test): bho 405159/1006/2006, mai 815470/1004/2011, mag 15652/1128/2013, ang 5221/747/1492, hin 508063/1654/2001.
Full table: `sizes.csv`. Per-source counts: `sources.csv`.
FLAG under 5 MB: mag (4.4 MB), ang (2.7 MB).
Ang test is 1492 (<2k): only 7190 sentences exist in AngikaMT.
BHTB and MGTB are test-only upstream, so all their sentences went to test (bho 359, mag 551).
Splits are by document; exact-duplicate sentences removed; overlap of tok_train/dev with test asserted zero.
Hin: 500k sentences from random row groups of one FineWeb2 shard, plus HDTB.
Sentences: split on danda/./?/!, kept if >=3 words and has Devanagari.
Test mixes web text and UD text. AngikaMT is translation text, not web.
tok_train files are gitignored (large); rerun the script to rebuild.
