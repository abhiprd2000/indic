"""Test: Qwen3 regex excludes \\p{M}, so matras split Devanagari words."""
import random, collections, unicodedata as U
import pandas as pd
import sys; sys.path.insert(0, "src")
from pretok_lib import REGEX, split_only
from baseline_fertility import SETS

print("Qwen3 regex:", REGEX["a_original"])
DEV = lambda c: "ऀ" <= c <= "ॿ"
words = [w for w in SETS["Bhojpuri (BHTB)"] for w in w.split()]
cat = collections.Counter(U.category(c) for w in words for c in w if DEV(c))
tot = sum(cat.values())
print("Devanagari chars by category (BHTB):", {k: f"{100*v/tot:.1f}%" for k, v in cat.most_common()})
print("chars that regex (a) treats as \\p{L}: %.1f%%; Mn/Mc (not \\p{L}): %.1f%%" % (100 * cat["Lo"] / tot, 100 * (cat["Mn"] + cat["Mc"]) / tot))
pre = {k: split_only(v) for k, v in REGEX.items()}
rows = []
for sn, sents in SETS.items():
    if sn.startswith("English"): continue
    ws = [w for s in sents for w in s.split()]
    r = dict(test_set=sn, words=len(ws), pct_words_with_mark=round(100 * sum(any(U.category(c) in ("Mn", "Mc") for c in w) for w in ws) / len(ws), 1))
    for k, p in pre.items():
        n = [len(p.pre_tokenize_str(w)) for w in ws]
        r[f"pretokens_per_word_{k[0]}"] = round(sum(n) / len(n), 3)
        r[f"pct_words_split_{k[0]}"] = round(100 * sum(x >= 2 for x in n) / len(n), 1)
    rows.append(r)
pd.DataFrame(rows).to_csv("results/pretok/hypothesis.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
uniq = sorted({w for w in words if len(w) >= 4 and sum(U.category(c) in ("Mn", "Mc") for c in w) >= 2})
ex = random.Random(0).sample(uniq, 5); erows = []
for w in ex:
    d = {"word": w}
    for k, p in pre.items(): d[k] = " | ".join(x for x, _ in p.pre_tokenize_str(w))
    erows.append(d)
pd.DataFrame(erows).to_csv("results/pretok/examples.csv", index=False)
print(pd.DataFrame(erows).to_string(index=False))
