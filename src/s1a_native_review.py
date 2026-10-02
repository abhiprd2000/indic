"""Native review shares with 95% Wilson intervals. Input: results/audit/sample_<lang>_labeled.txt = the sample file with a third tab column
label in {own, hindi, mixed, other}. Output: results/audit/native_review.csv. Nothing is dropped based on it."""
import os, sys, math, collections
import pandas as pd
D = os.environ.get("NATIVE_DIR", "results/audit"); LANGS = ["bho", "mai", "mag", "ang", "hin"]; CATS = ["own", "hindi", "mixed", "other"]
def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
rows = []
for l in LANGS:
    p = f"{D}/sample_{l}_labeled.txt"
    if not os.path.exists(p): print("no labels yet for", l); continue
    labs = [x.rstrip("\n").split("\t")[2].strip().lower() for x in open(p, encoding="utf-8") if x.strip() and len(x.split("\t")) >= 3]
    bad = [x for x in labs if x not in CATS]; assert not bad, f"{l}: unknown labels {set(bad)}"
    c = collections.Counter(labs); n = len(labs)
    for k in CATS:
        lo, hi = wilson(c[k], n); rows.append(dict(lang=l, label=k, n_labeled=n, count=c[k], share=round(c[k] / n, 4), wilson_lo=round(lo, 4), wilson_hi=round(hi, 4)))
if rows: pd.DataFrame(rows).to_csv(f"{D}/native_review.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
