"""Stage 1A step 3: filter TRAIN text only (near-duplicates of dev/test; letters from scripts other than Devanagari/Latin)."""
import sys, re, collections, unicodedata as U
import regex as RX
import pandas as pd
sys.path.insert(0, "src")
import data_access as DA
LANGS = ["bho", "mai", "mag", "ang", "hin"]; OUT = "results/audit"
def script_of(c):
    try: return U.name(c).split()[0]
    except ValueError: return "UNKNOWN"
rows, summ = [], []
for lang in LANGS:
    train = DA.load(lang, "tok_train"); near = {int(x) for x in open(f"{OUT}/neardup_train_ids_{lang}.txt") if x.strip()}
    chars = collections.Counter("".join(train)); bad = {c: script_of(c) for c in chars if U.category(c)[0] == "L" and not RX.match(r"[\p{Devanagari}\p{Latin}]", c)}
    rx = re.compile("[" + re.escape("".join(bad)) + "]") if bad else None
    keep, n_near, n_script, n_both = [], 0, 0, 0
    for i, s in enumerate(train):
        sc = sorted({bad[c] for c in s if c in bad}) if rx and rx.search(s) else []
        if i in near or sc:
            n_near += i in near and not sc; n_script += bool(sc) and i not in near; n_both += i in near and bool(sc)
            rows.append(dict(lang=lang, line=i, reason="+".join((["neardup"] if i in near else []) + (["script:" + "/".join(sc)] if sc else [])), text=s[:120]))
        else: keep.append(s)
    open(f"data/{lang}/tok_train_filtered.txt", "w", encoding="utf-8").write("\n".join(keep) + "\n")
    mb = lambda L: sum(len(x.encode()) + 1 for x in L) / 1e6
    summ.append(dict(lang=lang, n_train=len(train), removed_neardup_only=n_near, removed_script_only=n_script, removed_both=n_both, n_kept=len(keep),
                     MB_train=round(mb(train), 2), MB_kept=round(mb(keep), 2), bad_script_chars=len(bad)))
    print(summ[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/removed.csv", index=False); pd.DataFrame(summ).to_csv(f"{OUT}/removed_summary.csv", index=False)
r = pd.DataFrame(rows)
for l, g in r.groupby("lang"): print(l, g.reason.str.extract(r"script:(.*)")[0].str.split("/").explode().value_counts().head(4).to_dict())
