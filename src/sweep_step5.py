"""5.5 Leave-one-language-out: variant d, K=8000, lowonly mix minus one language."""
import sys, time
sys.path.insert(0, "src")
from sweep_lib import *
t0 = time.time(); K = 8000
plan, lines = target_plan("lowonly"); tok0 = load(vdir("d")); rows = []
NAME = {"bho": "Bhojpuri (BHTB)", "mai": "Maithili (web)", "mag": "Magahi (MGTB)"}
sw = pd.read_csv("results/sweep/fertility_vs_K.csv"); sw = sw[sw.variant == "d"]
for L in ["bho", "mai", "mag"]:
    p = {l: m for l, m in plan.items() if l != L}  # keep other languages' MB as in lowonly (total < 40 MB)
    cnt, info = T.pretok_counts(tok0, p, lines)
    merges, nu = train_merges(tok0, cnt, K); out = f"{SW}/dL_{L}"; T.build(vdir("d"), merges, K, out)
    res = eval_tok(load(out)); rows += flat(res, held_out=L, tokenizer="lolo", MB=round(sum(p.values()), 2))
    print("held out", L, "MB", round(sum(p.values()), 2), "unique", nu, f"{time.time()-t0:.0f}s", flush=True)
d = pd.DataFrame(rows)
tab = []
for L, sn in list(NAME.items()) + [("ang", "Angika (MT)")]:
    a = sw[(sw.test_set == sn) & (sw.K == K)].iloc[0]; b0 = sw[(sw.test_set == sn) & (sw.K == 0)].iloc[0]
    r = dict(language=sn, seen_in_training="no (unseen)" if L == "ang" else "in all-language run", base_K0=round(b0.fertility, 3),
             all_langs=round(a.fertility, 3), all_lo=round(a.fert_lo, 3), all_hi=round(a.fert_hi, 3))
    if L != "ang":
        x = d[(d.held_out == L) & (d.test_set == sn)].iloc[0]
        r.update(held_out=round(x.fertility, 3), held_lo=round(x.fert_lo, 3), held_hi=round(x.fert_hi, 3), delta_held_vs_all=round(x.fertility - a.fertility, 3))
    tab.append(r)
pd.DataFrame(tab).to_csv("results/sweep/lolo.csv", index=False); d.to_csv("results/sweep/lolo_all_sets.csv", index=False)
print(pd.DataFrame(tab).to_string(index=False)); log_time("5.5", t0)
