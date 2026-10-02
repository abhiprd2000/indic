"""Stage 1A step 6: FINAL eval of fixed candidates (a, d x K candidates, selected mix) on existing test sets and FLORES+ devtest."""
import time
os_env = __import__("os").environ; os_env["FINAL_EVAL"] = "1"  # final-eval script only
from s1a_lib import *
import data_access as DA
t0 = time.time(); R = "results/stage1a"; SEL = json.load(open(f"{R}/selection.json")); MIX, CAND = SEL["mix"], SEL["K"]
print("selected on dev:", MIX, CAND)
existing = test_sets()
flo, missing = {}, []
for code, name in [("bho_Deva", "FLORES devtest bho"), ("mag_Deva", "FLORES devtest mag"), ("mai_Deva", "FLORES devtest mai"), ("hin_Deva", "FLORES devtest hin"), ("eng_Latn", "FLORES devtest eng")]:
    try: flo[name] = DA.load_flores(code, "devtest")
    except Exception as e: missing.append((code, type(e).__name__))
print("FLORES devtest missing:", missing)
toks = {("base", 0): load(vdir("a"))}
for v in ["a", "d"]:
    merges = json.load(open(f"{S1}/merges_{MIX}_{v}.json"))
    for K in CAND: toks[(v, K)] = load(build_s1(v, MIX, merges, K))
rows_e, rows_f, h1 = [], [], []
stats_cache = {}
for (v, K), tok in toks.items():
    for grp, sets, rows in [("existing", existing, rows_e), ("flores", flo, rows_f)]:
        for sn, S in sets.items():
            st = sent_stats(tok, S); stats_cache[(v, K, sn)] = st; rows.append(dict(variant=v, K=K, test_set=sn, **{k: round(float(x), 4) for k, x in boot(*st).items()}))
for grp, sets in [("existing", existing), ("flores", flo)]:
    for sn in sets:
        for K in CAND:
            na, ta = stats_cache[("a", K, sn)][:2]; nd_, td = stats_cache[("d", K, sn)][:2]; n = len(na); idx = np.random.default_rng(SEED).integers(0, n, (1000, n))
            diff = ta[idx].sum(1) / na[idx].sum(1) - td[idx].sum(1) / nd_[idx].sum(1)
            h1.append(dict(group=grp, test_set=sn, K=K, fert_a=round(ta.sum() / na.sum(), 4), fert_d=round(td.sum() / nd_.sum(), 4), diff_a_minus_d=round(float(ta.sum() / na.sum() - td.sum() / nd_.sum()), 4),
                           ci_lo=round(float(np.percentile(diff, 2.5)), 4), ci_hi=round(float(np.percentile(diff, 97.5)), 4), supports_H1=bool(np.percentile(diff, 2.5) > 0)))
pd.DataFrame(rows_e).to_csv(f"{R}/test_fertility_existing.csv", index=False); pd.DataFrame(rows_f).to_csv(f"{R}/test_fertility_flores.csv", index=False)
h = pd.DataFrame(h1); h.to_csv(f"{R}/h1_paired.csv", index=False)
for name, rows in [("existing", rows_e), ("flores", rows_f)]:
    d = pd.DataFrame(rows); print(name); print(d[d.K.isin([0, 500, 2000, 8000])].pivot_table(index=["variant", "K"], columns="test_set", values="fertility").round(2).to_string())
print("H1 supported in", int(h.supports_H1.sum()), "of", len(h), "set x K cells; not supported:", h[~h.supports_H1][["test_set", "K"]].values.tolist())
log_time("s1a_6", t0)
