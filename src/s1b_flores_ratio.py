"""Stage 1B addition 1: Qwen3 base (K=0) vs variant d (K=500..8000) on FLORES+ devtest with 95% CIs and base/d ratios (paired bootstrap)."""
import time
from s1b_lib import *
from s1a_lib import *
t0 = time.time(); R = "results/scan"; SEL = json.load(open("results/stage1a/selection.json")); MIX = SEL["mix"]
FL = {"bho": "bho_Deva", "mag": "mag_Deva", "mai": "mai_Deva", "hin": "hin_Deva", "eng": "eng_Latn"}; CAND = SEL["K"]
merges = json.load(open(f"{S1}/merges_{MIX}_d.json")); toks = {0: load(vdir("a"))}
for K in CAND: toks[K] = load(build_s1("d", MIX, merges, K))
rows = []
for l, code in FL.items():
    S = DA.load_flores(code, "devtest"); st = {K: sent_stats(t, S) for K, t in toks.items()}; n = len(S); idx = np.random.default_rng(SEED).integers(0, n, (1000, n))
    nw, nt0 = st[0][0], st[0][1]; f0 = nt0[idx].sum(1) / nw[idx].sum(1)
    for K, (nw_, nt, ns) in st.items():
        f = nt[idx].sum(1) / nw[idx].sum(1); rows.append(dict(lang=l, K=K, variant="base" if K == 0 else "d", n_sents=n, fertility=round(nt.sum() / nw.sum(), 4), ci_lo=round(float(np.percentile(f, 2.5)), 4), ci_hi=round(float(np.percentile(f, 97.5)), 4),
                                                         ratio_base_over_d=round((nt0.sum() / nw.sum()) / (nt.sum() / nw.sum()), 3), ratio_lo=round(float(np.percentile(f0 / f, 2.5)), 3), ratio_hi=round(float(np.percentile(f0 / f, 97.5)), 3)))
d = pd.DataFrame(rows); d.to_csv(f"{R}/flores_qwen3_vs_d.csv", index=False)
print(d[d.K.isin([0, 500, 2000, 8000])].pivot_table(index=["K"], columns="lang", values="fertility").round(2).to_string())
print(d[d.K > 0].pivot_table(index="K", columns="lang", values="ratio_base_over_d").round(2).to_string())
print(d[d.K == 8000][["lang", "ratio_base_over_d", "ratio_lo", "ratio_hi"]].to_string(index=False))
