"""5.2 K sweep for variants a, b, d (mix lowonly). Trains once at K=32000 and truncates."""
import sys, time, json, os
sys.path.insert(0, "src")
from sweep_lib import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoTokenizer
t0 = time.time()
KS = [0, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
plan, lines = target_plan("lowonly"); rows = []
for v in VARIANTS:
    mp = f"{SW}/merges_{v}.json"
    if os.path.exists(mp): merges = json.load(open(mp))
    else:
        tok0 = load(vdir(v)); cnt, _ = T.pretok_counts(tok0, plan, lines)
        merges, nu = train_merges(tok0, cnt, 32000); json.dump(merges, open(mp, "w"), ensure_ascii=False)
        print(v, "unique pretokens", nu, "merges", len(merges), flush=True)
    for K in KS:
        res = eval_tok(load(build_k(v, merges, K)))
        rows += flat(res, variant=v, K=K)
    print("done", v, f"{time.time()-t0:.0f}s", flush=True)
g = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
rows += flat(eval_tok(g), variant="gemma3", K=-1)
d = pd.DataFrame(rows); d.to_csv("results/sweep/fertility_vs_K.csv", index=False)
gem = d[d.variant == "gemma3"].set_index("test_set").fertility
# smallest K reaching thresholds
sm = []
for v in VARIANTS:
    for sn in ALLSETS:
        x = d[(d.variant == v) & (d.test_set == sn)].sort_values("K")
        sm.append(dict(variant=v, test_set=sn, **{f"smallest_K_le_{t}": (int(x[x.fertility <= t].K.min()) if (x.fertility <= t).any() else "none") for t in (2.0, 1.5)}))
pd.DataFrame(sm).to_csv("results/sweep/smallest_K.csv", index=False)
plt.rcParams.update({"font.size": 12}); fig, ax = plt.subplots(2, 3, figsize=(17, 9)); col = {"a": "#7a7a7a", "b": "#3a9d5d", "d": "#1b6ca8"}
lab = {"a": "a original", "b": "b all marks", "d": "d Devanagari marks"}
for a, sn in zip(ax.ravel(), ALLSETS):
    for v in VARIANTS:
        x = d[(d.variant == v) & (d.test_set == sn)].sort_values("K"); xs = np.maximum(x.K, 60)
        a.plot(xs, x.fertility, "o-", color=col[v], label=lab[v], lw=2); a.fill_between(xs, x.fert_lo, x.fert_hi, color=col[v], alpha=0.25)
    a.axhline(gem[sn], color="#e08a1e", ls="--", lw=2, label="Gemma3 base"); a.axhline(2, color="k", lw=1, label="y = 2")
    a.set_xscale("log"); a.set_xticks([60, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]); a.set_xticklabels(["0", "125", "250", "500", "1k", "2k", "4k", "8k", "16k", "32k"], fontsize=9)
    a.set_title(sn); a.set_xlabel("extra merges K"); a.set_ylabel("tokens per word"); a.minorticks_off()
ax[0, 0].legend(frameon=False, fontsize=10); fig.tight_layout(); fig.savefig("results/sweep/fertility_vs_K.png", dpi=200)
print(d[(d.K.isin([0, 8000, 32000]))].pivot_table(index=["variant", "K"], columns="test_set", values="fertility").round(2)[ALLSETS].to_string())
print(pd.read_csv("results/sweep/smallest_K.csv").to_string(index=False))
log_time("5.2", t0)
