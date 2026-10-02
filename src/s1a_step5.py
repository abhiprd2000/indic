"""Stage 1A step 5: selection on DEV only (never sets FINAL_EVAL). Dev fertility curves per K; mix choice."""
import time
from s1a_lib import *
import data_access as DA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
assert os.environ.get("FINAL_EVAL") != "1", "selection script must not run with FINAL_EVAL"
t0 = time.time(); R = "results/stage1a"; CAND = [500, 1000, 2000, 4000, 8000]
dev = dict(get_dev_sets())
for code, name in [("bho_Deva", "FLORES dev bho"), ("mai_Deva", "FLORES dev mai"), ("mag_Deva", "FLORES dev mag"), ("hin_Deva", "FLORES dev hin"), ("eng_Latn", "FLORES dev eng")]:
    try: dev[name] = DA.load_flores(code, "dev")
    except Exception as e: print("FLORES dev missing:", code, type(e).__name__)
TARGET = ["Bhojpuri (dev)", "Magahi (dev)", "Maithili (dev)", "Angika (dev)"]
# dev curves, variants a and d on lowonly
rows = []
for v in ["a", "d"]:
    merges = json.load(open(f"{S1}/merges_lowonly_{v}.json"))
    for K in KS: rows += flat(eval_tok(load(build_s1(v, "lowonly", merges, K)), dev), variant=v, K=K, mix="lowonly")
cur = pd.DataFrame(rows); cur.to_csv(f"{R}/dev_fertility_vs_K.csv", index=False)
print(cur[cur.K.isin([0, 500, 2000, 8000])].pivot_table(index=["variant", "K"], columns="test_set", values="fertility").round(2).to_string(), flush=True)
# mix selection: variant d, candidate mixes, score = mean dev fertility over bho, mai, mag, ang and over candidate K
mixrows, score = [], []
for mix in ["hin_only", "natural4", "upsampled4", "lowonly"]:
    merges, prow = train_mix("d", mix, 8000); umb = sum(r["unique_MB"] for r in prow)
    for K in CAND:
        res = eval_tok(load(build_s1("d", mix, merges, K)), {k: dev[k] for k in TARGET})
        for sn, r in res.items(): mixrows.append(dict(mix=mix, K=K, test_set=sn, fertility=round(float(r["fertility"]), 4), unique_MB=round(umb, 2), n_langs=len(prow)))
    print("mix", mix, f"{time.time()-t0:.0f}s", flush=True)
m = pd.DataFrame(mixrows); m.to_csv(f"{R}/mix_dev_fertility.csv", index=False)
sc = m.groupby("mix").agg(score=("fertility", "mean"), unique_MB=("unique_MB", "first"), n_langs=("n_langs", "first")).reset_index().sort_values("score")
best = sc.score.min(); near = sc[sc.score <= best + 0.01].sort_values(["unique_MB", "n_langs"]); chosen = near.iloc[0].mix
sc["within_0.01_of_best"] = sc.score <= best + 0.01; sc.round(4).to_csv(f"{R}/mix_selection.csv", index=False)
print(sc.round(4).to_string(index=False)); print("chosen mix:", chosen)
json.dump(dict(K=CAND, mix=chosen, rule="mean dev fertility over bho/mai/mag/ang and K candidates; within 0.01 -> fewer unique MB, then fewer languages"), open(f"{R}/selection.json", "w"), indent=1)
# figure
plt.rcParams.update({"font.size": 11}); names = list(dev); fig, ax = plt.subplots(3, 4, figsize=(20, 12)); xs = np.maximum(KS, 120)
for a, sn in zip(ax.ravel(), names):
    for v, col in [("a", "#7a7a7a"), ("d", "#1b6ca8")]:
        x = cur[(cur.variant == v) & (cur.test_set == sn)].sort_values("K"); a.plot(np.maximum(x.K, 120), x.fertility, "o-", color=col, lw=2, label={"a": "a original", "d": "d scoped"}[v]); a.fill_between(np.maximum(x.K, 120), x.fert_lo, x.fert_hi, color=col, alpha=0.25)
    a.axhline(2, color="k", lw=1); a.set_xscale("log"); a.set_xticks([120, 250, 500, 1000, 2000, 4000, 8000, 16000]); a.set_xticklabels(["0", "250", "500", "1k", "2k", "4k", "8k", "16k"], fontsize=8); a.set_title(sn, fontsize=11); a.set_ylabel("tokens per word")
for a in ax.ravel()[len(names):]: a.axis("off")
ax.ravel()[0].legend(frameon=False); fig.tight_layout(); fig.savefig(f"{R}/dev_fertility_vs_K.png", dpi=200)
log_time("s1a_5", t0)
