"""5.4 Data efficiency: variant d, K=8000, merges from N MB of bho+mai+mag (natural mix, nested subsets)."""
import sys, time, random, collections
sys.path.insert(0, "src")
from sweep_lib import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0 = time.time(); K = 8000
NS = [0.25, 0.5, 1, 2, 5, 10, "all"]
tok0 = load(vdir("d")); pre = tok0.backend_tokenizer.pre_tokenizer; model = tok0.backend_tokenizer.model
items = [(l, s) for l in ["bho", "mai", "mag"] for s in T.read_lines(l)]
random.Random(SEED).shuffle(items)  # uniform shuffle = natural proportions; prefixes are nested
cnt, lb, used, j, snaps = collections.Counter(), collections.Counter(), 0, 0, {}
for l, s in items:
    for w, _ in pre.pre_tokenize_str(s): cnt[w] += 1
    b = len(s.encode()) + 1; used += b; lb[l] += b
    if j < len(NS) - 1 and used >= NS[j] * 1e6: snaps[j] = (collections.Counter(cnt), dict(lb), used); j += 1
snaps[len(NS) - 1] = (cnt, dict(lb), used)
print("counted", f"{time.time()-t0:.0f}s", flush=True)
cache, inv = {}, inv_vocab(); rows = []
res0 = eval_tok(tok0); rows += flat(res0, N_MB="0", actual_MB=0, unique_pretokens=0)
for j, N in enumerate(NS):
    c, lbytes, u = snaps[j]; words = list(c)
    for w in words:
        if w not in cache: cache[w] = [t.id for t in model.tokenize(w)]
    merges = T.learn([list(cache[w]) for w in words], [c[w] for w in words], dict(inv), K)
    out = f"{SW}/dN_{N}"; T.build(vdir("d"), merges, K, out)
    res = eval_tok(load(out))
    rows += flat(res, N_MB=str(N), actual_MB=round(u / 1e6, 2), unique_pretokens=len(words), **{f"MB_{l}": round(b / 1e6, 2) for l, b in lbytes.items()})
    print("N", N, "MB", round(u / 1e6, 2), "unique", len(words), "merges", len(merges), f"{time.time()-t0:.0f}s", flush=True)
d = pd.DataFrame(rows); d.to_csv("results/sweep/data_efficiency.csv", index=False)
plt.rcParams.update({"font.size": 12}); fig, ax = plt.subplots(2, 3, figsize=(17, 9))
for a, sn in zip(ax.ravel(), ALLSETS):
    x = d[(d.test_set == sn) & (d.N_MB != "0")]
    a.plot(x.actual_MB, x.fertility, "o-", color="#1b6ca8", lw=2, label="d, K=8k"); a.fill_between(x.actual_MB, x.fert_lo, x.fert_hi, color="#1b6ca8", alpha=0.25)
    a.axhline(d[(d.test_set == sn) & (d.N_MB == "0")].fertility.iloc[0], color="#7a7a7a", ls=":", lw=2, label="N=0 (K=0)"); a.axhline(2, color="k", lw=1, label="y = 2")
    a.set_xscale("log"); a.set_title(sn); a.set_xlabel("MB of bho+mai+mag text"); a.set_ylabel("tokens per word")
ax[0, 0].legend(frameon=False, fontsize=10); fig.tight_layout(); fig.savefig("results/sweep/data_efficiency.png", dpi=200)
print(d[d.N_MB != "0"].pivot_table(index="actual_MB", columns="test_set", values="fertility").round(2)[ALLSETS].to_string())
log_time("5.4", t0)
