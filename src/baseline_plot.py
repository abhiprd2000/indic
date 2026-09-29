"""Bar chart of baseline fertility (bytes omitted, off scale)."""
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
d = pd.read_csv("results/baseline/fertility.csv"); d = d[d.tokenizer != "Bytes"]
toks = list(dict.fromkeys(d.tokenizer)); sets = list(dict.fromkeys(d.test_set))
cols = ["#1b6ca8", "#e08a1e", "#3a9d5d", "#b04a7f", "#7a7a7a", "#8a5a2b"]
plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
for a, mode, ttl in zip(ax, ["context", "alone"], ["In context (sentence tokenised)", "Word alone"]):
    w = 0.13
    for i, t in enumerate(toks):
        v = [d[(d.tokenizer == t) & (d.test_set == s) & (d["mode"] == mode)].fertility.iloc[0] for s in sets]
        a.bar(np.arange(len(sets)) + (i - 2.5) * w, v, w, label=t, color=cols[i])
    a.axhline(2, color="k", ls="--", lw=1); a.set_ylabel("tokens per word"); a.set_title(ttl, loc="left")
ax[1].set_xticks(range(len(sets))); ax[1].set_xticklabels(sets)
ax[0].legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
fig.text(0.99, 0.005, "dashed line = target 2", ha="right", fontsize=11)
fig.tight_layout(); fig.savefig("results/baseline/fertility.png", dpi=200)
