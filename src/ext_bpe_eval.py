"""Eval extended tokenizers: fertility, regression (English, Hindi), figures."""
import os, random, sys
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download as dl
from transformers import AutoTokenizer
sys.path.insert(0, "src")
from baseline_fertility import get_test_sets, stats
import data_access as DA

SEED, SWEEP = 0, [500, 1000, 2000, 4000, 8000, 16000, 32000]
MIXES = ["hin_only", "natural4", "upsampled4", "lowonly"]
if not os.path.exists("data/en/test.txt"):  # 2k English sentences, eval only
    p = dl("snjev310/AngikaMT", "train.csv", repo_type="dataset", local_dir="data/raw/hf", token=os.environ.get("HF_TOKEN"))
    E = [" ".join(s.split()) for s in pd.read_csv(p)["english"].dropna()]
    random.Random(SEED).shuffle(E); os.makedirs("data/en", exist_ok=True)
    open("data/en/test.txt", "w", encoding="utf-8").write("\n".join(E[:2000]) + "\n")
SETS = dict(get_test_sets()); SETS["English (2k)"] = DA.load("en", "test")

CFG = [("base", 0, "artifacts/tok/base")] + [("upsampled4", k, f"artifacts/tok/upsampled4_K{k}") for k in SWEEP] \
    + [(m, 8000, f"artifacts/tok/{m}_K8000") for m in MIXES if m != "upsampled4"]
rows, reg, base_ids = [], [], {}
for mix, K, path in CFG:
    tok = AutoTokenizer.from_pretrained(path)
    for sn, sents in SETS.items():
        for mode in ["context", "alone"]:
            rows.append(dict(mix=mix, K=K, test_set=sn, mode=mode, **stats(tok, sents, mode)))
        if sn in ("English (2k)", "Hindi (HDTB)"):
            ids = tok(sents, add_special_tokens=False)["input_ids"]
            if mix == "base": base_ids[sn] = ids
            reg.append(dict(mix=mix, K=K, test_set=sn, pct_sents_changed=100 * np.mean([a != b for a, b in zip(ids, base_ids[sn])])))
    print("done", mix, K, flush=True)
d = pd.DataFrame(rows).round(4); d.to_csv("results/ext_bpe/fertility.csv", index=False)
c = d[d["mode"] == "context"]
r = pd.DataFrame(reg).merge(c[["mix", "K", "test_set", "fertility"]], on=["mix", "K", "test_set"])
b = r[r.mix == "base"].set_index("test_set").fertility
r["base_fertility"] = r.test_set.map(b); r["delta"] = (r.fertility - r.base_fertility).round(4)
r.round(4).to_csv("results/ext_bpe/regression.csv", index=False)

# smallest K reaching <=2 (sweep mix)
sw = c[(c.mix == "upsampled4")]
first = {sn: (int(g[g.fertility <= 2.0].K.min()) if (g.fertility <= 2.0).any() else None) for sn, g in sw.groupby("test_set")}
pd.Series(first, name="smallest_K_le_2").to_csv("results/ext_bpe/smallest_K.csv")

# figures
plt.rcParams.update({"font.size": 13})
bl = pd.read_csv("results/baseline/fertility.csv"); bl = bl[(bl["mode"] == "context") & (bl.tokenizer == "Gemma3")].set_index("test_set").fertility
hin_base = c[(c.mix == "base") & (c.test_set == "Hindi (HDTB)")].fertility.iloc[0]
langs = ["Bhojpuri (BHTB)", "Magahi (MGTB)", "Maithili (web)", "Angika (MT)", "Hindi (HDTB)"]
fig, ax = plt.subplots(2, 3, figsize=(16, 9)); ax = ax.ravel()
xs = [0] + SWEEP
for a, sn in zip(ax, langs):
    g = sw[sw.test_set == sn].set_index("K").fertility; y = [c[(c.mix == "base") & (c.test_set == sn)].fertility.iloc[0]] + [g[k] for k in SWEEP]
    a.plot(range(len(xs)), y, "o-", color="#1b6ca8", lw=2.5, label="Qwen3 + continued BPE")
    a.axhline(bl[sn], color="#e08a1e", ls="--", lw=2, label="Gemma3")
    if sn != "Hindi (HDTB)": a.axhline(hin_base, color="#7a7a7a", ls=":", lw=2, label="Qwen3 Hindi (base)")
    a.axhline(2, color="k", lw=1, label="target 2")
    a.set_xticks(range(len(xs))); a.set_xticklabels(["base"] + [f"{k//1000}k" if k >= 1000 else str(k) for k in SWEEP]); a.set_title(sn); a.set_ylabel("tokens per word"); a.set_xlabel("extra merges K")
h, l = ax[0].get_legend_handles_labels(); ax[5].axis("off"); ax[5].legend(h, l, loc="center", frameon=False, fontsize=15)
fig.tight_layout(); fig.savefig("results/ext_bpe/fertility_vs_K.png", dpi=200)
fig, a = plt.subplots(figsize=(15, 6)); names = ["base", "hin_only", "natural4", "upsampled4", "lowonly"]
cols = ["#7a7a7a", "#1b6ca8", "#e08a1e", "#3a9d5d", "#b04a7f"]; w = 0.16
for i, m in enumerate(names):
    v = [c[(c.mix == m) & (c.K.isin([0, 8000])) & (c.test_set == s)].fertility.iloc[0] for s in langs]
    a.bar(np.arange(len(langs)) + (i - 2) * w, v, w, label=m, color=cols[i])
a.axhline(2, color="k", lw=1); a.set_xticks(range(len(langs))); a.set_xticklabels(langs); a.set_ylabel("tokens per word"); a.set_title("K = 8k, 40 MB budget per mix", loc="left")
a.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15)); fig.tight_layout(); fig.savefig("results/ext_bpe/mix_bars.png", dpi=200)
print(c[c.mix == "upsampled4"].pivot(index="K", columns="test_set", values="fertility").round(2).to_string())
print(c[(c.K.isin([0, 8000]))].pivot_table(index="mix", columns="test_set", values="fertility").round(2).to_string())
print(r[r.K.isin([8000, 32000])][["mix", "K", "test_set", "delta", "pct_sents_changed"]].to_string(index=False))
print(first)
