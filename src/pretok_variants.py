"""Rerun continued BPE (K=8k, best mix) with 3 pre-tokenizers: a original, b marks attached, c akshara."""
import os, sys, json
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast
sys.path.insert(0, "src")
import ext_bpe_train as T
AutoTokenizer = type('L', (), {'from_pretrained': staticmethod(lambda d: PreTrainedTokenizerFast(tokenizer_file=d + '/tokenizer.json'))})  # AutoTokenizer ignores a custom regex
from pretok_lib import REGEX, variant_dir, split_only
from baseline_fertility import get_test_sets, stats
import data_access as DA

K = 8000
T3 = pd.read_csv("results/ext_bpe/fertility.csv")
t3 = T3[(T3["mode"] == "context") & (T3.K == K) & T3.test_set.str.contains("Bhojpuri|Magahi|Maithili|Angika")]
MIX = t3.groupby("mix").fertility.mean().idxmin()  # best mix = lowest mean fertility on 4 target sets
print("best mix from Task 3:", MIX)
SETS = dict(get_test_sets()); SETS["English (2k)"] = DA.load("en", "test")

langs = ["hin", "bho", "mai", "mag"]
lines = {l: T.read_lines(l) for l in langs}
sizes = {l: sum(len(s.encode()) + 1 for s in lines[l]) / 1e6 for l in langs}
ls, alpha = T.MIXES[MIX]; plan = T.mix_plan(ls, alpha, sizes)
inv0 = {v: k for k, v in json.load(open("artifacts/tok/base/tokenizer.json"))["model"]["vocab"].items()}
paths = {("a", 0): "artifacts/tok/base", ("a", K): f"artifacts/tok/{MIX}_K{K}"}
for v, name in [("b", "b_marks_attached"), ("c", "c_akshara")]:
    vd = variant_dir(v, REGEX[name]); paths[(v, 0)] = vd
    tok = AutoTokenizer.from_pretrained(vd)
    cnt, _ = T.pretok_counts(tok, plan, lines)
    words = list(cnt); seqs = [[t.id for t in tok.backend_tokenizer.model.tokenize(w)] for w in words]
    merges = T.learn(seqs, [cnt[w] for w in words], dict(inv0), K)
    os.makedirs(f"artifacts/tok/pre_{v}", exist_ok=True)
    json.dump(merges, open(f"artifacts/tok/pre_{v}/merges.json", "w", encoding="utf-8"), ensure_ascii=False)
    T.build(vd, merges, K, f"artifacts/tok/pre_{v}_K{K}"); paths[(v, K)] = f"artifacts/tok/pre_{v}_K{K}"
    print(v, "unique pretokens", len(words), "merges", len(merges), flush=True)

# English identity check: pre-tokens and token ids vs original
en = SETS["English (2k)"]; base = AutoTokenizer.from_pretrained(paths[("a", 0)])
bids = base(en, add_special_tokens=False)["input_ids"]; erows = []
for v, name in [("b", "b_marks_attached"), ("c", "c_akshara")]:
    pa, pv = split_only(REGEX["a_original"]), split_only(REGEX[name])
    same_pre = sum([x for x, _ in pa.pre_tokenize_str(s)] == [x for x, _ in pv.pre_tokenize_str(s)] for s in en)
    same_ids = sum(a == b for a, b in zip(bids, AutoTokenizer.from_pretrained(paths[(v, 0)])(en, add_special_tokens=False)["input_ids"]))
    erows.append(dict(variant=v, n_sentences=len(en), identical_pretokens=same_pre, identical_token_ids_K0=same_ids))
pd.DataFrame(erows).to_csv("results/pretok/english_check.csv", index=False); print(pd.DataFrame(erows).to_string(index=False))

rows = []
for (v, k), p in paths.items():
    tok = AutoTokenizer.from_pretrained(p)
    for sn, sents in SETS.items():
        for mode in ["context", "alone"]: rows.append(dict(variant=v, K=k, mix=MIX if k else "none", test_set=sn, mode=mode, **stats(tok, sents, mode)))
d = pd.DataFrame(rows).round(4); d.to_csv("results/pretok/fertility.csv", index=False)
c = d[d["mode"] == "context"]
langs5 = ["Bhojpuri (BHTB)", "Magahi (MGTB)", "Maithili (web)", "Angika (MT)", "Hindi (HDTB)"]
for m in ["fertility", "pct_tokens_mid_cluster", "pct_words_split"]:
    print(m); print(c.pivot_table(index=["variant", "K"], columns="test_set", values=m).round(2)[langs5 + ["English (2k)"]].to_string())
base_f = c[(c.variant == "a") & (c.K == 0)].set_index("test_set").fertility
h = c[c.K == K].copy(); h["delta_vs_base"] = h.apply(lambda r: r.fertility - base_f[r.test_set], axis=1)
chg = {}
for sn in ["Hindi (HDTB)", "English (2k)"]:
    b0 = AutoTokenizer.from_pretrained(paths[("a", 0)])(SETS[sn], add_special_tokens=False)["input_ids"]
    for (v, k), p in paths.items():
        ids = AutoTokenizer.from_pretrained(p)(SETS[sn], add_special_tokens=False)["input_ids"]
        chg[(v, k, sn)] = round(100 * np.mean([x != y for x, y in zip(ids, b0)]), 1)
h["pct_sents_changed_K8k"] = h.apply(lambda r: chg.get((r.variant, K, r.test_set)), axis=1)
h["pct_sents_changed_K0"] = h.apply(lambda r: chg.get((r.variant, 0, r.test_set)), axis=1)
R = h[h.test_set.isin(["Hindi (HDTB)", "English (2k)"])][["variant", "test_set", "fertility", "delta_vs_base", "pct_sents_changed_K0", "pct_sents_changed_K8k"]].round(4)
R.to_csv("results/pretok/regression.csv", index=False); print(R.to_string(index=False))

plt.rcParams.update({"font.size": 13}); fig, ax = plt.subplots(1, 2, figsize=(17, 6)); cols = {"a": "#7a7a7a", "b": "#3a9d5d", "c": "#b04a7f"}
names = {"a": "a original", "b": "b marks attached", "c": "c akshara"}
for a, m, t in zip(ax, ["fertility", "pct_tokens_mid_cluster"], ["tokens per word", "% tokens ending inside a grapheme cluster"]):
    for i, v in enumerate("abc"):
        y = [c[(c.variant == v) & (c.K == K) & (c.test_set == s)][m].iloc[0] for s in langs5]
        a.bar(np.arange(5) + (i - 1) * 0.27, y, 0.27, label=names[v], color=cols[v])
    a.set_xticks(range(5)); a.set_xticklabels([s.split(" (")[0] for s in langs5]); a.set_ylabel(t)
ax[0].axhline(2, color="k", lw=1); ax[0].legend(frameon=False); fig.suptitle(f"K = 8k, mix = {MIX}", x=0.02, ha="left")
fig.tight_layout(); fig.savefig("results/pretok/pretok_variants.png", dpi=200)
