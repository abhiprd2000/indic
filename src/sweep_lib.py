"""Shared helpers for the sweep experiments (tokenizer load, bootstrap fertility, variant training)."""
import os, re, sys, json, time
import numpy as np, pandas as pd
from transformers import PreTrainedTokenizerFast
sys.path.insert(0, "src")
import ext_bpe_train as T
from pretok_lib import REGEX, variant_dir
from baseline_fertility import get_test_sets, get_dev_sets
import data_access as DA

SEED = 0
SW = "artifacts/tok/sweep"
VARIANTS = {"a": "a_original", "b": "b_marks_attached", "d": "d_devanagari_marks"}
def test_sets():
    """Test sets incl. English; needs FINAL_EVAL=1."""
    s = dict(get_test_sets()); s["English (2k)"] = DA.load("en", "test"); return s
LANG5 = ["Bhojpuri (BHTB)", "Magahi (MGTB)", "Maithili (web)", "Angika (MT)", "Hindi (HDTB)"]
ALLSETS = LANG5 + ["English (2k)"]

def load(path): return PreTrainedTokenizerFast(tokenizer_file=path + "/tokenizer.json")  # AutoTokenizer ignores custom regex

def vdir(v):
    """Tokenizer dir for variant v at K=0."""
    d = f"{SW}/{v}_K0"
    if not os.path.exists(d): variant_dir(v, REGEX[VARIANTS[v]], out=d)
    return d

def sent_stats(tok, sents):
    """Per-sentence arrays: words, tokens, words that are exactly one token (in-context)."""
    enc = tok(sents, add_special_tokens=False, return_offsets_mapping=True)
    nw, nt, ns = [], [], []
    for s, off in zip(sents, enc["offset_mapping"]):
        spans = [m.start() for m in re.finditer(r"\S+", s)]
        ws = np.array(spans); cnt = np.zeros(len(spans), dtype=int)
        for a, e in off:
            while a < e and s[a].isspace(): a += 1
            if len(ws): cnt[max(np.searchsorted(ws, a, side="right") - 1, 0)] += 1
        nw.append(len(spans)); nt.append(len(off)); ns.append(int((cnt == 1).sum()))
    return np.array(nw), np.array(nt), np.array(ns)

_IDX = {}
def boot(nw, nt, ns, B=1000):
    """Point estimate and 95% bootstrap CI (resample sentences, seed 0) for fertility and % one-token words."""
    n = len(nw)
    if n not in _IDX: _IDX[n] = np.random.default_rng(SEED).integers(0, n, (B, n))
    i = _IDX[n]; f = nt[i].sum(1) / nw[i].sum(1); p = 100 * ns[i].sum(1) / nw[i].sum(1)
    return dict(n_sents=n, n_words=int(nw.sum()), fertility=nt.sum() / nw.sum(), fert_lo=np.percentile(f, 2.5), fert_hi=np.percentile(f, 97.5),
                pct_one_token=100 * ns.sum() / nw.sum(), one_lo=np.percentile(p, 2.5), one_hi=np.percentile(p, 97.5))

def eval_tok(tok, sets=None):
    return {sn: boot(*sent_stats(tok, s)) for sn, s in (sets or test_sets()).items()}

def target_plan(mix="lowonly", split="tok_train"):
    langs = ["hin", "bho", "mai", "mag"]
    lines = {l: T.read_lines(l, split) for l in langs}
    sizes = {l: sum(len(s.encode()) + 1 for s in lines[l]) / 1e6 for l in langs}
    ls, alpha = T.MIXES[mix]
    return T.mix_plan(ls, alpha, sizes), lines

def inv_vocab():
    return {v: k for k, v in json.load(open("artifacts/tok/base/tokenizer.json"))["model"]["vocab"].items()}

def train_merges(tok, cnt, K):
    words = list(cnt); model = tok.backend_tokenizer.model
    seqs = [[t.id for t in model.tokenize(w)] for w in words]
    return T.learn(seqs, [cnt[w] for w in words], inv_vocab(), K), len(words)

def build_k(v, merges, K):
    out = f"{SW}/{v}_K{K}"
    if K == 0: return vdir(v)
    if not os.path.exists(out + "/tokenizer.json"): T.build(vdir(v), merges, K, out)
    return out

def log_time(step, t0):
    os.makedirs("results/sweep", exist_ok=True)
    p = "results/sweep/timing.csv"; row = pd.DataFrame([dict(step=step, seconds=round(time.time() - t0))])
    row.to_csv(p, mode="a", header=not os.path.exists(p), index=False)

def flat(res, **kw):
    return [dict(test_set=sn, **kw, **{k: round(float(v), 4) for k, v in r.items()}) for sn, r in res.items()]
