"""Stage 1B: scoped fix + continued BPE (K=1000, 8000, filtered lowonly mix) for byte-level BPE tokenizers that cut at marks."""
import time, shutil, unicodedata as U
from s1b_lib import *
import ext_bpe_train as T
from sweep_lib import target_plan
from pretok_lib import ORIG_LETTER, _dm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
t0 = time.time(); R = "results/scan"; SC = "artifacts/tok/scan"; os.makedirs(SC, exist_ok=True)
base = pd.read_csv(f"{R}/scan_base.csv"); sets = all_sets(); en_test = DA.load("en", "test"); sets_en = {"FLORES eng": sets.pop("FLORES eng"), "English (2k)": en_test}
SCOPED = f"[^\\r\\n\\p{{L}}\\p{{N}}{_dm}]?[\\p{{L}}{_dm}]+"
plan, lines = target_plan("lowonly", "train_f"); KS = [0, 1000, 8000]
def norm(tok, s):
    n = tok.backend_tokenizer.normalizer; return n.normalize_str(s) if n is not None else s
def load_file(d): return PreTrainedTokenizerFast(tokenizer_file=d + "/tokenizer.json")
def measure(ad, sets_):
    out = {}
    for sn, S in sets_.items():
        nw, nt, m_, d_ = token_stats(ad, S); b = boot(nw, nt, np.zeros_like(nt)); out[sn] = (b, m_, d_)
    return out
rows, summ = [], []
elig = base[(base.type == "byte-level BPE with regex") & base.cuts_at_marks & base.regex.fillna("").str.contains(ORIG_LETTER, regex=False)]
print("eligible:", elig.tokenizer.tolist(), flush=True)
for hid in elig.tokenizer:
    tt0 = time.time(); name = hid.replace("/", "__"); d0 = f"{SC}/{name}/orig"; os.makedirs(d0, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(hid); tok.save_pretrained(d0)
    j = json.load(open(d0 + "/tokenizer.json")); items = j["pre_tokenizer"].get("pretokenizers", [j["pre_tokenizer"]]); n_rep = 0
    for it in items:
        if it["type"] == "Split" and ORIG_LETTER in it["pattern"].get("Regex", ""): it["pattern"]["Regex"] = it["pattern"]["Regex"].replace(ORIG_LETTER, SCOPED); n_rep += 1
    assert n_rep == 1, (hid, n_rep)
    dv = f"{SC}/{name}/scoped_K0"; os.makedirs(dv, exist_ok=True); json.dump(j, open(dv + "/tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)
    tv = load_file(dv); inv = {v: k for k, v in j["model"]["vocab"].items()}
    cnt, _ = T.pretok_counts(tv, plan, lines); words = list(cnt); mdl = tv.backend_tokenizer.model
    seqs = [[t.id for t in mdl.tokenize(w)] for w in words]; merges = T.learn(seqs, [cnt[w] for w in words], dict(inv), 8000)
    json.dump(merges, open(f"{SC}/{name}/merges_8000.json", "w"), ensure_ascii=False)
    variants = {("orig", 0): (tok, HF(tok)), ("scoped", 0): (tv, HF(tv))}
    for K in (1000, 8000):
        out = f"{SC}/{name}/scoped_K{K}"; T.build(dv, merges, K, out); t_ = load_file(out); variants[("scoped", K)] = (t_, HF(t_))
    # control: original regex + continued BPE on the same mix (separates the regex effect from the merge effect)
    to = load_file(d0); cnt0, _ = T.pretok_counts(to, plan, lines); w0 = list(cnt0); m0 = to.backend_tokenizer.model
    inv0 = {v: k for k, v in json.load(open(d0 + "/tokenizer.json"))["model"]["vocab"].items()}
    merges0 = T.learn([[t.id for t in m0.tokenize(w)] for w in w0], [cnt0[w] for w in w0], dict(inv0), 8000); json.dump(merges0, open(f"{SC}/{name}/merges_orig_8000.json", "w"), ensure_ascii=False)
    for K in (1000, 8000):
        out = f"{SC}/{name}/orig_cbpe_K{K}"; T.build(d0, merges0, K, out); t_ = load_file(out); variants[("orig_cbpe", K)] = (t_, HF(t_))
    ids_orig_en = {sn: tok(S, add_special_tokens=False)["input_ids"] for sn, S in sets_en.items()}
    for (v, K), (t_, ad) in variants.items():
        for grp, S_all in [("dev", sets), ("en", sets_en)]:
            for sn, (b, m_, d_) in measure(ad, S_all).items():
                rows.append(dict(tokenizer=hid, variant=v, K=K, test_set=sn, fertility=round(float(b["fertility"]), 4), ci_lo=round(float(b["fert_lo"]), 4), ci_hi=round(float(b["fert_hi"]), 4),
                                 pct_tokens_end_inside_cluster=round(100 * m_ / max(d_, 1), 2) if grp == "dev" else None))
        allS = {**sets, **sets_en}; rt = 0
        for sn, S in allS.items():
            dec = t_.batch_decode(t_(S, add_special_tokens=False)["input_ids"], clean_up_tokenization_spaces=False); rt += sum(norm(t_, a) != b for a, b in zip(S, dec))
        r = dict(tokenizer=hid, variant=v, K=K, roundtrip_failures=rt, roundtrip_n=sum(map(len, allS.values())))
        for sn, S in sets_en.items():
            ids = t_(S, add_special_tokens=False)["input_ids"]; r[f"pct_sents_changed {sn}"] = round(100 * np.mean([a != b for a, b in zip(ids, ids_orig_en[sn])]), 2)
        summ.append(r)
    print(hid, f"{time.time()-tt0:.0f}s", [x for x in summ if x["tokenizer"] == hid and x["K"] == 8000], flush=True)
pd.DataFrame(rows).to_csv(f"{R}/fix_long.csv", index=False); pd.DataFrame(summ).to_csv(f"{R}/fix_summary.csv", index=False)
