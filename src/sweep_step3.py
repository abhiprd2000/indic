"""5.3 Safety of d vs b: round trip on all test sets, multi-script regression on FLORES+ devtest."""
import sys, time, json, os, unicodedata as U
sys.path.insert(0, "src")
from sweep_lib import *
from huggingface_hub import hf_hub_download as dl
t0 = time.time()
K = 8000
toks = {("a", 0): load(vdir("a"))}
for v in VARIANTS:
    toks[(v, K)] = load(build_k(v, json.load(open(f"{SW}/merges_{v}.json")), K))
    if v != "a": toks[(v, 0)] = load(vdir(v))
rows = []
# (i) round trip
for (v, k), tok in toks.items():
    if k not in (0, K) or (v != "a" and k == 0): continue
    for sn, sents in SETS.items():
        dec = tok.batch_decode(tok(sents, add_special_tokens=False)["input_ids"], clean_up_tokenization_spaces=False)
        fail = [(s, d) for s, d in zip(sents, dec) if d != s]
        nfc = sum(1 for s, d in fail if d == U.normalize("NFC", s))
        rows.append(dict(check="roundtrip", variant=v, K=k, lang=sn, n=len(sents), failures=len(fail), failures_only_nfc=nfc))
print(pd.DataFrame(rows)[["variant", "K", "lang", "n", "failures", "failures_only_nfc"]].query("failures>0").to_string(index=False) or "roundtrip: 0 failures", flush=True)
print("roundtrip failures total:", sum(r["failures"] for r in rows), flush=True)
# (ii) multi-script
FL = ["eng_Latn", "fra_Latn", "spa_Latn", "ben_Beng", "tam_Taml", "urd_Arab", "tha_Thai", "jpn_Jpan", "arb_Arab", "hin_Deva"]
skipped, fl = [], {}
for l in FL:
    try:
        p = dl("openlanguagedata/flores_plus", f"devtest/{l}.jsonl", repo_type="dataset", local_dir="data/raw/hf", token=os.environ.get("HF_TOKEN"))
        fl[l] = [json.loads(x)["text"] for x in open(p, encoding="utf-8")]
    except Exception as e: skipped.append((l, type(e).__name__))
print("skipped FLORES+ langs:", skipped)
base = toks[("a", 0)]
for l, S in fl.items():
    bids = base(S, add_special_tokens=False)["input_ids"]; bt = sum(map(len, bids)); nw = sum(len(s.split()) for s in S)
    for (v, k), tok in toks.items():
        ids = tok(S, add_special_tokens=False)["input_ids"]; nt = sum(map(len, ids))
        rows.append(dict(check="multiscript", variant=v, K=k, lang=l, n=len(S), pct_sents_changed=round(100 * np.mean([a != b for a, b in zip(ids, bids)]), 2),
                         fertility=round(nt / nw, 4), base_fertility=round(bt / nw, 4), token_change_pct=round(100 * (nt / bt - 1), 3)))
d = pd.DataFrame(rows); d.to_csv("results/sweep/safety.csv", index=False)
m = d[d.check == "multiscript"]
print(m[m.K == 0].pivot_table(index="lang", columns="variant", values="pct_sents_changed").to_string())
print(m[m.K == K].pivot_table(index="lang", columns="variant", values="pct_sents_changed").to_string())
print(m[m.K == K].pivot_table(index="lang", columns="variant", values="token_change_pct").to_string())
log_time("5.3", t0)
