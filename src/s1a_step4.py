"""Stage 1A step 4: merges at K=32000 for a and d on the filtered lowonly mix; script check, round trip, FLORES+ dev safety."""
import time, unicodedata as U
from s1a_lib import *
import data_access as DA
t0 = time.time(); MIX = "lowonly"; R = "results/stage1a"; os.makedirs(R, exist_ok=True)
MERGES, rows_chk, mixrows = {}, [], []
for v in ["a", "d"]:
    MERGES[v], rows = train_mix(v, MIX, 32000); mixrows += [dict(variant=v, mix=MIX, **r) for r in rows]
    c = status_counts(MERGES[v], 32000); rows_chk.append(dict(variant=v, K=32000, **{k: c[k] for k in ["ok", "full", "fragment", "mark_other", "ambiguous"]}))
    for K in KS[1:]:
        c = status_counts(MERGES[v], K); rows_chk.append(dict(variant=v, K=K, **{k: c[k] for k in ["ok", "full", "fragment", "mark_other", "ambiguous"]}))
    print(v, "merges", len(MERGES[v]), f"{time.time()-t0:.0f}s", flush=True)
chk = pd.DataFrame(rows_chk); chk.to_csv(f"{R}/newtoken_check.csv", index=False); pd.DataFrame(mixrows).to_csv(f"{R}/mixes_filtered.csv", index=False)
print(chk[chk.K.isin([8000, 32000])].to_string(index=False)); assert chk[chk.K <= 16000].full.sum() == 0 and chk[chk.K <= 16000].fragment.sum() == 0, "new tokens with letters outside Devanagari/Latin"
# round trip on every dev/test set (test via audit purpose: integrity only)
sets = {f"{l} dev": DA.load(l, "dev") for l in ["bho", "mai", "mag", "ang", "hin", "en"]}
sets.update({f"{l} test": DA.load(l, "test", purpose="audit") for l in ["bho", "mai", "mag", "ang", "hin", "en"]})
sets.update({f"UD {n}": DA.load_ud(n, purpose="audit") for n in ["bho_bhtb-ud-test", "mag_mgtb-ud-test", "hi_hdtb-ud-test"]})
FL = ["ben_Beng", "tam_Taml", "urd_Arab", "tha_Thai", "jpn_Jpan", "arb_Arab", "fra_Latn", "spa_Latn", "eng_Latn", "hin_Deva"]
fl, skipped = {}, []
for l in FL:
    try: fl[l] = DA.load_flores(l, "dev")
    except Exception as e: skipped.append((l, type(e).__name__))
print("FLORES+ dev skipped:", skipped)
toks = {("a", 0): load(vdir("a")), ("d", 0): load(vdir("d"))}; rt, fs = [], []
for v in ["a", "d"]:
    for K in KS[1:]: toks[(v, K)] = load(build_s1(v, MIX, MERGES[v], K))
base = toks[("a", 0)]
for (v, K), tok in toks.items():
    for sn, S in {**sets, **{f"FLORES dev {l}": s for l, s in fl.items()}}.items():
        dec = tok.batch_decode(tok(S, add_special_tokens=False)["input_ids"], clean_up_tokenization_spaces=False)
        rt.append(dict(variant=v, K=K, set=sn, n=len(S), failures_raw=sum(a != b for a, b in zip(S, dec)), failures_vs_nfc=sum(U.normalize("NFC", a) != b for a, b in zip(S, dec)), n_not_nfc=sum(U.normalize("NFC", a) != a for a in S)))
pd.DataFrame(rt).to_csv(f"{R}/roundtrip.csv", index=False); print("roundtrip failures vs NFC text:", sum(r["failures_vs_nfc"] for r in rt), "| raw:", sum(r["failures_raw"] for r in rt), "over", len(rt), "checks (tokenizer applies NFC)")
for l, S in fl.items():
    bids = base(S, add_special_tokens=False)["input_ids"]; bt = sum(map(len, bids))
    for (v, K), tok in toks.items():
        ids = tok(S, add_special_tokens=False)["input_ids"]
        fs.append(dict(lang=l, variant=v, K=K, n=len(S), pct_sents_changed=round(100 * np.mean([a != b for a, b in zip(ids, bids)]), 2), token_change_pct=round(100 * (sum(map(len, ids)) / bt - 1), 3)))
f = pd.DataFrame(fs); f.to_csv(f"{R}/flores_safety.csv", index=False)
print(f[f.variant == "d"].pivot_table(index="lang", columns="K", values="pct_sents_changed").to_string())
print(f[(f.variant == "a") & (f.K == 8000)][["lang", "pct_sents_changed"]].to_string(index=False))
log_time("s1a_4", t0)
