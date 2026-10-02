"""Stage 1B addition 2: variant d at K=8000 with danda (U+0964, U+0965) forbidden inside new tokens."""
import time
from s1a_lib import *
import data_access as DA
from baseline_fertility import get_dev_sets
t0 = time.time(); R = "results/scan"; S1B = "artifacts/tok/s1b"; os.makedirs(S1B, exist_ok=True)
assert os.environ.get("FINAL_EVAL") != "1", "dev-only script"
FWD = {b: u for u, b in B2U.items()}
def dstr(cp): return "".join(FWD[b] for b in chr(cp).encode())
DAN = [dstr(0x964), dstr(0x965)]; BARE = set(DAN)
forbid = lambda s: any(x in s for x in DAN) and s not in BARE  # the bare danda token may exist; nothing may be merged with it
plan, lines = target_plan("lowonly", "train_f"); tok0 = load(vdir("d")); cnt, _ = T.pretok_counts(tok0, plan, lines)
mp = f"{S1B}/merges_d_nodanda.json"
if os.path.exists(mp): merges = json.load(open(mp))
else:
    words = list(cnt); mdl = tok0.backend_tokenizer.model; seqs = [[t.id for t in mdl.tokenize(w)] for w in words]
    merges = T.learn(seqs, [cnt[w] for w in words], inv_vocab(), 8000, forbid=forbid); json.dump(merges, open(mp, "w"), ensure_ascii=False)
print("merges", len(merges), "with danda:", sum(any(x in a + b for x in DAN) and (a + b) not in BARE for a, b in merges), f"{time.time()-t0:.0f}s", flush=True)
std = json.load(open(f"{S1}/merges_lowonly_d.json")); print("std d K=8000 tokens containing danda:", sum(any(x in a + b for x in DAN) and (a + b) not in BARE for a, b in std[:8000]))
out = f"{S1B}/d_nodanda_K8000"; T.build(vdir("d"), merges, 8000, out); t_nd = load(out); t_d = load(build_s1("d", "lowonly", std, 8000)); base = load(vdir("a"))
dev = dict(get_dev_sets())
for code, name in [("bho_Deva", "FLORES dev bho"), ("mai_Deva", "FLORES dev mai"), ("mag_Deva", "FLORES dev mag"), ("hin_Deva", "FLORES dev hin"), ("eng_Latn", "FLORES dev eng")]: dev[name] = DA.load_flores(code, "dev")
rows = []
for sn, S in dev.items():
    a, b = eval_tok(t_d, {sn: S})[sn], eval_tok(t_nd, {sn: S})[sn]
    rows.append(dict(kind="fertility", set=sn, fert_d=round(float(a["fertility"]), 4), fert_nodanda=round(float(b["fertility"]), 4), diff=round(float(b["fertility"] - a["fertility"]), 4)))
FL = ["ben_Beng", "tam_Taml", "urd_Arab", "tha_Thai", "jpn_Jpan", "arb_Arab", "fra_Latn", "spa_Latn", "eng_Latn", "hin_Deva"]
for l in FL:
    S = DA.load_flores(l, "dev"); bi = base(S, add_special_tokens=False)["input_ids"]
    r = dict(kind="pct_sents_changed_vs_base", set=l)
    for nm, t in [("d", t_d), ("nodanda", t_nd)]: r[f"pct_{nm}"] = round(100 * np.mean([x != y for x, y in zip(t(S, add_special_tokens=False)["input_ids"], bi)]), 2)
    rows.append(r)
d = pd.DataFrame(rows); d.to_csv(f"{R}/danda_ablation.csv", index=False)
f = d[d.kind == "fertility"]; print(f[["set", "fert_d", "fert_nodanda", "diff"]].to_string(index=False)); print("max |diff|:", f["diff"].abs().max())
print(d[d.kind != "fertility"][["set", "pct_d", "pct_nodanda"]].to_string(index=False))
log_time("s1b_danda", t0)
