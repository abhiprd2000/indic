"""Stage 1B: merge scan + fix results into results/scan/scan.csv and scan.md (one row per tokenizer)."""
import os, json, hashlib
import pandas as pd
R = "results/scan"; b = pd.read_csv(f"{R}/scan_base.csv"); f = pd.read_csv(f"{R}/fix_long.csv"); s = pd.read_csv(f"{R}/fix_summary.csv")
b["regex_has_marks"] = b.regex.fillna("").str.contains(r"\p{M}", regex=False)
def reason(r):
    if r.cuts_at_marks and r.type == "byte-level BPE with regex": return "applied"
    if r.type.startswith("byte-level") or r.type.startswith("tiktoken"):
        return f"not needed: pre-tokenizer does not cut at marks ({r.pct_words_cut_at_mark}% of Devanagari words); regex letter class includes \\p{{M}}: {r.regex_has_marks}"
    return f"not applicable: {r.type} has no regex pre-tokenizer that cuts at marks; continued BPE with a scoped regex does not apply"
b["fix_status"] = b.apply(reason, axis=1)
def h(name):
    p = f"artifacts/tok/scan/{name.replace('/', '__')}/orig/tokenizer.json"
    return hashlib.md5(json.dumps(json.load(open(p))["model"]["merges"]).encode()).hexdigest()[:8] if os.path.exists(p) else ""
b["merges_hash"] = b.tokenizer.map(h)
def val(t, v, K, sn, col="fertility"):
    x = f[(f.tokenizer == t) & (f.variant == v) & (f.K == K) & (f.test_set == sn)]; return x[col].iloc[0] if len(x) else None
for sn, tag in [("FLORES bho", "flores_bho"), ("Bhojpuri (BHTB)", "bhtb"), ("FLORES hin", "flores_hin")]:
    for v, K, nm in [("orig", 0, "orig_K0"), ("scoped", 0, "scoped_K0"), ("orig_cbpe", 1000, "origCBPE_K1000"), ("scoped", 1000, "scoped_K1000"), ("orig_cbpe", 8000, "origCBPE_K8000"), ("scoped", 8000, "scoped_K8000")]:
        b[f"fix {tag} {nm}"] = b.tokenizer.map(lambda t: val(t, v, K, sn))
for v, K, nm in [("orig", 0, "orig"), ("scoped", 8000, "scoped_K8000"), ("orig_cbpe", 8000, "origCBPE_K8000")]:
    b[f"fix pct_tokens_end_inside_cluster flores_bho {nm}"] = b.tokenizer.map(lambda t: val(t, v, K, "FLORES bho", "pct_tokens_end_inside_cluster"))
sc = s[s.variant.isin(["scoped", "orig_cbpe"])]
b["fix roundtrip_failures_total"] = b.tokenizer.map(lambda t: int(sc[sc.tokenizer == t].roundtrip_failures.sum()) if (sc.tokenizer == t).any() else None)
b["fix roundtrip_checks_total"] = b.tokenizer.map(lambda t: int(sc[sc.tokenizer == t].roundtrip_n.sum()) if (sc.tokenizer == t).any() else None)
b["fix english_pct_sents_changed_max"] = b.tokenizer.map(lambda t: float(sc[sc.tokenizer == t][["pct_sents_changed FLORES eng", "pct_sents_changed English (2k)"]].max().max()) if (sc.tokenizer == t).any() else None)
b["fix english_fertility_delta_scoped_K8000"] = b.tokenizer.map(lambda t: round(val(t, "scoped", 8000, "FLORES eng") - val(t, "orig", 0, "FLORES eng"), 4) if (f.tokenizer == t).any() else None)
b.to_csv(f"{R}/scan.csv", index=False)
def fm(r, sn): return f"{r[f'fert {sn}']:.2f} [{r[f'lo {sn}']:.2f}, {r[f'hi {sn}']:.2f}]"
L = ["# Tokenizer scan (Devanagari: BHTB, MGTB, Maithili web, Angika MT, HDTB, FLORES+ devtest bho/mag/mai/hin; measurement only)", "",
     "| Tokenizer | Vocab | Type | Regex has \\p{M} | Fert BHTB [95% CI] | Fert FLORES bho | Words split % | Cut at mark % | Tokens ending inside cluster % | Top cut characters | Fix (FLORES bho fert: orig, orig+cBPE K8000, scoped K8000) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in b.iterrows():
    top = "; ".join(x.split(" ")[0] + " " + x.split(" ")[-1] for x in str(r.top_cut_chars).split("; ")[:3]) if r.top_cut_chars != "none" else "none"
    fx = f"{r['fix flores_bho orig_K0']:.2f}, {r['fix flores_bho origCBPE_K8000']:.2f}, {r['fix flores_bho scoped_K8000']:.2f}" if r.fix_status == "applied" else r.fix_status.split(":")[0]
    L.append(f"| {r.tokenizer} | {r.vocab_size} | {r.type} | {r.regex_has_marks} | {fm(r, 'Bhojpuri (BHTB)')} | {fm(r, 'FLORES bho')} | {r.pct_words_split} | {r.pct_words_cut_at_mark} | {r.pct_tokens_end_inside_cluster} | {top} | {fx} |")
L += ["", "Fix status per tokenizer:", ""] + [f"- {r.tokenizer}: {r.fix_status}" for _, r in b.iterrows()] + ["", "Notes: " + "; ".join(f"{r.tokenizer}: {r.note}" for _, r in b.iterrows() if isinstance(r.note, str) and r.note)]
open(f"{R}/scan.md", "w", encoding="utf-8").write("\n".join(L) + "\n"); print(open(f"{R}/scan.md", encoding="utf-8").read()[:3000])
