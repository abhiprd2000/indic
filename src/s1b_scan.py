"""Stage 1B: scan of open tokenizers (measurement only)."""
import time, sys
from s1b_lib import *
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download as dl
R = "results/scan"; os.makedirs(R, exist_ok=True); t0 = time.time()
HFIDS = ["Qwen/Qwen2.5-7B", "Qwen/Qwen3-1.7B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "google/gemma-3-4b-pt", "mistralai/Mistral-Nemo-Base-2407",
         "microsoft/phi-4", "allenai/Olmo-3-1025-7B", "deepseek-ai/DeepSeek-V3", "HuggingFaceTB/SmolLM3-3B", "xlm-roberta-base", "google/mt5-base", "ai4bharat/IndicBERTv2-MLM-only"]
def describe(tok):
    j = json.loads(tok.backend_tokenizer.to_str()); m = j["model"]; pre = j.get("pre_tokenizer"); norm = j.get("normalizer")
    items = (pre or {}).get("pretokenizers", [pre] if pre else []); rx = []; bl = False; usebl = False
    for it in items:
        if it["type"] == "Split":
            p = it["pattern"]; rx.append(p.get("Regex") or p.get("String"))
        if it["type"] == "ByteLevel": bl = True; usebl = it.get("use_regex", False)
    if usebl and not rx: rx.append(r"GPT-2 default: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
    if m["type"] == "BPE": typ = "byte-level BPE with regex" if bl and rx else "byte-level BPE (no regex)" if bl else "SentencePiece BPE" + (" (byte fallback)" if m.get("byte_fallback") else "")
    elif m["type"] == "Unigram": typ = "SentencePiece unigram"
    elif m["type"] == "WordPiece": typ = "WordPiece"
    else: typ = m["type"]
    return typ, " || ".join(rx), (pre or {}).get("type", "none"), (norm or {}).get("type", "none"), j
rows, long, trig_rows, skipped, sets = [], [], [], [], all_sets()
print("sets", {k: len(v) for k, v in sets.items()}, flush=True)
adapters = []
for hid in HFIDS:
    try:
        tok = AutoTokenizer.from_pretrained(hid)
        if not getattr(tok, "is_fast", False): raise RuntimeError("no fast tokenizer")
        typ, rx, pret, norm, j = describe(tok); note = ""
        try:
            raw = json.load(open(dl(hid, "tokenizer.json"))); same = json.dumps(raw.get("pre_tokenizer"), sort_keys=True) == json.dumps(j.get("pre_tokenizer"), sort_keys=True)
            note = "" if same else "AutoTokenizer pre-tokenizer differs from repo tokenizer.json"
        except Exception: note = "no tokenizer.json to compare"
        adapters.append((hid, HF(tok), typ, rx, pret, norm, note, tok))
    except Exception as e: skipped.append((hid, f"{type(e).__name__}: {str(e)[:80]}"))
try:
    import tiktoken; enc = tiktoken.get_encoding("o200k_base"); adapters.append(("tiktoken o200k_base", TT(enc), "tiktoken (BPE over bytes with regex)", enc._pat_str, "regex", "none", "", None))
except Exception as e: skipped.append(("tiktoken o200k_base", f"{type(e).__name__}: {str(e)[:80]}"))
print("loaded", [a[0] for a in adapters], "skipped", skipped, flush=True)
for hid, ad, typ, rx, pret, norm, note, tok in adapters:
    row = dict(tokenizer=hid, vocab_size=ad.n_vocab(), type=typ, pre_tokenizer=pret, normalizer=norm, regex=rx, note=note)
    nwords = nsplit = nmark = mid = dtok = 0; trig = collections.Counter()
    for sn, S in sets.items():
        nw, nt, m_, d_ = token_stats(ad, S); b = boot(nw, nt, np.zeros_like(nt))
        row[f"fert {sn}"] = round(float(b["fertility"]), 3); row[f"lo {sn}"] = round(float(b["fert_lo"]), 3); row[f"hi {sn}"] = round(float(b["fert_hi"]), 3)
        long.append(dict(tokenizer=hid, test_set=sn, fertility=round(float(b["fertility"]), 4), ci_lo=round(float(b["fert_lo"]), 4), ci_hi=round(float(b["fert_hi"]), 4), n_sents=len(S)))
        if sn in DEVSETS:
            n, sa, sm, tr = cut_stats(ad, S); nwords += n; nsplit += sa; nmark += sm; trig += tr; mid += m_; dtok += d_
    row.update(devanagari_words=nwords, pct_words_split=round(100 * nsplit / nwords, 2), pct_words_cut_at_mark=round(100 * nmark / nwords, 2), pct_tokens_end_inside_cluster=round(100 * mid / max(dtok, 1), 2))
    tot = sum(trig.values()); top = trig.most_common(8)
    row["top_cut_chars"] = "; ".join(f"U+{ord(c):04X} {U.name(c, '?')} {100*n/tot:.1f}%" for c, n in top) if tot else "none"
    for c, n in top: trig_rows.append(dict(tokenizer=hid, codepoint=f"U+{ord(c):04X}", name=U.name(c, "?"), count=n, pct_of_cuts=round(100 * n / tot, 2)))
    row["cuts_at_marks"] = row["pct_words_cut_at_mark"] >= 10.0; rows.append(row)
    print(hid, row["vocab_size"], typ, "split", row["pct_words_split"], "cut@mark", row["pct_words_cut_at_mark"], f"{time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(f"{R}/scan_base.csv", index=False); pd.DataFrame(long).to_csv(f"{R}/scan_fertility_long.csv", index=False); pd.DataFrame(trig_rows).to_csv(f"{R}/scan_cut_chars.csv", index=False)
pd.DataFrame(skipped, columns=["tokenizer", "reason"]).to_csv(f"{R}/scan_skipped.csv", index=False); print("skipped:", skipped)
