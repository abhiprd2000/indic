"""Baseline fertility on test sets. Needs HF_TOKEN env for gated tokenizers."""
import os, sys, unicodedata as U
import pandas as pd
from transformers import AutoTokenizer

TOKS = {"Qwen3": "Qwen/Qwen3-1.7B", "Gemma3": "google/gemma-3-4b-pt", "Llama3.1": "meta-llama/Llama-3.1-8B",
        "XLM-R": "xlm-roberta-base", "mBERT": "bert-base-multilingual-cased",
        "IndicBERTv2": "ai4bharat/IndicBERTv2-MLM-only"}
def ud_text(f):
    return [l[8:].strip() for l in open(f, encoding="utf-8") if l.startswith("# text =")]
def txt(f):
    return [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
SETS = {"Bhojpuri (BHTB)": ud_text("data/ud/bho_bhtb-ud-test.conllu"),
        "Magahi (MGTB)": ud_text("data/ud/mag_mgtb-ud-test.conllu"),
        "Hindi (HDTB)": ud_text("data/ud/hi_hdtb-ud-test.conllu"),
        "Maithili (web)": txt("data/mai/test.txt"),
        "Angika (MT)": txt("data/ang/test.txt")}

def is_mark(c): return U.category(c) in ("Mn", "Mc")

def enc(tok, s):
    """Return (ends, mid_flags, word_index) per token; ends are char offsets."""
    if tok == "bytes":
        b, out, ci = s.encode(), [], 0
        pos = []  # char index of each byte
        for i, ch in enumerate(s): pos += [i] * len(ch.encode())
        n = len(b)
        starts = [pos[j] for j in range(n)]
        mid = []
        for j in range(n):
            nxt = j + 1
            if nxt >= n: mid.append(False)
            else: mid.append(pos[nxt] == pos[j] or is_mark(s[pos[nxt]]))
        return [pos[j] for j in range(n)], mid
    r = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    off = r["offset_mapping"]
    # trim leading space in offsets (sentencepiece)
    fixed = []
    for a, e in off:
        while a < e and s[a].isspace(): a += 1
        fixed.append((a, e))
    n, mid = len(fixed), []
    for i, (a, e) in enumerate(fixed):
        if i + 1 >= n or e >= len(s): mid.append(False); continue
        na = fixed[i + 1][0]
        mid.append(na < e or is_mark(s[e]))
    return [a for a, e in fixed], mid

def stats(tok, sents, mode):
    nw = nt = nsplit = mids = chars = byts = 0
    cache = {}
    for s in sents:
        words = s.split()
        nw += len(words)
        if mode == "alone":
            for w in words:
                if w not in cache:
                    ends, mid = enc(tok, w); cache[w] = (len(ends), sum(mid))
                k, m = cache[w]
                nt += k; nsplit += k >= 2; mids += m
                chars += len(w); byts += len(w.encode())
        else:
            ends, mid = enc(tok, s)
            nt += len(ends); mids += sum(mid); chars += len(s); byts += len(s.encode())
            # words split: count tokens per word by token start char
            starts = ends
            bounds, p = [], 0
            for w in words:
                p = s.index(w, p); bounds.append((p, p + len(w))); p += len(w)
            cnt = [0] * len(words); wi = 0
            for a in starts:
                while wi < len(words) - 1 and a >= bounds[wi][1]: wi += 1
                cnt[wi] += 1
            nsplit += sum(c >= 2 for c in cnt)
    return dict(n_words=nw, n_tokens=nt, fertility=nt / nw, pct_words_split=100 * nsplit / nw,
                chars_per_token=chars / nt, bytes_per_token=byts / nt, pct_tokens_mid_cluster=100 * mids / nt)

if __name__ == "__main__":
    rows, skipped = [], []
    for name in list(TOKS) + ["Bytes"]:
        if name == "Bytes": tok = "bytes"
        else:
            try: tok = AutoTokenizer.from_pretrained(TOKS[name])
            except Exception as e: skipped.append((name, type(e).__name__)); continue
        for sn, sents in SETS.items():
            for mode in ["alone", "context"]:
                rows.append(dict(tokenizer=name, test_set=sn, mode=mode, **stats(tok, sents, mode)))
        print("done", name, flush=True)
    os.makedirs("results/baseline", exist_ok=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv("results/baseline/fertility.csv", index=False)
    print("skipped:", skipped)
    print(df[df["mode"] == "context"].pivot(index="tokenizer", columns="test_set", values="fertility").round(2).to_string())
