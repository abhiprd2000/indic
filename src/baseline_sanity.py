"""Check Qwen3/Gemma3 fertility on FLORES+ devtest vs published numbers. Needs HF_TOKEN."""
import json, pandas as pd
from huggingface_hub import hf_hub_download as dl
from transformers import AutoTokenizer
PUB = {"Qwen3": {"hin": 4.11, "bho": 4.43, "mag": 4.90}, "Gemma3": {"hin": 1.59, "bho": 1.76, "mag": 1.74}}
TOK = {"Qwen3": "Qwen/Qwen3-1.7B", "Gemma3": "google/gemma-3-4b-pt"}
rows = []
for tn, tid in TOK.items():
    t = AutoTokenizer.from_pretrained(tid)
    for l, pub in PUB[tn].items():
        p = dl("openlanguagedata/flores_plus", f"devtest/{l}_Deva.jsonl", repo_type="dataset", local_dir="data/raw/hf")
        S = [json.loads(x)["text"] for x in open(p, encoding="utf-8")]
        W = [w for s in S for w in s.split()]
        V = {"context": sum(len(t.tokenize(s)) for s in S), "alone": sum(len(t.tokenize(w)) for w in W),
             "alone_space": sum(len(t.tokenize(" " + w)) for w in W)}
        for m, nt in V.items():
            f = nt / len(W)
            rows.append(dict(tokenizer=tn, lang=l, mode=m, n_sents=len(S), ours=round(f, 2), published=pub, diff=round(f - pub, 2)))
df = pd.DataFrame(rows); df.to_csv("results/baseline/sanity.csv", index=False); print(df.to_string(index=False))
