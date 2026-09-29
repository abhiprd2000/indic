"""Fetch UD, FineWeb2, AngikaMT, data/mine. NFC, split, write sizes.csv. Needs HF_TOKEN env for AngikaMT."""
import os, re, glob, hashlib, random, unicodedata as U
import pandas as pd, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download as dl

SEED = 0
HIN_N = 500_000
TOK = os.environ.get("HF_TOKEN")
RAW = "data/raw/hf"
LANGS = ["bho", "mai", "mag", "ang", "hin"]
FW = {"bho": "bho_Deva", "mai": "mai_Deva", "mag": "mag_Deva", "hin": "hin_Deva"}
UD = {"bho": ["bho_bhtb-ud-test"], "mag": ["mag_mgtb-ud-test"],
      "hin": ["hi_hdtb-ud-train", "hi_hdtb-ud-dev", "hi_hdtb-ud-test"]}
FORCE = {"train": "tok_train", "dev": "dev", "test": "test"}
DEV = re.compile(r"[ऀ-ॿ]")
SPLIT = re.compile(r"(?<=[।॥?!.])\s+|\n+")

nfc = lambda s: U.normalize("NFC", s).strip()

def sents(text):
    for s in SPLIT.split(nfc(text)):
        s = " ".join(s.split())
        if len(s.split()) >= 3 and DEV.search(s):
            yield s

def fw_docs(lang):
    """FineWeb2 docs: full small subsets; random row groups of one shard for hin."""
    rng = random.Random(SEED)
    docs = []
    for part in ["train", "test"]:
        f = f"data/{FW[lang]}/{part}/000_00000.parquet"
        p = dl("HuggingFaceFW/fineweb-2", f, repo_type="dataset", local_dir=RAW, token=TOK)
        pf = pq.ParquetFile(p)
        if lang == "hin":
            if part == "test": continue
            order = list(range(pf.num_row_groups)); rng.shuffle(order)
            n = 0
            for g in order:
                for t in pf.read_row_group(g, columns=["text"]).column("text").to_pylist():
                    d = list(sents(t)); docs.append(d); n += len(d)
                if n >= HIN_N: break
        else:
            for t in pf.read(columns=["text"]).column("text").to_pylist():
                docs.append(list(sents(t)))
    if lang == "hin":  # cut to exactly HIN_N sentences
        rng.shuffle(docs); out, n = [], 0
        for d in docs:
            if n >= HIN_N: break
            d = d[: HIN_N - n]; out.append(d); n += len(d)
        docs = out
    return [("fineweb2", d) for d in docs if d]

def ud_docs(lang):
    os.makedirs("data/ud", exist_ok=True)
    forced = []
    for name in UD.get(lang, []):
        raw = open(f"data/raw/ud/{name}.conllu", encoding="utf-8").read()
        open(f"data/ud/{name}.conllu", "w", encoding="utf-8").write(U.normalize("NFC", raw))
        part = name.rsplit("-", 1)[1]
        for l in U.normalize("NFC", raw).splitlines():
            if l.startswith("# text ="):
                s = " ".join(l[8:].split())
                if s: forced.append((f"ud_{name}", s, FORCE[part]))
    return forced

def ang_docs():
    out = []
    for f in ["train.csv", "dev.csv"]:
        p = dl("snjev310/AngikaMT", f, repo_type="dataset", local_dir=RAW, token=TOK)
        for t in pd.read_csv(p)["angika"].dropna():
            d = list(sents(t))
            if d: out.append(("angikamt", d))
    return out

def mine_docs(lang):
    p = f"data/mine/{lang}.txt"
    if not os.path.exists(p): return []
    return [("mine", list(sents(l))) for l in open(p, encoding="utf-8") if l.strip()]

def split_lang(lang, docs, forced):
    rng = random.Random(SEED)
    seen, sp = set(), {"tok_train": [], "dev": [], "test": []}
    src = {}
    for s, x, part in forced:  # UD official splits first
        if x not in seen: seen.add(x); sp[part].append(x); src[(s, part)] = src.get((s, part), 0) + 1
    tot = sum(len(d) for _, d in docs) + sum(len(v) for v in sp.values())
    nt, nd = (2000, 1000) if tot >= 10000 else (int(.2 * tot), int(.1 * tot))
    docs = sorted(docs, key=lambda x: hashlib.md5((str(SEED) + x[1][0]).encode()).hexdigest())
    for s, d in docs:
        d = [x for x in d if x not in seen]
        if not d: continue
        part = "test" if len(sp["test"]) < nt else "dev" if len(sp["dev"]) < nd else "tok_train"
        seen.update(d); sp[part].extend(d); src[(s, part)] = src.get((s, part), 0) + len(d)
    assert not set(sp["tok_train"]) & set(sp["test"]) and not set(sp["dev"]) & set(sp["test"]), lang
    os.makedirs(f"data/{lang}", exist_ok=True)
    for k, v in sp.items():
        open(f"data/{lang}/{k}.txt", "w", encoding="utf-8").write("\n".join(v) + "\n")
    return sp, src

if __name__ == "__main__":
    rows, srcrows = [], []
    for lang in LANGS:
        docs = fw_docs(lang) if lang in FW else ang_docs()
        docs += mine_docs(lang)
        if not docs: raise SystemExit(f"no data for {lang}")
        sp, src = split_lang(lang, docs, ud_docs(lang))
        allb = sum(len(x.encode()) for v in sp.values() for x in v) / 1e6
        for k, v in sp.items():
            rows.append(dict(lang=lang, split=k, sentences=len(v), words=sum(len(x.split()) for x in v),
                             MB=round(sum(len(x.encode()) for x in v) / 1e6, 3), lang_total_MB=round(allb, 3),
                             under_5MB=allb < 5))
        srcrows += [dict(lang=lang, source=s, split=p, sentences=n) for (s, p), n in src.items()]
        print(lang, {k: len(v) for k, v in sp.items()}, f"{allb:.1f}MB")
    os.makedirs("results/data", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/data/sizes.csv", index=False)
    pd.DataFrame(srcrows).to_csv("results/data/sources.csv", index=False)
