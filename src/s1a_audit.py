"""Stage 1A step 2: audit (near-duplicates, language ID, sample files, character shares). Report only; reads test for audit."""
import os, sys, re, random, hashlib, collections, unicodedata as U
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import data_access as DA

SEED, LANGS = 0, ["bho", "mai", "mag", "ang", "hin"]
OWN = {"bho": "bho_Deva", "mai": "mai_Deva", "mag": "mag_Deva", "ang": "anp_Deva", "hin": "hin_Deva"}
OUT = "results/audit"; os.makedirs(OUT, exist_ok=True)
def split_text(lang, split): return DA.load(lang, split, purpose="audit")

# ---- (a) near duplicates: 5-gram word shingles, Jaccard >= 0.5
def shingles(s):
    w = s.split()
    if len(w) < 5: return {hash(tuple(w))}
    return {hash(tuple(w[i:i + 5])) for i in range(len(w) - 4)}
def neardup(lang):
    """Returns per-split counts and the train line indices that have a >=0.5 neighbour in dev or test."""
    train = DA.load(lang, "tok_train"); q = {sp: split_text(lang, sp) for sp in ("dev", "test")}
    qsh = {sp: [shingles(s) for s in v] for sp, v in q.items()}
    index = collections.defaultdict(list)  # query shingle -> (split, idx)
    for sp, L in qsh.items():
        for i, sh in enumerate(L):
            for h in sh: index[h].append((sp, i))
    hit_q, flagged = collections.defaultdict(set), set()
    for ti, s in enumerate(train):
        sh = shingles(s); cand = {c for h in sh if h in index for c in index[h]}
        for sp, i in cand:
            a = qsh[sp][i]; j = len(sh & a) / len(sh | a)
            if j >= 0.5: hit_q[sp].add(i); flagged.add(ti)
    rows = [dict(lang=lang, pair=f"train-{sp}", n_query=len(q[sp]), n_query_with_train_neighbour=len(hit_q[sp])) for sp in q]
    rows[0]["n_train_flagged"] = len(flagged)
    # dev vs test
    idx = collections.defaultdict(list)
    for i, sh in enumerate(qsh["test"]):
        for h in sh: idx[h].append(i)
    n = 0
    for sh in qsh["dev"]:
        if any(len(sh & qsh["test"][i]) / len(sh | qsh["test"][i]) >= 0.5 for i in {c for h in sh if h in idx for c in idx[h]}): n += 1
    rows.append(dict(lang=lang, pair="dev-test", n_query=len(q["dev"]), n_query_with_train_neighbour=n))
    return rows, sorted(flagged)

# ---- (b) language ID
def langid(lang, sents, model):
    labs = collections.Counter(model.f.predict(s, 1, 0.0, "strict")[0][1].replace("__label__", "") for s in sents)
    n = sum(labs.values()); own = labs[OWN[lang]]
    return dict(n=n, pct_not_own=round(100 * (n - own) / n, 2), pct_hin=round(100 * labs["hin_Deva"] / n, 2), top5=";".join(f"{k}:{round(100*v/n,1)}" for k, v in labs.most_common(5)))

# ---- (d) character shares
def script_of(c):
    try: return U.name(c).split()[0]
    except ValueError: return "UNKNOWN"
def char_row(counter):
    tot = sum(counter.values()); cls = collections.Counter(); other = collections.Counter()
    for c, n in counter.items():
        cat = U.category(c); sc = script_of(c)
        if c in "‌‍": k = "joiner"
        elif cat[0] == "Z" or c in "\t\n": k = "space"
        elif cat[0] == "P": k = "punct"
        elif cat == "Nd" and (c.isascii() or sc == "DEVANAGARI"): k = "digit"
        elif sc == "DEVANAGARI": k = "devanagari"
        elif sc == "LATIN": k = "latin"
        else: k = "other"; other[sc if cat[0] in "LM" else f"{cat}"] += n
        cls[k] += n
    r = {f"pct_{k}": round(100 * cls[k] / tot, 3) for k in ["devanagari", "latin", "digit", "punct", "space", "joiner", "other"]}
    r["pct_outside_dev_latin_punct"] = round(100 * (cls["other"]) / tot, 3); r["top_other"] = ";".join(f"{k}:{v}" for k, v in other.most_common(5)); return r

if __name__ == "__main__":
    import fasttext; model = fasttext.load_model("data/raw/hf/glotlid/model.bin")
    nd, lid, ch, samp = [], [], [], []
    for lang in LANGS:
        rows, flagged = neardup(lang); nd += rows
        open(f"{OUT}/neardup_train_ids_{lang}.txt", "w").write("\n".join(map(str, flagged)) + "\n"); print(lang, "neardup", rows, flush=True)
        pools = {sp: (DA.load(lang, sp) if sp == "tok_train" else split_text(lang, sp)) for sp in ("tok_train", "dev", "test")}
        for sp, S in pools.items():
            sub = S if len(S) <= 100000 else random.Random(SEED).sample(S, 100000)
            lid.append(dict(lang=lang, split=sp, own_label=OWN[lang], sampled=len(sub) < len(S), **langid(lang, sub, model)))
            ch.append(dict(lang=lang, split=sp, **char_row(collections.Counter("".join(S)))))
        allsent = [(sp, s) for sp, S in pools.items() for s in S]; pick = random.Random(SEED).sample(allsent, 100)
        open(f"{OUT}/sample_{lang}.txt", "w", encoding="utf-8").write("\n".join(f"{sp}\t{s}" for sp, s in pick) + "\n")
        print(lang, "lid", [(r["split"], r["pct_not_own"], r["pct_hin"]) for r in lid if r["lang"] == lang], flush=True)
    pd.DataFrame(nd).to_csv(f"{OUT}/neardup.csv", index=False); pd.DataFrame(lid).to_csv(f"{OUT}/langid.csv", index=False); pd.DataFrame(ch).to_csv(f"{OUT}/charshare.csv", index=False)
