"""Continued BPE: K extra merges over Qwen3 token sequences. Writes artifacts/tok/<mix>/merges.json and tokenizers."""
import os, sys, json, random, heapq, collections, time
sys.path.insert(0, "src")
import pandas as pd
from transformers import AutoTokenizer
import data_access as DA

SEED, BUDGET_MB, ALPHA, KMAX = 0, 40, 0.3, 32000
SWEEP = [500, 1000, 2000, 4000, 8000, 16000, 32000]
BASE = "Qwen/Qwen3-1.7B"
MIXES = {"hin_only": (["hin"], None), "natural4": (["hin", "bho", "mai", "mag"], None),
         "upsampled4": (["hin", "bho", "mai", "mag"], ALPHA), "lowonly": (["bho", "mai", "mag"], ALPHA)}
SWEEP_MIX = "upsampled4"  # fixed before seeing results

def read_lines(lang, split="tok_train"):
    L = DA.load(lang, split)
    random.Random(SEED).shuffle(L)
    return L

def mix_plan(langs, alpha, sizes):
    """Target MB per language: natural (size share) or size**alpha share."""
    w = {l: sizes[l] ** (alpha if alpha else 1) for l in langs}
    z = sum(w.values())
    return {l: BUDGET_MB * w[l] / z for l in langs}

def pretok_counts(tok, plan, lines):
    """Weighted counts of byte-level pre-tokens. Weight >1 when target MB exceeds unique text."""
    pre = tok.backend_tokenizer.pre_tokenizer
    cnt, rows = collections.Counter(), []
    for lang, target in plan.items():
        L, used, take = lines[lang], 0, []
        for s in L:
            if used >= target * 1e6: break
            take.append(s); used += len(s.encode()) + 1
        uniq_mb = used / 1e6; mult = max(1.0, target / uniq_mb) if uniq_mb < target * .999 else 1.0
        c = collections.Counter()
        for s in take:
            for w, _ in pre.pre_tokenize_str(s): c[w] += 1
        for w, n in c.items(): cnt[w] += n * mult
        rows.append(dict(lang=lang, target_MB=round(target, 2), unique_MB=round(uniq_mb, 2), repeat=round(mult, 2)))
    return cnt, rows

def learn(seqs, wts, tokstr, K):
    """Greedy BPE merges over weighted id sequences. tokstr: id->str (mutated). Returns list of (a_str, b_str)."""
    str2id = {s: i for i, s in tokstr.items()}
    pc, where = collections.defaultdict(float), collections.defaultdict(set)
    for i, s in enumerate(seqs):
        for p in zip(s, s[1:]): pc[p] += wts[i]; where[p].add(i)
    heap = [(-c, p) for p, c in pc.items()]; heapq.heapify(heap)
    nxt, merges = max(tokstr) + 1, []
    while len(merges) < K and heap:
        c, p = heapq.heappop(heap)
        if pc.get(p) != -c: continue
        a, b = p; s = tokstr[a] + tokstr[b]
        if s in str2id: pc.pop(p); continue
        new = nxt; nxt += 1; tokstr[new] = s; str2id[s] = new; merges.append((tokstr[a], tokstr[b]))
        changed = set()
        for i in where.pop(p):
            seq = seqs[i]; w = wts[i]
            if not any(x == a and y == b for x, y in zip(seq, seq[1:])): continue
            for q in zip(seq, seq[1:]): pc[q] -= w; changed.add(q)
            out, j = [], 0
            while j < len(seq):
                if j + 1 < len(seq) and seq[j] == a and seq[j + 1] == b: out.append(new); j += 2
                else: out.append(seq[j]); j += 1
            seqs[i] = out
            for q in zip(out, out[1:]): pc[q] += w; where[q].add(i); changed.add(q)
        pc.pop(p, None)
        for q in changed:
            v = pc.get(q, 0)
            if v > 1e-9: heapq.heappush(heap, (-v, q))
            else: pc.pop(q, None)
    return merges

def build(base_dir, merges, K, out):
    """Write HF tokenizer with first K new merges appended after base merges."""
    j = json.load(open(f"{base_dir}/tokenizer.json")); m = j["model"]
    nid = max(max(m["vocab"].values()), max(a["id"] for a in j["added_tokens"])) + 1
    # tokenizers renumbers special tokens to len(vocab) unless they are in the model vocab; keep their ids fixed
    for a in j["added_tokens"]: m["vocab"].setdefault(a["content"], a["id"])
    for k, (a, b) in enumerate(merges[:K]):
        m["vocab"][a + b] = nid + k; m["merges"].append([a, b])
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(base_dir):
        if f != "tokenizer.json": open(f"{out}/{f}", "wb").write(open(f"{base_dir}/{f}", "rb").read())
    json.dump(j, open(f"{out}/tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(BASE); bdir = "artifacts/tok/base"; tok.save_pretrained(bdir)
    inv = {v: k for k, v in json.load(open(f"{bdir}/tokenizer.json"))["model"]["vocab"].items()}
    langs = ["hin", "bho", "mai", "mag"]
    lines = {l: read_lines(l) for l in langs}
    sizes = {l: sum(len(s.encode()) + 1 for s in lines[l]) / 1e6 for l in langs}
    model, allrows = tok.backend_tokenizer.model, []
    for name, (ls, alpha) in MIXES.items():
        t0 = time.time(); plan = mix_plan(ls, alpha, sizes)
        cnt, rows = pretok_counts(tok, plan, lines)
        words = list(cnt); seqs = [[t.id for t in model.tokenize(w)] for w in words]; wts = [cnt[w] for w in words]
        ts = dict(inv); K = KMAX if name == SWEEP_MIX else 8000
        chk = list(zip(words, [list(s) for s in seqs]))[:20000]
        merges = learn(seqs, wts, ts, K)
        os.makedirs(f"artifacts/tok/{name}", exist_ok=True)
        json.dump(merges, open(f"artifacts/tok/{name}/merges.json", "w", encoding="utf-8"), ensure_ascii=False)
        for k in (SWEEP if name == SWEEP_MIX else [8000]): build(bdir, merges, k, f"artifacts/tok/{name}_K{k}")
        # check: built tokenizer reproduces learner sequences on 20k words
        t2 = AutoTokenizer.from_pretrained(f"artifacts/tok/{name}_K{K if name == SWEEP_MIX else 8000}")
        m2 = t2.backend_tokenizer.model
        bad = sum(len(m2.tokenize(w)) != len(seqs[i]) for i, (w, _) in enumerate(chk))
        for r in rows: allrows.append(dict(mix=name, K_max=K, **r))
        print(name, "merges", len(merges), "unique words", len(words), "mismatch", bad, f"{time.time()-t0:.0f}s", flush=True)
        assert bad == 0, name
    pd.DataFrame(allrows).assign(eff_MB_mix=lambda d: d.groupby("mix").target_MB.transform("sum")).to_csv("results/ext_bpe/mixes.csv", index=False)
