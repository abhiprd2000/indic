"""Kaggle run: arms A (base), B (extended, mean-init), C (B + new rows trained); BPB, POS probe, speed.
Env: MODEL=real|tiny, SMOKE=1 (tiny run), LR=<lr> (skip LR trials), STEPS_MULT=<k> (repeat data k times)."""
import os, sys, json, time, math, glob, random, re, copy, unicodedata
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

SMOKE = os.environ.get("SMOKE") == "1"
MODEL = os.environ.get("MODEL", "real")
PACK = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/train.tsv", recursive=True)), "data/adapt_pack/train.tsv"))
OUT = os.environ.get("OUT", "/kaggle/working/results" if os.path.exists("/kaggle/working") else "results/adapt_local")
os.makedirs(OUT, exist_ok=True)
CUDA = torch.cuda.is_available(); DEV = "cuda" if CUDA else "cpu"
N_OLD, N_NEW, EOS, D_SEED = 151669, 8000, 151643, 0
SEQ, MB, ACCUM = 512, 2, 8
CAP_S = 2.5 * 3600
N_DEV, N_TEST = (20, 20) if SMOKE else (300, 10 ** 9)
TRIAL_STEPS = 2 if SMOKE else 300
POS_N, POS_EPOCHS = (200, 2) if SMOKE else (10 ** 9, 10)
SPEED_RUNS = 3 if SMOKE else 20
LRS = [2e-4, 1e-3]
T0 = time.time(); INFO = {}
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
def ac(): return torch.autocast(device_type=DEV, dtype=torch.float16, enabled=CUDA)
def sync():
    if CUDA: torch.cuda.synchronize()
def tok(name): return PreTrainedTokenizerFast(tokenizer_file=f"{PACK}/{name}/tokenizer.json")
def _lock(p):
    if ("test" in os.path.basename(p)) and os.environ.get("FINAL_EVAL") != "1": raise PermissionError(f"{p} is a test file: set FINAL_EVAL=1")
def lines(p, n=None):
    _lock(p); return [l.rstrip("\n") for l in open(f"{PACK}/{p}", encoding="utf-8") if l.strip()][:n]

# ---------- model ----------
class Emb(nn.Module):
    """Input embedding: frozen old rows + trainable new rows (fp32)."""
    def __init__(s, old, new):
        super().__init__(); s.register_buffer("old", old); s.new = nn.Parameter(new)
    def forward(s, ids):
        o = F.embedding(ids.clamp(max=N_OLD - 1), s.old); n = F.embedding((ids - N_OLD).clamp(min=0), s.new)
        return torch.where((ids >= N_OLD)[..., None], n.to(o.dtype), o)
class Head(nn.Module):
    """Tied output head sharing the same old/new rows."""
    def __init__(s, emb): super().__init__(); s.emb = emb
    def forward(s, h): return torch.cat([F.linear(h, s.emb.old.to(h.dtype)), F.linear(h, s.emb.new.to(h.dtype))], -1)

def load_base():
    if MODEL == "tiny":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        return Qwen3ForCausalLM(Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                            head_dim=16, vocab_size=151936, tie_word_embeddings=True)).float().to(DEV).eval()
    return AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype=torch.float32).to(DEV).eval()

def init_rows(old):
    """New row = mean of leaf base-token rows via the merge tree (no decoding to text)."""
    merges = json.load(open(f"{PACK}/merges_d8000.json")); vocab = dict(json.load(open(f"{PACK}/tok_base/tokenizer.json"))["model"]["vocab"])
    D = old.shape[1]; sums = torch.zeros(N_NEW, D, device=old.device); cnt = torch.zeros(N_NEW, device=old.device); ids = dict(vocab)
    for k, (a, b) in enumerate(merges):
        parts = []
        for x in (a, b):
            i = ids[x]
            if i >= N_OLD: parts.append((sums[i - N_OLD], cnt[i - N_OLD]))
            else: assert i < EOS, x; parts.append((old[i], 1.0))
        sums[k] = parts[0][0] + parts[1][0]; cnt[k] = parts[0][1] + parts[1][1]; ids[a + b] = N_OLD + k
    return sums / cnt[:, None]

def build(kind, base_state=None):
    m = load_base() if base_state is None else base_state
    if kind == "A": return m
    old = m.model.embed_tokens.weight.data[:N_OLD].clone()
    emb = Emb(old, init_rows(old)); m.model.embed_tokens = emb; m.lm_head = Head(emb)
    for p in m.parameters(): p.requires_grad = False
    emb.new.requires_grad = True
    return m

# ---------- data ----------
def pack_blocks(t, seq):
    units = [l.rstrip("\n").split("\t", 1) for l in open(f"{PACK}/train.tsv", encoding="utf-8")]
    if SMOKE: units = units[:3000]
    enc = t([u[1] for u in units], add_special_tokens=False)["input_ids"]
    stream = np.concatenate([np.array(e + [EOS]) for e in enc]); n = len(stream) // seq
    return torch.tensor(stream[: n * seq].reshape(n, seq), dtype=torch.long), len(stream)

# ---------- BPB ----------
def batches(items, max_tok):
    cur = []
    for it in items:
        if cur and (len(cur) + 1) * (it[1] + 1) > max_tok: yield cur; cur = []
        cur.append(it)
    if cur: yield cur

@torch.no_grad()
def bpb(model, tz, sents, autocast=True):
    keep = [s for s in sents if len(s.encode()) <= 1500]
    enc = tz(keep, add_special_tokens=False)["input_ids"]; order = sorted(range(len(keep)), key=lambda i: len(enc[i]))
    nll, ntok = 0.0, 0
    for b in batches([(i, len(enc[i])) for i in order], 2048):
        idx = [x[0] for x in b]; L = max(len(enc[i]) for i in idx) + 1
        x = torch.zeros(len(idx), L, dtype=torch.long); m = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx): e = [EOS] + enc[i]; x[r, :len(e)] = torch.tensor(e); m[r, :len(e)] = 1
        x, m = x.to(DEV), m.to(DEV)
        with (ac() if autocast else torch.autocast("cpu", enabled=False)):
            lg = model(input_ids=x, attention_mask=m).logits
        l = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none").view(len(idx), -1)
        mk = m[:, 1:].float(); nll += float((l * mk).sum()); ntok += int(mk.sum())
    nb = sum(len(s.encode()) for s in keep)
    return dict(bpb=nll / (math.log(2) * nb), n_sents=len(keep), n_dropped=len(sents) - len(keep), bytes=nb, tokens=ntok)

SETS = ["bho", "mai", "mag", "ang", "hin", "en"]
def eval_bpb(model, tz, arm, split, rows, sets=None):
    for l in sets or (SETS if split == "test" else [x for x in SETS if x != "ang"]):
        r = bpb(model, tz, lines(f"{split}_{l}.txt", N_DEV if split == "dev" else N_TEST)); rows.append(dict(arm=arm, split=split, lang=l, **r))
    log(arm, split, {r["lang"]: round(r["bpb"], 4) for r in rows if r["arm"] == arm and r["split"] == split})

# ---------- training ----------
def train(model, blocks, steps, lr, tag, ckpt_dir=None, seq=SEQ, log_rows=None):
    emb = model.model.embed_tokens; opt = torch.optim.AdamW([emb.new], lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=CUDA); V = N_OLD + N_NEW; t0 = time.time(); old_before = emb.old.clone() if SMOKE else None
    for step in range(steps):
        f = min(1.0, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups: g["lr"] = lr * f
        tot = 0.0
        for mi in range(ACCUM):
            j = ((step * ACCUM + mi) * MB) % max(len(blocks) - MB, 1); x = blocks[j:j + MB, :seq].to(DEV)
            with ac():
                h = model.model(input_ids=x).last_hidden_state; lg = model.lm_head(h[:, :-1])
            loss = F.cross_entropy(lg.float().reshape(-1, V), x[:, 1:].reshape(-1)) / ACCUM
            scaler.scale(loss).backward(); tot += float(loss)
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([emb.new], 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if log_rows is not None and (step % 25 == 0 or step == steps - 1):
            log_rows.append(dict(run=tag, step=step, loss=round(tot, 4), lr=lr * f, elapsed_s=round(time.time() - t0, 1)))
        if step % 25 == 0: log(tag, "step", step, "loss", round(tot, 4))
        if ckpt_dir and (step + 1) % 250 == 0 and step + 1 < steps: torch.save(emb.new.detach().cpu(), f"{ckpt_dir}/new_rows_step{step+1}.pt")
    sync()
    if SMOKE: assert torch.equal(old_before, emb.old), "old rows changed"
    return time.time() - t0

# ---------- POS probe ----------
def read_conllu(p, n):
    _lock(p); out, cur = [], []
    for l in open(f"{PACK}/{p}", encoding="utf-8"):
        l = l.rstrip("\n")
        if l.startswith("#"): continue
        if not l:
            if cur: out.append(cur); cur = []
            continue
        f = l.split("\t")
        if "-" in f[0] or "." in f[0]: continue
        cur.append((f[1], f[3]))
    return [s for s in out if all(" " not in w for w, _ in s)][:n]

@torch.no_grad()
def pos_feats(model, tz, sents):
    texts = [" ".join(w for w, _ in s) for s in sents]
    enc = tz(texts, add_special_tokens=False, return_offsets_mapping=True); order = sorted(range(len(texts)), key=lambda i: len(enc["input_ids"][i]))
    feats = [None] * len(texts)
    for b in batches([(i, len(enc["input_ids"][i])) for i in order], 4096):
        idx = [x[0] for x in b]; L = max(len(enc["input_ids"][i]) for i in idx) + 1
        x = torch.zeros(len(idx), L, dtype=torch.long); m = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx): e = [EOS] + enc["input_ids"][i]; x[r, :len(e)] = torch.tensor(e); m[r, :len(e)] = 1
        with ac(): h = model.model(input_ids=x.to(DEV), attention_mask=m.to(DEV)).last_hidden_state
        for r, i in enumerate(idx):
            s = texts[i]; ws = np.array([q.start() for q in re.finditer(r"\S+", s)]); first = {}
            for t, (a, e) in enumerate(enc["offset_mapping"][i]):
                while a < e and s[a].isspace(): a += 1
                w = max(int(np.searchsorted(ws, a, side="right")) - 1, 0); first.setdefault(w, t)
            assert len(first) == len(ws), (s, len(first), len(ws))
            feats[i] = h[r, [1 + first[w] for w in range(len(ws))]].half().cpu()
    return feats

def pos_probe(model, tz, arm, rows):
    tr, tes = read_conllu("hi_hdtb-ud-train.conllu", POS_N), {"BHTB": read_conllu("bho_bhtb-ud-test.conllu", POS_N), "MGTB": read_conllu("mag_mgtb-ud-test.conllu", POS_N)}
    tags = sorted({t for s in tr for _, t in s}); tid = {t: i for i, t in enumerate(tags)}
    def prep(S):
        f = pos_feats(model, tz, S); y = [torch.tensor([tid.get(t, -1) for _, t in s]) for s in S]; return f, y
    ftr, ytr = prep(tr); fte = {k: prep(v) for k, v in tes.items()}; D = ftr[0].shape[1]; log(arm, "pos feats", len(tr), "train sents", D)
    for seed in range(3):
        torch.manual_seed(seed); random.seed(seed)
        mlp = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.LayerNorm(256), nn.Linear(256, len(tags))).to(DEV)
        opt = torch.optim.AdamW(mlp.parameters(), lr=2e-4, weight_decay=0.01); nb = math.ceil(len(ftr) / 32); total = POS_EPOCHS * nb; warm = max(1, int(0.05 * total)); step = 0
        for ep in range(POS_EPOCHS):
            perm = torch.randperm(len(ftr)).tolist()
            for b in range(nb):
                ids = perm[b * 32:(b + 1) * 32]; x = torch.cat([ftr[i] for i in ids]).to(DEV).float(); y = torch.cat([ytr[i] for i in ids]).to(DEV)
                f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
                for g in opt.param_groups: g["lr"] = 2e-4 * f
                loss = F.cross_entropy(mlp(x), y); opt.zero_grad(); loss.backward(); opt.step(); step += 1
        for k, (f_, y_) in fte.items():
            with torch.no_grad(): p = torch.cat([mlp(torch.cat(f_[i:i + 64]).to(DEV).float()).argmax(-1).cpu() for i in range(0, len(f_), 64)])
            yy = torch.cat(y_); rows.append(dict(arm=arm, seed=seed, test=k, acc=float((p == yy).float().mean()), n_words=len(yy), n_sents=len(f_)))
    log(arm, "pos", [(r["test"], r["seed"], round(r["acc"], 4)) for r in rows if r["arm"] == arm])

# ---------- speed ----------
@torch.no_grad()
def speed(model, tz, arm, rows):
    """64 BHTB test sentences, pure fp16 weights, 5 warmups, timed runs."""
    _lock("bho_bhtb-ud-test.conllu"); txt = [l[8:].strip() for l in open(f"{PACK}/bho_bhtb-ud-test.conllu", encoding="utf-8") if l.startswith("# text =")]
    sents = random.Random(D_SEED).sample(txt, 64); enc = tz(sents, add_special_tokens=False)["input_ids"]; L = max(map(len, enc)) + 1
    x = torch.zeros(64, L, dtype=torch.long); m = torch.zeros(64, L, dtype=torch.long)
    for r, e in enumerate(enc): e = [EOS] + e; x[r, :len(e)] = torch.tensor(e); m[r, :len(e)] = 1
    x, m = x.to(DEV), m.to(DEV); mh = copy.deepcopy(model).half() if CUDA else copy.deepcopy(model)
    if CUDA: torch.cuda.reset_peak_memory_stats()
    for _ in range(5): mh(input_ids=x, attention_mask=m)
    sync(); ts = []
    for _ in range(SPEED_RUNS):
        t = time.time(); mh(input_ids=x, attention_mask=m); sync(); ts.append(time.time() - t)
    nb = sum(len(s.encode()) for s in sents)
    rows.append(dict(arm=arm, tokens_per_sentence=np.mean([len(e) for e in enc]), ms_per_batch=1000 * np.mean(ts), ms_std=1000 * np.std(ts),
                     bytes_per_sec=nb / np.mean(ts), peak_mem_GiB=(torch.cuda.max_memory_allocated() / 2 ** 30 if CUDA else float("nan")), padded_len=L, dtype="fp16 weights"))
    del mh

def save(name, rows): pd.DataFrame(rows).to_csv(f"{OUT}/{name}.csv", index=False)

def main():
    INFO.update(smoke=SMOKE, model=MODEL, gpu=torch.cuda.get_device_name(0) if CUDA else "cpu", torch=torch.__version__, pack=PACK)
    tA, tD = tok("tok_base"), tok("tok_d8k"); bpb_rows, gpu_s = [], 0.0
    blocks, ntok = pack_blocks(tD, SEQ); INFO.update(train_tokens=int(ntok), blocks=len(blocks)); log("blocks", len(blocks), "tokens", ntok)
    # arms A and B (no training)
    A = load_base()
    eval_bpb(A, tA, "A", "dev", bpb_rows); eval_bpb(A, tA, "A", "test", bpb_rows)
    B = build("B", load_base()); eval_bpb(B, tD, "B", "dev", bpb_rows); eval_bpb(B, tD, "B", "test", bpb_rows); save("bpb", bpb_rows)
    # dtype check on arm A: fp16 autocast vs bf16 weights
    if CUDA:
        A16 = bpb(A, tA, lines("test_bho.txt", 200)); Ab = copy.deepcopy(A).to(torch.bfloat16)
        Abf = bpb(Ab, tA, lines("test_bho.txt", 200), autocast=False); del Ab
        save("dtype_check", [dict(set="bho first 200 test", fp16_autocast_bpb=A16["bpb"], bf16_bpb=Abf["bpb"], diff_pct=100 * (A16["bpb"] / Abf["bpb"] - 1))])
    # throughput
    A.cpu(); B.cpu(); torch.cuda.empty_cache() if CUDA else None
    def fresh(): return build("B", load_base())
    seq = SEQ; steps_full = len(blocks) // (MB * ACCUM) * int(os.environ.get("STEPS_MULT", 1))
    if SMOKE: steps_full = min(steps_full, 4)
    while True:
        m = fresh(); tt = train(m, blocks, 20 if not SMOKE else 2, 1e-4, "throughput", seq=seq); t_step = tt / (20 if not SMOKE else 2); del m
        n_trial = 0 if os.environ.get("LR") else 2 * TRIAL_STEPS
        proj = (n_trial + steps_full) * t_step; log(f"throughput {t_step:.2f}s/step seq {seq}; full steps {steps_full}; projected {proj/3600:.2f} h (cap {CAP_S/3600} h)")
        if proj <= CAP_S: break
        fit = int((CAP_S - n_trial * t_step) / t_step)
        if fit >= 500: steps_full = fit; log("cut data to", steps_full, "steps"); break
        seq //= 2; log("halving seq to", seq); blocks = blocks.reshape(-1, seq); steps_full = len(blocks) // (MB * ACCUM)
    INFO.update(sec_per_step=t_step, seq=seq, steps_full=steps_full, projected_h=proj / 3600); gpu_s += 20 * t_step
    save("throughput", [dict(sec_per_step=t_step, seq=seq, steps_full=steps_full, projected_hours=proj / 3600, cap_hours=CAP_S / 3600)])
    # LR trials
    trial_rows, tr_log = [], []
    if os.environ.get("LR"): lr = float(os.environ["LR"])
    else:
        for l in LRS:
            m = fresh(); t = train(m, blocks, TRIAL_STEPS, l, f"trial_lr{l}", seq=seq, log_rows=tr_log); gpu_s += t; r = []
            eval_bpb(m, tD, f"trial_lr{l}", "dev", r, ["bho", "mai", "mag", "hin", "en"])
            mean3 = float(np.mean([x["bpb"] for x in r if x["lang"] in ("bho", "mai", "mag")])); mean3 = mean3 if math.isfinite(mean3) else float("inf")
            trial_rows.append(dict(lr=l, steps=TRIAL_STEPS, mean_dev_bpb_bho_mai_mag=mean3, train_seconds=t, **{f"dev_{x['lang']}": x["bpb"] for x in r})); del m
        save("lr_trials", trial_rows); lr = min(trial_rows, key=lambda r: r["mean_dev_bpb_bho_mai_mag"])["lr"]; log("chosen lr", lr)
    # full run
    ck = f"{OUT}/ckpt"; os.makedirs(ck, exist_ok=True)
    C = fresh(); t = train(C, blocks, steps_full, lr, "full", ckpt_dir=ck, seq=seq, log_rows=tr_log); gpu_s += t
    torch.save(C.model.embed_tokens.new.detach().cpu(), f"{ck}/new_rows_final.pt"); save("train_log", tr_log)
    INFO.update(lr=lr, train_seconds=t, gpu_seconds_total=gpu_s, peak_mem_GiB=(torch.cuda.max_memory_allocated() / 2 ** 30 if CUDA else None)); log("full run", round(t), "s")
    eval_bpb(C, tD, "C", "dev", bpb_rows); eval_bpb(C, tD, "C", "test", bpb_rows); save("bpb", bpb_rows)
    # POS probe and speed
    pos_rows, sp_rows = [], []
    for arm, mdl, tz in [("A", A, tA), ("B", B, tD), ("C", C, tD)]:
        mdl.to(DEV); pos_probe(mdl, tz, arm, pos_rows); save("pos_probe", pos_rows)
        if arm != "C": mdl.cpu(); torch.cuda.empty_cache() if CUDA else None
    d = pd.DataFrame(pos_rows).groupby(["arm", "test"]).acc.agg(["mean", "std"]).reset_index(); d.to_csv(f"{OUT}/pos_summary.csv", index=False)
    for arm, mdl, tz in [("A", A, tA), ("C", C, tD)]: mdl.to(DEV); speed(mdl, tz, arm, sp_rows); mdl.cpu() if arm == "A" else None
    save("speed", sp_rows); INFO.update(total_seconds=time.time() - T0); json.dump(INFO, open(f"{OUT}/run_info.json", "w"), indent=1); log("done")

if __name__ == "__main__":
    main()
