"""Build the data pack for the Kaggle adaptation run (tokenizers, train mix, dev, test, UD, code)."""
import os, sys, json, random, shutil, re
sys.path.insert(0, "src")
import ext_bpe_train as T
import data_access as DA
from datasets import load_dataset

SEED, OUT = 0, "data/adapt_pack"
EN_TRAIN_MB, EN_DEV_MB, HIN_MB, LOW_MB, ALPHA = 4.5, 0.5, 1.5, 24.0, 0.3
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
for d, s in [("tok_base", "artifacts/tok/base/tokenizer.json"), ("tok_d8k", "artifacts/tok/sweep/d_K8000/tokenizer.json")]:
    os.makedirs(f"{OUT}/{d}"); shutil.copy(s, f"{OUT}/{d}/tokenizer.json")
json.dump(json.load(open("artifacts/tok/sweep/merges_d.json"))[:8000], open(f"{OUT}/merges_d8000.json", "w"), ensure_ascii=False)
for f in ["hi_hdtb-ud-train", "bho_bhtb-ud-test", "mag_mgtb-ud-test"]: shutil.copy(DA.ud_path(f), f"{OUT}/{f}.conllu")
for l in ["bho", "mai", "mag", "ang", "hin", "en"]: open(f"{OUT}/test_{l}.txt", "w", encoding="utf-8").write("\n".join(DA.load(l, "test")) + "\n")
# train mix
sizes = {l: sum(len(s.encode()) + 1 for s in T.read_lines(l)) / 1e6 for l in ["bho", "mai", "mag"]}
w = {l: v ** ALPHA for l, v in sizes.items()}; tgt = {l: LOW_MB * w[l] / sum(w.values()) for l in w}; tgt["hin"] = HIN_MB
units, used = [], {}
for l, mb in tgt.items():
    n = 0
    for s in T.read_lines(l):
        if n >= mb * 1e6: break
        units.append((l, s)); n += len(s.encode()) + 1
    used[l] = round(n / 1e6, 2)
# English replay + dev (streamed in order)
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
en, dev_docs, n = [], [], 0
for r in ds:
    t = " ".join(r["text"].split())
    if n < EN_TRAIN_MB * 1e6: en.append(t); n += len(t.encode()) + 1
    else:
        dev_docs.append(t)
        if sum(len(x.encode()) for x in dev_docs) >= EN_DEV_MB * 1e6: break
test_en = DA.load("en", "test")
big = "\n".join(en); leak = sum(s in big for s in test_en); assert leak == 0, f"English test sentences found in replay: {leak}"
units += [("en", t) for t in en]; used["en"] = round(n / 1e6, 2)
random.Random(SEED).shuffle(units)
with open(f"{OUT}/train.tsv", "w", encoding="utf-8") as f:
    for l, t in units: f.write(f"{l}\t{t}\n")
for l in ["bho", "mai", "mag", "hin"]:
    open(f"{OUT}/dev_{l}.txt", "w", encoding="utf-8").write("\n".join(DA.load(l, "dev")[:300]) + "\n")
sent = [s for d in dev_docs for s in re.split(r"(?<=[.!?])\s+", d) if len(s.split()) >= 5][:300]
open(f"{OUT}/dev_en.txt", "w", encoding="utf-8").write("\n".join(sent) + "\n")
os.makedirs(f"{OUT}/code"); [shutil.copy(f"src/{f}", f"{OUT}/code/{f}") for f in os.listdir("src") if f.startswith("adapt_") and f != "adapt_prep.py"]
json.dump(dict(MB=used, units=len(units), english_test_leaks=leak, dev_en_sentences=len(sent)), open(f"{OUT}/pack_info.json", "w"))
print("pack:", used, "units", len(units), "leaks", leak, "dev_en", len(sent))
os.makedirs("results/adapt", exist_ok=True); json.dump(dict(MB=used, units=len(units), english_test_leaks=leak), open("results/adapt/train_mix.json", "w"))
