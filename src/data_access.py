"""Only entry point for text. Test splits need FINAL_EVAL=1; purpose="audit" reads (leak, LID, script stats) are allowed and logged."""
import os, sys, time, json

LANGS = ["bho", "mai", "mag", "ang", "hin", "en"]
LOG = "results/audit/test_access.log"

def _gate(what, purpose):
    final = os.environ.get("FINAL_EVAL") == "1"
    if not final and purpose != "audit":
        raise PermissionError(f"{what} is a test split: set FINAL_EVAL=1 in a final-eval script (never in selection scripts)")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    open(LOG, "a").write(f"{time.strftime('%F %T')}\t{os.path.basename(sys.argv[0])}\t{what}\t{'final' if final else 'audit'}\n")

def _lines(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]

def load(lang, split, purpose=None):
    """Sentences of data/<lang>/<split>.txt; split in tok_train, dev, test."""
    assert split in ("tok_train", "dev", "test") and lang in LANGS, (lang, split)
    if split == "test": _gate(f"{lang}/test", purpose)
    return _lines(f"data/{lang}/{split}.txt")

def ud_path(name, purpose=None):
    """CoNLL-U path, e.g. bho_bhtb-ud-test; test files are locked."""
    if name.endswith("-test"): _gate(f"ud/{name}", purpose)
    return f"data/ud/{name}.conllu"

def load_ud(name, purpose=None):
    """Sentence texts ('# text =') of a UD file."""
    return [l[8:].strip() for l in open(ud_path(name, purpose), encoding="utf-8") if l.startswith("# text =")]

def load_flores(code, split, purpose=None):
    """FLORES+ sentences (code like hin_Deva); devtest is locked, dev is not. Cached under data/flores/."""
    assert split in ("dev", "devtest"), split
    if split == "devtest": _gate(f"flores/{code}/devtest", purpose)
    p = f"data/flores/{split}/{code}.txt"
    if not os.path.exists(p):
        from huggingface_hub import hf_hub_download as dl
        raw = dl("openlanguagedata/flores_plus", f"{split}/{code}.jsonl", repo_type="dataset", local_dir="data/raw/hf", token=os.environ.get("HF_TOKEN"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write("\n".join(" ".join(json.loads(x)["text"].split()) for x in open(raw, encoding="utf-8")) + "\n")
    return _lines(p)
