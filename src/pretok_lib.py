"""Pre-tokenizer regex variants for Qwen3 and helpers."""
import json, os, shutil
from tokenizers import Regex, pre_tokenizers

A = json.load(open("artifacts/tok/base/tokenizer.json"))["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
ORIG_LETTER = r"[^\r\n\p{L}\p{N}]?\p{L}+"
assert ORIG_LETTER in A
B = A.replace(ORIG_LETTER, r"[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+")
_c = lambda *r: "".join(chr(x) for x in r)
_cons = f"[{_c(0x915)}-{_c(0x939)}{_c(0x958)}-{_c(0x95F)}]"
_marks = f"[{_c(0x900)}-{_c(0x903)}{_c(0x93A)}{_c(0x93B)}{_c(0x93E)}-{_c(0x94D)}{_c(0x94E)}{_c(0x94F)}{_c(0x951)}-{_c(0x957)}{_c(0x962)}{_c(0x963)}]"
_vow = f"[{_c(0x904)}-{_c(0x914)}{_c(0x960)}{_c(0x961)}]"
AKSH = f"(?:{_cons}{_c(0x93C)}?{_c(0x94D)})*{_cons}{_c(0x93C)}?{_marks}*|{_vow}{_marks}*"
_first, _rest = A.split("|", 1)
C = f"{_first}|[^\\r\\n\\p{{L}}\\p{{M}}\\p{{N}}]?(?:{AKSH})|{_rest}"
REGEX = {"a_original": A, "b_marks_attached": B, "c_akshara": C}

def split_only(rx):
    """Pre-tokenizer that shows plain text pieces (no byte mapping)."""
    return pre_tokenizers.Split(Regex(rx), "isolated")

def variant_dir(name, rx):
    """Copy base tokenizer to artifacts/tok/pre_<name> with the given split regex."""
    out = f"artifacts/tok/pre_{name}"
    if os.path.exists(out): shutil.rmtree(out)
    shutil.copytree("artifacts/tok/base", out)
    j = json.load(open(f"{out}/tokenizer.json"))
    j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = rx
    json.dump(j, open(f"{out}/tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)
    return out
