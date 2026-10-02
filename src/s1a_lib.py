"""Stage 1A helpers: filtered-train merges, new-token script check, build."""
import os, sys, json, regex as RX, unicodedata as U
sys.path.insert(0, "src")
from sweep_lib import *

S1 = "artifacts/tok/s1a"; os.makedirs(S1, exist_ok=True)
KS = [0, 250, 500, 1000, 2000, 4000, 8000, 16000]
def _b2u():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
B2U = _b2u()
OKSCRIPT = RX.compile(r"[\p{Devanagari}\p{Latin}]"); NEUTRAL = RX.compile(r"[\p{Common}\p{Inherited}\p{Devanagari}\p{Latin}]")
def _all_letters_other(prefix):
    """True if every completion of a split UTF-8 character is a letter/mark of a script other than Devanagari/Latin; None if undecidable."""
    n = 4 if prefix[0] >= 0xF0 else 3 if prefix[0] >= 0xE0 else 2; miss = n - len(prefix)
    if miss <= 0 or miss > 2: return None
    seen = False
    for x in range(64 ** miss):
        tail = bytes(0x80 + ((x >> (6 * k)) & 63) for k in range(miss - 1, -1, -1))
        try: c = (bytes(prefix) + tail).decode("utf-8")
        except UnicodeDecodeError: continue
        seen = True
        if U.category(c)[0] not in "LM" or OKSCRIPT.match(c): return False
    return True if seen else None
def token_status(tok_str):
    """full: complete letter of another script; fragment: split character whose completions are all other-script letters/marks;
    mark_other: complete mark of another script (not a letter); ambiguous: cannot tell; ok."""
    bs = bytes(B2U[c] for c in tok_str); i, n, amb, mark = 0, len(bs), False, False
    while i < n:
        b = bs[i]
        if b < 0x80: i += 1; continue
        need = 1 if 0xC2 <= b <= 0xDF else 2 if 0xE0 <= b <= 0xEF else 3 if 0xF0 <= b <= 0xF4 else 0
        if need == 0: amb = True; i += 1; continue
        chunk = bs[i:i + 1 + need]
        if len(chunk) == 1 + need and all(0x80 <= x <= 0xBF for x in chunk[1:]):
            try: c = chunk.decode("utf-8")
            except UnicodeDecodeError: amb = True; i += 1 + need; continue
            if U.category(c)[0] == "L" and not OKSCRIPT.match(c): return "full"
            if U.category(c)[0] != "L" and not NEUTRAL.match(c): mark = True
            i += 1 + need
        else:
            r = _all_letters_other(chunk) if len(chunk) >= 2 else None
            if r: return "fragment"
            amb = amb or r is None; i = n
    return "mark_other" if mark else "ambiguous" if amb else "ok"
def status_counts(merges, K):
    import collections
    return collections.Counter(token_status(a + b) for a, b in merges[:K])
def train_mix(v, mix, K, split="train_f"):
    """Merges for variant v on mix (filtered train); cached in S1."""
    mp = f"{S1}/merges_{mix}_{v}.json"
    plan, lines = target_plan(mix, split)
    tok0 = load(vdir(v)); cnt, rows = T.pretok_counts(tok0, plan, lines)
    if os.path.exists(mp) and len(json.load(open(mp))) >= K: return json.load(open(mp)), rows
    merges, nu = train_merges(tok0, cnt, K); json.dump(merges, open(mp, "w"), ensure_ascii=False); return merges, rows
def build_s1(v, mix, merges, K):
    out = f"{S1}/{v}_{mix}_K{K}"
    if K == 0: return vdir(v)
    if not os.path.exists(out + "/tokenizer.json"): T.build(vdir(v), merges, K, out)
    return out
