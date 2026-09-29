"""5.3 follow-up: why do d merges change Bengali/Tamil? Script mix of training text and of new tokens."""
import sys, json, collections
sys.path.insert(0, "src")
from sweep_lib import *
def script(cp):
    for lo, hi, n in [(0x900, 0x97F, "Devanagari"), (0x980, 0x9FF, "Bengali"), (0xB80, 0xBFF, "Tamil"), (0x600, 0x6FF, "Arabic"), (0x0E00, 0x0E7F, "Thai")]:
        if lo <= cp <= hi: return n
    return "Latin/ASCII" if cp < 0x250 else "other"
rows = []
for l in ["bho", "mai", "mag"]:
    c = collections.Counter(script(ord(ch)) for s in open(f"data/{l}/tok_train.txt", encoding="utf-8") for ch in s if not ch.isspace())
    t = sum(c.values()); rows += [dict(what=f"train_chars_{l}", script=k, count=v, pct=round(100 * v / t, 3)) for k, v in c.items()]
b2u = {}  # byte-level unicode char -> byte (GPT-2 map)
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
b2u = {chr(c): b for b, c in zip(bs, cs)}
merges = json.load(open(f"{SW}/merges_d.json"))[:8000]; c = collections.Counter()
for a, b in merges:
    by = bytes(b2u[ch] for ch in (a + b).replace("Ġ", chr(288)) if ch in b2u)
    txt = by.decode("utf-8", errors="ignore").strip()
    c[max((script(ord(ch)) for ch in txt), key=lambda s: sum(script(ord(x)) == s for x in txt)) if txt else "bytes/space only"] += 1
rows += [dict(what="new_tokens_d_K8000_by_main_script", script=k, count=v, pct=round(100 * v / 8000, 2)) for k, v in c.items()]
pd.DataFrame(rows).to_csv("results/sweep/script_leak.csv", index=False)
d = pd.DataFrame(rows); print(d[d.pct >= 0.05].to_string(index=False))
