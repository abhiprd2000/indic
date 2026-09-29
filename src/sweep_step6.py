"""5.6 Tokens per word at K=8000 next to ByteChunk (H-Net) chunks per word."""
import sys, time
sys.path.insert(0, "src")
from sweep_lib import *
t0 = time.time()
d = pd.read_csv("results/sweep/fertility_vs_K.csv")
rows = []
for sn in ["Bhojpuri (BHTB)", "Magahi (MGTB)"]:
    for name, v, K in [("Qwen3 base (K=0)", "a", 0), ("a original, K=8k", "a", 8000), ("b all marks, K=8k", "b", 8000), ("d Devanagari marks, K=8k", "d", 8000)]:
        r = d[(d.variant == v) & (d.K == K) & (d.test_set == sn)].iloc[0]
        rows.append(dict(test_set=sn, system=name, unit="tokens/word", value=round(r.fertility, 3), ci_lo=round(r.fert_lo, 3), ci_hi=round(r.fert_hi, 3),
                         note="our tokenizers, whitespace words"))
    rows.append(dict(test_set=sn, system="Qwen3 H-Net (ByteChunk Table 1)", unit="chunks/word", value=1.17, ci_lo="", ci_hi="",
                     note="value as given by user; chunk and token definitions differ (learned byte chunks vs vocabulary tokens); not directly comparable"))
pd.DataFrame(rows).to_csv("results/sweep/granularity.csv", index=False)
print(pd.DataFrame(rows)[["test_set", "system", "unit", "value", "ci_lo", "ci_hi"]].to_string(index=False)); log_time("5.6", t0)
