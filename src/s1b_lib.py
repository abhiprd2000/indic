"""Stage 1B helpers: tokenizer adapters (HF fast / tiktoken), sets, measurements. Measurement only; FINAL_EVAL=1."""
import os, re, sys, json, collections, unicodedata as U
os.environ["FINAL_EVAL"] = "1"
import numpy as np, pandas as pd, regex as RX
sys.path.insert(0, "src")
from sweep_lib import boot, test_sets
import data_access as DA

DEVSETS = ["Bhojpuri (BHTB)", "Magahi (MGTB)", "Maithili (web)", "Angika (MT)", "Hindi (HDTB)", "FLORES bho", "FLORES mag", "FLORES mai", "FLORES hin"]
FLO = {"FLORES bho": "bho_Deva", "FLORES mag": "mag_Deva", "FLORES mai": "mai_Deva", "FLORES hin": "hin_Deva", "FLORES eng": "eng_Latn"}
def all_sets():
    s = dict(test_sets()); s.pop("English (2k)", None)
    for k, c in FLO.items(): s[k] = DA.load_flores(c, "devtest")
    return s
def is_dmark(c): return "ऀ" <= c <= "ॿ" and U.category(c) in ("Mn", "Mc")
def is_dev(c): return "ऀ" <= c <= "ॿ"
def dev_words(s): return [(m.start(), m.end()) for m in re.finditer(r"\S+", s) if any(is_dev(c) for c in m.group())]

class HF:
    """Hugging Face fast tokenizer adapter."""
    def __init__(self, tok): self.t = tok; self.pre = getattr(getattr(tok, "backend_tokenizer", None), "pre_tokenizer", None)
    def n_vocab(self): return len(self.t)
    def spans(self, s):
        """Per token: (start, end) char offsets, start trimmed of leading spaces."""
        out = []
        for a, e in self.t(s, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]:
            while a < e and s[a].isspace(): a += 1
            out.append((a, e))
        return out
    def pieces(self, s):
        if self.pre is None: return [(0, len(s))]
        return [o for _, o in self.pre.pre_tokenize_str(s)]
class TT:
    """tiktoken adapter (byte offsets mapped to chars)."""
    def __init__(self, enc): self.e = enc; self.rx = RX.compile(enc._pat_str)
    def n_vocab(self): return self.e.n_vocab
    def spans(self, s):
        b = s.encode(); cmap = []
        for i, ch in enumerate(s): cmap += [i] * len(ch.encode())
        out, p = [], 0
        for t in self.e.encode(s, disallowed_special=()):
            n = len(self.e.decode_single_token_bytes(t)); a = cmap[p]; e = cmap[p + n - 1] + 1 if p + n - 1 < len(cmap) else len(s)
            # a token that ends in the middle of a character gets end = that character's start (a split inside the character)
            end_mid = p + n < len(b) and (b[p + n] & 0xC0) == 0x80
            out.append((a, cmap[p + n - 1] + (0 if end_mid else 1) if n else a)); p += n
        return out
    def pieces(self, s): return [(m.start(), m.end()) for m in self.rx.finditer(s)]

def token_stats(ad, sents):
    """Per-sentence words, tokens, and counts of tokens ending inside a Devanagari cluster (next char a mark, or a character split across tokens)."""
    nw, nt, mid, dtok = [], [], 0, 0
    for s in sents:
        sp = ad.spans(s); nw.append(len(s.split())); nt.append(len(sp))
        for i, (a, e) in enumerate(sp):
            if any(is_dev(c) for c in s[a:e]) or (e < len(s) and is_dev(s[e])):
                dtok += 1
                if (i + 1 < len(sp) and sp[i + 1][0] < e) or (e < len(s) and is_dmark(s[e])): mid += 1
    return np.array(nw), np.array(nt), mid, dtok

def cut_stats(ad, sents):
    """Devanagari words split by the pre-tokenizer: any split, split right before a combining mark, and the trigger characters."""
    n = split_any = split_mark = 0; trig = collections.Counter()
    for s in sents:
        ws = dev_words(s)
        if not ws: continue
        pcs = ad.pieces(s); starts = sorted({p[0] for p in pcs})
        for a, b in ws:
            n += 1; inner = [x for x in starts if a < x < b]
            if inner: split_any += 1; split_mark += any(is_dmark(s[x]) for x in inner)
            for x in inner: trig[s[x]] += 1
    return n, split_any, split_mark, trig
