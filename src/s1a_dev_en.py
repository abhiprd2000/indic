"""English dev set = AngikaMT English dev column; must not overlap the English test sample."""
import os, sys, pandas as pd
sys.path.insert(0, "src")
import data_access as DA
from huggingface_hub import hf_hub_download as dl
p = dl("snjev310/AngikaMT", "dev.csv", repo_type="dataset", local_dir="data/raw/hf", token=os.environ.get("HF_TOKEN"))
dev = [" ".join(s.split()) for s in pd.read_csv(p)["english"].dropna()]
test = set(DA.load("en", "test", purpose="audit")); overlap = sum(s in test for s in dev); assert overlap == 0, overlap
open("data/en/dev.txt", "w", encoding="utf-8").write("\n".join(dev) + "\n"); print("en dev", len(dev), "overlap with test", overlap)
