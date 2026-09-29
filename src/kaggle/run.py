import glob, os, sys, runpy
os.environ["SMOKE"] = "0"; os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
p = glob.glob("/kaggle/input/**/adapt_run.py", recursive=True)[0]
sys.path.insert(0, os.path.dirname(p)); runpy.run_path(p, run_name="__main__")
