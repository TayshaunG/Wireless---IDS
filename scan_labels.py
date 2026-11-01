"""scan_labels.py - label counts per CSV file in a folder (reads only the label column)."""
import sys, glob, os, pandas as pd
folder = sys.argv[1]
files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
print(f"{len(files)} csv files")
for f in files:
    cols = list(pd.read_csv(f, nrows=0, low_memory=False).columns)
    lab = [c for c in cols if c.strip().lower() in ("label", "class", "attack")]
    if not lab:
        print(f"{os.path.basename(f)}: no label column ({len(cols)} cols)")
        continue
    s = pd.read_csv(f, usecols=[lab[0]], low_memory=False)[lab[0]].value_counts()
    print(f"{os.path.basename(f)} ({os.path.getsize(f)/1e6:.0f} MB): " + ", ".join(f"{k}={v}" for k, v in s.items()))
