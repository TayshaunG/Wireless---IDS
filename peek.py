"""peek.py - inspect a huge CSV without loading it all.   python peek.py <file.csv>"""
import sys, pandas as pd
f = sys.argv[1]
cols = list(pd.read_csv(f, nrows=0, low_memory=False).columns)
print(f"{len(cols)} columns. Saved full list to columns.txt")
open("columns.txt", "w").write("\n".join(cols))
keys = ("time", "label", "class", "type", "subtype", "wlan.sa", "wlan.da", "wlan.ta", "wlan.ra",
        "addr", "signal", "dbm", "rssi", "deauth", "reason")
hit = [c for c in cols if any(k in c.lower() for k in keys)]
print("Likely useful columns:")
for c in hit:
    print("  ", c)
lab = [c for c in cols if "label" in c.lower() or c.lower() in ("class", "attack")]
if lab:
    s = pd.read_csv(f, usecols=lab[:1], nrows=500000, low_memory=False)[lab[0]]
    print(f"\nFirst 500000 rows - values of {lab[0]}:")
    print(s.value_counts().head(20))
print("\nFirst 3 rows of the useful columns:")
print(pd.read_csv(f, usecols=hit[:15], nrows=3, low_memory=False).to_string())