"""
convert_awid.py - turn AWID3 CSV files into the frame CSV that wids.py trains on.

  python convert_awid.py C:\\Users\\taysh\\awid [--files 17-32] [--out awid_frames.csv]

Reads only 7 of the 254 columns, file by file.
Output columns: ts, src, dst, type, subtype, rssi, label   (label: 0 = Normal, 1 = attack)
"""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--files", default="17-32", help="e.g. 17-32 or 0,5,21-30")
ap.add_argument("--out", default="awid_frames.csv")
a = ap.parse_args()

nums = []
for part in a.files.split(","):
    if "-" in part:
        lo, hi = part.split("-"); nums += range(int(lo), int(hi) + 1)
    else:
        nums.append(int(part))

USE = ["frame.time_epoch", "radiotap.dbm_antsignal", "wlan.sa", "wlan.ta", "wlan.da", "wlan.ra",
       "wlan.fc.type", "wlan.fc.subtype", "Label"]
parts = []
for n in nums:
    path = os.path.join(a.folder, f"Deauth_{n}.csv")
    if not os.path.exists(path):
        print(f"skip (missing): {path}"); continue
    d = pd.read_csv(path, usecols=USE, low_memory=False)
    d = d.dropna(subset=["frame.time_epoch", "wlan.fc.type", "wlan.fc.subtype"])
    out = pd.DataFrame({
        "ts": d["frame.time_epoch"].astype(float),
        "src": d["wlan.sa"].fillna(d["wlan.ta"]).fillna("unknown").astype(str).str.lower(),
        "dst": d["wlan.da"].fillna(d["wlan.ra"]).fillna("ff:ff:ff:ff:ff:ff").astype(str).str.lower(),
        "type": d["wlan.fc.type"].astype(int),
        "subtype": d["wlan.fc.subtype"].astype(int),
        # AWID stores one value per antenna glued together (e.g. "-34-37-34"): keep the first
        "rssi": pd.to_numeric(d["radiotap.dbm_antsignal"].astype(str).str.extract(r"(-\d+)")[0],
                              errors="coerce").fillna(-70.0),
        "label": (d["Label"].astype(str).str.strip().str.lower() != "normal").astype(int),
    })
    parts.append(out)
    print(f"Deauth_{n}.csv: {len(out)} frames, {int(out['label'].sum())} attack")
if not parts:
    raise SystemExit("No files loaded.")
res = pd.concat(parts, ignore_index=True).sort_values("ts")
res.to_csv(a.out, index=False)
print(f"\nWrote {a.out}: {len(res)} frames, {int(res['label'].sum())} labelled attack")
print(f"Next: python wids.py train --csv {a.out}")