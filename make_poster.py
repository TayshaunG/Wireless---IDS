"""
make_poster.py - builds the A1 poster from your REAL results.

Run in the same folder as wids.py and awid_frames.csv, after:
    python wids.py detect --csv awid_frames.csv --prevent
Then:
    python make_poster.py
Options: --csv FILE   --random-split   --png preview.png
"""
import argparse, json, os, textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import wids  # your own wids.py, same folder

# ---------------- EDIT THESE ----------------
AUTHOR = ("Tayshaun Gitonga  |  BSc Computer Networks and Cyber Security"
          "  |  Strathmore University")
DATASET = "AWID3 Deauth capture (Kaggle copy), files 17-32"
# --------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="awid_frames.csv")
ap.add_argument("--random-split", action="store_true")
ap.add_argument("--png")
a = ap.parse_args()

# ---------------- compute results from the data ----------------
frames = pd.read_csv(a.csv)
feats = wids.window_features(frames)
FE = wids.FEATURES
X, y = feats[FE], feats["label"]
if a.random_split:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    split_txt = "70/30 stratified random split of windows"
else:
    order = feats.sort_values("win").index
    cut = int(len(order) * 0.7)
    Xtr, ytr = X.loc[order[:cut]], y.loc[order[:cut]]
    Xte, yte = X.loc[order[cut:]], y.loc[order[cut:]]
    split_txt = ("Time-ordered split: train on the earliest 70% of windows, "
                 "test on the latest 30%")
clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
acc = accuracy_score(yte, pred)
cm = confusion_matrix(yte, pred, labels=[0, 1])
p, r, f1, _ = precision_recall_fscore_support(
    yte, pred, labels=[1], zero_division=0)
p, r, f1 = float(p[0]), float(r[0]), float(f1[0])
fp, fn = int(cm[0][1]), int(cm[1][0])
imp = sorted(zip(FE, clf.feature_importances_), key=lambda t: -t[1])
flagged = int(clf.predict(X).sum())
n_frames, n_att = len(frames), int(frames["label"].sum())
att = frames.loc[frames["label"] == 1, "rssi"]
nrm = frames.loc[frames["label"] == 0, "rssi"]
rssi_att = float(att.median()) if n_att else float("nan")
rssi_norm = float(nrm.median())
blocked = []
if os.path.exists("blocked_macs.json"):
    blocked = list(json.load(open("blocked_macs.json")).keys())
alert_lines = []
if os.path.exists("alerts.log"):
    first = open("alerts.log").readline().strip()
    if "[ALERT]" in first:
        parts = first[first.index("[ALERT]"):].split(" | ")
        mac = parts[0].split("from ")[-1]
        alert_lines = ["[ALERT] Deauth flood from", mac + " | " + parts[1]]
        if mac in blocked:
            alert_lines += ["[WIPS ] " + mac + " added", "to blocked_macs.json"]
        alert_lines += ["(all alerts also in alerts.log)"]
if not alert_lines:
    alert_lines = ["Run wids.py detect --prevent", "then re-run make_poster.py"]
print("accuracy=%.4f cm=%s flagged=%d blocked=%d"
      % (acc, cm.tolist(), flagged, len(blocked)))

# ---------------- poster ----------------
W, H = 33.1, 23.4
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
F = "DejaVu Sans"
M, G = 0.8, 0.7
CW = (W - 2 * M - 2 * G) / 3
X0 = [M, M + CW + G, M + 2 * (CW + G)]
TOP = H - 4.2
BODY = 23

def para(x, y, text, size=BODY, w=CW, lh=1.4):
    width = int(w * 72 / (size * 0.60))
    lines = []
    for pp in text.split("\n"):
        lines += textwrap.wrap(pp, width) or [""]
    for i, l in enumerate(lines):
        ax.text(x, y - i * size * lh / 72, l, fontsize=size, va="top", family=F)
    return y - len(lines) * size * lh / 72

def heading(x, y, text):
    ax.add_patch(Rectangle((x, y - 0.9), CW, 0.9, fc="black"))
    ax.text(x + 0.3, y - 0.45, text, fontsize=30, color="white",
            weight="bold", va="center", family=F)
    return y - 1.2

def bullets(x, y, items, size=BODY):
    for it in items:
        ax.text(x, y, "\u25aa", fontsize=size, va="top", family=F)
        y = para(x + 0.5, y, it, size, CW - 0.5) - 0.18
    return y

def label(x, y, text, size, **kw):
    ax.text(x, y, text, fontsize=size, family=F, **kw)

# header
ax.add_patch(Rectangle((0, H - 3.8), W, 3.8, fc="black"))
label(0.9, H - 1.2, "Wireless Intrusion Detection and Prevention "
      "with Machine Learning", 58, color="white", weight="bold", va="center")
label(0.9, H - 2.35, "Detecting Wi-Fi deauthentication flood attacks with a "
      "Random Forest classifier (WIPS prototype)", 32, color="white", va="center")
label(0.9, H - 3.15, AUTHOR, 26, color="white", va="center")

# column 1
y = heading(X0[0], TOP, "1. Problem and aim")
y = para(X0[0], y,
         "Deauthentication frames are unauthenticated management frames. "
         "An attacker can spoof them to repeatedly disconnect clients from "
         "an access point, causing denial of service and enabling evil twin "
         "and handshake-capture attacks. Aim: detect deauth floods "
         "automatically, raise an alert and block the offender.") - 0.4
y = heading(X0[0], y, "2. How the system works")
steps = ["Wi-Fi frames (pcap, CSV or live)", "1-second windows per source MAC",
         "8 features per window", "Random Forest (scikit-learn)",
         "Label: normal or deauth flood", "ALERT + log; WIPS blocks the MAC"]
bh, gap = 0.8, 0.3
for i, s in enumerate(steps):
    yy = y - i * (bh + gap) - bh
    ax.add_patch(FancyBboxPatch((X0[0] + 0.4, yy), CW - 0.8, bh,
                 boxstyle="round,pad=0.04", fc="white", ec="black", lw=3))
    label(X0[0] + CW / 2, yy + bh / 2, s, 22, ha="center", va="center")
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch(
            (X0[0] + CW / 2, yy - 0.01), (X0[0] + CW / 2, yy - gap + 0.01),
            arrowstyle="-|>", mutation_scale=30, lw=3, color="black"))
y = y - len(steps) * (bh + gap) - 0.1
y = heading(X0[0], y, "3. Features (per MAC, per second)")
y = bullets(X0[0], y, [
    "Frame count; deauth/disassoc count and ratio",
    "Management-frame ratio; unique destinations",
    "Broadcast ratio; signal strength (RSSI) mean and std"])

# column 2
y = heading(X0[1], TOP, "4. Data and model")
y = bullets(X0[1], y, [
    "Dataset: %s: %s frames, %s labelled attack."
    % (DATASET, format(n_frames, ","), format(n_att, ",")),
    "Frames grouped per source MAC per second: %s windows."
    % format(len(feats), ","),
    "Random Forest, 100 trees, depth 8. %s." % split_txt,
    "Same code also reads own monitor-mode pcaps and live capture "
    "on an authorised lab."]) - 0.1
y = heading(X0[1], y, "5. Results")
label(X0[1], y - 0.1, "Accuracy: %.1f%%" % (acc * 100), 36,
      weight="bold", va="top")
if fp == 0 and fn == 0:
    errs = "No false alarms, no missed attacks."
else:
    errs = "%d false alarms, %d missed attacks." % (fp, fn)
y = para(X0[1], y - 0.85,
         "%s unseen test windows. Attack-class precision %.2f, recall %.2f, "
         "F1 %.2f. %s" % (format(len(yte), ","), p, r, f1, errs)) - 0.3
cs = 1.55
mx = X0[1] + 1.3
my = y - 0.7
label(mx + cs, y - 0.1, "Confusion matrix", 22, weight="bold",
      ha="center", va="center")
for i in range(2):
    for j in range(2):
        dark = i == j
        ax.add_patch(Rectangle((mx + j * cs, my - (i + 1) * cs), cs, cs,
                     fc="black" if dark else "white", ec="black", lw=3))
        label(mx + j * cs + cs / 2, my - i * cs - cs / 2,
              format(int(cm[i][j]), ","), 30, weight="bold", ha="center",
              va="center", color="white" if dark else "black")
for j, l in enumerate(["Normal", "Flood"]):
    label(mx + j * cs + cs / 2, my - 2 * cs - 0.25, l, 20,
          ha="center", va="center")
for i, l in enumerate(["Normal", "Flood"]):
    label(mx - 0.1, my - i * cs - cs / 2, l, 20, ha="right", va="center")
label(mx + cs, my - 2 * cs - 0.65, "Predicted", 20, ha="center", va="center")
label(X0[1], my - cs, "Actual", 20, rotation=90, ha="center", va="center")
bx = X0[1] + 6.3
bw = CW - 6.3 - 0.1
label(bx - 1.0 + (bw + 1.0) / 2, y - 0.1, "Feature importance", 22,
      weight="bold", ha="center", va="center")
top = imp[0][1] or 1.0
for i, (n, v) in enumerate(imp):
    yy = my - i * 0.4 - 0.1
    label(bx + 1.5, yy - 0.15, n, 15, ha="right", va="center")
    ax.add_patch(Rectangle((bx + 1.6, yy - 0.3),
                 max(v, 0.002) / top * (bw - 1.7), 0.28, fc="black"))
y = my - 2 * cs - 1.0
y = para(X0[1], y, "Note: %s is the strongest feature (%.2f) because attack "
         "frames are deauth frames. Subtler attacks would be harder."
         % (imp[0][0], imp[0][1]))

# column 3
y = heading(X0[2], TOP, "6. Alert and WIPS action")
bh3 = 2.9
ax.add_patch(Rectangle((X0[2], y - bh3), CW, bh3, fc="white", ec="black", lw=3))
for i, l in enumerate(alert_lines[:5]):
    ax.text(X0[2] + 0.3, y - 0.3 - i * 0.5, l, fontsize=20,
            family="DejaVu Sans Mono", va="top")
y -= bh3 + 0.25
y = para(X0[2], y, "The detector flagged %s of %s windows and block-listed %d "
         "source MACs. The list can feed an AP MAC filter or controller."
         % (format(flagged, ","), format(len(feats), ","), len(blocked))) - 0.3
y = heading(X0[2], y, "7. Limitations")
y = bullets(X0[2], y, [
    "One attack type, one capture. RSSI values are implausibly low "
    "(median %.0f dBm attack, %.0f dBm normal), so signal features "
    "are unreliable." % (rssi_att, rssi_norm),
    "Spoofed MACs: %d addresses were block-listed, possibly victim "
    "devices, so MAC blocking can hit innocent devices." % len(blocked),
    "WIPS only records the MAC; it does not yet push it to an AP."]) - 0.1
y = heading(X0[2], y, "8. What I would improve")
y = bullets(X0[2], y, [
    "Test on other AWID3 attacks (disassoc, evil twin) and own lab captures.",
    "Evaluate on a completely separate capture.",
    "Use an allow-list of known APs; detect spoofing with sequence numbers.",
    "Push blocks to the AP or controller automatically."])

# footer
HALF = (W - 2 * M) / 2
ax.add_patch(Rectangle((M, 0.5), W - 2 * M, 2.3, fc="white", ec="black", lw=3))
label(M + 0.3, 2.55, "AI use statement", 26, weight="bold", va="top")
para(M + 0.3, 1.95,
     "Claude (AI assistant) helped write the detection code, the AWID3 data "
     "converter and this poster generator, and helped me debug. I ran and "
     "tested everything myself and can explain how the code works.",
     23, HALF - 0.9)
label(M + 0.3 + HALF, 2.55, "Ethics", 26, weight="bold", va="top")
para(M + 0.3 + HALF, 1.95,
     "Training used the public AWID3 dataset. I generated no attacks; any "
     "own capture is limited to an authorised isolated testbed.",
     23, HALF - 0.6)

fig.savefig("WIDS_Poster_A1.pdf", format="pdf")
if a.png:
    fig.savefig(a.png, dpi=40)
print("Saved WIDS_Poster_A1.pdf (A1 landscape)")