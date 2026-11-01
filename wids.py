#!/usr/bin/env python3
"""
wids.py - Wireless Intrusion Detection / Prevention prototype (deauth flood).

Pipeline:  frames -> 1-second windows per source MAC -> features -> RandomForest
           -> label normal / deauth_flood -> ALERT (+ WIPS: block-list the MAC)

Modes
  python wids.py demo                                   synthetic data: train, accuracy, confusion matrix, replay + alerts
  python wids.py train  --csv frames.csv [--time-split] train on a frame CSV (cols: ts,src,dst,type,subtype,rssi,label)
  python wids.py train  --pcap a.pcap --attack-macs aa:bb:..   train on a capture from your own lab
  python wids.py detect --csv frames.csv [--prevent]    offline detection on a frame CSV
  python wids.py detect --pcap lab.pcap [--prevent]     offline detection on a pcap
  python wids.py live   --iface wlan0mon [--prevent]    live detection (monitor mode, authorised lab only)

Only analyse traffic from your own / an approved isolated testbed.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

MODEL_PATH = "wids_model.joblib"
BLOCKLIST = "blocked_macs.json"
ALERT_LOG = "alerts.log"
WINDOW = 1.0  # seconds
FEATURES = ["total_frames", "deauth_count", "deauth_ratio", "mgmt_ratio",
            "unique_dst", "bcast_ratio", "rssi_mean", "rssi_std"]
BCAST = "ff:ff:ff:ff:ff:ff"
DEAUTH, DISASSOC = 12, 10  # management subtypes
COLS = ["ts", "src", "dst", "type", "subtype", "rssi", "label"]

# ---------------------------------------------------------------- features
def window_features(frames: pd.DataFrame) -> pd.DataFrame:
    """frames cols: ts, src, dst, type, subtype, rssi[, label] -> one row per (src, window)."""
    df = frames.copy()
    df["win"] = (df["ts"] // WINDOW).astype(int)
    rows = []
    for (src, win), g in df.groupby(["src", "win"]):
        n = len(g)
        deauth = ((g["type"] == 0) & g["subtype"].isin([DEAUTH, DISASSOC])).sum()
        row = {
            "src": src, "win": win, "total_frames": n, "deauth_count": deauth,
            "deauth_ratio": deauth / n,
            "mgmt_ratio": (g["type"] == 0).mean(),
            "unique_dst": g["dst"].nunique(),
            "bcast_ratio": (g["dst"] == BCAST).mean(),
            "rssi_mean": g["rssi"].mean(),
            "rssi_std": g["rssi"].std(ddof=0),
        }
        if "label" in g:
            row["label"] = int(g["label"].max())  # 1 if any malicious frame in window
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- data sources
def synthetic_frames(seed=1, duration=120):
    """Simulated lab traffic: APs, clients, probe noise, and a deauth-flood attacker."""
    rng = np.random.default_rng(seed)
    mac = lambda i: "02:00:00:00:%02x:%02x" % (i // 256, i % 256)
    aps = [mac(i) for i in range(1, 4)]
    clients = [mac(i) for i in range(100, 130)]
    out = []
    for t in range(duration):
        for ap in aps:  # beacons ~10/s + occasional legit deauth
            for _ in range(rng.poisson(10)):
                out.append((t + rng.random(), ap, BCAST, 0, 8, rng.normal(-55, 2), 0))
            if rng.random() < 0.03:
                for _ in range(rng.integers(1, 4)):
                    out.append((t + rng.random(), ap, rng.choice(clients), 0, DEAUTH, rng.normal(-55, 2), 0))
        for c in clients:  # data + probe requests
            for _ in range(rng.poisson(rng.choice([2, 8, 25]))):
                out.append((t + rng.random(), c, rng.choice(aps), 2, 0, rng.normal(-60, 6), 0))
            if rng.random() < 0.1:
                out.append((t + rng.random(), c, BCAST, 0, 4, rng.normal(-60, 6), 0))
            if rng.random() < 0.01:  # normal client disassoc
                out.append((t + rng.random(), c, rng.choice(aps), 0, DISASSOC, rng.normal(-60, 6), 0))
        # attackers: bursts of deauth frames, spoofed AP address or unknown MAC
        if rng.random() < 0.35:
            src = rng.choice(aps + [mac(200), mac(201)])
            rate = rng.integers(25, 300)
            bcast = rng.random() < 0.5
            base = rng.normal(-45, 5)
            for _ in range(rate):
                dst = BCAST if bcast else rng.choice(clients)
                out.append((t + rng.random(), src, dst, 0, DEAUTH, base + rng.normal(0, 1), 1))
    df = pd.DataFrame(out, columns=COLS)
    return df.sort_values("ts").reset_index(drop=True)

def frame_from_packet(p):
    from scapy.all import Dot11
    if not p.haslayer(Dot11):
        return None
    d = p[Dot11]
    rssi = getattr(p, "dBm_AntSignal", None)
    return (float(p.time), (d.addr2 or "00:00:00:00:00:00").lower(),
            (d.addr1 or BCAST).lower(), d.type, d.subtype,
            float(rssi) if rssi is not None else -70.0)

def pcap_to_frames(path, attack_macs=()):
    from scapy.all import PcapReader
    attack_macs = {m.lower() for m in attack_macs}
    rows = []
    with PcapReader(path) as pr:
        for p in pr:
            fr = frame_from_packet(p)
            if fr:
                fr = list(fr)
                fr.append(int(fr[1] in attack_macs and fr[3] == 0 and fr[4] in (DEAUTH, DISASSOC)))
                rows.append(fr)
    return pd.DataFrame(rows, columns=COLS)

# ---------------------------------------------------------------- training
def train(frames, tag="", time_split=False):
    feats = window_features(frames)
    X, y = feats[FEATURES], feats["label"]
    if time_split:  # train on the earliest 70% of windows, test on the latest 30%
        order = feats.sort_values("win").index
        cut = int(len(order) * 0.7)
        Xtr, ytr = X.loc[order[:cut]], y.loc[order[:cut]]
        Xte, yte = X.loc[order[cut:]], y.loc[order[cut:]]
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred)
    print(f"\n=== Results {tag}===")
    print(f"Windows: {len(feats)}  (train {len(Xtr)}, test {len(Xte)})")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion matrix [rows=true, cols=pred] (0=normal, 1=deauth flood):")
    print(cm)
    print(classification_report(yte, pred, target_names=["normal", "deauth_flood"], zero_division=0))
    imp = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1])
    print("Feature importance:", ", ".join(f"{n}={v:.2f}" for n, v in imp))
    ConfusionMatrixDisplay(cm, display_labels=["normal", "deauth_flood"]).plot(cmap="Greys")
    plt.title(f"Confusion matrix (accuracy {acc*100:.1f}%)")
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(6, 3.5))
    plt.barh([n for n, _ in imp][::-1], [v for _, v in imp][::-1], color="black")
    plt.title("Feature importance"); plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=200); plt.close()
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved {MODEL_PATH}, confusion_matrix.png, feature_importance.png")
    return clf

# ---------------------------------------------------------------- alert / prevent
def load_blocklist():
    return json.load(open(BLOCKLIST)) if os.path.exists(BLOCKLIST) else {}

def raise_alert(src, row, prevent):
    msg = (f"[ALERT] Deauth flood from {src} | {int(row['deauth_count'])} deauth frames/s "
           f"| bcast={row['bcast_ratio']:.0%} | rssi={row['rssi_mean']:.0f} dBm")
    print(msg)
    with open(ALERT_LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    if prevent and src != BCAST:  # WIPS action: block-list the offender (never the broadcast address)
        bl = load_blocklist()
        if src not in bl:
            bl[src] = time.strftime("%Y-%m-%d %H:%M:%S")
            json.dump(bl, open(BLOCKLIST, "w"), indent=2)
            print(f"[WIPS ] {src} added to {BLOCKLIST}")

def classify_windows(clf, frames, prevent):
    feats = window_features(frames)
    if feats.empty:
        return
    feats["pred"] = clf.predict(feats[FEATURES])
    for _, r in feats[feats["pred"] == 1].sort_values("win").iterrows():
        raise_alert(r["src"], r, prevent)
    print(f"\nProcessed {len(feats)} windows, {int(feats['pred'].sum())} flagged malicious.")

def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit("No model found. Run 'python wids.py demo' or 'train' first.")
    return joblib.load(MODEL_PATH)

# ---------------------------------------------------------------- live
def live(iface, prevent):
    from scapy.all import sniff
    clf = load_model()
    buf, state = [], {"start": None}
    def handle(p):
        fr = frame_from_packet(p)
        if not fr:
            return
        buf.append(fr + (0,))
        if state["start"] is None:
            state["start"] = fr[0]
        if fr[0] - state["start"] >= WINDOW:
            df = pd.DataFrame(buf, columns=COLS)
            buf.clear(); state["start"] = None
            f = window_features(df)
            f["pred"] = clf.predict(f[FEATURES])
            for _, r in f[f["pred"] == 1].iterrows():
                raise_alert(r["src"], r, prevent)
    print(f"Sniffing on {iface} (Ctrl+C to stop)...")
    sniff(iface=iface, prn=handle, store=False)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["demo", "train", "detect", "live"])
    ap.add_argument("--csv"); ap.add_argument("--pcap"); ap.add_argument("--iface")
    ap.add_argument("--attack-macs", nargs="*", default=[])
    ap.add_argument("--prevent", action="store_true", help="WIPS: block-list offending MACs")
    ap.add_argument("--time-split", action="store_true", help="train on earlier windows, test on later ones")
    a = ap.parse_args()

    if a.mode == "demo":
        print("Generating synthetic training traffic...")
        clf = train(synthetic_frames(seed=1), "(synthetic) ")
        print("\nReplaying unseen traffic (new seed) through the detector...")
        classify_windows(clf, synthetic_frames(seed=99, duration=30), a.prevent)
    elif a.mode == "train":
        if a.csv:
            frames = pd.read_csv(a.csv)
        elif a.pcap:
            frames = pcap_to_frames(a.pcap, a.attack_macs)
        else:
            sys.exit("Provide --csv or --pcap")
        train(frames, time_split=a.time_split)
    elif a.mode == "detect":
        if a.csv:
            frames = pd.read_csv(a.csv)
        elif a.pcap:
            frames = pcap_to_frames(a.pcap)
        else:
            sys.exit("Provide --csv or --pcap")
        classify_windows(load_model(), frames, a.prevent)
    else:
        if not a.iface:
            sys.exit("Provide --iface")
        live(a.iface, a.prevent)

if __name__ == "__main__":
    main()
