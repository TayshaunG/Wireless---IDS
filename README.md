# Wireless IDS/IPS with machine learning

Detects Wi-Fi deauthentication flood attacks with a Random Forest
(scikit-learn) and block-lists the offending MAC (WIPS action).

## Setup
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Quick demo (no dataset needed)
    python wids.py demo --prevent

## Real data: AWID3 (not included)
AWID3 needs a registration request, so it is not in this repo.
Get it from https://icsdweb.aegean.gr/awid/ (I used a Kaggle copy,
Deauth folder, CSV files 17-32).

    python convert_awid.py <folder-with-csvs> --files 17-32
    python wids.py train --csv awid_frames.csv --time-split
    python wids.py detect --csv awid_frames.csv --prevent
    python make_poster.py

## Files
- wids.py: features, training, alerts, WIPS block-list
- convert_awid.py: AWID3 CSV to the frame format wids.py reads
- make_poster.py: builds the A1 poster from the results
- peek.py, scan_labels.py: helpers for inspecting the CSVs
- results/: confusion matrix, feature importance, sample output

## Results (AWID3 Deauth, time-ordered split)
4,226 windows, 1,268 test windows: accuracy 100%,
confusion matrix [[1084, 0], [0, 184]].
Limitations: one attack type, one capture; MAC blocking can hit
spoofed or innocent devices; RSSI values in the data look unreliable.

## AI use
Claude (AI assistant) wrote most of the code (detector, data
converter, poster generator) and helped me debug. I ran and tested
it all and can explain how it works.

Ethics: I used the public AWID3 dataset and generated no attacks.