"""Hodisa klasterlariga nom qo'yish uchun varaq: har klasterdan 4 namuna.   py events_sheet.py [--data ../data]"""
import argparse
import os

import cv2
import numpy as np

import data as dt
import labels as lb
from train import HERE, OUT

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
a = ap.parse_args()
D = dt.load_all(a.data)
cp = os.path.join(OUT, "events_centers.npy")
cls, centers = lb.event_classes(D["evt_px"], np.load(cp) if os.path.exists(cp) else None)
rows = []
for k in range(lb.K_EVT):
    idx = np.where(cls == k)[0]
    pick = idx[np.linspace(0, len(idx) - 1, 4).astype(int)] if len(idx) else []
    cells = [cv2.resize(D["evt_px"][i], (256, 80)) for i in pick] + [np.zeros((80, 256), np.uint8)] * (4 - len(pick))
    lab = np.zeros((80, 90), np.uint8)
    cv2.putText(lab, f"#{k} n={len(idx)}", (2, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, 255, 1)
    rows.append(np.hstack([lab] + cells))
os.makedirs(OUT, exist_ok=True)
cv2.imwrite(os.path.join(OUT, "events_sheet.png"), np.vstack(rows))
print("yozildi:", os.path.join(OUT, "events_sheet.png"))
