"""Reward tizimini audit qilish: komponentlar hissasi, progress (DIST%) bilan mosligi, reward-hack alomatlari.
    py reward_audit.py [--data ../data]
Tekshiruvlar:
  1. Har komponent (tezlik, devor, yo'ldan chiqish, rul, progress, nitro) jami reward'ga qancha hissa qo'shadi.
  2. Reward va haqiqiy progress: 20 qadamlik (5 s) oynada o'rtacha reward <-> DIST% o'sishi korrelyatsiyasi.
  3. Hack alomati: yuqori tezlik (>60) lekin 8 s davomida progress yo'q (aylanish/teskari/devor bo'ylab) — bunday oynalarda reward qancha?"""
import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["step", "now", "speed", "a_steer", "a_drift", "a_nitro", "a_gas", "r", "eps", "loss", "age", "r_speed", "r_wall", "r_off",
        "r_steer", "r_prog", "dist", "ra", "off", "nitro", "evt", "coll_pred"]


def load(path):
    a = np.genfromtxt(path, delimiter=",")
    if a.ndim == 1:
        a = a[None]
    if a.shape[1] < len(COLS):
        a = np.hstack([a, np.full((len(a), len(COLS) - a.shape[1]), np.nan)])
    return {c: a[:, i] for i, c in enumerate(COLS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.data, "agent_*.csv")))
    if not files:
        raise SystemExit("agent_*.csv topilmadi")
    S = [load(f) for f in files]
    tot = sum(len(s["r"]) for s in S)
    print(f"sessiyalar {len(S)}, qadamlar {tot}")
    comp = {k: np.nansum(np.concatenate([np.abs(s[k]) for s in S])) for k in ("r_speed", "r_wall", "r_off", "r_steer", "r_prog")}
    comp["nitro"] = 0.0
    z = sum(comp.values()) or 1
    print("komponentlar hissasi (|reward| bo'yicha):", {k: f"{100 * v / z:.0f}%" for k, v in comp.items()})
    # 2) reward <-> progress
    xs, ys, hack = [], [], []
    W = 20
    for s in S:
        d, r, sp = s["dist"], np.nan_to_num(s["r"]), s["speed"]
        for i in range(0, len(r) - W, 5):
            dd = d[i:i + W + 1]
            if np.isnan(dd).any() or (dd < 0).any():
                continue
            prog = dd[-1] - dd[0]
            xs.append(r[i:i + W].mean()); ys.append(prog)
            if sp[i:i + W].mean() > 60 and prog <= 0:
                hack.append(r[i:i + W].mean())
    if len(xs) > 10:
        print(f"reward <-> progress (DIST% oshishi, 5 s oyna): korrelyatsiya {np.corrcoef(xs, ys)[0, 1]:+.2f}  (n={len(xs)})")
        allr = np.mean(xs)
        print(f"hack alomati: yuqori tezlikda progress yo'q oynalar {len(hack)} ta; ularning o'rtacha reward'i {np.mean(hack) if hack else float('nan'):.2f} (umumiy o'rtacha {allr:.2f})")
    else:
        print("DIST% ma'lumoti yetarli emas (dist o'qigich shablonlari kerak) — progress bilan solishtirib bo'lmadi")
    sp = np.concatenate([s["speed"] for s in S]); r = np.concatenate([np.nan_to_num(s["r"]) for s in S])
    for lo, hi in ((0, 30), (30, 80), (80, 140), (140, 700)):
        m = (sp >= lo) & (sp < hi)
        if m.any():
            print(f"  tezlik {lo:3d}-{hi:3d}: o'rtacha reward {r[m].mean():+.2f}  ({m.mean() * 100:.0f}% qadam)")


if __name__ == "__main__":
    main()
