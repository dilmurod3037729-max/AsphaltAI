"""PC HUD tezligini raqam shabloni bilan o'qish (winocr 2 xonali tezlikni ko'pincha o'qiy olmaydi).
Shablonlar OCR to'g'ri o'qigan 3 xonali kadrlardan yig'iladi: data/pc_speed_digits.npz.
    py tools/pc_digits.py build <race_grab.npz>"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "..", "data", "pc_speed_digits.npz")
BOX = (.058, .132, .845, .949)   # mijoz sohasiga nisbatan: MPH yorlig'i + raqamlar
GW, GH = 12, 20


def segments(f):
    """Raqam bo'laklari (chapdan o'ngga), har biri GHxGW 0/1. MPH yorlig'i yuqori 35% da — ustun proyeksiyasi pastki qismdan."""
    H, W = f.shape[:2]
    c = f[int(BOX[0] * H):int(BOX[1] * H), int(BOX[2] * W):int(BOX[3] * W)]
    on = c.astype(np.int16).mean(2) > 150
    h = on.shape[0]
    cols = on[int(h * .4):].sum(0) > 1
    out, x = [], 0
    while x < len(cols):
        if not cols[x]:
            x += 1; continue
        x0 = x
        while x < len(cols) and cols[x]:
            x += 1
        if x - x0 < h * .15:
            continue
        g = on[int(h * .2):, x0:x]
        ys = np.where(g.any(1))[0]
        g = g[ys[0]:ys[-1] + 1]
        yi = np.arange(GH) * g.shape[0] // GH
        xi = np.arange(GW) * g.shape[1] // GW
        out.append(g[yi][:, xi].astype(np.float32))
    return out


class SpeedReader:
    def __init__(self):
        z = np.load(TPL)
        self.T, self.y = z["T"].reshape(len(z["y"]), -1), z["y"]

    def __call__(self, f):
        seg = segments(f)
        if not 1 <= len(seg) <= 3:
            return -1, 1.0
        v, worst = 0, 0.0
        for s in seg:
            d = np.abs(self.T - s.reshape(-1)).mean(1)
            k = int(np.argmin(d)); v = v * 10 + int(self.y[k]); worst = max(worst, float(d[k]))
        return v, worst


def build(path):
    sys.path.insert(0, HERE)
    from pc_agent import read_speed
    d = np.load(path); R = d["R"]
    F = np.zeros((1711, 3146, 3), np.uint8)
    T, Y = [], []
    for i in range(len(R)):
        F[:R.shape[1], -R.shape[2]:] = R[i]
        v = read_speed(F)
        seg = segments(F)
        if 100 <= v <= 450 and len(seg) == 3:
            for s, ch in zip(seg, f"{v:03d}"):
                T.append(s); Y.append(int(ch))
    T, Y = np.array(T), np.array(Y)
    np.savez(TPL, T=T, y=Y)
    print("shablonlar", len(Y), "raqamlar", np.bincount(Y, minlength=10))


if __name__ == "__main__":
    build(sys.argv[2])
