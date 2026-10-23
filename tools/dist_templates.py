"""DIST% shablonlarini yaratish (tezlik o'qigichdagi usul): klasterlash -> varaqda nomlash -> shablonlar fayli.
    py dist_templates.py learn <qirqimlar_papkasi>        data/dist/learn/centers_big.png chiqaradi
    py dist_templates.py name 1,2,0,10,-,...               klaster tartibida nomlar (0-9 raqam, 10 = '%', - = tashla)
Natija: App/Assets/dist_templates.txt"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SPEED_DLL = os.path.join(ROOT, "Speed", "bin", "Release", "net9.0-windows", "Speed.dll")
OUT = os.path.join(ROOT, "data", "dist", "learn")
ASSET = os.path.join(ROOT, "App", "Assets", "dist_templates.txt")
W, H, S, COLS = 20, 24, 5, 8


def learn(crops):
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ, MAXG="4")
    subprocess.run(["dotnet", SPEED_DLL, "learn", crops, OUT, "24"], env=env, check=True)
    L = [list(map(float, l.split())) for l in open(os.path.join(OUT, "centers.txt"))]
    rows = (len(L) + COLS - 1) // COLS
    img = Image.new("RGB", (COLS * (W * S + 10), rows * (H * S + 24)), "white")
    d = ImageDraw.Draw(img)
    for i, v in enumerate(L):
        a = np.array(v).reshape(H, W)
        a = (255 * (1 - (a - a.min()) / (a.max() - a.min() + 1e-9))).astype("uint8")
        x, y = (i % COLS) * (W * S + 10), (i // COLS) * (H * S + 24)
        img.paste(Image.fromarray(a).resize((W * S, H * S), Image.NEAREST).convert("RGB"), (x, y + 20))
        d.text((x + 2, y + 3), f"#{i}", fill="red")
    img.save(os.path.join(OUT, "centers_big.png"))
    print("varaq:", os.path.join(OUT, "centers_big.png"))


def name(labels):
    os.makedirs(os.path.dirname(ASSET), exist_ok=True)
    subprocess.run(["dotnet", SPEED_DLL, "name", os.path.join(OUT, "centers.txt"), labels, ASSET], check=True)
    print("yozildi:", ASSET)


if __name__ == "__main__":
    if sys.argv[1] == "learn":
        learn(sys.argv[2])
    else:
        name(sys.argv[2])
