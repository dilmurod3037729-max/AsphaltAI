"""Emulatorda o'yin menyusida yurish: ekranni etalon miniatyuralar bilan solishtirib, kerakli tugmani bosadi.
    py flow.py ref <nom>     joriy ekranni etalon sifatida saqlaydi (tools/screens/<nom>.png)
    py flow.py where         qaysi ekran ekanini aytadi
    py flow.py step          ekranni aniqlab, unga mos tugmani bosadi
"""
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image

ADB = [r"C:\platform-tools\adb.exe", "-s", "emulator-5554"]
HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "screens")
# ekran nomi -> bosiladigan nuqta (1560x720 koordinatalari); None = faqat kut
ACTIONS = {
    "results": (1374, 621), "rewards": (1356, 612), "dq": (1356, 612), "lostaccess": (1413, 126),
    "privateroom": (1010, 440), "lobby": (1254, 621), "pause": (1332, 340), "starterpack": (1240, 125),
    "home": (1230, 630), "exitgame": (580, 533), "multiplayer": (1060, 330), "racing": None, "loading": None,
}
THUMB = (52, 24)
# menyu tartibi o'zgarib turadi (kartalar surilishi): bir xil ekranda ketma-ket urinishlarda navbatma-navbat bosiladigan nuqtalar
ALT = {"multiplayer": [(1060, 330), (1200, 330), (880, 330)], "home": [(1230, 630)]}


def shot_rgb():
    png = subprocess.run(ADB + ["exec-out", "screencap", "-p"], capture_output=True).stdout
    p = os.path.join(os.environ.get("TEMP", "."), "flow.png")
    open(p, "wb").write(png)
    return Image.open(p).convert("RGB")


def shot():
    return shot_rgb().convert("L")


def thumb(im):
    a = np.asarray(im.resize(THUMB, Image.BILINEAR), np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


PAUSE_BOX = (55, 50, 140, 140)   # poyga HUD'idagi pauza tugmasi (1560x720 koordinatalari)


def pause_patch(im):
    c = im.crop(PAUSE_BOX).resize((32, 32), Image.BILINEAR)
    a = np.asarray(c, np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def td_state(im):
    """HUD'dagi TOUCHDRIVE holati: True = ON (och yashil, ~(192,247,18)), False = OFF (qizil), None = noaniq. im = RGB."""
    a = np.asarray(im.crop((270, 138, 335, 168)).convert("RGB"), np.int32)
    green = int(((a[:, :, 1] > 200) & (a[:, :, 0] > 120) & (a[:, :, 2] < 90)).sum())
    red = int(((a[:, :, 0] > 190) & (a[:, :, 1] < 100) & (a[:, :, 2] < 130)).sum())
    if green > 40 and green > red:
        return True
    if red > 40 and red > green:
        return False
    return None


def classify(im):
    t = thumb(im)
    best, bd = None, 1e9
    for f in glob.glob(os.path.join(SC, "*.png")):
        if os.path.basename(f).startswith("hud_"):
            continue
        r = thumb(Image.open(f).convert("L"))
        d = float(np.mean((t - r) ** 2))
        if d < bd:
            best, bd = os.path.basename(f)[:-4].split("_")[0], d
    if best is not None and bd < 0.3:                       # ma'lum menyu ekraniga juda o'xshash
        return best, bd
    # poyga: pauza tugmasi shabloniga NCC (kadr mazmuni qanday bo'lmasin tugma bir xil)
    tp = pause_patch(im)
    ncc = [float(np.mean(tp * pause_patch(Image.open(f).convert("L")))) for f in glob.glob(os.path.join(SC, "hud_*.png"))]
    if ncc and max(ncc) > 0.55:
        return "racing", 0.0
    return best, bd


if __name__ == "__main__":
    os.makedirs(SC, exist_ok=True)
    cmd = sys.argv[1]
    im = shot()
    if cmd == "ref":
        n = len(glob.glob(os.path.join(SC, sys.argv[2] + "_*.png")))
        im.save(os.path.join(SC, f"{sys.argv[2]}_{n}.png"))
        print("saqlandi", sys.argv[2], n)
    else:
        name, d = classify(im)
        print(name, f"{d:.2f}")
        if cmd == "step" and name in ACTIONS and ACTIONS[name] and d < 0.9:
            subprocess.run(ADB + ["shell", "input", "tap", str(ACTIONS[name][0]), str(ACTIONS[name][1])])
            print("tap", ACTIONS[name])
