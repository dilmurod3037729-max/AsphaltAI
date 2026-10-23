"""Poyga videosi (alohida jarayon): o'yin oynasini 15 kadr/s yozadi, ustiga agentning data/dream_live.json dagi oxirgi qarorini chizadi.
data/video_go fayli bor paytda yozadi; yo'qolsa videoni yopadi.   py tools/pc_video.py <chiqish.mp4>"""
import json
import os
import sys
import time

import cv2
import numpy as np

from pc_env import Game

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
LIVE, GO = os.path.join(DATA, "dream_live.json"), os.path.join(DATA, "video_go")
NAMES = ["chap", "chap-", "to'g'ri", "o'ng-", "o'ng"]
FPS, OW, OH = 15, 1280, 720

out = sys.argv[1]
g = Game()
while not os.path.exists(GO):
    time.sleep(0.5)
wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OW, OH))
t_next, n = time.time(), 0
while os.path.exists(GO):
    t_next += 1 / FPS
    time.sleep(max(0.0, t_next - time.time()))
    f = g.frame()
    if f is None:
        continue
    im = cv2.resize(np.ascontiguousarray(f[..., :3]), (OW, OH), interpolation=cv2.INTER_AREA)
    try:
        L = json.load(open(LIVE))
    except (OSError, ValueError):
        L = None
    if L:
        x0, y0, bw = 20, OH - 150, 60
        for s in range(5):
            r = L["risk"][s]
            col = (0, 0, 255) if r > 0.5 else (0, 200, 255) if r > 0.25 else (0, 200, 0)
            h = int(100 * r)
            cv2.rectangle(im, (x0 + s * (bw + 10), y0 + 100 - h), (x0 + s * (bw + 10) + bw, y0 + 100), col, -1)
            cv2.rectangle(im, (x0 + s * (bw + 10), y0), (x0 + s * (bw + 10) + bw, y0 + 100), (255, 255, 255), 1)
            mark = (255, 255, 0) if s == L["final"] else (200, 200, 200)
            cv2.putText(im, NAMES[s], (x0 + s * (bw + 10), y0 + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mark, 2 if s == L["final"] else 1)
        msg = (f"FAQAT DREAMER  joystik: {L.get('sx', 0):+.2f}  nitro: {'HA' if L.get('nitro') else 'yo`q'}  drift: {'HA' if L.get('drift') else 'yo`q'}  tezlik {L.get('speed', 0)}" if L.get("only")
               else f"Dreamer xavfi (1 s)   siyosat: {NAMES[L['policy']]}  ->  yakuniy: {NAMES[L['final']]}")
        if L["override"]:
            msg += "   << DREAMER ALMASHTIRDI"
        cv2.putText(im, msg, (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(im, msg, (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if L["override"] else (255, 255, 255), 2)
    wr.write(im); n += 1
wr.release()
print(f"video {out}: {n} kadr ({n / FPS:.0f} s)", flush=True)
