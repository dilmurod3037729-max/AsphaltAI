"""Kadrdan yo'l ko'rinishi: HUD'ni inpaint -> Depth Anything V2 Small -> tekis yer modeli -> har ustun uchun bo'sh masofa.
Chuqurlik NISBIY (disparity: katta = yaqin). Yer tekisligi har kadrda qayta moslanadi; undan "yaqinroq" nuqta = to'siq.
Ishlatish: py perceive.py <chiqish.png> <kadr...>"""
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

NAME = "depth-anything/Depth-Anything-V2-Small-hf"
SIZE = (252, 518)           # model kirishi (balandlik, kenglik), 14 ga karrali
HUD = [(0, 0, .28, .32), (.30, 0, .68, .08), (.80, 0, 1, .48), (0, .30, .20, .56), (.70, .30, 1, .50)]  # x0,y0,x1,y1 ulushda
EGO = (.36, .62, .66, 1.0)  # o'z mashinasi (kadr ulushi)
BINS = 15

_proc = _model = None


def _load():
    global _proc, _model
    if _model is None:
        _proc = AutoImageProcessor.from_pretrained(NAME)
        _model = AutoModelForDepthEstimation.from_pretrained(NAME, torch_dtype=torch.float16).to("cuda").eval()


def clean(bgr):
    h, w = bgr.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in HUD:
        m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
    return cv2.inpaint(bgr, m, 5, cv2.INPAINT_TELEA)


def disparity(rgb):
    _load()
    x = _proc(images=Image.fromarray(rgb), return_tensors="pt", size={"height": SIZE[0], "width": SIZE[1]},
              do_resize=True, keep_aspect_ratio=False)["pixel_values"].to("cuda").half()
    with torch.no_grad():
        return _model(pixel_values=x).predicted_depth[0].float().cpu().numpy()


def free_space(d):
    """v-disparity usuli: yer = disparity qator bo'yicha silliq o'sadi (gradient ~ a); devor/mashina = gradient ~ 0.
    Har bin uchun pastdan yuqoriga yer gradienti saqlanguncha yuriladi; to'xtagan qator = to'siq (None = ufqqacha ochiq)."""
    H, W = d.shape
    g = cv2.Sobel(cv2.GaussianBlur(d, (0, 0), 2.0), cv2.CV_32F, 0, 1, ksize=3)
    y0 = int(.38 * H)
    road = g[int(.72 * H):int(.97 * H), int(.15 * W):int(.85 * W)]
    a = float(np.median(road[road > 0])) if (road > 0).any() else 1.0
    ground = g > 0.45 * a
    top = int(.38 * H)
    out = []
    for k in range(BINS):
        c0, c1 = int(k * W / BINS), int((k + 1) * W / BINS)
        ego = c1 > EGO[0] * W and c0 < EGO[2] * W
        start = int(EGO[1] * H) if ego else H - 2
        col = ground[:, c0:c1].mean(1) > .5
        bad = 0
        hit = None
        for r in range(start, top, -1):
            bad = 0 if col[r] else bad + 1
            if bad >= 4:
                hit = r + 3
                break
        out.append(hit)
    obst = ~ground
    obst[:top] = False
    return (a, 0.0), out, obst


def perceive(bgr):
    rgb = cv2.cvtColor(clean(bgr), cv2.COLOR_BGR2RGB)
    d = disparity(rgb)
    plane, free, obst = free_space(d)
    return d, plane, free, obst


def render(bgr, d, free, obst):
    H, W = bgr.shape[:2]
    n = (d - d.min()) / (d.max() - d.min() + 1e-9)
    dm = cv2.applyColorMap((255 * n).astype(np.uint8), cv2.COLORMAP_TURBO)
    dm = cv2.resize(dm, (W, H))
    ov = bgr.copy()
    ob = cv2.resize(obst.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    ov[ob] = (0.5 * ov[ob] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    h = d.shape[0]
    for k, r in enumerate(free):
        x0, x1 = int(k * W / BINS), int((k + 1) * W / BINS)
        y = H if r is None else int(r / h * H)
        cv2.rectangle(ov, (x0 + 2, y), (x1 - 2, H), (0, 255, 0) if r is None else (0, 200, 255), 3)
        cv2.line(ov, (x0, y), (x1, y), (255, 255, 255), 2)
    return np.hstack([cv2.resize(ov, (520, 240)), cv2.resize(dm, (520, 240))])


if __name__ == "__main__":
    out, files = sys.argv[1], sys.argv[2:]
    rows = []
    for f in files:
        bgr = cv2.imread(f)
        t = time.time()
        d, plane, free, obst = perceive(bgr)
        print(f, f"{(time.time() - t) * 1000:.0f} ms", "bo'sh qator:", free)
        rows.append(render(bgr, d, free, obst))
    cv2.imwrite(out, np.vstack(rows))
