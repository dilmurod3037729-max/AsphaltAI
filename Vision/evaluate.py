"""Ko'rish kodlovchisini tekshirish:
(1) to'qnashuvni 1 soniya oldin aytish AUC — faqat tezlik bilan solishtirib (kodlovchi vizual ma'lumot qo'shyaptimi?);
(2) saliency: kodlovchi kadrning QAYSI qismiga qarab to'qnashuvni bashorat qilyapti (devor/mashina/yo'l?).
    py evaluate.py [--data ../data]"""
import argparse
import os

import cv2
import numpy as np
import torch

import data as dt
import gpuguard
import labels as lb
from model import Net
from train import HERE, OUT, auc, evaluate, make_batch, split


def speed_baseline(D, L, tr, va):
    """Faqat [tezlik, oxirgi o'zgarish] -> to'qnashuv (logistik regressiya)."""
    v = D["speed"].astype(np.float32)
    dv = np.r_[0, np.diff(v)].astype(np.float32)
    X = np.stack([v / 100, dv / 20], 1)
    y = L["coll"].astype(np.float32)
    m = L["valid"]
    w = torch.zeros(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], 0.05)
    Xt, yt = torch.tensor(X[tr][m[tr]]), torch.tensor(y[tr][m[tr]])
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    s = (torch.tensor(X[va]) @ w + b).detach().numpy()
    return auc(s[m[va]], L["coll"][va][m[va]])


def saliency(net, D, L, ids, dev, path, k=8):
    net.eval()
    rows = []
    for i in ids[:k]:
        x = make_batch(D, L, np.array([i]), False, dev).requires_grad_(True)
        _, c, *_ = net(x)
        c.sum().backward()
        g = x.grad[0].abs().reshape(3, 3, 48, 96).sum((0, 1)).cpu().numpy()
        g = cv2.GaussianBlur(g / (g.max() + 1e-9), (0, 0), 1.5)
        cur = D["frame"][i]
        heat = cv2.applyColorMap((255 * g).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        ov = (0.55 * cur + 0.45 * heat).astype(np.uint8)
        rows.append(np.hstack([cv2.resize(cur, (384, 192)), cv2.resize(ov, (384, 192))]))
    if rows:
        cv2.imwrite(path, np.vstack(rows)[:, :, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    a = ap.parse_args()
    gpuguard.preflight()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpuguard.cap_vram()
    D = dt.load_all(a.data)
    centers = np.load(os.path.join(OUT, "events_centers.npy"))
    L, _ = lb.build(D, centers)
    tr, va = split(D)
    net = Net().to(dev)
    net.load_state_dict(torch.load(os.path.join(OUT, "vision.pt")))
    net.eval()
    res, P = evaluate(net, D, L, va, dev)
    base = speed_baseline(D, L, tr, va)
    print(f"to'qnashuvni 1 s oldin aytish (tekshiruv epizodlari): kodlovchi AUC={res['auc_coll']:.3f}   faqat tezlik AUC={base:.3f}")
    print("boshqa:", {k: round(v, 3) for k, v in res.items() if k != "auc_coll"})   # auc_wrongway: WRONG WAY boshlanishini oldindan aytish
    m = L["valid"][va]
    order = np.argsort(-P["c"])
    tp = [va[j] for j in order if L["coll"][va][j] and m[j]][:8]
    fp = [va[j] for j in order if not L["coll"][va][j] and m[j]][:8]
    os.makedirs(OUT, exist_ok=True)
    saliency(net, D, L, tp, dev, os.path.join(OUT, "saliency_true_collisions.png"))
    saliency(net, D, L, fp, dev, os.path.join(OUT, "saliency_false_alarms.png"))
    print("saliency rasmlari:", OUT)


if __name__ == "__main__":
    main()
