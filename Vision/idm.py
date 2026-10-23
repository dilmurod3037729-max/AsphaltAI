"""Teskari dinamika modeli (IDM, VPT usuli): kadrlar t, t+1, t+2 dan t qadamdagi RUL harakatini topadi.
O'z ma'lumotimizda (harakat ma'lum) o'rgatiladi, ko'rilmagan epizodlarda tekshiriladi, so'ng YouTube kadrlariga
taxminiy rul yorlig'i qo'yadi (faqat ishonchli joylar). Yo'nalish 3 sinf: chap / to'g'ri / o'ng (rul miqdori mashinaga bog'liq).
    py tools/safe_run.py 1500 -- py Vision/idm.py            (emulator YOPIQ)
Natija: Vision/out/idm.pt, data/yt/labels.npz"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn

import data as dt
import gpuguard

HERE = os.path.dirname(os.path.abspath(__file__))
DIR3 = np.array([0, 0, 1, 2, 2])          # rul 0..4 -> chap / to'g'ri / o'ng


class IDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(9, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(), nn.Conv2d(96, 128, 3, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 3 * 6, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        return self.f(x)


MASK = (slice(15, 31), slice(78, 96))     # videodagi qo'l kamerasi oynasi joyi: ikkala manbada ham yopiladi


def stack(frames, idx):
    fr = frames[np.stack([idx, idx + 1, idx + 2], 1)].copy()
    fr[:, :, MASK[0], MASK[1]] = 90
    return torch.from_numpy(fr).float().div(255).permute(0, 1, 4, 2, 3).reshape(len(idx), 9, 48, 96)


def augment(x):
    # domen farqi (video: boshqa mashina, yorug'lik, xira) uchun: yorqinlik/kontrast + gorizontal aks (rul ham almashadi)
    b = x.shape[0]
    x = x * (0.7 + 0.6 * torch.rand(b, 1, 1, 1, device=x.device)) + 0.1 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
    return x.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.7)
    a = ap.parse_args()
    gpuguard.preflight()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpuguard.cap_vram()
    torch.set_num_threads(3)
    th = gpuguard.Throttle()
    D = dt.load_all(a.data, None)
    ep, n = D["ep"], len(D["frame"])
    ok = np.zeros(n, bool)
    ok[:-2] = (ep[:-2] == ep[2:]) & (D["speed"][:-2] >= 30)
    y = DIR3[np.clip(D["a_steer"].astype(int), 0, 4)]
    eps = np.unique(ep); rs = np.random.RandomState(0); rs.shuffle(eps)
    va_eps = set(eps[: len(eps) // 6].tolist())
    isva = np.array([e in va_eps for e in ep])
    tr, va = np.where(ok & ~isva)[0], np.where(ok & isva)[0]
    print(f"o'rgatish {len(tr)}, tekshiruv {len(va)}; sinflar {np.bincount(y[tr], minlength=3)}", flush=True)
    w = torch.tensor(1.0 / np.maximum(np.bincount(y[tr], minlength=3), 1), dtype=torch.float32)
    w = (w / w.sum() * 3).to(dev)
    net = IDM().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss(weight=w)
    best, best_state = 0, None
    for e in range(a.epochs):
        net.train(); t0 = time.time()
        perm = rs.permutation(tr)
        for i in range(0, len(perm) - 63, 128):
            idx = perm[i:i + 128]
            x = augment(stack(D["frame"], idx).to(dev)); yy = torch.from_numpy(y[idx]).to(dev)
            flip = torch.rand(len(idx), device=dev) < 0.5
            x[flip] = x[flip].flip(-1); yy[flip] = 2 - yy[flip]
            loss = ce(net(x), yy)
            opt.zero_grad(); loss.backward(); opt.step(); th.step()
        net.eval(); P = []
        with torch.no_grad():
            for i in range(0, len(va), 512):
                P.append(net(stack(D["frame"], va[i:i + 512]).to(dev)).softmax(1).cpu().numpy())
        P = np.concatenate(P); pred = P.argmax(1); yt = y[va]
        acc = (pred == yt).mean(); bal = np.mean([(pred[yt == k] == k).mean() for k in range(3)])
        conf = P.max(1) >= a.conf
        print(f"epoch {e + 1}: aniqlik {acc:.3f}, muvozanatli {bal:.3f}, ishonchli ({conf.mean():.0%}) da {(pred[conf] == yt[conf]).mean():.3f}  {time.time() - t0:.0f}s", flush=True)
        if bal > best:
            best, best_state = bal, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    torch.save(best_state, os.path.join(HERE, "out", "idm.pt"))
    # YouTube kadrlari: qo'l kamerasi oynasi (o'ng-o'rta) kulrang bilan yopiladi
    out = {}
    for f in sorted(glob.glob(os.path.join(a.data, "yt", "*.rgb"))):
        fr = np.fromfile(f, np.uint8).reshape(-1, 48, 96, 3)
        idx = np.arange(len(fr) - 2); P = []
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                P.append(net(stack(fr, idx[i:i + 512]).to(dev)).softmax(1).cpu().numpy())
        P = np.concatenate(P)
        out[os.path.basename(f)[:-4]] = P
        print(f"{os.path.basename(f)}: {len(P)} kadr, ishonchli {(P.max(1) >= a.conf).mean():.0%}, yo'nalish {np.bincount(P.argmax(1), minlength=3)}", flush=True)
    np.savez(os.path.join(a.data, "yt", "labels.npz"), **out)
    print("eng yaxshi muvozanatli aniqlik", round(best, 3))


if __name__ == "__main__":
    main()
