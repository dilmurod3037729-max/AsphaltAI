"""Eng yaxshi epizodlardan o'rganish (filtrlangan xulq-nusxa): barcha poygalardan progress/daq bo'yicha yuqori epizodlar tanlanadi,
siyosat ularning (holat -> harakat) juftliklarida o'rgatiladi. Onlayn DQN barqaror emas edi (G10 dan keyin yiqildi), bu usul esa barqaror
va har davradan keyin yaxshiroq epizodlar bilan qayta o'rgatiladi. Chiqish APK Q-tarmog'i formatida: Q = logit.
Kodlovchi: qurilmada ishlatilayotgan ONNX (data/live/vision.onnx) — holat embeddingi shu bilan hisoblanadi.
    py tools/safe_run.py 900 -- py Vision/offline_bc.py [--top 0.35]"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import data as dt
from offline_q import STEER_V, write_policy

HERE = os.path.dirname(os.path.abspath(__file__))
OFF, CNT = [0, 5, 8], [5, 3, 3]


def embed_onnx(D, path):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    n = len(D["frame"])
    E = np.zeros((n, 146), np.float32)
    for i in range(n):
        idx = [max(i - 2, 0), max(i - 1, 0), i]
        x = D["frame"][idx].astype(np.float32).transpose(0, 3, 1, 2).reshape(1, 9, 48, 96) / 255
        e, a = sess.run(None, {"x": x})
        E[i, :128], E[i, 128:] = e[0], a[0]
    return E


def states(D, E):
    sp = D["speed"].astype(np.float32)
    n = len(sp)
    S = np.zeros((n, 152), np.float32)
    S[:, :146] = E
    dS = np.zeros(n, np.float32); dS[1:] = sp[1:] - sp[:-1]
    S[:, 146] = sp / 200
    S[:, 147] = np.clip(dS / 20, -2, 2)
    st, dr = D["a_steer"].astype(int), D["a_drift"].astype(int)
    S[1:, 148] = STEER_V[np.clip(st[:-1], 0, 4)]
    S[1:, 149] = (dr[:-1] == 1)
    S[:, 150] = 1.0
    ep, t = D["ep"], D["t"]
    # poyga boshi APK dagidek: HUD 8 s dan ko'p yo'qolgandan keyingi birinchi qadam (qisqa uzilishlar poygani bo'lmaydi)
    new = np.r_[True, (np.diff(t) > 8000) | (D["session"][1:] != D["session"][:-1])]
    start = t[np.maximum.accumulate(np.where(new, np.arange(len(t)), 0))]
    S[:, 151] = np.clip((t - start) / 180000.0, 0, 1)    # poyga soati, APK: (now-epStart)/180000
    return S


def episode_scores(D):
    ep, now, d = D["ep"], D["t"], D["dist"]
    out = {}
    for e in np.unique(ep):
        m = np.where(ep == e)[0]
        if len(m) < 80:
            continue
        dd = np.diff(d[m]); g = (d[m][:-1] >= 0) & (d[m][1:] >= 0) & (dd >= 0) & (dd <= 3)
        mins = (now[m][-1] - now[m][0]) / 60000.0
        out[e] = float(dd[g].sum()) / max(mins, 0.1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--top", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--beta", type=float, default=0.5, help="AWR: advantage vazn kuchi (0 = oddiy BC)")
    ap.add_argument("--video", type=float, default=0.0, help="YouTube kadrlari (IDM yorlig'i) bilan rul qismiga qo'shimcha o'rgatish vazni; 0 = o'chiq")
    ap.add_argument("--min-rate", type=float, default=0.0)
    a = ap.parse_args()
    torch.set_num_threads(3)
    D = dt.load_all(a.data, None)
    t0 = time.time()
    E = embed_onnx(D, os.path.join(a.data, "live", "vision.onnx"))
    print(f"embedding {len(E)} kadr {time.time() - t0:.0f} s", flush=True)
    sc = episode_scores(D)
    rates = np.array(list(sc.values()))
    thr = max(np.quantile(rates, 1 - a.top), a.min_rate)
    keep = {e for e, r in sc.items() if r >= thr}
    print(f"epizodlar {len(sc)}, tanlangan {len(keep)} (progress/daq >= {thr:.1f}; o'rtacha {rates.mean():.1f}, eng yaxshi {rates.max():.1f})", flush=True)
    S = states(D, E)
    sel = np.where(np.isin(D["ep"], list(keep)))[0]
    sel = sel[(D["speed"][sel] >= 0) & (sel > 1) & ((np.nan_to_num(D["expl"][sel]).astype(int) & 7) == 0)]    # tasodifiy tadqiqot harakatlarini nusxalamaymiz (8 = bo'sh-tomon qoidasi, u nusxalanadi)
    A = np.stack([D["a_steer"], D["a_drift"], D["a_nitro"]], 1).astype(int)
    rs = np.random.RandomState(0)
    eps = np.array(sorted(keep)); rs.shuffle(eps)
    hold = set(eps[:max(1, len(eps) // 6)].tolist())
    va = np.array([e in hold for e in D["ep"][sel]])
    tr, vi = sel[~va], sel[va]
    print(f"o'rgatish {len(tr)}, tekshiruv {len(vi)} qadam", flush=True)
    net = nn.Sequential(nn.Linear(152, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 11))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    St, At = torch.from_numpy(S), torch.from_numpy(A)
    ce = nn.CrossEntropyLoss(reduction='none')

    def run(idx):
        o = net(St[idx]); l = 0; acc = []
        w = Wt[idx]
        for h in range(3):
            lg = o[:, OFF[h]:OFF[h] + CNT[h]]
            l = l + (ce(lg, At[idx, h]) * w).sum() / w.sum() / 3
            acc.append((lg.argmax(1) == At[idx, h]).float().mean().item())
        return l, acc

    # segment bo'yicha advantage: keyingi 12 qadamda (3 s) progress o'sishi, epizod o'rtachasidan farq
    d = D["dist"].astype(np.float32); W = np.ones(len(S), np.float32)
    for e in keep:
        m = np.where(D["ep"] == e)[0]
        dd = np.where((d[m] >= 0), d[m], np.nan)
        fut = np.full(len(m), np.nan)
        fut[:-12] = dd[12:] - dd[:-12]
        fut = np.where((fut >= 0) & (fut <= 12), fut, np.nan)
        mu = np.nanmean(fut) if np.isfinite(fut).any() else 0
        sd = np.nanstd(fut) if np.isfinite(fut).any() else 1
        adv = np.nan_to_num((fut - mu) / max(sd, 1e-3))
        W[m] = np.clip(np.exp(adv * a.beta), 0.2, 5.0)
    Wt = torch.from_numpy(W)
    Vs = Vy = None
    if a.video > 0:
        # video: IDM ishonchli (>=0.7) rul yo'nalishi -> yumshoq rul (chap -0.5 / to'g'ri / o'ng +0.5); tezlik va boshqa skalyarlar noma'lum -> tipik qiymat
        L = np.load(os.path.join(a.data, "yt", "labels.npz"))
        vs, vy = [], []
        sp_typ = float(np.median(D["speed"][sel])) / 200
        for k in L.files:
            fr = np.fromfile(os.path.join(a.data, "yt", k + ".rgb"), np.uint8).reshape(-1, 48, 96, 3).copy()
            fr[:, 15:31, 78:96] = 90
            P = L[k]; lab = P.argmax(1); conf = P.max(1) >= 0.7
            Ev = embed_onnx({"frame": fr}, os.path.join(a.data, "live", "vision.onnx"))
            Sv = np.zeros((len(fr), 152), np.float32); Sv[:, :146] = Ev
            Sv[:, 146] = sp_typ; Sv[:, 150] = 1.0; Sv[:, 151] = 0.5
            prev = np.r_[1, lab[:-1]]; Sv[:len(lab), 148] = np.array([-0.5, 0, 0.5], np.float32)[prev]
            idx = np.where(conf)[0]
            vs.append(Sv[idx]); vy.append(np.array([1, 2, 3])[lab[idx]])
        Vs, Vy = torch.from_numpy(np.concatenate(vs)), torch.from_numpy(np.concatenate(vy))
        cnt = np.bincount(Vy.numpy(), minlength=5).astype(float); vp = 1.0 / cnt[Vy.numpy()]; vp /= vp.sum()
        print(f"video namunalari {len(Vy)}, rul {np.bincount(Vy.numpy(), minlength=5)}", flush=True)
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        net.train()
        perm = rs.permutation(tr)
        for i in range(0, len(perm), 256):
            l, _ = run(torch.from_numpy(perm[i:i + 256]))
            if Vs is not None:
                j = torch.from_numpy(rs.choice(len(Vy), 128, p=vp))     # sinflar muvozanatda: IDM o'ngga og'gan (domen farqi)
                l = l + a.video * ce(net(Vs[j])[:, 0:5], Vy[j]).mean()
            opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl, acc = run(torch.from_numpy(vi))
        if vl.item() < best:
            best, best_state = vl.item(), {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 5 == 4:
            print(f"epoch {ep + 1} val_ce {vl.item():.3f} aniqlik rul/drift/nitro {np.round(acc, 2)}", flush=True)
    net.load_state_dict(best_state)

    class QWrap:
        f = [net[0], None, net[3], None, net[6]]

    q = QWrap()
    q.f = [net[0], None, net[3], None, net[6]]
    out = os.path.join(a.data, "offline")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(a.data, "live", "vision.ver")) as f:
        ver = f.read().strip()
    write_policy(q, os.path.join(out, f"policy_v2_{ver}.bin"), a.steps)
    open(os.path.join(out, "pending"), "w").close()
    print(f"yozildi: policy_v2_{ver}.bin, val_ce", round(best, 3), flush=True)


if __name__ == "__main__":
    main()
