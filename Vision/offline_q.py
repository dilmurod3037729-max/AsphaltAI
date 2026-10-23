"""Ofsayn Q-o'rgatish: barcha yozilgan poygalardan (rec.bin + agent.csv) shoxlangan DQN'ni PC'da o'rgatadi.
APK holati bilan bir xil: [embedding(128), aux(18), tezlik/200, dS/20, oxirgi rul, oxirgi drift, gaz]. n-qadam 5, gamma 0.97.
Natija: data/offline/policy_<kind>.bin + replay_<kind>.bin (APK formati) — emulatorga push_policy bilan yuboriladi.
    py tools/safe_run.py 1500 -- py Vision/offline_q.py [--steps 20000]   (emulator YOPIQ bo'lishi shart)"""
import argparse
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn

import data as dt
from model import Export, Net

HERE = os.path.dirname(os.path.abspath(__file__))
NS, ND, NN = 5, 3, 3
OFF, CNT = [0, NS, NS + ND], [NS, ND, NN]
NOUT = 11
GAMMA, NSTEP = 0.97, 5
EXTREME_W = 0.10
STEER_V = np.array([-1, -.5, 0, .5, 1], np.float32)


def embed(D, net_path, bs=256):
    net = Net()
    net.load_state_dict(torch.load(net_path, map_location="cpu"))
    net.eval()
    ex = Export(net)
    n = len(D["frame"])
    E = np.zeros((n, 128 + 18), np.float32)
    with torch.no_grad():
        for i in range(0, n, bs):
            idx = np.arange(i, min(n, i + bs))
            i0, i1 = np.maximum(idx - 2, 0), np.maximum(idx - 1, 0)
            fr = D["frame"][np.stack([i0, i1, idx], 1)]
            x = torch.from_numpy(fr).float().div(255).permute(0, 1, 4, 2, 3).reshape(len(idx), 9, 48, 96)
            e, a = ex(x)
            E[idx] = torch.cat([e, a], 1).numpy()
    return E


def transitions(D, E):
    """(s, a[3], R_n, s2, episode) — APK Loop() bilan bir xil: qator t dagi r = oldingi harakatning mukofoti."""
    sp = D["speed"].astype(np.float32)
    ep = D["ep"]
    n = len(sp)
    S = np.zeros((n, 152), np.float32)
    S[:, :146] = E
    dS = np.zeros(n, np.float32)
    dS[1:] = sp[1:] - sp[:-1]
    S[:, 146] = sp / 200
    S[:, 147] = np.clip(dS / 20, -2, 2)
    st, dr = D["a_steer"].astype(int), D["a_drift"].astype(int)
    S[1:, 148] = STEER_V[np.clip(st[:-1], 0, 4)]
    S[1:, 149] = (dr[:-1] == 1).astype(np.float32)
    S[:, 150] = 1.0
    S[:, 151] = 0.0
    S[:, 150], S[:, 151] = 1.0, 0.0
    # State(): s[V+2..V+4] = last[0..2] = rul, drift-flag, gaz; s[V+5] = 0 (vision)
    # V=146: 146 tezlik, 147 dS, 148 rul, 149 drift, 150 gaz, 151 0
    r = np.nan_to_num(D["r"]) + np.nan_to_num(D["bonus"])
    ext = np.zeros(n, np.float32)
    ext[1:] = -EXTREME_W * ((st[:-1] == 0) | (st[:-1] == 4))      # Agent.cs ExtremeW bilan bir xil: oldingi harakat keskin rul bo'lsa
    r = r + ext
    a = np.stack([st, dr, D["a_nitro"].astype(int)], 1)
    ok = (sp >= 0) & (np.abs(dS) < 80)
    out = []
    i = 0
    while i < n:
        j = i
        while j < n and ep[j] == ep[i]:
            j += 1
        for k in range(i + 2, j - NSTEP - 1):
            if not ok[k:k + NSTEP + 2].all():
                continue
            R = sum((GAMMA ** m) * r[k + 1 + m] for m in range(NSTEP))
            out.append((k, k + NSTEP, R, ep[i]))
        i = j
    ks = np.array([o[0] for o in out]); k2 = np.array([o[1] for o in out])
    R = np.array([o[2] for o in out], np.float32); epi = np.array([o[3] for o in out])
    return S[ks], a[ks], R, S[k2], epi


class Q(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nin, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, NOUT))
        with torch.no_grad():
            self.f[4].bias.copy_(torch.tensor([0, 0, .3, 0, 0, .3, 0, 0, .3, 0, 0]))

    def forward(self, x):
        return self.f(x)


def heads(q):
    return [q[:, OFF[h]:OFF[h] + CNT[h]] for h in range(3)]


def loss_fn(q, tq, s, a, R, s2, cql):
    with torch.no_grad():
        nxt = torch.stack([h.max(1).values for h in heads(tq(s2))], 1).mean(1)
        y = R + (GAMMA ** NSTEP) * nxt
    qs = heads(q(s))
    tot, reg = 0, 0
    for h in range(3):
        qa = qs[h].gather(1, a[:, h:h + 1])[:, 0]
        tot = tot + nn.functional.smooth_l1_loss(qa, y) / 3
        reg = reg + (torch.logsumexp(qs[h], 1) - qa).mean() / 3     # ko'rilmagan harakatlarni oshirib yubormaslik (CQL)
    return tot + cql * reg, tot


def write_policy(q, path, steps):
    with open(path, "wb") as f:
        f.write(struct.pack("<q", steps))
        sizes = [152, 128, 128, NOUT]
        f.write(struct.pack("<i", len(sizes)))
        for s in sizes:
            f.write(struct.pack("<i", s))
        for lin in (q.f[0], q.f[2], q.f[4]):
            f.write(lin.weight.detach().cpu().numpy().astype("<f4").tobytes())
            f.write(lin.bias.detach().cpu().numpy().astype("<f4").tobytes())


def write_replay(S, A, R, S2, path, cap=20000):
    S, A, R, S2 = S[-cap:], A[-cap:], R[-cap:], S2[-cap:]
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", len(S), S.shape[1]))
        for i in range(len(S)):
            f.write(S[i].astype("<f4").tobytes()); f.write(S2[i].astype("<f4").tobytes())
            f.write(struct.pack("<4b", int(A[i, 0]), int(A[i, 1]), int(A[i, 2]), 1))
            f.write(struct.pack("<f", float(R[i])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--tags", default="")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--cql", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max", type=int, default=0, help="sinov uchun: faqat birinchi N qator")
    ap.add_argument("--start-steps", type=int, default=8000, help="policy fayliga yoziladigan qadam (eps jadvali: 8000 -> 0.08)")
    a = ap.parse_args()
    torch.set_num_threads(3)
    D = dt.load_all(a.data, [t for t in a.tags.split(",") if t] or None)
    if a.max:
        D = {k: v[:a.max] for k, v in D.items()}
    ver = open(os.path.join(HERE, "..", "App", "Assets", "vision.ver")).read().strip()
    t0 = time.time()
    E = embed(D, os.path.join(HERE, "out", "vision.pt"))
    print(f"embedding {len(E)} kadr, {time.time() - t0:.0f} s; versiya {ver}", flush=True)
    S, A, R, S2, ep = transitions(D, E)
    print(f"o'tishlar {len(S)}, epizodlar {len(np.unique(ep))}, reward o'rt {R.mean():.3f} std {R.std():.3f}", flush=True)
    eps = np.unique(ep)
    rs = np.random.RandomState(0)
    hold = set(rs.choice(eps, max(1, len(eps) // 8), replace=False).tolist())
    va = np.array([e in hold for e in ep])
    tr = np.where(~va)[0]; vi = np.where(va)[0]
    tt = lambda x, dt_=torch.float32: torch.from_numpy(x).to(dt_)
    Str, Atr, Rtr, S2tr = tt(S), tt(A, torch.long), tt(R), tt(S2)
    q = Q(152); tq = Q(152); tq.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), a.lr)
    for it in range(1, a.steps + 1):
        b = torch.from_numpy(rs.choice(tr, 256))
        l, td = loss_fn(q, tq, Str[b], Atr[b], Rtr[b], S2tr[b], a.cql)
        opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad():
            for p, tp in zip(q.parameters(), tq.parameters()):
                tp.mul_(0.995).add_(0.005 * p)
        if it % 2000 == 0:
            with torch.no_grad():
                vb = torch.from_numpy(rs.choice(vi, min(2048, len(vi))))
                _, vtd = loss_fn(q, tq, Str[vb], Atr[vb], Rtr[vb], S2tr[vb], 0)
                qq = q(Str[vb]); g = [h.argmax(1).numpy() for h in heads(qq)]
            print(f"it {it} td {td.item():.3f} val_td {vtd.item():.3f}  greedy rul {np.bincount(g[0], minlength=5)} drift {np.bincount(g[1], minlength=3)} nitro {np.bincount(g[2], minlength=3)}", flush=True)
    out = os.path.join(a.data, "offline")
    os.makedirs(out, exist_ok=True)
    kind = f"v2_{ver}"
    write_policy(q, os.path.join(out, f"policy_{kind}.bin"), a.start_steps)
    write_replay(S, A, R, S2, os.path.join(out, f"replay_{kind}.bin"))
    print("yozildi:", out, kind, flush=True)


if __name__ == "__main__":
    main()
