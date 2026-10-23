"""Curriculum bosqichi: data/ds2 ustida AWR (advantage-vaznli xulq-nusxa), reward = rewards.STAGES[k].
Siyosat APK/PC formatida alohida faylga yoziladi (jonli siyosatga tegmaydi): data/offline/policy_v2_<ver>_s<k>.bin
    py Vision/curriculum.py --stage 0 [--init data/offline/policy_v2_<ver>.bin]
Holat APK bilan bir xil (152): kodlovchi emb+aux (146), tezlik/200, dS/20, oldingi rul X, oldingi drift, gaz, poyga soati.
Advantage: 20 qadamlik diskontlangan reward-to-go (epizod ichida) minus tezlik bo'lagi bo'yicha o'rtacha; tadqiqot harakatlari ham kiradi
(AWR ularning yaxshisini o'zi tanlaydi). Kodlovchi chiqishi data/ds2/emb_<ver>.npy ga keshlanadi. Faqat CPU."""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

import build_ds2 as B
import rewards as RW
from offline_q import write_policy

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "data", "ds2")
OFF, CNT = [0, 5, 8], [5, 3, 3]
G, W = 0.97, 20


def embeddings(n, ver):
    path = os.path.join(DS, f"emb_{ver}.npy")
    if os.path.exists(path) and np.load(path, mmap_mode="r").shape[0] == n:
        return np.load(path)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 3
    sess = ort.InferenceSession(os.path.join(HERE, "..", "data", "live", "vision.onnx"), so, providers=["CPUExecutionProvider"])
    fr = np.memmap(os.path.join(DS, "frames.u8"), np.uint8, "r", shape=(n, 48, 96, 3))
    S = B.load_steps(os.path.join(DS, "steps.csv"))
    first = S["is_first"] == 1
    E = np.zeros((n, 146), np.float32)
    t0 = time.time()
    for i in range(n):
        a = i if first[i] else i - 1
        b = a if (first[i] or first[a]) else a - 1         # APK halqasi: segment boshida eng eski kadr takrorlanadi
        x = fr[[b, a, i]].astype(np.float32).transpose(0, 3, 1, 2).reshape(1, 9, 48, 96) / 255
        e, au = sess.run(None, {"x": x})
        E[i, :128], E[i, 128:] = e[0], au[0]
        if i % 20000 == 0:
            print(f"  embedding {i}/{n} {time.time() - t0:.0f} s", flush=True)
    np.save(path, E)
    return E


def states(S, E):
    n = len(E)
    X = np.zeros((n, 152), np.float32)
    X[:, :146] = E
    sp = S["speed"].astype(np.float32)
    dS = np.r_[0, np.diff(sp)]; dS[S["is_first"] == 1] = 0
    X[:, 146] = sp / 200
    X[:, 147] = np.clip(dS / 20, -2, 2)
    X[:, 148] = np.array([-1, -.5, 0, .5, 1], np.float32)[S["prev_steer"].astype(int).clip(0, 4)]
    X[:, 149] = S["prev_drift"] == 1
    X[:, 150] = 1.0
    X[:, 151] = np.clip(S["t_race_ms"] / 180000.0, 0, 1)
    return X


def advantage(S, r, ends):
    n = len(r)
    rtg = np.zeros(n, np.float32)
    for i in range(n):
        j = min(ends[i], i + W)
        rtg[i] = (r[i:j] * G ** np.arange(j - i)).sum()
    b = np.clip(S["speed"] // 10, 0, 30).astype(int)
    base = np.zeros(31, np.float32)
    for k in range(31):
        m = b == k
        if m.any():
            base[k] = rtg[m].mean()
    adv = rtg - base[b]
    return adv / max(adv.std(), 1e-6), rtg


def load_init(net, path):
    import struct
    buf = open(path, "rb").read()
    nl = struct.unpack_from("<i", buf, 8)[0]; o = 12 + 4 * nl
    for lin in (net[0], net[3], net[6]):
        w = np.frombuffer(buf, "<f4", lin.weight.numel(), o).reshape(lin.weight.shape); o += 4 * w.size
        bb = np.frombuffer(buf, "<f4", lin.bias.numel(), o); o += 4 * bb.size
        lin.weight.data = torch.from_numpy(w.copy()); lin.bias.data = torch.from_numpy(bb.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--init", default="")
    ap.add_argument("--source", default="emulator,pc")
    a = ap.parse_args()
    torch.set_num_threads(3)
    ver = open(os.path.join(HERE, "..", "data", "live", "vision.ver")).read().strip()
    S = B.load_steps(os.path.join(DS, "steps.csv"))
    n = len(S["speed"])
    E = embeddings(n, ver)
    X = states(S, E)
    r = RW.STAGES[a.stage](S)
    eps = [e for e in csv.DictReader(open(os.path.join(DS, "episodes.csv"))) if not e["excluded"] and e["start_idx"] != "-1"]
    ends = np.zeros(n, np.int64)
    for e in eps:
        ends[int(e["start_idx"]):int(e["end_idx"])] = int(e["end_idx"])
    adv, rtg = advantage(S, r, ends)
    Wt = np.clip(np.exp(a.beta * adv), 0.1, 10.0).astype(np.float32)
    srcs = set(a.source.split(","))
    use = np.isin(S["source"], list(srcs)) & (S["speed_glitch"] == 0)
    rs = np.random.RandomState(0)
    ids = np.array(sorted({e["episode_id"] for e in eps}))
    rs.shuffle(ids)
    hold = set(ids[:max(1, len(ids) // 6)].tolist())
    va = np.isin(S["episode_id"], list(hold))
    tr, vi = np.where(use & ~va)[0], np.where(use & va)[0]
    A = np.stack([S["a_steer"], S["a_drift"], S["a_nitro"]], 1).astype(np.int64).clip(0, 4)
    A[:, 1:] = A[:, 1:].clip(0, 2)
    print(f"{a.stage}-bosqich: qadam {n}, o'rgatish {len(tr)}, tekshiruv {len(vi)}, vazn o'rtacha {Wt[tr].mean():.2f} max {Wt[tr].max():.1f}", flush=True)
    net = nn.Sequential(nn.Linear(152, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 11))
    if a.init:
        load_init(net, a.init)
    opt = torch.optim.AdamW(net.parameters(), 5e-4 if a.init else 1e-3, weight_decay=1e-2)
    Xt, At, Wtt = torch.from_numpy(X), torch.from_numpy(A), torch.from_numpy(Wt)
    ce = nn.CrossEntropyLoss(reduction="none")

    def run(idx, weighted=True):
        o = net(Xt[idx]); l = 0
        w = Wtt[idx] if weighted else torch.ones(len(idx))
        for h in range(3):
            l = l + (ce(o[:, OFF[h]:OFF[h] + CNT[h]], At[idx, h]) * w).sum() / w.sum() / 3
        return l, o

    best, best_state = 1e9, None
    for ep in range(a.epochs):
        net.train()
        perm = rs.permutation(tr)
        for i in range(0, len(perm), 512):
            l, _ = run(torch.from_numpy(perm[i:i + 512]))
            opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl, _ = run(torch.from_numpy(vi))
        if vl.item() < best:
            best, best_state = vl.item(), {k: v.clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep + 1} val_awr {vl.item():.3f}", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        o = net(Xt[torch.from_numpy(vi)]).numpy()
    g = [o[:, OFF[h]:OFF[h] + CNT[h]].argmax(1) for h in range(3)]
    print("tekshiruv: ma'lumot vs yangi siyosat (greedy)")
    for h, name in enumerate(("rul", "drift", "nitro")):
        print(f"  {name}: ma'lumot {np.round(np.bincount(A[vi, h], minlength=CNT[h]) / len(vi), 3)}  siyosat {np.round(np.bincount(g[h], minlength=CNT[h]) / len(vi), 3)}")
    straight = S["a_steer"][vi] == 2
    print(f"  to'g'ri rulda drift: ma'lumot {(A[vi, 1][straight] > 0).mean():.3f}  siyosat {(g[1][straight] > 0).mean():.3f}")
    hi, lo = adv[vi] > 0.5, adv[vi] < -0.5
    for h, name in enumerate(("rul", "drift", "nitro")):
        agree = g[h] == A[vi, h]
        print(f"  {name} mosligi: yaxshi oqibatli harakatlar {agree[hi].mean():.3f}  yomon oqibatlilar {agree[lo].mean():.3f}")

    class QWrap:
        f = [net[0], None, net[3], None, net[6]]

    out = os.path.join(HERE, "..", "data", "offline", f"policy_v2_{ver}_s{a.stage}.bin")
    write_policy(QWrap(), out, 80000)
    print("yozildi:", out, flush=True)


if __name__ == "__main__":
    main()
