"""Epizodlar bo'yicha o'zaro tekshiruv: har safar BITTA epizod tekshiruvga, qolganlari o'rgatishga. Kodlovchi ko'rilmagan epizodda
to'qnashuvni oldindan aytayaptimi (AUC) va bu faqat-tezlik bazasidan yaxshimi?   py crossval.py [--epochs 8] [--tags a1]"""
import argparse
import os

import numpy as np
import torch

import data as dt
import gpuguard
import labels as lb
from model import Net
from train import BATCH, HERE, act_batch, auc, evaluate, losses, make_batch, pos_weights, targets


def speed_only(D, L, tr, va):
    v = D["speed"].astype(np.float32)
    X = np.stack([v / 100, np.r_[0, np.diff(v)].astype(np.float32) / 20], 1)
    y = L["coll"].astype(np.float32)
    m = L["valid"]
    w = torch.zeros(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], 0.05)
    Xt, yt = torch.tensor(X[tr][m[tr]]), torch.tensor(y[tr][m[tr]])
    for _ in range(200):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (torch.tensor(X[va]) @ w + b).detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    gpuguard.preflight()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpuguard.cap_vram()
    torch.set_num_threads(3)
    th = gpuguard.Throttle()
    D = dt.load_all(a.data, [t for t in a.tags.split(",") if t] or None)
    L, _ = lb.build(D)
    eps = np.unique(D["ep"])
    allc, alls, ally = [], [], []
    for e in eps:
        va = np.where(D["ep"] == e)[0]
        tr = np.where(D["ep"] != e)[0]
        m = L["valid"][va]
        if L["coll"][va][m].sum() < 2 or len(va) < 60:
            continue
        pos_w, ww_w = pos_weights(L, tr, dev)
        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
        for ep in range(a.epochs):
            perm = np.random.permutation(tr)
            for s in range(max(1, len(tr) // BATCH)):
                ids = perm[s * BATCH:(s + 1) * BATCH]
                if len(ids) < 8:
                    continue
                with torch.autocast(dev, enabled=dev == "cuda"):
                    ls = losses(net(make_batch(D, L, ids, True, dev), act_batch(L, ids, dev)), targets(L, ids, dev), pos_w, ww_w)
                loss = ls[0] + ls[1] + 0.5 * ls[2] + 0.5 * ls[3] + ls[4]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                th.step()
        res, P = evaluate(net, D, L, va, dev)
        base = speed_only(D, L, tr, va)
        y = L["coll"][va][m]
        a_enc, a_base = auc(P["c"][m], y), auc(base[m], y)
        print(f"epizod {int(e):2d}: n={len(va):4d} to'qnashuv={int(y.sum()):3d}  kodlovchi AUC={a_enc:.3f}  faqat-tezlik AUC={a_base:.3f}  WRONG-WAY AUC={res['auc_wrongway']:.3f}", flush=True)
        allc.append(P["c"][m]); alls.append(base[m]); ally.append(y)
        del net, opt
        torch.cuda.empty_cache()
    if allc:
        c, s, y = np.concatenate(allc), np.concatenate(alls), np.concatenate(ally)
        print(f"\nUMUMIY (ko'rilmagan epizodlar, n={len(y)}, to'qnashuv {int(y.sum())}): kodlovchi AUC={auc(c, y):.3f}   faqat-tezlik AUC={auc(s, y):.3f}")


if __name__ == "__main__":
    main()
