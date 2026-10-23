"""Tun bo'yi trening: bir necha sozlama epizodlar bo'yicha o'zaro tekshiruvda taqqoslanadi, eng yaxshisi bilan yakuniy model o'rgatiladi.
GPU himoyasi: gpuguard (preflight: emulator YOPIQ bo'lishi shart; VRAM cheklovi; ish aylanishi ~50%; util>75% da pauza).
ALBATTA safe_run ostida ishga tushiring (80% dan oshsa o'ldiradi):
    py tools/safe_run.py 14400 -- py Vision/overnight.py [--tags a1,L1] [--folds 6]
Natija: data/overnight_report.md, Vision/out/vision.pt, App/Assets/vision.onnx (+ data/live/)"""
import argparse
import json
import os
import time

import numpy as np
import torch

import data as dt
import gpuguard
import labels as lb
from crossval import speed_only
from model import Net
from train import BATCH, HERE, act_batch, auc, evaluate, export, losses, make_batch, pos_weights, targets

CONFIGS = [
    {"name": "bazaviy", "epochs": 8, "lr": 2e-3, "wd": 1e-2, "dropout": 0.2},
    {"name": "kuchli-regul", "epochs": 15, "lr": 1e-3, "wd": 5e-2, "dropout": 0.4},
    {"name": "uzun-regul", "epochs": 25, "lr": 5e-4, "wd": 5e-2, "dropout": 0.5},
    {"name": "qisqa-sekin", "epochs": 6, "lr": 5e-4, "wd": 1e-1, "dropout": 0.5},
]


def fit(D, L, tr, cfg, dev, th, epochs=None):
    net = Net(cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), cfg["lr"] / 2, weight_decay=cfg["wd"])
    ne = epochs or cfg["epochs"]
    steps = max(1, len(tr) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=ne * steps)
    pos_w, ww_w = pos_weights(L, tr, dev)
    for _ in range(ne):
        perm = np.random.permutation(tr)
        for s in range(steps):
            ids = perm[s * BATCH:(s + 1) * BATCH]
            if len(ids) < 8:
                continue
            with torch.autocast(dev, enabled=dev == "cuda"):
                ls = losses(net(make_batch(D, L, ids, True, dev), act_batch(L, ids, dev)), targets(L, ids, dev), pos_w, ww_w)
            loss = ls[0] + ls[1] + 0.5 * ls[2] + 0.5 * ls[3] + ls[4]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            th.step()
    return net


def folds_of(D, k):
    eps = np.unique(D["ep"])
    groups = [eps[i::k] for i in range(k)]
    return [g for g in groups if len(g)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--tags", default="")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--final-epochs", type=int, default=0)
    a = ap.parse_args()
    gpuguard.preflight()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpuguard.cap_vram()
    torch.set_num_threads(3)
    th = gpuguard.Throttle()
    D = dt.load_all(a.data, [t for t in a.tags.split(",") if t] or None)
    L, centers = lb.build(D)
    np.save(os.path.join(HERE, "out", "events_centers.npy"), centers)
    fl = folds_of(D, a.folds)
    print(f"namunalar {len(D['speed'])}, epizodlar {len(np.unique(D['ep']))}, foldlar {len(fl)}; to'qnashuv {L['coll'][L['valid']].mean():.3f}, WRONG WAY {L['ww_next'][L['valid']].mean():.3f}", flush=True)
    results = []
    for cfg in CONFIGS:
        t0 = time.time()
        pc, pw, ps, yc, yw = [], [], [], [], []
        for g in fl:
            va = np.where(np.isin(D["ep"], g))[0]
            tr = np.where(~np.isin(D["ep"], g))[0]
            if len(va) < 60:
                continue
            net = fit(D, L, tr, cfg, dev, th)
            _, P = evaluate(net, D, L, va, dev)
            m = L["valid"][va]
            pc.append(P["c"][m]); pw.append(P["ww"][m]); ps.append(speed_only(D, L, tr, va)[m])
            yc.append(L["coll"][va][m]); yw.append(L["ww_next"][va][m])
            del net
            torch.cuda.empty_cache()
        c, w, s_, y1, y2 = [np.concatenate(x) for x in (pc, pw, ps, yc, yw)]
        r = {"cfg": cfg, "auc_coll": auc(c, y1), "auc_ww": auc(w, y2), "auc_speed_base": auc(s_, y1), "n": int(len(y1)), "coll_pos": int(y1.sum()),
             "ww_pos": int(y2.sum()), "sec": round(time.time() - t0)}
        r["score"] = (r["auc_coll"] - r["auc_speed_base"]) + (r["auc_ww"] - 0.5 if not np.isnan(r["auc_ww"]) else 0)
        results.append(r)
        print(f"{cfg['name']:14s} to'qnashuv AUC {r['auc_coll']:.3f} (faqat-tezlik {r['auc_speed_base']:.3f})  WRONG-WAY AUC {r['auc_ww']:.3f}  ({r['sec']} s)", flush=True)
    best = max(results, key=lambda r: r["score"])
    print("eng yaxshi sozlama:", best["cfg"]["name"], flush=True)
    # yakuniy model: barcha ma'lumotda
    allidx = np.arange(len(D["speed"]))
    net = fit(D, L, allidx, best["cfg"], dev, th, epochs=a.final_epochs or best["cfg"]["epochs"])
    out = os.path.join(HERE, "out")
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, os.path.join(out, "vision.pt"))
    export(net.cpu(), os.path.join(HERE, "..", "data", "live"))
    rep = os.path.join(HERE, "..", "data", "overnight_report.md")
    with open(rep, "w", encoding="utf8") as f:
        f.write(f"# Tun bo'yi trening hisoboti ({time.strftime('%Y-%m-%d %H:%M')})\n\nNamunalar {len(allidx)}, foldlar {len(fl)}. Baholash: ko'rilmagan epizod guruhlari (o'zaro tekshiruv).\n\n")
        f.write("| sozlama | to'qnashuv AUC | faqat-tezlik AUC | WRONG WAY AUC | to'qnashuv/WW soni | vaqt |\n|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['cfg']['name']} | {r['auc_coll']:.3f} | {r['auc_speed_base']:.3f} | {r['auc_ww']:.3f} | {r['coll_pos']}/{r['ww_pos']} | {r['sec']} s |\n")
        f.write(f"\nEng yaxshi: **{best['cfg']['name']}**. 2-bosqich mezoni: AUC >= 0.70 va faqat-tezlik bazasidan yuqori.\n")
    json.dump(results, open(os.path.join(out, "overnight.json"), "w"), indent=1, default=str)
    print("hisobot:", rep)


if __name__ == "__main__":
    main()
