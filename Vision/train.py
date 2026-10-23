"""Ko'rish kodlovchisini o'rgatish va ONNX'ga eksport (GPU himoyasi bilan: gpuguard.py).
    py train.py [--data ../data] [--epochs 15] [--tags a1,b2] [--export]
Ma'lumot: ../data/rec_<tag>.bin + agent_<tag>.csv (APK yozadi). Natija: out/vision.pt, ../App/Assets/vision.onnx + vision.ver
Xavfsiz ishga tushirish: py ../tools/safe_run.py 300 -- py Vision/train.py ..."""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data as dt
import gpuguard
import labels as lb
from model import Export, Net, onehot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ASSETS = os.path.join(HERE, "..", "App", "Assets")
BATCH = 128


def stack_index(i, start):
    """i-2, i-1, i (epizod boshidan o'tmasdan)."""
    return np.stack([np.maximum(i - 2, start[i]), np.maximum(i - 1, start[i]), i], 1)


def make_batch(D, L, ids, aug, dev):
    idx = stack_index(ids, L["start"])
    x = torch.from_numpy(D["frame"][idx]).to(dev).float().div(255)          # [B,3,48,96,3]
    x = x.permute(0, 1, 4, 2, 3).reshape(len(ids), 9, 48, 96)
    if aug:
        b = len(ids)
        x = x * (0.8 + 0.4 * torch.rand(b, 1, 1, 1, device=dev)) + 0.1 * (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
        x = x + 0.02 * torch.randn_like(x)
        x = torch.roll(x, int(np.random.randint(-3, 4)), dims=3)
    return x.clamp(0, 1)


def act_batch(L, ids, dev):
    t = lambda a: torch.from_numpy(a[ids]).to(dev)
    return onehot(t(L["steer"]), t(L["drift"]), t(L["nitro_a"]))


def targets(L, ids, dev):
    def t(a, dtype=torch.float32):
        return torch.from_numpy(a[ids]).to(dev).to(dtype)
    return {"valid": t(L["valid"]), "coll": t(L["coll"]), "reg": torch.stack([t(L["drop"]), t(L["gain"]), t(L["ngain"])], 1),
            "evt": t(L["evt"], torch.long), "val": t(L["val"]), "ww": t(L["ww_next"])}


def losses(out, T, pos_w, ww_w):
    e, c, reg, ev, val, ww = out
    m = T["valid"]
    ms = m.sum().clamp(min=1)
    l_coll = (F.binary_cross_entropy_with_logits(c.float(), T["coll"], pos_weight=pos_w, reduction="none") * m).sum() / ms
    l_reg = (F.smooth_l1_loss(reg.float(), T["reg"], reduction="none").sum(1) * m).sum() / ms
    l_evt = F.cross_entropy(ev.float(), T["evt"])
    l_val = F.smooth_l1_loss(val.float(), T["val"])
    l_ww = (F.binary_cross_entropy_with_logits(ww.float(), T["ww"], pos_weight=ww_w, reduction="none") * m).sum() / ms
    return l_coll, l_reg, l_evt, l_val, l_ww


def auc(score, y):
    y = np.asarray(y).astype(bool)
    if y.sum() < 3 or (~y).sum() < 3:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


@torch.no_grad()
def evaluate(net, D, L, ids, dev):
    net.eval()
    th = gpuguard.Throttle(0.7)
    P = {"c": [], "reg": [], "ev": [], "val": [], "ww": []}
    for k in range(0, len(ids), 128):
        b = ids[k:k + 128]
        e, c, reg, ev, val, ww = net(make_batch(D, L, b, False, dev), act_batch(L, b, dev))
        P["c"].append(torch.sigmoid(c).cpu().numpy())
        P["reg"].append(reg.cpu().numpy())
        P["ev"].append(ev.argmax(1).cpu().numpy())
        P["val"].append(val.cpu().numpy())
        P["ww"].append(torch.sigmoid(ww).cpu().numpy())
        th.step()
    P = {k: np.concatenate(v) for k, v in P.items()}
    m = L["valid"][ids]
    res = {"auc_coll": auc(P["c"][m], L["coll"][ids][m]),
           "auc_wrongway": auc(P["ww"][m], L["ww_next"][ids][m]),
           "gain_mae_mph": float(np.abs(P["reg"][m, 1] - L["gain"][ids][m]).mean() * 50),
           "evt_acc": float((P["ev"] == L["evt"][ids]).mean()),
           "val_corr": float(np.corrcoef(P["val"], L["val"][ids])[0, 1]) if len(ids) > 5 else float("nan")}
    net.train()
    return res, P


def split(D):
    eps = np.unique(D["ep"])
    n_val = max(1, int(len(eps) * 0.15))
    val_eps = set(eps[-n_val:].tolist())
    is_val = np.array([e in val_eps for e in D["ep"]])
    return np.where(~is_val)[0], np.where(is_val)[0]


def pos_weights(L, tr, dev):
    pos = max(1.0, (~L["coll"][tr]).sum() / max(1, L["coll"][tr].sum()))
    pw = max(1.0, (~L["ww_next"][tr]).sum() / max(1, L["ww_next"][tr].sum()))
    return torch.tensor(min(20.0, float(pos)), device=dev), torch.tensor(min(20.0, float(pw)), device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--tags", default="")
    ap.add_argument("--export", action="store_true", help="ONNX'ni APK assets'iga yozish")
    ap.add_argument("--init", default="", help="oldingi vision.pt dan davom ettirish (fine-tune)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--allow-emulator", action="store_true", help="emulator ishlayotganda ham (faqat safe_run ostida!)")
    ap.add_argument("--teacher", action="store_true", help="V-JEPA 2 distillyatsiyasi: data/teacher/*.npz (teacher.py) — o'quvchi o'tmish 3 kadrdan o'qituvchining kelajakni ko'rgan embeddingini bashorat qiladi")
    ap.add_argument("--tw", type=float, default=1.0, help="distillyatsiya vazni")
    ap.add_argument("--live-dir", default="", help="onnx+ver ni shu papkaga ham yozish (adb push uchun)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gpuguard.preflight(allow_emulator=a.allow_emulator)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpuguard.cap_vram()
    torch.set_num_threads(3)
    th = gpuguard.Throttle()
    D = dt.load_all(a.data, [t for t in a.tags.split(",") if t] or None)
    L, centers = lb.build(D)
    np.save(os.path.join(OUT, "events_centers.npy"), centers)
    tr, va = split(D)
    v = L["valid"]
    print(f"namunalar {len(D['speed'])}, epizodlar {len(np.unique(D['ep']))}, o'rgatish {len(tr)}, tekshiruv {len(va)}; "
          f"to'qnashuv {L['coll'][v].mean():.3f}, WRONG WAY boshlanishi {L['ww_next'][v].mean():.3f}")
    pos_w, ww_w = pos_weights(L, tr, dev)
    net = Net(a.dropout).to(dev)
    if a.init and os.path.exists(a.init):
        net.load_state_dict(torch.load(a.init))
        print("davom ettirildi:", a.init)
    srcs, proj = [], None
    if a.teacher:
        import glob as _g
        tdir = os.path.join(a.data, "teacher")
        z = np.load(os.path.join(tdir, "ours.npz"))
        assert z["idx"].max() < len(D["frame"]), "o'qituvchi embeddinglari boshqa ma'lumot to'plamiga tegishli: teacher.py ni qayta ishga tushiring"
        # xom V-JEPA embeddinglari umumiy o'rtacha bilan ustun (tasodifiy juft cos 0.96) -> markazlashtirmasak cos yo'qotish bo'sh
        cen = lambda e: torch.from_numpy(e.astype(np.float32) - e.astype(np.float32).mean(0))
        srcs.append((D["frame"], z["idx"], cen(z["emb"]), False))
        for f in sorted(_g.glob(os.path.join(tdir, "yt_*.npz"))):
            z = np.load(f)
            fr = np.fromfile(os.path.join(a.data, "yt", os.path.basename(f)[3:-4] + ".rgb"), np.uint8).reshape(-1, 48, 96, 3).copy()
            fr[:, 15:31, 78:96] = 90
            srcs.append((fr, z["idx"], cen(z["emb"]), True))
        proj = torch.nn.Sequential(torch.nn.Linear(128, 512), torch.nn.ReLU(), torch.nn.Linear(512, 1024)).to(dev)
        print("distillyatsiya manbalari:", [len(s_[1]) for s_ in srcs], flush=True)
    params = list(net.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, a.lr / 2, weight_decay=a.wd)
    steps_per = max(1, len(tr) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        t0 = time.time()
        perm = np.random.permutation(tr)
        tot = np.zeros(5); dsum = 0.0
        for s in range(steps_per):
            ids = perm[s * BATCH:(s + 1) * BATCH]
            if len(ids) < 8:
                continue
            x, act, T = make_batch(D, L, ids, True, dev), act_batch(L, ids, dev), targets(L, ids, dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                ls = losses(net(x, act), T, pos_w, ww_w)
            loss = ls[0] + ls[1] + 0.5 * ls[2] + 0.5 * ls[3] + 1.0 * ls[4]
            if srcs:
                fr_, ix_, te_, is_yt = srcs[np.random.randint(len(srcs))]
                j = np.random.randint(0, len(ix_), BATCH)
                ii = ix_[j]
                xs = torch.from_numpy(fr_[np.stack([ii - 2, ii - 1, ii], 1)]).to(dev).float().div(255).permute(0, 1, 4, 2, 3).reshape(len(ii), 9, 48, 96)
                xs = (xs * (0.8 + 0.4 * torch.rand(len(ii), 1, 1, 1, device=dev))).clamp(0, 1)
                with torch.autocast(dev, enabled=dev == "cuda"):
                    pe = proj(net.enc(xs))
                dl = 1 - F.cosine_similarity(pe.float(), te_[j].to(dev), dim=1).mean()
                loss = loss + a.tw * dl
                dsum += float(dl.detach())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += np.array([float(l.detach()) for l in ls])
            th.step()
        if len(va):
            res, _ = evaluate(net, D, L, va, dev)
            f = lambda k: 0.0 if np.isnan(res[k]) else res[k]
            score = -f("auc_coll") - f("auc_wrongway") - f("val_corr") + res["gain_mae_mph"] / 100
        else:
            res, score = {}, float(tot.sum())
        print(f"epoch {ep + 1:2d}/{a.epochs}  yo'qotish coll/reg/evt/val/ww {np.round(tot / steps_per, 3)}  tekshiruv {({k: round(x, 3) for k, x in res.items()})}  {time.time() - t0:.0f}s  distil {dsum / steps_per:.3f}", flush=True)
        if score < best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT, "vision.pt"))
    res, _ = evaluate(net, D, L, va if len(va) else tr, dev)
    json.dump(res, open(os.path.join(OUT, "vision_report.json"), "w"), indent=1)
    print("yakuniy:", res)
    if a.export:
        export(net.cpu(), a.live_dir)


def export(net, live_dir=""):
    net.eval()
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, "vision.onnx")
    torch.onnx.export(Export(net), torch.zeros(1, 9, 48, 96), path, input_names=["x"], output_names=["emb", "aux"], opset_version=17, dynamo=False)
    h = hashlib.sha1()
    for k, v in net.state_dict().items():
        h.update(v.numpy().tobytes())
    ver = h.hexdigest()[:8]
    open(os.path.join(ASSETS, "vision.ver"), "w").write(ver)
    if live_dir:
        os.makedirs(live_dir, exist_ok=True)
        import shutil
        shutil.copy(path, os.path.join(live_dir, "vision.onnx"))
        open(os.path.join(live_dir, "vision.ver"), "w").write(ver)
    print("eksport:", path, "versiya", ver)


if __name__ == "__main__":
    main()
