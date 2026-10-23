"""Dreamer nimani ko'rmayapti: PC poyga yozuvini (rec_<tag>.bin + agent_<tag>.csv) world model orqali o'tkazadi.
  - video: haqiqiy kadr | Dreamer ichki tasviri (posterior rekonstruksiya) | 1 s oldinga tasavvur, + xatolik xaritasi
  - ko'r nuqtalar: sezilmagan urilishlar, tiqilishlar, bekor xavotirlar (kadrlari bilan)
  - holat bo'yicha xato: qorong'i/yorug', burilish, past tezlik, nitro
    py Vision/dream_diag.py <tag> [cpu]        -> data/diag_<tag>/"""
import csv
import os
import sys

import cv2
import numpy as np
import torch

import dreamer as DR

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REC = 12 + 48 * 96 * 3 + 40 * 128 + 8 * 96 * 3
NAMES = ["chap", "chap-", "to'g'ri", "o'ng-", "o'ng"]


def load(tag):
    raw = np.fromfile(os.path.join(DATA, f"rec_{tag}.bin"), np.uint8)
    n = len(raw) // REC
    raw = raw[: n * REC].reshape(n, REC)
    steps = raw[:, :4].copy().view("<i4")[:, 0]
    ms = raw[:, 4:12].copy().view("<i8")[:, 0]
    fr = raw[:, 12:12 + 48 * 96 * 3].reshape(n, 48, 96, 3)
    act = {}
    for r in csv.reader(open(os.path.join(DATA, f"agent_{tag}.csv"))):
        sx = float(r[25]) if len(r) > 25 else DR.STEER_V[int(r[3])]
        act[int(r[0])] = (int(r[2]), int(r[3]), int(r[4]), int(r[5]), sx)   # tezlik, rul, drift, nitro, joystik
    A = np.array([act.get(int(s), (-1, 2, 0, 0, 0.0)) for s in steps])
    return fr, ms, A


def main():
    tag = sys.argv[1]
    dev = "cpu" if len(sys.argv) > 2 and sys.argv[2] == "cpu" else "cuda"
    torch.set_num_threads(3)
    out = os.path.join(DATA, f"diag_{tag}"); os.makedirs(out, exist_ok=True)
    fr, ms, A = load(tag)
    n = len(fr); sp = A[:, 0].astype(float); st = A[:, 1].astype(int)
    wm = DR.WorldModel().to(dev); ck = torch.load(DR.CKPT, map_location="cpu"); wm.load_state_dict(ck["wm"], strict=False); wm.eval()
    has_st = "stuckh.0.weight" in ck["wm"]
    oh = np.zeros((n, DR.NA), np.float32)
    oh[:, :5] = DR.steer_soft(A[:, 4]); oh[np.arange(n), 5 + A[:, 2].astype(int).clip(0, 2)] = 1; oh[np.arange(n), 8 + A[:, 3].astype(int).clip(0, 2)] = 1
    gap = np.r_[True, np.diff(ms) > 1500]                                   # poyga/segment chegarasi
    rec = np.zeros((n, 48, 96, 3), np.uint8); fut = np.zeros_like(rec)
    risk = np.zeros(n); srisk = np.zeros(n); spd = np.zeros(n); err = np.zeros(n); err_map = np.zeros((48, 96))
    with torch.no_grad():
        h, z = wm.initial(1, dev); a_prev = torch.zeros(1, DR.NA, device=dev)
        for i in range(n):
            if gap[i]:
                h, z = wm.initial(1, dev); a_prev = torch.zeros(1, DR.NA, device=dev)
            x = torch.from_numpy(fr[i]).to(dev).permute(2, 0, 1)[None].float().div(255).sub(0.5)
            h, _ = wm.img_step(h, z, a_prev)
            z = wm.sample(wm.dist(wm.post(torch.cat([h, wm.enc(x)], -1))))
            f = torch.cat([h, z], -1)
            r = wm.dec(f)[0]
            rec[i] = ((r.clamp(-0.5, 0.5) + 0.5) * 255).permute(1, 2, 0).byte().cpu().numpy()
            e = (r - x[0]).pow(2).mean(0).cpu().numpy(); err[i] = e.mean(); err_map += e
            risk[i] = float(torch.sigmoid(wm.collh(f))); spd[i] = float(wm.spdh(f)) * 100
            srisk[i] = float(torch.sigmoid(wm.stuckh(f))) if has_st else risk[i]
            hh, zz = h, z                                                    # 1 s (4 qadam) haqiqiy harakatlar bilan tasavvur
            for k in range(1, 5):
                ak = torch.from_numpy(oh[min(i + k - 1, n - 1)])[None].to(dev)
                hh, p = wm.img_step(hh, zz, ak); zz = wm.sample(p)
            fut[i] = ((wm.dec(torch.cat([hh, zz], -1))[0].clamp(-0.5, 0.5) + 0.5) * 255).permute(1, 2, 0).byte().cpu().numpy()
            a_prev = torch.from_numpy(oh[i])[None].to(dev)
    err_map /= n

    # haqiqiy hodisalar
    crash = np.zeros(n, bool); stuck = np.zeros(n, bool)
    for i in range(n - 4):
        if not gap[i + 1:i + 5].any():
            crash[i] = sp[i] > 60 and sp[i] - sp[i + 1:i + 5].min() >= 30
    for i in range(n - 8):
        stuck[i] = (sp[i:i + 8] < 30).all() and (sp[i:i + 8] >= 0).all() and not gap[i + 1:i + 8].any()

    def auc(p, y):
        y = np.asarray(y, bool)
        if y.sum() == 0 or (~y).sum() == 0:
            return float("nan")
        a = p[y][:, None]; b = p[~y][None]
        return float((a > b).mean() + 0.5 * (a == b).mean())

    # emulator bilan solishtirish: o'rgatishda ko'rilmagan emulator epizodlarida rekonstruksiya xatosi
    D = DR.Data(64, ["emulator"])
    with torch.no_grad():
        b = DR.to(D.batch(8, val=True), dev)
        H, Z, _, _, x = wm.observe(b["obs"], b["act"], b["first"])
        rr = wm.dec(torch.cat([H, Z], -1).reshape(-1, H.shape[-1] + Z.shape[-1])).view(x.shape[0], x.shape[1], 3, 48, 96)
        emu_err = float((rr - x).pow(2).mean())
    bright = fr.reshape(n, -1).mean(1)
    L = [f"tag {tag}: {n} qadam, world model qadam {ck['step']}",
         f"ichki tasvir xatosi (MSE): PC {err.mean():.4f}  vs  emulator (ko'rilmagan) {emu_err:.4f}  -> PC {err.mean() / emu_err:.1f}x yomonroq",
         f"urilishni sezish AUC {auc(risk, crash):.3f} ({crash.sum()} urilish oldi qadam);  tiqilishni sezish AUC {auc(srisk, stuck):.3f} ({stuck.sum()} qadam, {'tiqilish boshi' if has_st else 'urilish boshi orqali'})",
         f"tezlik bashorati: o'rtacha xato {np.mean(spd[sp >= 0] - sp[sp >= 0]):+.1f} mph, |xato| {np.mean(np.abs(spd[sp >= 0] - sp[sp >= 0])):.1f}",
         "holat bo'yicha ichki tasvir xatosi va tezlik |xato|:"]
    groups = {"qorong'i (yorug'lik past 25%)": bright < np.percentile(bright, 25), "yorug' (yuqori 25%)": bright > np.percentile(bright, 75),
              "burilish (rul chap/o'ng)": np.isin(st, [0, 4]), "to'g'ri": st == 2, "past tezlik <40": (sp >= 0) & (sp < 40),
              "yuqori tezlik >120": sp > 120, "nitro": A[:, 3] > 0, "urilish oldi": crash, "tiqilish": stuck}
    for k, m in groups.items():
        if m.sum():
            L.append(f"   {k:32s} n={m.sum():4d}  tasvir xatosi {err[m].mean():.4f}  tezlik |xato| {np.mean(np.abs(spd[m] - sp[m])):5.1f}  o'rtacha xavf {risk[m].mean():.2f}")
    # xatolik xaritasi: kadrning qaysi qismini eng yomon ko'radi
    em = err_map / err_map.max()
    reg = {"yuqori (osmon/HUD)": em[:16].mean(), "o'rta (yo'l, uzoq)": em[16:32].mean(), "pastki (mashina, yaqin yo'l)": em[32:].mean(),
           "chap chet": em[:, :20].mean(), "o'rta ustun": em[:, 38:58].mean(), "o'ng chet": em[:, 76:].mean()}
    L.append("kadr qismi bo'yicha nisbiy xato (1 = eng yomon piksel): " + ", ".join(f"{k} {v:.2f}" for k, v in reg.items()))

    # ko'r nuqtalar: sezilmagan urilishlar (xavf past), bekor xavotir (xavf yuqori, urilish yo'q)
    starts = [i for i in range(1, n) if crash[i] and not crash[i - 1]]
    missed = sorted(starts, key=lambda i: risk[max(0, i - 2):i + 1].max())[:6]
    false = [i for i in np.argsort(-risk) if not crash[max(0, i - 4):i + 5].any()][:200]
    fsel = []
    for i in false:
        if all(abs(i - j) > 20 for j in fsel):
            fsel.append(i)
        if len(fsel) == 3:
            break
    ss = [i for i in range(1, n) if stuck[i] and not stuck[i - 1]][:3]

    def big(im, s=4):
        return cv2.resize(im, (96 * s, 48 * s), interpolation=cv2.INTER_NEAREST)

    def heat(i):
        e = np.abs(rec[i].astype(float) - fr[i].astype(float)).mean(2)
        return cv2.applyColorMap(np.clip(e * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]

    def row(i, label):
        tiles = [big(fr[i]), big(rec[i]), big(fut[i]), big(fr[min(i + 4, n - 1)]), big(heat(i))]
        im = np.concatenate([np.pad(t, ((24, 2), (2, 2), (0, 0))) for t in tiles], 1)
        cv2.putText(im, label + f"  tezlik {sp[i]:.0f}  xavf {risk[i]:.2f}  rul {NAMES[st[i]]}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return im
    head = np.zeros((22, 5 * (96 * 4 + 4), 3), np.uint8)
    for k, t in enumerate(["haqiqiy kadr", "Dreamer ichki tasviri", "Dreamer: 1 s keyin", "haqiqat: 1 s keyin", "ko'rmagan joy (yorug' = xato)"]):
        cv2.putText(head, t, (k * (96 * 4 + 4) + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    rows = [head] + [row(max(0, i - 2), "SEZILMAGAN URILISH") for i in missed] + [row(i, "TIQILISH") for i in ss] + [row(i, "BEKOR XAVOTIR") for i in fsel]
    cv2.imwrite(os.path.join(out, "blind_spots.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))

    # video: butun poyga, 4 kadr/s (agent qadami)
    vw = cv2.VideoWriter(os.path.join(out, "dreamer_eyes.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (3 * 96 * 4, 48 * 4 + 40))
    for i in range(n):
        im = np.concatenate([big(fr[i]), big(rec[i]), big(fut[i])], 1)
        im = np.pad(im, ((40, 0), (0, 0), (0, 0)))
        bar = int(risk[i] * 300)
        cv2.rectangle(im, (6, 6), (6 + bar, 16), (255, 60, 60) if risk[i] > 0.5 else (255, 200, 0) if risk[i] > 0.25 else (60, 200, 60), -1)
        cv2.putText(im, f"xavf {risk[i]:.2f}  tezlik HUD {sp[i]:.0f} / Dreamer {spd[i]:.0f}  rul {NAMES[st[i]]}" + ("  URILISH" if crash[i] else "") + ("  TIQILISH" if stuck[i] else ""),
                    (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for k, t in enumerate(["haqiqiy", "Dreamer ko'rgani", "Dreamer: 1 s keyin"]):
            cv2.putText(im, t, (k * 384 + 300, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        vw.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    vw.release()
    open(os.path.join(out, "report.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
