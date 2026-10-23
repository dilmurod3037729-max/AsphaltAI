""""Nima qilarding?" — world model faqat ko'rilmagan (tekshiruv) epizodlarda: urilishdan 0.5 s oldin va devorga tiqilgan paytda
har rul varianti uchun 2 s kelajakni tasavvur qiladi, urilish xavfi va tezlikni bashorat qiladi.
    py Vision/dreamer_whatif.py          -> data/whatif/sheet.png, data/whatif/report.txt"""
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

import dreamer as DR
import gpuguard

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "whatif")
CTX, LEAD, HOR, K = 16, 2, 8, 8          # kontekst qadam, urilishdan oldin qaror (0.5 s), gorizont 2 s, namuna
STEER = ["chapga", "chap-", "to'g'ri", "o'ng-", "o'ngga"]
SC = 3


def onehot(steer, drift=0, nitro=0):
    a = torch.zeros(DR.NA); a[steer] = 1; a[5 + drift] = 1; a[8 + nitro] = 1
    return a


@torch.no_grad()
def imagine(wm, D, t0, plans, dev):
    """t0 gacha kuzatadi; har reja uchun HOR qadam tasavvur. plans: [HOR, NA] tensorlar ro'yxati.
    Qaytaradi: xavf [P] (1 s ichida urilish ehtimolining maksimumi), tezlik [P, HOR], kadr [P, 3,48,96] (+1 s)."""
    idx = np.arange(t0 - CTX + 1, t0 + 1)
    obs = torch.from_numpy(D.fr[idx].copy())[None].to(dev)
    act = torch.from_numpy(D.act[idx])[None].to(dev)
    first = torch.zeros(1, CTX, device=dev); first[:, 0] = 1
    H, Z, _, _, _ = wm.observe(obs, act, first)
    P = len(plans)
    h = H[:, -1].repeat(P * K, 1); z = Z[:, -1].repeat(P * K, 1)
    A = torch.stack(plans).to(dev).repeat_interleave(K, 0)            # [P*K, HOR, NA]
    risk, spd, frames = [], [], None
    for t in range(HOR):
        h, p = wm.img_step(h, z, A[:, t])
        z = wm.sample(p)
        f = torch.cat([h, z], -1)
        risk.append(torch.sigmoid(wm.collh(f)).squeeze(-1)); spd.append(wm.spdh(f).squeeze(-1) * 100)
        if t == 3:
            frames = wm.dec(f)
    R = torch.stack(risk, 1).view(P, K, HOR)[:, :, :4].amax(2).mean(1)   # keyingi 1 s ichidagi xavf
    S = torch.stack(spd, 1).view(P, K, HOR).mean(1)
    Fm = frames.view(P, K, 3, 48, 96).mean(1)
    return R.cpu().numpy(), S.cpu().numpy(), Fm.cpu().numpy()


def real_plan(D, t0):
    return torch.from_numpy(D.act[t0:t0 + HOR].copy())


def opt_plan(s):
    return torch.stack([onehot(s)] * 4 + [onehot(2)] * (HOR - 4))


def events(D, S):
    va = D.va
    crash, stuck = [], []
    for a, b in va:
        c = S["coll"][a:b]; v = S["speed"][a:b]; tr = S["t_race_ms"][a:b]
        for i in range(CTX + LEAD, b - a - HOR):
            if c[i] == 1 and c[i - 1] == 0:
                t0 = a + i - LEAD
                if D.first[t0 - CTX + 1:t0 + 1].sum() == 0:
                    crash.append(t0)
        i = CTX
        while i < b - a - HOR:
            if v[i] < 25 and tr[i] > 8000 and np.all(v[i:i + 8] < 25) and D.first[a + i - CTX + 1:a + i + 1].sum() == 0:
                stuck.append(a + i); i += 40
            else:
                i += 1
    return crash, stuck


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def tile(img, label):
    im = Image.fromarray(img).resize((96 * SC, 48 * SC), Image.NEAREST)
    canvas = Image.new("RGB", (96 * SC, 48 * SC + 18), (20, 20, 20))
    canvas.paste(im, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), label, fill=(235, 235, 235))
    return canvas


def to_img(x):
    return (np.clip(x.transpose(1, 2, 0) + 0.5, 0, 1) * 255).astype(np.uint8)


def main():
    cpu = bool(os.environ.get("WHATIF_CPU"))                  # trening GPU'da ketayotganda mantiqni sinash
    dev = "cpu" if cpu else "cuda"
    if not cpu:
        gpuguard.preflight(); gpuguard.cap_vram()
    torch.set_num_threads(3)
    D = DR.Data(64, ["emulator", "pc"])
    import build_ds2 as B
    S = B.load_steps(os.path.join(DR.DS, "steps.csv"))
    wm = DR.WorldModel().to(dev)
    ck = torch.load(DR.CKPT, map_location=dev)
    wm.load_state_dict(ck["wm"]); wm.eval()
    os.makedirs(OUT, exist_ok=True)
    crash, stuck = events(D, S)
    rs = np.random.RandomState(0)
    lines = [f"model qadam {ck['step']}; tekshiruv epizodlari {len(D.va)} (o'rganishda ko'rilmagan); urilish oldi holatlar {len(crash)}, tiqilish {len(stuck)}"]

    # 1) haqiqiy harakatlar bilan: urilishni oldindan bilyaptimi (AUC), tezlik-baza bilan solishtirish
    neg = []
    for a, b in D.va:
        for t in rs.randint(a + CTX, b - HOR, size=6):
            if np.nansum(S["coll"][t:t + 6]) == 0 and D.first[t - CTX + 1:t + 1].sum() == 0:
                neg.append(t)
    lim = 12 if cpu else 300
    pos = crash[:lim]; neg = neg[:lim]
    rp = [imagine(wm, D, t, [real_plan(D, t)], dev)[0][0] for t in pos]
    rn = [imagine(wm, D, t, [real_plan(D, t)], dev)[0][0] for t in neg]
    a_model = auc(rp, rn)
    a_speed = auc(S["speed"][pos], S["speed"][neg])
    lines.append(f"urilishni 0.5 s oldin bashorat (haqiqiy harakat bilan): AUC model {a_model:.3f}  |  faqat-tezlik baza {a_speed:.3f}  ({len(pos)} urilish / {len(neg)} oddiy)")

    # 2) "nima qilarding": har rul varianti
    better, same, chose = 0, 0, []
    rows = []
    for t in pos:
        plans = [real_plan(D, t)] + [opt_plan(s) for s in range(5)]
        R, Sp, Fm = imagine(wm, D, t, plans, dev)
        best = int(np.argmin(R[1:]))
        chose.append(best)
        if R[1 + best] < R[0] - 0.15:
            better += 1
        if best == int(np.argmax(D.act[t, :5])):
            same += 1
        rows.append((t, R, Sp, Fm))
    n = max(1, len(pos))
    lines.append(f"urilish oldidan: model {better}/{n} ({100 * better / n:.0f}%) holatda xavfi 15+ punkt past variant topdi; "
                 f"tanlovi agent harakati bilan bir xil {100 * same / n:.0f}%")
    lines.append("tanlangan rul taqsimoti: " + ", ".join(f"{STEER[s]} {chose.count(s)}" for s in range(5)))

    # 3) rasm: 6 urilish + 3 tiqilish
    pick = [rows[i] for i in rs.choice(len(rows), min(6, len(rows)), replace=False)] if rows else []
    for t in stuck[:3]:
        plans = [real_plan(D, t)] + [opt_plan(s) for s in range(5)]
        pick.append((t, *imagine(wm, D, t, plans, dev), ))
    sheet = []
    for k, (t, R, Sp, Fm) in enumerate(pick):
        kind = "URILISH" if k < 6 and t in crash else "TIQILISH"
        tiles = [tile(D.fr[t], f"{kind} hozir {S['speed'][t]:.0f} mph"),
                 tile(D.fr[t + 4], f"haqiqat +1s {S['speed'][t + 4]:.0f} mph"),
                 tile(to_img(Fm[0]), f"tasavvur(haqiqiy) xavf {100 * R[0]:.0f}%")]
        best = int(np.argmin(R[1:]))
        for s in range(5):
            mark = " <" if s == best else ""
            tiles.append(tile(to_img(Fm[1 + s]), f"{STEER[s]} xavf {100 * R[1 + s]:.0f}% {Sp[1 + s].mean():.0f}mph{mark}"))
        row = Image.new("RGB", (sum(x.width for x in tiles) + 4 * len(tiles), tiles[0].height + 4), (0, 0, 0))
        x = 0
        for tl in tiles:
            row.paste(tl, (x, 2)); x += tl.width + 4
        sheet.append(row)
    if sheet:
        W = max(r.width for r in sheet); Hh = sum(r.height for r in sheet)
        img = Image.new("RGB", (W, Hh), (0, 0, 0)); y = 0
        for r in sheet:
            img.paste(r, (0, y)); y += r.height
        img.save(os.path.join(OUT, "sheet.png"))
    open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
