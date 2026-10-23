"""1-bosqich: reward tizimini TASDIQLASH. O'zini yaxshilash tsikliga o'tishdan oldin bajarilishi shart bo'lgan mezonlar.
    py reward_validate.py [--tags a1,L1]
Mezonlar (PASS/FAIL chop etiladi):
  A  kelajak reward (20 qadam, gamma .97) <-> haqiqiy progress (DIST% o'sishi, shu 20 qadamda) Spearman korrelyatsiyasi >= 0.6
  B  WRONG WAY boshlanishidan keyingi 1 s reward-to-go o'rtacha <= umumiy medianadan past
  C  hech bir komponent |reward| ning 50% idan ko'pini bermaydi; reward chegarasiga (clamp) urilgan qadamlar < 2%
  D  qo'rqoqlik yo'q: o'rtacha tezlik >= 45 mph va 100+ mph ulushi > 0
  E  bazalar: har qadam 0 tezlik ("to'xtab turish") va "doim o'rtacha tezlik, progress yo'q" agentdan past return olishi
Faqat toza oynalar (HUD uzilishsiz, DIST toza o'sish) ishlatiladi."""
import argparse
import glob
import os

import numpy as np

import data as dt
import labels as lb

HERE = os.path.dirname(os.path.abspath(__file__))
G, W = 0.97, 20


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def rebuild(D, off_w=0.1, prog_w=0.5, ww_w=0.5, wall_w=2.0, steer_w=0.05, speed_w=1.0, ww=None):
    """Yozilgan komponentlardan reward'ni yangi og'irliklar bilan qayta hisoblaydi (emulatorsiz). a1 (eski APK) komponentlari:
    r_off 0.4 bilan, r_prog 1.0 bilan, r_wall 2.0 bilan, r_steer 0.05 bilan yozilgan. DIST xatolari (raqam tashlash) filtrlanadi."""
    n = len(D["speed"])
    nz = lambda k: np.nan_to_num(D[k])
    nitro = nz("r") - (nz("r_speed") + nz("r_wall") + nz("r_off") + nz("r_steer") + nz("r_prog"))
    nitro = np.where(np.abs(nitro) > 4, 0, nitro)                       # clamp ta'siri
    d = D["dist"]
    prog = np.zeros(n)
    for i in range(1, n):
        if D["ep"][i] == D["ep"][i - 1] and d[i] >= 0 and d[i - 1] >= 0 and 0 < d[i] - d[i - 1] <= 3:
            prog[i] = d[i] - d[i - 1]
    r = speed_w * nz("r_speed") + (wall_w / 2.0) * nz("r_wall") + (off_w / 0.4) * nz("r_off") + (steer_w / 0.05) * nz("r_steer") + prog_w * prog + nitro
    if ww is not None:
        r = r - ww_w * ww
    return np.clip(r, -8, 5), prog


def evaluate_criteria(D, r, ww, verbose=True):
    n = len(r)
    ep = D["ep"]; sp = D["speed"]; d = D["dist"]
    start, end = dt.episode_bounds(ep)
    rtg = np.zeros(n); prog = np.full(n, np.nan)
    for i in range(n):
        j = min(end[i], i + W)
        rtg[i] = sum(r[k] * G ** (k - i) for k in range(i, j))
        if j - i >= W // 2 and d[i] >= 0 and d[j - 1] >= 0:
            seg = np.diff(d[i:j])
            if (seg >= 0).all() and (seg <= 3).all():
                prog[i] = d[j - 1] - d[i]
    m = ~np.isnan(prog)
    A = spearman(rtg[m], prog[m]) if m.sum() > 30 else float("nan")
    onset = [i for i in range(1, n) if ww[i] and not ww[i - 1] and end[i] - i > 4]
    B = (np.mean([r[i:i + 4].sum() for i in onset]) - np.median([r[i:min(end[i], i + 4)].sum() for i in range(0, n, 4)])) if onset else float("nan")
    comp = {k: np.abs(np.nan_to_num(D[k])).sum() for k in ("r_speed", "r_wall", "r_off", "r_steer", "r_prog")}
    tot = sum(comp.values()) or 1
    clamp = float(((r <= -7.99) | (r >= 4.99)).mean())
    return {"A": A, "B": B, "clamp": clamp, "speed": float(sp.mean()), "fast": float((sp > 100).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--tags", default="")
    ap.add_argument("--grid", action="store_true", help="komponent og'irliklari bo'yicha qayta hisoblash jadvali")
    a = ap.parse_args()
    D = dt.load_all(a.data, [t for t in a.tags.split(",") if t] or None)
    n = len(D["speed"])
    ep = D["ep"]
    r = np.nan_to_num(D["r"]); sp = D["speed"]; d = D["dist"]
    ww = lb.wrongway(D["frame"])
    start, end = lb.episode_bounds(ep) if hasattr(lb, "episode_bounds") else dt.episode_bounds(ep)
    # reward-to-go (W qadam) va progress (W qadam)
    rtg = np.zeros(n); prog = np.full(n, np.nan)
    for i in range(n):
        j = min(end[i], i + W)
        rtg[i] = sum(r[k] * G ** (k - i) for k in range(i, j))
        if j - i >= W // 2 and d[i] >= 0 and d[j - 1] >= 0:
            seg = np.diff(d[i:j])
            if (seg >= 0).all() and (seg <= 3).all():
                prog[i] = d[j - 1] - d[i]
    m = ~np.isnan(prog)
    resA = spearman(rtg[m], prog[m]) if m.sum() > 30 else float("nan")
    print(f"A  reward-to-go <-> progress: Spearman {resA:+.2f} (n={int(m.sum())})   {'PASS' if resA >= 0.6 else 'FAIL'}  (mezon >= 0.6)")
    # B
    onset = [i for i in range(1, n) if ww[i] and not ww[i - 1] and end[i] - i > 4]
    if onset:
        after = np.mean([r[i:i + 4].sum() for i in onset])
        base = np.median([r[i:min(end[i], i + 4)].sum() for i in range(0, n, 4)])
        print(f"B  WRONG WAY boshlanishidan keyingi 1 s reward {after:+.2f}  vs  umumiy mediana {base:+.2f}  (hodisalar {len(onset)})   {'PASS' if after <= base else 'FAIL'}")
    else:
        print("B  WRONG WAY hodisasi topilmadi")
    # C
    comp = {k: np.abs(np.nan_to_num(D[k])).sum() for k in ("r_speed", "r_wall", "r_off", "r_steer", "r_prog")}
    tot = sum(comp.values()) or 1
    share = {k: v / tot for k, v in comp.items()}
    clamp = float(((r <= -7.99) | (r >= 4.99)).mean())
    okC = max(share.values()) <= 0.5 and clamp < 0.02
    print(f"C  komponent hissalari { {k: f'{100 * v:.0f}%' for k, v in share.items()} }  clamp {100 * clamp:.1f}%   {'PASS' if okC else 'FAIL'}")
    # D
    fast = float((sp > 100).mean())
    print(f"D  o'rtacha tezlik {sp.mean():.0f} mph, 100+ mph ulushi {100 * fast:.1f}%   {'PASS' if sp.mean() >= 45 and fast > 0 else 'FAIL'}")
    # E: bazalar — progress yo'q qadamlarda reward agentning shunday qadamlaridan past bo'lishi kerak
    stall = m & (prog == 0)
    move = m & (prog > 0)
    if stall.sum() > 10 and move.sum() > 10:
        print(f"E  progress=0 oynalar reward-to-go {rtg[stall].mean():+.2f}  <  progress>0 oynalar {rtg[move].mean():+.2f}   {'PASS' if rtg[stall].mean() < rtg[move].mean() else 'FAIL'}")
    print(f"   (jami qadam {n}, epizodlar {len(np.unique(ep))})")
    if a.grid:
        print("\nQAYTA QURILGAN reward (komponentlar og'irliklari o'zgartirilgan):  A>=0.6, B<0, clamp<2%")
        print(f"{'off':>5s} {'prog':>5s} {'ww':>4s} {'speed':>6s} | {'A':>6s} {'B':>7s} {'clamp%':>7s}")
        for off_w in (0.0, 0.1):
            for prog_w in (0.5, 1.0, 2.0):
                for ww_w in (0.5, 1.0, 2.0):
                    for speed_w in (1.0,):
                        rr, _ = rebuild(D, off_w, prog_w, ww_w, speed_w=speed_w, ww=ww.astype(float))
                        c = evaluate_criteria(D, rr, ww)
                        flag = "  <-- o'tdi" if (c["A"] >= 0.6 and c["B"] < 0 and c["clamp"] < 0.02) else ""
                        print(f"{off_w:5.1f} {prog_w:5.1f} {ww_w:4.1f} {speed_w:6.1f} | {c['A']:+6.2f} {c['B']:+7.2f} {100 * c['clamp']:7.1f}{flag}")


if __name__ == "__main__":
    main()
