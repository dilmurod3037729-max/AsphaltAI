"""Davra auditi: reward + hodisalar + diskvalifikatsiya sababi + urilishdan oldingi holat + devor yorlig'i + oldindan ko'rish.
    py audit_round.py <tag> [--foresight]       (--foresight GPU ishlatadi: emulator YOPIQ bo'lishi shart, safe_run orqali)
Natija: data/audit_<tag>.md"""
import argparse
import os
import subprocess
import sys

import numpy as np

import data as dt
import labels as lb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")


def episode_ends(D):
    ep = D["ep"]
    n = len(ep)
    start, end = dt.episode_bounds(ep)
    out = []
    for s in sorted(set(start)):
        e = end[s]
        out.append((s, e))
    return out


def dq_analysis(D, ww, lines):
    sp, dist = D["speed"], D["dist"]
    rows = []
    for s, e in episode_ends(D):
        if e - s < 20:
            continue
        last = slice(max(s, e - 24), e)                         # oxirgi ~6 s
        ww_last = float(ww[slice(max(s, e - 80), e)].mean())      # oxirgi ~20 s
        dmax = float(np.nanmax(dist[s:e])) if (dist[s:e] >= 0).any() else -1
        stall = False
        d = dist[s:e][dist[s:e] >= 0]
        if len(d) > 40:
            stall = d[-1] - d[-40] <= 0                          # oxirgi ~10 s da progress yo'q
        kind = "tugatdi (>=95%)" if dmax >= 95 else ("WRONG WAY bilan tugadi" if ww_last > 0.10 else ("progress turib qoldi" if stall else "boshqa"))
        rows.append((int(D["ep"][s]), e - s, dmax, float(sp[last].mean()), ww_last, kind))
    lines.append("## 3. Epizod tugashi (diskvalifikatsiya sababi)\n")
    lines.append("| epizod | qadam | maks DIST% | oxirgi 6 s tezlik | oxirgi 20 s WRONG WAY ulushi | tur |\n|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]:.0f} | {r[4]:.2f} | {r[5]} |")
    kinds = {}
    for r in rows:
        kinds[r[5]] = kinds.get(r[5], 0) + 1
    lines.append(f"\nXulosa: {kinds}\n")


def preimpact(D, lines):
    sp = D["speed"]; st = D["a_steer"]; dr = D["a_drift"]; ni = D["a_nitro"]; now = D["now"]
    n = len(sp)
    ok = np.r_[True, np.diff(now) <= 1500]
    ev = []
    i = 8
    while i < n - 4:
        if ok[i - 7:i + 4].all() and sp[i - 1] - sp[i:i + 4].min() >= 20 and (sp[i - 4:i - 1] - sp[i - 3:i]).max() < 10:
            ev.append(i); i += 8
        else:
            i += 1
    lines.append("## 5. Urilishdan oldingi harakatlar va to'plangan rul (holat)\n")
    lines.append(f"Keskin tezlik yo'qotish boshlanishlari (>=20 mph, oldin barqaror): **{len(ev)}**\n")
    if len(ev) < 10:
        lines.append("Hodisa soni juda kam — xulosa chiqmaydi.\n")
        return
    f = {"rul keskin": np.isin(st, [0, 4]), "drift ushlash": dr == 1, "360": dr == 2, "nitro x2": ni == 2}
    lines.append("| harakat | baza | 1 qadam oldin | 2 | 3 | 4 | 6 |\n|---|---|---|---|---|---|---|")
    for k, v in f.items():
        b = v.mean()
        lines.append(f"| {k} | {b:.2f} | " + " | ".join(f"{np.mean([v[i - l] for i in ev]) / b:.1f}x" for l in (1, 2, 3, 4, 6)) + " |")
    # to'plangan rul (holat o'rnini bosuvchi): oxirgi 8 qadamdagi (rul-2) yig'indisi moduli
    acc = np.array([abs(np.nansum(st[i - 8:i] - 2)) for i in ev])
    base = np.array([abs(np.nansum(st[j - 8:j] - 2)) for j in range(8, n)])
    lines.append(f"\nTo'plangan rul og'ishi (|sum(rul-2)| oxirgi 2 s): urilishdan oldin o'rtacha **{acc.mean():.1f}** vs baza {base.mean():.1f}  "
                 f"({'holat gipotezasi QO\'LLAB-QUVVATLANDI' if acc.mean() > 1.25 * base.mean() else 'farq yo\'q'})\n")


def wall_section(D, lines):
    lines.append("## 6. Devorni bilish\n")
    lines.append("Ma'lumotda devorga tegish yorlig'i **yo'q** (qo'lda ham, avtomatik ham). 'To'qnashuv' yorlig'i = tezlik tushishi (agent o'zi gaz/drift qilganda ham tushadi). "
                 "Tayyor segmentatsiya tunnel/off-road'da devorni topolmaydi. Natija: ko'rish devorni HALI bilmaydi.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--foresight", action="store_true")
    a = ap.parse_args()
    D = dt.load_all(DATA, [a.tag])
    ww = lb.wrongway(D["frame"])
    lines = [f"# Audit: {a.tag}\n", f"Qadamlar {len(D['speed'])}, epizodlar {len(np.unique(D['ep']))}, o'rtacha tezlik {D['speed'].mean():.0f} mph, "
             f"WRONG WAY kadrlar {100 * ww.mean():.1f}%\n"]
    r = subprocess.run([sys.executable, os.path.join(HERE, "reward_validate.py"), "--tags", a.tag], capture_output=True, text=True, cwd=HERE)
    lines.append("## 1-2. Reward mezonlari (A-E) — joriy yozilgan reward bo'yicha\n")
    lines.append("```\n" + r.stdout.strip() + "\n```\n")
    r2 = subprocess.run([sys.executable, os.path.join(HERE, "reward_validate.py"), "--tags", a.tag, "--grid"], capture_output=True, text=True, cwd=HERE)
    grid = r2.stdout.split("QAYTA QURILGAN")[-1] if "QAYTA QURILGAN" in r2.stdout else ""
    lines.append("Og'irliklar bo'yicha qayta hisoblash (DIST filtri bilan):\n```\nQAYTA QURILGAN" + grid.strip() + "\n```\n")
    dq_analysis(D, ww, lines)
    lines.append("## 4. Hodisa detektorlari\n")
    lines.append(f"WRONG WAY kadr ulushi {100 * ww.mean():.1f}%; nitro/DIST/hodisa ustuni o'qilishi: DIST o'qilgan qadamlar {100 * (D['dist'] >= 0).mean():.1f}%.\n")
    preimpact(D, lines)
    wall_section(D, lines)
    lines.append("## 7. Oldindan ko'rish (bashorat)\n")
    if a.foresight:
        r3 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "safe_run.py"), "420", "--", sys.executable, os.path.join(HERE, "crossval.py"),
                             "--epochs", "8"], capture_output=True, text=True, cwd=ROOT)
        keep = [l for l in r3.stdout.splitlines() if l.startswith(("epizod", "UMUMIY", "natija"))]
        lines.append("```\n" + "\n".join(keep) + "\n```\nMezon: harakatga shartlangan bashorat AUC >= 0.70 (ko'rilmagan epizodlarda) va faqat-tezlik bazasidan yuqori.\n")
    else:
        lines.append("(`--foresight` bilan ishga tushirilmagan)\n")
    out = os.path.join(DATA, f"audit_{a.tag}.md")
    open(out, "w", encoding="utf8").write("\n".join(lines))
    print("\n".join(lines))
    print("yozildi:", out)


if __name__ == "__main__":
    main()
