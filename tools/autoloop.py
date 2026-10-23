"""Bir xarita (Nevada, Private Room) bo'yicha o'zini yaxshilash tsikli. Har davra:
  1. emulator + APK, agent 10 poyga o'ynaydi va o'rganadi (ko'rish: eng so'nggi kodlovchi)
  2. loglar olinadi, emulator YOPILADI (GPU/CPU bo'shashi uchun)
  3. davra ko'rsatkichlari hisoblanadi, plato tekshiriladi
  4. kodlovchi BARCHA yig'ilgan ma'lumotda qayta o'rgatiladi (gpuguard + safe_run: GPU/VRAM < 80%), APK qayta quriladi
Plato: oxirgi 3 davrada eng yaxshi progress/daqiqa >5% ga oshmasa to'xtaydi va xulosa chiqaradi.
    py autoloop.py [--races 10] [--max-rounds 12] [--plateau 3] [--minutes 50]
Natija: data/autoloop.json, data/autoloop_summary.md"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import devctl  # noqa: E402
import trainloop  # noqa: E402

DATA = os.path.join(devctl.ROOT, "data")
VISION = os.path.join(devctl.ROOT, "Vision")
LOG = os.path.join(DATA, "autoloop.json")


def metrics(tag):
    """Davra ko'rsatkichlari agent_<tag>.csv dan. Progress faqat DIST% ning toza o'sishidan (0..3 oralig'ida)."""
    p = os.path.join(DATA, f"agent_{tag}.csv")
    if not os.path.exists(p):
        return None
    a = np.genfromtxt(p, delimiter=",")
    a = a[None] if a.ndim == 1 else a
    if len(a) < 100:
        return {"steps": len(a), "ok": False}
    now, sp, r, d = a[:, 1], a[:, 2], a[:, 7], a[:, 16] if a.shape[1] > 16 else np.full(len(a), -1.0)
    ok = np.r_[True, (np.diff(now) <= 1500) & (np.diff(now) > 0)]
    minutes = float((np.diff(now)[ok[1:]]).sum()) / 60000.0
    dd = np.diff(d)
    good = (d[:-1] >= 0) & (d[1:] >= 0) & (dd >= 0) & (dd <= 3) & ok[1:]
    prog = float(dd[good].sum())
    ep = np.cumsum(~ok)
    finished = int(sum(1 for e in np.unique(ep) if d[ep == e].max() >= 95))
    ww = a[:, 22] if a.shape[1] > 22 else np.full(len(a), np.nan)
    # haqiqiy marra = natijalar ekranidan keyin "rewards" ekrani (DQ da yo'q); finish rate + Wilson 95% oralig'i
    ed = os.path.join(DATA, f"ends_{tag}")
    names = os.listdir(ed) if os.path.isdir(ed) else []
    n_end = len({f.split("_")[0] for f in names if f.endswith(("_dq.png", "_results.png", "_rewards.png"))})
    # marra: agentning o'z FINISH bonusi (DIST>=99, poyga >20 s). Ekran klassifikatori marra va DQ ekranlarini adashtiradi
    n_fin = int((np.nan_to_num(a[:, 23]) >= 20).sum()) if a.shape[1] > 23 else 0
    n_end = max(n_end, n_fin)
    p_, z = n_fin / max(n_end, 1), 1.96
    c = (p_ + z * z / (2 * max(n_end, 1))) / (1 + z * z / max(n_end, 1))
    h = z * np.sqrt(p_ * (1 - p_) / max(n_end, 1) + z * z / (4 * max(n_end, 1) ** 2)) / (1 + z * z / max(n_end, 1))
    return {"races_ended": n_end, "finishes": n_fin, "finish_rate_ci95": [round(max(0, c - h), 2), round(min(1, c + h), 2)],
            "steps": len(a), "ok": True, "minutes": round(minutes, 1), "progress_per_min": round(prog / max(minutes, 0.1), 2),
            "avg_speed": round(float(sp[sp >= 0].mean()), 1), "mean_reward": round(float(np.nan_to_num(r).mean()), 3),
            "fast_share_100mph": round(float((sp > 100).mean()), 3), "dist95_reached": finished,
            "wrong_way_rate": None if np.isnan(ww).all() else round(float(np.nanmean(ww)), 3)}


def train_encoder(tags):
    devctl.stop_emulator()
    cmd = [sys.executable, os.path.join(devctl.ROOT, "tools", "safe_run.py"), "900", "--", sys.executable, os.path.join(VISION, "train.py"),
           "--epochs", "12", "--lr", "5e-4", "--wd", "5e-2", "--dropout", "0.5", "--export", "--tags", ",".join(tags)]
    init = os.path.join(VISION, "out", "vision.pt")
    if os.path.exists(init):
        cmd += ["--init", init]
    if subprocess.run(cmd, cwd=devctl.ROOT).returncode != 0:
        return False
    # eksport tekshiruvi: ONNX == PyTorch, embedding tirik. Buzilgan bo'lsa yangi kodlovchi ishlatilmaydi.
    chk = subprocess.run([sys.executable, os.path.join(VISION, "check_export.py"), "--tags", tags[-1]], cwd=VISION)
    return chk.returncode == 0


def live(a):
    """JONLI rejim: har poyga = bitta davra. Emulator yoqiq turadi; poygadan keyin kodlovchi ustiga o'rgatiladi (fine-tune, 3 epoch),
    yangi onnx emulatorga nusxalanadi (APK qayta qurilmaydi); siyosat saqlanadi.
    GPU xavfsizligi: emulator yoqiq paytda GPU band (>45%) bo'lsa o'rgatish o'tkazib yuboriladi; o'rgatish safe_run ostida (80% dan oshsa o'ldiriladi).
    Plato: o'rtacha(oxirgi 3 poyga) progress/daq eng yaxshi o'rtachadan >5% oshmasa a.plateau poyga ketma-ket."""
    import gpuguard_helper as gh
    live_dir = os.path.join(DATA, "live")
    hist = json.load(open(LOG)) if os.path.exists(LOG) else []
    if not devctl.emulator_running() or a.reinstall:
        if not devctl.build():
            print("APK qurilmadi — to'xtadim"); return
        devctl.start_emulator()
        devctl.install(); devctl.clear(); devctl.capture(); devctl.launch()
        time.sleep(5)
    vals = [h["metrics"]["progress_per_min"] for h in hist if h.get("metrics", {}).get("ok")]
    best_roll, stale, fails = 0.0, 0, 0
    for i in range(3, len(vals) + 1):
        r = float(np.mean(vals[i - 3:i]))
        if r > best_roll * 1.05:
            best_roll, stale = r, 0
        else:
            stale += 1
    start = len(hist) + 1
    for rnd in range(start, start + a.max_rounds):
        tag = f"{a.prefix}{rnd}"
        print(f"\n=== JONLI poyga {rnd} ({tag}) ===", flush=True)
        trainloop.run(a.minutes, tag, "start", max_races=1)
        m = metrics(tag)
        print("ko'rsatkichlar:", m, flush=True)
        if not m or not m.get("ok"):
            fails += 1
            if fails >= 3:
                print("3 poyga ketma-ket ma'lumot yig'ilmadi — to'xtadim"); break
            continue
        fails = 0
        hist.append({"round": rnd, "tag": tag, "races_played": 1, "metrics": m, "time": time.strftime("%Y-%m-%d %H:%M")})
        json.dump(hist, open(LOG, "w"), indent=1)
        vals.append(m["progress_per_min"])
        if len(vals) >= 3:
            roll = float(np.mean(vals[-3:]))
            if roll > best_roll * 1.05:
                best_roll, stale = roll, 0
            else:
                stale += 1
            print(f"progress/daq {m['progress_per_min']}  o'rtacha(3) {roll:.1f}  eng yaxshi {best_roll:.1f}  yaxshilanmagan {stale}/{a.plateau}", flush=True)
            if stale >= a.plateau:
                print("PLATO: yaxshilanish to'xtadi"); break
        tags = [h["tag"] for h in hist if os.path.exists(os.path.join(DATA, f"rec_{h['tag']}.bin"))]
        util, mem = gh.gpu_state()
        if util is not None and util > 45:
            print(f"GPU band ({util:.0f}%) — bu poygadan keyin o'rgatish o'tkazib yuborildi", flush=True)
            continue
        cmd = [sys.executable, os.path.join(devctl.ROOT, "tools", "safe_run.py"), "300", "--", sys.executable, os.path.join(VISION, "train.py"),
               "--epochs", "3", "--lr", "3e-4", "--export", "--allow-emulator", "--live-dir", live_dir, "--tags", ",".join(tags)]
        init = os.path.join(VISION, "out", "vision.pt")
        if os.path.exists(init):
            cmd += ["--init", init]
        ok = subprocess.run(cmd, cwd=devctl.ROOT).returncode == 0
        if ok:
            devctl.push_vision(live_dir)
            print("kodlovchi yangilandi va emulatorga yuborildi", flush=True)
        else:
            print("o'rgatish o'tmadi/to'xtatildi (GPU himoyasi) — avvalgi kodlovchi bilan davom etaman", flush=True)
    print("\nJONLI rejim tugadi. Xulosa: data/autoloop.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", type=int, default=10)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--plateau", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=50)
    ap.add_argument("--prefix", default="L")
    ap.add_argument("--live", action="store_true", help="har poyga = bitta davra (emulator yoqiq, jonli o'rgatish)")
    ap.add_argument("--reinstall", action="store_true")
    ap.add_argument("--bc", action="store_true", help="har davradan keyin siyosat eng yaxshi epizodlardan qayta o'rgatiladi (offline_bc.py)")
    ap.add_argument("--fixed", action="store_true", help="kodlovchi qotirilgan, siyosat davralar bo'yi saqlanadi (plato yo'q)")
    a = ap.parse_args()
    if a.live:
        return live(a)
    hist = json.load(open(LOG)) if os.path.exists(LOG) else []
    best, stale, fails = 0.0, 0, 0
    for h in hist:
        v = h["metrics"].get("progress_per_min", 0) if h.get("metrics", {}).get("ok") else 0
        if v > best * 1.05:
            best, stale = v, 0
        else:
            stale += 1
    start_round = len(hist) + 1
    for rnd in range(start_round, start_round + a.max_rounds):
        tag = f"{a.prefix}{rnd}"
        print(f"\n=== DAVRA {rnd} ({tag}): {a.races} poyga ===", flush=True)
        if not devctl.build():
            print("APK qurilmadi — to'xtadim"); break
        devctl.start_emulator()
        devctl.install(); devctl.clear()
        devctl.push_vision(os.path.join(DATA, "live"))
        pend = os.path.join(DATA, "offline", "pending")
        if os.path.exists(pend):
            devctl.push_offline(os.path.join(DATA, "offline")); os.remove(pend)
            print("ofsayn siyosat yuborildi", flush=True)
        devctl.capture(); devctl.launch()
        time.sleep(5)
        races = trainloop.run(a.minutes, tag, "start", max_races=a.races)
        devctl.stop_emulator()
        m = metrics(tag)
        print("ko'rsatkichlar:", m, flush=True)
        if not m or not m.get("ok"):
            fails += 1
            if fails >= 2:
                print("ikki davra ketma-ket ma'lumot yig'ilmadi — to'xtadim"); break
            continue
        fails = 0
        hist.append({"round": rnd, "tag": tag, "races_played": races, "metrics": m, "time": time.strftime("%Y-%m-%d %H:%M")})
        json.dump(hist, open(LOG, "w"), indent=1)
        v = m["progress_per_min"]
        if v > best * 1.05:
            best, stale = v, 0
        else:
            stale += 1
        print(f"progress/daq {v}  (eng yaxshi {best}, yaxshilanmagan davralar {stale}/{a.plateau})", flush=True)
        tags = sorted(f[4:-4] for f in os.listdir(DATA) if f.startswith("rec_") and f.endswith(".bin") and os.path.exists(os.path.join(DATA, f"agent_{f[4:-4]}.csv")))
        if stale >= a.plateau and not a.fixed:
            print("PLATO: yaxshilanish to'xtadi"); break
        if a.bc:
            cmd = [sys.executable, os.path.join(devctl.ROOT, "tools", "safe_run.py"), "1500", "--", sys.executable, os.path.join(VISION, "offline_bc.py")]
            print("BC:", "OK" if subprocess.run(cmd, cwd=devctl.ROOT).returncode == 0 else "muvaffaqiyatsiz — avvalgi siyosat", flush=True)
        if tags and not a.fixed and not train_encoder(tags):
            print("kodlovchi o'rgatish muvaffaqiyatsiz — avvalgisi bilan davom etaman")
    with open(os.path.join(DATA, "autoloop_summary.md"), "w", encoding="utf8") as f:
        f.write("| davra | poyga | progress/daq | o'rt. tezlik | 100+ mph ulushi | tugatilgan | WRONG WAY | reward |\n|---|---|---|---|---|---|---|---|\n")
        for h in hist:
            m = h["metrics"]
            f.write(f"| {h['round']} | {h['races_played']} | {m['progress_per_min']} | {m['avg_speed']} | {m['fast_share_100mph']} | {m.get('finishes', m.get('races_finished'))} | {m['wrong_way_rate']} | {m['mean_reward']} |\n")
    print("\nTUGADI. Xulosa: data/autoloop_summary.md")


if __name__ == "__main__":
    main()
