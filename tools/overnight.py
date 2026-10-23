"""Tun bo'yi o'zi o'rganish: [Dreamer 10 poyga (tajriba eps) -> o'yin yopiladi -> ds2 ga qo'shiladi -> Dreamer +1500 qadam ->
PCD1 (chetda) da tekshiruv, yomonlashsa qaytariladi] x N, belgilangan soatgacha. Natija data/overnight.csv, jurnal data/overnight.log.
    py tools/overnight.py --until 08:00"""
import argparse
import csv
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "Vision", "out", "dreamer_wm.pt")
PY = sys.executable
GAME_APP = r"shell:AppsFolder\A278AB0D.Asphalt9_h6adky7gbf63m!Asphalt9"
NOWIN = 0x08000000


def log(m):
    line = f"{time.strftime('%H:%M:%S')}  {m}"
    print(line, flush=True)
    with open(os.path.join(DATA, "overnight.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, timeout, out=None):
    with open(out or os.devnull, "a", encoding="utf-8") as f:
        try:
            return subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, creationflags=NOWIN).returncode
        except subprocess.TimeoutExpired:
            log(f"vaqt tugadi: {os.path.basename(args[1]) if len(args) > 1 else args[0]}")
            return -1


def game_running():
    return "Asphalt9_gdk_x64_rtl.exe" in subprocess.run(["tasklist"], capture_output=True, text=True).stdout


def start_game():
    for _ in range(3):
        if game_running():
            return
        subprocess.Popen(["explorer.exe", GAME_APP])
        time.sleep(50)
        if not game_running():                     # osilib qolgan gamelaunchhelper keyingi ishga tushirishni to'sadi
            log("o'yin ochilmadi — gamelaunchhelper yopilib qayta urinilyapti")
            subprocess.run(["taskkill", "/F", "/IM", "gamelaunchhelper.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "gamingservicesui.exe"], capture_output=True)
            time.sleep(3)


def stop_game():
    subprocess.run(["taskkill", "/F", "/IM", "Asphalt9_gdk_x64_rtl.exe"], capture_output=True)
    time.sleep(8)


def corp_net():
    """Korporativ LAN (tashqi IP 45.9.229.x) orqali Asphalt serverga ulanmaydi — CONNECTING TO SERVER da qoladi."""
    try:
        import urllib.request
        return urllib.request.urlopen("https://api.ipify.org", timeout=8).read().decode().startswith("45.9.229.")
    except OSError:
        return True


def ckpt_step():
    out = subprocess.run([PY, "-c", f"import torch;print(torch.load(r'{CKPT}',map_location='cpu',weights_only=False)['step'])"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def evaluate():
    run([PY, "-u", "Vision/dream_diag.py", "PCD1", "cpu"], 1800, os.path.join(DATA, "overnight_eval.log"))
    rep = open(os.path.join(DATA, "diag_PCD1", "report.txt"), encoding="utf-8").read()
    g = lambda pat: float(re.search(pat, rep).group(1)) if re.search(pat, rep) else float("nan")
    return g(r"urilishni sezish AUC ([0-9.]+)"), g(r"tiqilishni sezish AUC ([0-9.]+)"), g(r"\|xato\| ([0-9.]+)")


def race_stats(tag):
    rows = [r for r in csv.reader(open(os.path.join(DATA, "pc_races.csv"))) if r and r[0] == tag]
    fin = [r for r in rows if r[2] == "True" and float(r[4] or 0) > 30]
    f = lambda k, rs: round(sum(float(r[k]) for r in rs) / len(rs), 2) if rs else ""
    return len(rows), len(fin), f(4, fin), f(5, fin), f(6, fin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="08:00")
    ap.add_argument("--races", type=int, default=10)
    ap.add_argument("--train-steps", type=int, default=1500)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--base", default="", help="boshlang'ich PCD1 metrikalari (urilish,tiqilish,xato) — qayta tekshiruvsiz")
    a = ap.parse_args()
    hh, mm = map(int, a.until.split(":"))
    now = dt.datetime.now(); end = now.replace(hour=hh, minute=mm, second=0)
    if end <= now:
        end += dt.timedelta(days=1)
    summ = os.path.join(DATA, "overnight.csv")
    if not os.path.exists(summ):
        open(summ, "w").write("round,tag,time,races,finished,avg_time_s,avg_mph,crash_min,ckpt_step,coll_auc,stuck_auc,speed_err,accepted\n")
    log(f"=== tun boshlandi, tugash {end:%H:%M}, aylanish = {a.races} poyga + {a.train_steps} qadam ===")
    subprocess.Popen([PY, os.path.join(HERE, "keep_awake.py"), str(os.getpid())], creationflags=NOWIN)   # o'qitish paytida ham qulflanmasin
    sys.path.insert(0, HERE)
    from pc_env import locked
    base = tuple(map(float, a.base.split(','))) if a.base else evaluate()
    log(f"boshlang'ich PCD1: urilish AUC {base[0]:.3f} tiqilish AUC {base[1]:.3f} tezlik |xato| {base[2]:.1f}")
    k = 0
    while dt.datetime.now() + dt.timedelta(minutes=40) < end:
        k += 1; tag = f"N{time.strftime('%m%d_%H%M')}"
        if locked():
            log("kompyuter qulflangan — qulf ochilishini kutyapman (qulfda pad inputi o'yinga yetmaydi)")
            while locked() and dt.datetime.now() + dt.timedelta(minutes=40) < end:
                time.sleep(60)
            if locked():
                break
            log("qulf ochildi — davom")
        if corp_net():
            log("internet korporativ LAN orqali (o'yin serverga ulanmaydi) — Wi-Fi'ni kutyapman")
            while corp_net() and dt.datetime.now() + dt.timedelta(minutes=40) < end:
                time.sleep(60)
            if corp_net():
                break
            log("Wi-Fi ulandi — davom")
        log(f"--- aylanish {k} ({tag}): poygalar")
        start_game()
        run([PY, "-u", "tools/pc_agent.py", "--tag", tag, "--races", str(a.races), "--dream-only", "--dream-eps", str(a.eps), "--private"],
            60 * 60, os.path.join(DATA, f"overnight_{tag}.log"))
        stop_game()
        n, fin, t, sp, cr = race_stats(tag)
        log(f"poygalar: {n}, marra {fin}, o'rtacha vaqt {t} s, tezlik {sp} mph, urilish {cr}/daq")
        if n == 0:
            log("yangi poyga yo'q — o'qitish o'tkazib yuborildi"); time.sleep(60); continue
        run([PY, "-u", "Vision/build_ds2.py"], 1800, os.path.join(DATA, "overnight_build.log"))
        shutil.copy(CKPT, CKPT + ".prev")
        step = ckpt_step()
        log(f"o'qitish {step} -> {step + a.train_steps}")
        for attempt in range(3):
            if attempt:
                log(f"o'qitish bajarilmadi (qadam {ckpt_step()}, GPU chegarasi?) — 60 s dan keyin qayta, urinish {attempt + 1}")
                time.sleep(60)
            run([PY, "tools/safe_run.py", "3000", "--", PY, "-u", "Vision/dreamer.py", "--steps", str(step + a.train_steps), "--bs", "8", "--T", "48",
                 "--resume", "--exclude", "PCD1", "--pc-frac", "0.5"], 3600, os.path.join(DATA, "overnight_train.log"))
            if ckpt_step() > step:
                break
        else:
            log("o'qitish 3 marta bajarilmadi — model o'zgarmadi, tekshiruv o'tkazib yuborildi"); continue
        ev = evaluate()
        ok = ev[0] >= base[0] - 0.02 and ev[1] >= base[1] - 0.05            # chetdagi poygalarda sezilarli yomonlashmasin (urilish asosiy)
        if ok:
            base = ev
        else:
            shutil.copy(CKPT + ".prev", CKPT)
        log(f"tekshiruv: urilish AUC {ev[0]:.3f} tiqilish {ev[1]:.3f} tezlik |xato| {ev[2]:.1f} -> {'QABUL' if ok else 'RAD (oldingi model qaytarildi)'}")
        with open(summ, "a") as f:
            f.write(f"{k},{tag},{time.strftime('%H:%M')},{n},{fin},{t},{sp},{cr},{ckpt_step()},{ev[0]:.3f},{ev[1]:.3f},{ev[2]:.1f},{int(ok)}\n")
    log("=== tun tugadi ===")


if __name__ == "__main__":
    main()
