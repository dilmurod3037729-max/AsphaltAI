"""To'liq tsikl: yig'ish (agent o'ynaydi) -> PC'da ko'rish kodlovchisini o'rgatish -> APK'ga o'rnatish -> yana yig'ish.
    py cycle.py [--rounds 3] [--minutes 20] [--epochs 30] [--tag c]
Har davrada yangi kodlovchi chiqadi; agent siyosati kodlovchi versiyasiga bog'langan (yangi versiya = toza siyosat, lekin
kodlovchi ketma-ket yig'ilgan BARCHA ma'lumotda o'rganadi)."""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import devctl  # noqa: E402
import trainloop  # noqa: E402

VISION = os.path.join(devctl.ROOT, "Vision")


def sh(cmd, cwd):
    print(">", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default="c")
    a = ap.parse_args()
    for r in range(1, a.rounds + 1):
        tag = f"{a.tag}{int(time.time()) % 100000}"
        print(f"=== davra {r}/{a.rounds}: yig'ish {a.minutes} daqiqa ({tag})", flush=True)
        devctl.install(); devctl.capture(); devctl.launch()
        time.sleep(3)
        trainloop.run(a.minutes, tag, "start")
        print("=== kodlovchini o'rgatish", flush=True)
        if sh([sys.executable, "train.py", "--epochs", str(a.epochs), "--export"], VISION) != 0:
            print("o'rgatish muvaffaqiyatsiz — to'xtadim")
            break
        sh([sys.executable, "evaluate.py"], VISION)
        if not devctl.build():
            print("qurish muvaffaqiyatsiz — to'xtadim")
            break
