"""O'yinsiz uzun fine-tune: bo'laklab (chunk qadam) o'qitadi, har bo'lakdan keyin PCD1 (chetda) da tekshiradi, har birini saqlaydi.
    py tools/finetune_long.py --chunks 2 --chunk 3000"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from overnight import CKPT, DATA, PY, ROOT, ckpt_step, evaluate, run  # noqa: E402


def log(m):
    line = f"{time.strftime('%H:%M:%S')}  {m}"
    print(line, flush=True)
    with open(os.path.join(DATA, "finetune_long.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=3000)
    a = ap.parse_args()
    base = evaluate(); best = (base, ckpt_step())
    shutil.copy(CKPT, CKPT.replace(".pt", f"_{best[1]}.pt"))
    log(f"boshlang'ich {best[1]}: urilish AUC {base[0]:.3f} tiqilish {base[1]:.3f} tezlik |xato| {base[2]:.1f}")
    for c in range(a.chunks):
        s0 = ckpt_step(); s1 = s0 + a.chunk
        log(f"bo'lak {c + 1}: o'qitish {s0} -> {s1}")
        run([PY, "tools/safe_run.py", "3000", "--", PY, "-u", "Vision/dreamer.py", "--steps", str(s1), "--bs", "8", "--T", "48",
             "--resume", "--exclude", "PCD1", "--pc-frac", "0.5"], 4 * 3600, os.path.join(DATA, "finetune_long_train.log"))
        if ckpt_step() <= s0:
            log("o'qitish bajarilmadi (GPU chegarasi?) — to'xtadim"); break
        ev = evaluate(); st = ckpt_step()
        shutil.copy(CKPT, CKPT.replace(".pt", f"_{st}.pt"))
        log(f"bo'lak {c + 1} ({st}): urilish AUC {ev[0]:.3f} tiqilish {ev[1]:.3f} tezlik |xato| {ev[2]:.1f}")
        if ev[0] > best[0][0]:
            best = (ev, st)
    snap = CKPT.replace(".pt", f"_{best[1]}.pt")
    if os.path.exists(snap):
        shutil.copy(snap, CKPT)                     # agent dreamer_wm.pt ni o'qiydi — eng yaxshisi shu yerda qoladi
    log(f"eng yaxshisi: {best[1]} (urilish AUC {best[0][0]:.3f}) -> {os.path.basename(CKPT)}")


if __name__ == "__main__":
    main()
