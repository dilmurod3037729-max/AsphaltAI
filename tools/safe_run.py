"""Xavfsiz sinov: buyruqni ishga tushiradi, har 1 s GPU yuklamasi/VRAM'ni o'lchaydi, chegaradan oshsa yoki vaqt tugasa o'zi o'ldiradi.
    py safe_run.py <soniya> -- <buyruq ...>
Chegara: util > 80% (3 ketma-ket o'lchov) yoki VRAM > 80% -> darhol o'ldiriladi. Oxirida cho'qqi qiymatlar chop etiladi."""
import subprocess
import sys
import time

MAX_UTIL, MAX_MEM = 80.0, 80.0


def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    u, used, tot = [float(x) for x in out.split(",")]
    return u, used / tot * 100, used, tot


def main():
    secs = float(sys.argv[1])
    cmd = sys.argv[sys.argv.index("--") + 1:]
    u0, m0, used0, tot = smi()
    print(f"boshlang'ich: util {u0:.0f}%  VRAM {m0:.0f}% ({used0:.0f}/{tot:.0f} MB)", flush=True)
    p = subprocess.Popen(cmd, creationflags=0x00004000)       # BELOW_NORMAL_PRIORITY_CLASS
    peak_u = peak_m = 0.0
    over, killed, t0 = 0, "", time.time()
    while p.poll() is None:
        time.sleep(1)
        u, m, _, _ = smi()
        peak_u, peak_m = max(peak_u, u), max(peak_m, m)
        over = over + 1 if u > MAX_UTIL else 0
        print(f"  t={time.time() - t0:4.0f}s util {u:3.0f}%  VRAM {m:3.0f}%", flush=True)
        if over >= 3 or m > MAX_MEM:
            killed = "CHEGARADAN OSHDI"
        elif time.time() - t0 > secs:
            killed = "vaqt tugadi (rejalashtirilgan)"
        if killed:
            p.kill()
            break
    p.wait()
    ok = peak_u <= MAX_UTIL and peak_m <= MAX_MEM and killed != "CHEGARADAN OSHDI"
    print(f"\nnatija: {'TEST OTDI' if ok else 'TEST OTMADI'}  cho'qqi util {peak_u:.0f}%  cho'qqi VRAM {peak_m:.0f}%  ({killed or 'o`zi tugadi'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
