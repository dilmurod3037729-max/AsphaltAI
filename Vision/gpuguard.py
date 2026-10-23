"""GPU himoyasi: GPU yuklamasi ham, VRAM ham 80% dan oshmasin (oshsa kompyuter qotadi / qora ekran).
Sabab (2026-10-20): trening hech qanday cheklovsiz GPU'ni 100% band qilib turdi; GPU displey va emulator (-gpu host) bilan bo'lishilgan.
    preflight()  — GPU band bo'lsa yoki emulator ishlayotgan bo'lsa to'xtatadi
    cap_vram()   — boshqalar ishlatayotgan VRAM'ni hisobga olib, o'z ulushini cheklaydi (jami <= 80%)
    Throttle     — har qadamdan keyin dam beradi (ish aylanishi ~duty) va util/VRAM yuqori bo'lsa to'xtab turadi"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # bo'laklanish: busiz reserved 5-13 GB, bilan 1.6 GB (bs8 T48 o'lchangan)
import subprocess
import time

import torch

MAX_UTIL = 75           # % — shundan yuqori bo'lsa to'xtab turamiz
MAX_MEM = 78            # % — jami VRAM (barcha jarayonlar)
VRAM_LIMIT = 0.75       # CUDA konteksti (~300-500 MB) va boshqa dasturlar (SolidWorks) tebranishi hisobga olinmaydi -> 80% dan ancha past
CTX_OVERHEAD = 1.2 * 2 ** 30
DUTY = 0.5              # o'rtacha GPU ish aylanishi (qolgan vaqt dam)


def smi():
    """(util %, ishlatilgan MB, jami MB) yoki None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        u, used, tot = [float(x) for x in out.split(",")]
        return u, used, tot
    except Exception:
        return None


def emulator_running():
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=10).stdout.lower()
        return "qemu-system" in out
    except Exception:
        return False


def preflight(allow_emulator=False):
    if emulator_running() and not allow_emulator:
        raise SystemExit("Emulator ishlayapti — GPU/CPU trening kompyuterni qotiradi. Avval emulatorni yoping.")
    s = smi()
    if s and (s[0] > 50 or s[1] / s[2] * 100 > 60):
        raise SystemExit(f"GPU allaqachon band: util {s[0]:.0f}%, VRAM {s[1] / s[2] * 100:.0f}% — trening boshlanmadi.")


def cap_vram():
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    allow = VRAM_LIMIT * total - (total - free) - CTX_OVERHEAD    # kontekst + cuDNN/cuBLAS kernellari allocator'dan tashqarida (o'lchangan ~1 GB)
    if allow < 400 * 2 ** 20:
        raise SystemExit(f"VRAM yetarli emas (boshqalar band): ruxsat etilgan ulush {allow / 2 ** 20:.0f} MB")
    torch.cuda.set_per_process_memory_fraction(min(0.5, allow / total))


class Throttle:
    def __init__(self, duty=DUTY):
        self.duty, self.n, self.t = duty, 0, time.time()

    def step(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        busy = time.time() - self.t
        time.sleep(min(1.0, busy * (1 / self.duty - 1)))      # ish aylanishini ~duty ga tushirish
        self.n += 1
        if self.n % 20 == 0:
            self.wait_safe()
        self.t = time.time()

    @staticmethod
    def wait_safe(max_wait=300):
        t0 = time.time()
        while time.time() - t0 < max_wait:
            s = smi()
            if not s or (s[0] <= MAX_UTIL and s[1] / s[2] * 100 <= MAX_MEM):
                return
            time.sleep(2)
        raise SystemExit("GPU 5 daqiqa davomida 80% dan past tushmadi — trening to'xtatildi.")
