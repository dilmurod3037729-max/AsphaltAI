"""PC Asphalt (MS Store) muhiti: arvoh oyna, WGC kadr, fokussiz klaviatura/sichqoncha, ekran matni.
Menyuda virtual pad BO'LMASIN: pad bor paytda o'yin gamepad rejimiga o'tib Enter'ni yutadi. Pad faqat poyga ichida."""
import ctypes
import json
import subprocess
import sys
import time

import numpy as np
import win32api
import win32con
import win32gui

sys.path.insert(0, r"V:\Projects\A9Agent\pc")
import ghost  # noqa: E402
from a9.capture import Capture  # noqa: E402
from look import ocr  # noqa: E402

VK = {"ENTER": win32con.VK_RETURN, "ESC": win32con.VK_ESCAPE, "LEFT": win32con.VK_LEFT, "RIGHT": win32con.VK_RIGHT,
      "UP": win32con.VK_UP, "DOWN": win32con.VK_DOWN, "SPACE": win32con.VK_SPACE}
EXT = {win32con.VK_LEFT, win32con.VK_RIGHT, win32con.VK_UP, win32con.VK_DOWN}


class Game:
    def __init__(self):
        for _ in range(45):                       # o'yin endigina ishga tushgan bo'lsa oyna ~1 daqiqada chiqadi
            self.h = ghost.find_window()
            if self.h:
                break
            time.sleep(2)
        if not self.h:
            raise SystemExit("o'yin oynasi topilmadi")
        self.hide()
        self.cap = Capture(self.h)
        self.cap.start()
        time.sleep(0.5)

    def hide(self):
        if win32gui.GetWindowRect(self.h)[0] != ghost.OFF_X:   # o'yin ruxsat o'zgarganda oynani (0,0) ga qaytaradi
            ghost.hide(self.h)

    def raw(self):
        return self.cap.latest()

    def frame(self):
        """Faqat mijoz sohasi (sarlavha va ramkasiz), BGR."""
        f = self.cap.latest()
        for _ in range(100):                      # o'yin endigina ochilganda WGC birinchi kadrni hali bermagan bo'ladi
            if f is not None:
                break
            time.sleep(0.1); f = self.cap.latest()
        if f is None:
            raise RuntimeError("WGC kadr bermayapti (10 s)")
        r = win32gui.GetWindowRect(self.h)
        s = f.shape[1] / (r[2] - r[0])
        ox, oy = win32gui.ClientToScreen(self.h, (0, 0))
        _, _, cw, ch = win32gui.GetClientRect(self.h)
        x0, y0 = int((ox - r[0]) * s), int((oy - r[1]) * s)
        return f[y0:y0 + int(ch * s), x0:x0 + int(cw * s)]

    def key(self, name):
        vk = VK[name]
        sc = win32api.MapVirtualKey(vk, 0)
        ext = (1 << 24) if vk in EXT else 0
        win32gui.PostMessage(self.h, win32con.WM_KEYDOWN, vk, 1 | (sc << 16) | ext)
        time.sleep(0.08)
        win32gui.PostMessage(self.h, win32con.WM_KEYUP, vk, 1 | (sc << 16) | ext | (3 << 30))

    def click(self, rx, ry):
        """rx, ry: mijoz sohasiga nisbatan 0..1."""
        _, _, cw, ch = win32gui.GetClientRect(self.h)
        lp = win32api.MAKELONG(int(rx * cw), int(ry * ch))
        win32gui.PostMessage(self.h, win32con.WM_MOUSEMOVE, 0, lp)
        time.sleep(0.05)
        win32gui.PostMessage(self.h, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lp)
        time.sleep(0.08)
        win32gui.PostMessage(self.h, win32con.WM_LBUTTONUP, 0, lp)

    def texts(self):
        """[(matn, rx, ry)] — markaz, mijoz sohasiga nisbatan."""
        f = self.frame()
        H, W = f.shape[:2]
        return [(t["text"], (t["x"] + t["w"] / 2) / W, (t["y"] + t["h"] / 2) / H) for t in ocr(f)]

    def tap(self, *words, texts=None):
        """Birinchi mos yozuvni bosadi (katta harf, to'liq so'z). Topilmasa None."""
        for t, x, y in texts if texts is not None else self.texts():
            u = t.upper().strip()
            if any(u == w for w in words):
                self.click(x, y)
                return t
        return None


def locked():
    """Kompyuter qulflanganmi — WTS seans bayrog'i (SessionFlags: 0 = qulf, 1 = ochiq). Tana loyihasida 2026-09-06 o'lchangan rasmiy manba.
    Avvalgi OpenInputDesktop+SwitchDesktop mezoni qulf ekrani o'chib-yonganda "ochiq" deb aldagan (2026-10-23 tungi sinov)."""
    w = ctypes.windll.wtsapi32
    sid = ctypes.c_uint(); ctypes.windll.kernel32.ProcessIdToSessionId(ctypes.windll.kernel32.GetCurrentProcessId(), ctypes.byref(sid))
    buf = ctypes.c_void_p(); n = ctypes.c_uint()
    if not w.WTSQuerySessionInformationW(None, sid, 25, ctypes.byref(buf), ctypes.byref(n)) or not buf.value:
        return False
    try:
        return n.value >= 20 and ctypes.c_int.from_address(buf.value + 16).value == 0
    finally:
        w.WTSFreeMemory(buf)


HIDHIDE = r"C:\Program Files\Nefarius Software Solutions\HidHide\x64\HidHideCLI.exe"
_hh_cache = [0.0, False]


def _hh(*args):
    return subprocess.run([HIDHIDE, *args], capture_output=True, text=True, timeout=10, creationflags=0x08000000).stdout


def hidhide_ready():
    """HidHide TESKARI rejimda (--inv-on): ro'yxatdagi ilovalar (qulf ekrani LogonUI) padni KO'RMAYDI, o'yin esa odatdagidek ko'radi.
    Oddiy rejim (faqat ro'yxatdagilar ko'radi) GDK o'yinida ishlamadi: pad o'yindan ham yashirindi (2026-10-22 sinov). 30 s kesh."""
    if time.time() - _hh_cache[0] > 30:
        try:
            _hh_cache[1] = "--cloak-on" in _hh("--cloak-state") and "--inv-on" in _hh("--inv-state") and "logonui.exe" in _hh("--app-list").lower()
        except (OSError, subprocess.SubprocessError):
            _hh_cache[1] = False
        _hh_cache[0] = time.time()
    return _hh_cache[1]


def hide_pads():
    """Yangi ulangan virtual X360 pad qurilmalarini (HID nomi har ulanishda o'zgaradi) HidHide'da yashirish. Admin kerak."""
    try:
        devs = [d for g in json.loads(_hh("--dev-gaming") or "[]") for d in g["devices"] if "VID_045E&PID_028E" in d["deviceInstancePath"].upper()]
        hidden = _hh("--dev-list")
        for path in {p for d in devs for p in (d["deviceInstancePath"], d["baseContainerDeviceInstancePath"])}:
            if path not in hidden:
                _hh("--dev-hide", path)
        return len(devs)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return 0


def blocked():
    """Qulf bo'lsa-yu HidHide padni yashira olmasa — buyruq yuborish xavfli (qulf ekranini boshqarib yuboradi)."""
    return locked() and not hidhide_ready()


def small(f, w=96, h=48):
    """APK Snap.Frame bilan bir xil: butun kadr, eng yaqin qo'shni, RGB."""
    H, W = f.shape[:2]
    ys = (np.arange(h) * H // h)
    xs = (np.arange(w) * W // w)
    return np.ascontiguousarray(f[ys][:, xs, ::-1])
