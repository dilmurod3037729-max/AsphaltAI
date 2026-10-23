"""Emulator bilan ishlash yordamchilari: qurish, o'rnatish, root sozlamalari, ekran yozishni boshlash, loglarni olish.
    py devctl.py build|install|setup|capture|launch|pull <tag>|clear|all"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ADB = [r"C:\platform-tools\adb.exe", "-s", "emulator-5554"]
PKG = "space.asphaltai"
GAME = "com.gameloft.android.ANMP.GloftA9HM"
FILES = f"/sdcard/Android/data/{PKG}/files"
APK = os.path.join(ROOT, "App", "bin", "Release", "net9.0-android", "android-x64", f"{PKG}-Signed.apk")
SDK = r"C:\Users\User\AppData\Local\Android\Sdk"
JDK = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.12.8-hotspot"


def adb(*a, text=True):
    return subprocess.run(ADB + list(a), capture_output=True, text=text).stdout


def build():
    r = subprocess.run(["dotnet", "build", os.path.join(ROOT, "App"), "-c", "Release", "-v", "q", f"-p:AndroidSdkDirectory={SDK}",
                        f"-p:JavaSdkDirectory={JDK}", "-p:RuntimeIdentifier=android-x64"], capture_output=True, text=True)
    errs = [l for l in r.stdout.splitlines() if " error " in l]
    print("qurish:", "OK" if r.returncode == 0 else "XATO")
    for e in errs[:8]:
        print("  ", e)
    return r.returncode == 0


def install():
    adb("shell", "am", "force-stop", PKG)
    out = adb("install", "-r", "-g", APK).strip().splitlines()
    print("o'rnatish:", out[-1] if out else "javob yo'q")
    setup()


def setup():
    adb("shell", "su -c 'appops set space.asphaltai PROJECT_MEDIA allow; setenforce 0; chmod 666 /dev/uinput'")


def launch(pkg=GAME):
    adb("shell", "monkey", "-p", pkg, "-c", "android.intent.category.LAUNCHER", "1")


def capture():
    """Ilovani ochib pad yaratadi va ekran yozishni boshlaydi (tapt.py orqali)."""
    adb("shell", "am", "start", "-n", f"{PKG}/crc64122e5dbf1d74b4ba.MainActivity")
    time.sleep(2)
    tapt = os.path.join(HERE, "tapt.py")
    subprocess.run([sys.executable, tapt, "Virtual gamepad yaratish"], capture_output=True)
    subprocess.run([sys.executable, tapt, "Ekranni yozishni boshlash"], capture_output=True)
    for _ in range(6):
        time.sleep(1.5)
        if subprocess.run([sys.executable, tapt, "Start now"], capture_output=True).returncode == 0:
            break
    time.sleep(2)


def emulator_running():
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    return "qemu-system" in out


def start_emulator():
    """Emulatorni (toza yuklash) ishga tushiradi va tayyor bo'lguncha kutadi."""
    if emulator_running():
        return
    env = dict(os.environ, ANDROID_HOME=r"V:\Android\sdk", ANDROID_SDK_ROOT=r"V:\Android\sdk", ANDROID_AVD_HOME=r"V:\Android\avd")
    log = open(r"V:\Android\emu_stdout.log", "w")
    subprocess.Popen([r"V:\Android\sdk\emulator\emulator.exe", "-avd", "a9", "-gpu", "host", "-no-boot-anim", "-no-snapshot-load",
                      "-no-snapshot-save", "-crash-report-mode", "never", "-no-metrics"], env=env, stdout=log, stderr=log, creationflags=0x00000208)
    subprocess.run([ADB[0], "wait-for-device"], timeout=300)
    for _ in range(120):
        if adb("shell", "getprop", "sys.boot_completed").strip() == "1":
            break
        time.sleep(3)
    time.sleep(25)


def stop_emulator():
    """Emulatorni yopadi (GPU/CPU bo'shashi uchun) va jarayon tugashini kutadi."""
    adb("emu", "kill")
    for _ in range(40):
        if not emulator_running():
            return
        time.sleep(2)
    subprocess.run(["taskkill", "/F", "/IM", "qemu-system-x86_64.exe"], capture_output=True)


def push_vision(live_dir):
    """Yangi kodlovchini (onnx + versiya) emulatorga nusxalaydi — APK qayta qurilmaydi; agent keyingi 'ai start' da yuklaydi."""
    tmp = "/data/local/tmp"
    for f in ("vision.onnx", "vision.ver"):
        subprocess.run(ADB + ["push", os.path.join(live_dir, f), f"{tmp}/{f}"], capture_output=True)
    adb("shell", f"su -c 'cp {tmp}/vision.onnx {FILES}/vision.onnx; cp {tmp}/vision.ver {FILES}/vision.ver; chmod 666 {FILES}/vision.onnx {FILES}/vision.ver'")


def push_offline(off_dir):
    """Ofsayn o'rgatilgan siyosat + replay'ni app fayllariga qo'yadi (app to'xtagan bo'lishi kerak)."""
    tmp = "/data/local/tmp"
    for f in os.listdir(off_dir):
        if f.endswith(".bin"):
            subprocess.run(ADB + ["push", os.path.join(off_dir, f), f"{tmp}/{f}"], capture_output=True)
            adb("shell", f"su -c 'cp {tmp}/{f} {FILES}/{f}; chmod 666 {FILES}/{f}'")


def pull(tag):
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    got = []
    for src, dst in (("rec.bin", f"rec_{tag}.bin"), ("agent.csv", f"agent_{tag}.csv"), ("perc.csv", f"perc_{tag}.csv")):
        p = os.path.join(ROOT, "data", dst)
        subprocess.run(ADB + ["pull", f"{FILES}/{src}", p], capture_output=True)
        if os.path.exists(p):
            got.append((dst, os.path.getsize(p)))
    for f in os.listdir(os.path.join(ROOT, "data")):
        pass
    evdir = os.path.join(ROOT, "data", f"evt_{tag}")
    os.makedirs(evdir, exist_ok=True)
    for name in adb("shell", f"ls {FILES}").split():
        if name.startswith("evt_"):
            subprocess.run(ADB + ["pull", f"{FILES}/{name}", os.path.join(evdir, name)], capture_output=True)
    print("olindi:", got)


def clear():
    adb("shell", f"rm -f {FILES}/rec.bin {FILES}/agent.csv {FILES}/perc.csv {FILES}/evt_*")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        sys.exit(0 if build() else 1)
    elif cmd == "install":
        install(); setup()
    elif cmd == "setup":
        setup()
    elif cmd == "capture":
        capture()
    elif cmd == "launch":
        launch()
    elif cmd == "pull":
        pull(sys.argv[2])
    elif cmd == "clear":
        clear()
    elif cmd == "all":
        if build():
            install(); clear(); capture(); launch()
