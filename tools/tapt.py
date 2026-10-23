"""Emulatorda matn bo'yicha bosish: py tapt.py "Start now" [serial]  (uiautomator dump)."""
import re
import subprocess
import sys

ADB = r"C:\platform-tools\adb.exe"


def adb(*a):
    return subprocess.run([ADB, "-s", SERIAL, *a], capture_output=True, text=True).stdout


SERIAL = sys.argv[2] if len(sys.argv) > 2 else "emulator-5554"
adb("shell", "uiautomator", "dump", "/sdcard/u.xml")
xml = adb("shell", "cat", "/sdcard/u.xml")
want = sys.argv[1].lower()
for m in re.finditer(r'<node [^>]*?text="([^"]*)"[^>]*?bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml):
    if m.group(1).lower() == want:
        x0, y0, x1, y1 = map(int, m.groups()[1:])
        adb("shell", "input", "tap", str((x0 + x1) // 2), str((y0 + y1) // 2))
        print("tap", (x0 + x1) // 2, (y0 + y1) // 2)
        break
else:
    print("yo'q")
    sys.exit(1)
