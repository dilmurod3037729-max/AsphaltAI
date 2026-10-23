"""Agent ishlayotganda kompyuter o'zi qulflanmasin (qulfda pad inputi qulf ekraniga ketadi).
SetThreadExecutionState (tizim + ekran band) + har 30 s 1 px borib-qaytuvchi sichqoncha hodisasi (ekran saqlovchi 600 s qulfli — uni shu to'xtatadi).
Berilgan jarayon (pc_agent) tugaganda o'zi to'xtaydi va odatdagi rejim qaytadi.
    py tools/keep_awake.py <kuzatiladigan_pid>"""
import ctypes
import sys
import time

import psutil

ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
MOUSEEVENTF_MOVE = 0x0001


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]


def nudge():
    # ekran saqlovchi (10 daq, qulfli) harakatsizlik taymerini nolga qaytarish: 1 px borib qaytish — kursor joyida qoladi
    for d in (1, -1):
        i = INPUT(type=0, mi=MOUSEINPUT(d, 0, 0, MOUSEEVENTF_MOVE, 0, None))
        ctypes.windll.user32.SendInput(1, ctypes.byref(i), ctypes.sizeof(i))


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    k = ctypes.windll.kernel32
    try:
        while pid is None or psutil.pid_exists(pid):
            k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
            nudge()
            time.sleep(30)
    finally:
        k.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
