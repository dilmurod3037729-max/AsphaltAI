"""DIST% o'qigich shablonlari uchun HUD qirqimlarini yig'adi (TouchDrive yoqiq, to'liq poyga; har 1.5 s da qirqim).
Faqat shablon ma'lumoti uchun — reward baholashga ishlatilmaydi.   py collect_dist.py <papka>
Keyin: Speed learn (MAXG=4) -> klaster varag'i -> nomlash -> App/Assets/dist_templates.txt  (tools/dist_templates.py)"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devctl  # noqa: E402
import flow  # noqa: E402

BOX = (230, 70, 360, 130)       # DIST% qatori (yakka poyga); qirqim keng, keyin aniqlashtiriladi


def pad(cmd):
    devctl.adb("shell", f"am broadcast -a space.asphaltai.PAD --es cmd '{cmd}'")


def td_on(im):
    a = im.crop((270, 115, 330, 150))
    px = a.load()
    g = sum(1 for x in range(a.width) for y in range(a.height) if px[x, y][1] > 180 and px[x, y][0] < 120 and px[x, y][2] < 120)
    return g > 30


def to_lobby():
    for _ in range(60):
        n, _d = flow.classify(flow.shot())
        if n == "lobby":
            return True
        if n in flow.ACTIONS and flow.ACTIONS[n]:
            devctl.adb("shell", "input", "tap", str(flow.ACTIONS[n][0]), str(flow.ACTIONS[n][1]))
        time.sleep(4)
    return False


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    if not to_lobby():
        sys.exit("lobby topilmadi")
    devctl.adb("shell", "input", "tap", "1254", "621")
    t0, seen = time.time(), 0
    while time.time() - t0 < 200 and seen < 2:
        time.sleep(2)
        seen = seen + 1 if flow.classify(flow.shot())[0] == "racing" else 0
    time.sleep(6)
    if flow.td_state(flow.shot_rgb()) is not True:
        pad("touchdrive")
        time.sleep(1.5)
    pad("gas 1")
    i, ended = 0, 0
    while time.time() - t0 < 600:
        im = flow.shot_rgb()
        n, _d = flow.classify(im.convert("L"))
        ended = ended + 1 if n in ("dq", "results", "rewards") else 0
        if ended >= 2:
            break
        if n == "racing":
            im.crop(BOX).save(os.path.join(out, f"d{i:04d}.png"))
            i += 1
        time.sleep(1.5)
    pad("reset")
    print("qirqimlar:", i)
