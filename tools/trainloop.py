"""Agent o'rganayotganda menyularda yuradi: poyga tugasa/diskvalifikatsiya bo'lsa yangi poygaga qaytaradi.
    py trainloop.py <daqiqa> [tag] [start|eval|watch]
Oxirida agentni to'xtatib loglarni data/ ga oladi (rec_<tag>.bin, agent_<tag>.csv, perc_<tag>.csv, evt_<tag>/).
Noma'lum ekran 3 marta ketma-ket chiqsa pad 'B' (orqaga) bosiladi."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devctl  # noqa: E402
import flow  # noqa: E402

D = os.path.join(devctl.ROOT, "data")


def pad(cmd):
    devctl.adb("shell", f"am broadcast -a space.asphaltai.PAD --es cmd '{cmd}'")


def run(minutes, tag, mode="start", max_races=None, track="nevada_narrows"):
    devctl.clear()
    pad(f"ai track {track}")       # rekord shu trassa kaliti bo'yicha saqlanadi
    pad(f"ai {mode}")
    t_end = time.time() + minutes * 60
    last, streak, races, last_td, last_race_t, same = None, 0, 0, 0.0, time.time(), 0
    prev_rgb = None
    with open(os.path.join(D, f"flow_{tag}.log"), "a") as log:
        while time.time() < t_end:
            rgb = flow.shot_rgb()
            name, d = flow.classify(rgb.convert("L"))
            known = d < 0.9
            if name == "racing":
                prev_rgb = rgb
            elif last == "racing" and name in ("dq", "results", "rewards") and prev_rgb is not None:
                # diskvalifikatsiya sababini aniqlash uchun dalil: oxirgi poyga kadri (soat/HUD ko'rinadi) va natija ekrani
                rgb_dir = os.path.join(D, f"ends_{tag}")
                os.makedirs(rgb_dir, exist_ok=True)
                prev_rgb.save(os.path.join(rgb_dir, f"race{races + 1}_last.png"))
                rgb.save(os.path.join(rgb_dir, f"race{races + 1}_{name}.png"))
            if name == "racing":
                last_race_t = time.time()
            elif time.time() - last_race_t > 600:      # 10 daqiqa poygasiz: o'yinni qayta ochamiz
                devctl.launch()
                last_race_t = time.time()
            if name == "racing" and time.time() - last_td > 8 and flow.td_state(rgb) is True:
                pad("touchdrive")                    # TouchDrive YOQIQ bo'lsa agent o'rganmaydi — majburan o'chiramiz
                last_td = time.time()
                log.write(f"{time.strftime('%H:%M:%S')} TouchDrive o'chirildi\n")
            log.write(f"{time.strftime('%H:%M:%S')} {name if known else '?'} {d:.2f}\n")
            log.flush()
            if known and name in flow.ACTIONS and flow.ACTIONS[name]:
                if last == "racing" and name in ("dq", "results", "rewards"):
                    races += 1
                    if max_races and races >= max_races:
                        break
                same = same + 1 if last == name else 0
                pts = flow.ALT.get(name) or [flow.ACTIONS[name]]
                pt = pts[same % len(pts)]
                devctl.adb("shell", "input", "tap", str(pt[0]), str(pt[1]))
                streak = 0
            elif not known:
                streak = streak + 1 if last == "?" else 1
                if streak >= 3:
                    pad("tap B 150")
                    streak = 0
            last = name if known else "?"
            time.sleep(4)
    pad("ai stop")
    time.sleep(3)
    devctl.pull(tag)
    print("poygalar:", races)
    return races


if __name__ == "__main__":
    run(float(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "loop", sys.argv[3] if len(sys.argv) > 3 else "start")
