"""evt_<n>_<tezlik>.png kompozitlarini (1056x640: 4 vaqt nuqtasi x [kadr | segmentatsiya | chuqurlik]) mustaqil tasvirlarga ajratadi.
    py split_events.py [--data ../data]
Chiqish: ../data/dataset/events/{rgb,seg,depth}/<hodisa>_f00..f03.png (f00 eng eski, f03 hodisa paytidagi eng so'nggi) + events_index.csv
Vaqt belgisi YO'Q (perception chiqishlari orasidagi vaqt noma'lum) — bo'sh qoldiriladi, taxmin qilinmaydi.
Eslatma: seg panelida pastki 40 pikselda yo'l-xususiyat ustunlari (debug chizma) bor — bu sof klass maskasi emas."""
import argparse
import csv
import glob
import os
import re

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    a = ap.parse_args()
    out = os.path.join(a.data, "dataset", "events")
    for k in ("rgb", "seg", "depth"):
        os.makedirs(os.path.join(out, k), exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(a.data, "evt_*", "evt_*.png"))):
        name = os.path.basename(p)[:-4]
        session = os.path.basename(os.path.dirname(p))[4:]
        m = re.match(r"evt_(\d+)_(\d+)", name)
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if (w, h) != (1056, 640):
            continue
        for f in range(4):
            for j, k in enumerate(("rgb", "seg", "depth")):
                im.crop((j * 352, f * 160, (j + 1) * 352, (f + 1) * 160)).save(os.path.join(out, k, f"{session}_{name}_f{f:02d}.png"))
            rows.append([f"{session}_{name}", f, f"rgb/{session}_{name}_f{f:02d}.png", f"seg/{session}_{name}_f{f:02d}.png",
                         f"depth/{session}_{name}_f{f:02d}.png", "", int(m.group(2)) if f == 3 else "", "evt_a1" if False else session])
    with open(os.path.join(out, "events_index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "frame_idx", "rgb_path", "seg_path", "depth_path", "timestamp_ms", "speed_at_trigger", "session"])
        w.writerows(rows)
    print("hodisalar:", len(rows) // 4, "-> rasmlar", len(rows) * 3)


if __name__ == "__main__":
    main()
