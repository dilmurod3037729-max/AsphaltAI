"""Xom yozuvlarni (agent_<tag>.csv + rec_<tag>.bin) o'zgartirmasdan, aniq sxemali "kuzatuv -> harakat -> keyingi kuzatuv -> natija" jadvallariga aylantiradi.
    py export_dataset.py [--data ../data] [--no-png]
Chiqish (../data/dataset/):  trajectory.csv, outcomes.csv, schema.json, rgb/<episode>/<step>.png
Qoidalar: vaqt/qiymat faqat manba kodida (App/Agent.cs, AiService.cs) tasdiqlangan bo'lsa yoziladi; noma'lum/mavjud bo'lmagan narsa BO'SH qoladi (taxmin qilinmaydi).
Epizod = uzluksiz yozuv: qo'shni qadamlar orasida vaqt oralig'i > 1500 ms yoki qadam raqami sakrasa yangi epizod.
transition_valid = keyingi qadam shu epizodda va 150 <= dt_ms <= 800 (agent qadami 250 ms)."""
import argparse
import csv
import glob
import json
import os

import numpy as np
from PIL import Image

import data as dt
import labels as lb

HERE = os.path.dirname(os.path.abspath(__file__))
GAP_MS, DT_MIN, DT_MAX = 1500, 150, 800

SCHEMA = {
    "agent_csv": {
        "note": "Sarlavhasiz fayl. Ustunlar ma'nosi App/Agent.cs (Loop) va AiService.cs dan olingan; eski fayllarda faqat birinchi 11 ustun bor.",
        "columns": [
            ["c00", "step", "agent qadam hisoblagichi (sessiyalar aro davom etishi mumkin)", "verified"],
            ["c01", "now_ms", "Android SystemClock.ElapsedRealtime, ms (monoton, qurilma yoqilgandan beri)", "verified"],
            ["c02", "speed", "HUD'dan o'qilgan tezlik (emulatorda MPH)", "verified"],
            ["c03", "a_steer", "rul indeksi 0..4 -> stik X = -1,-0.5,0,0.5,1", "verified"],
            ["c04", "a_drift", "0 qo'yib yubor, 1 LB ushla, 2 ikki marta bos (360)", "verified"],
            ["c05", "a_nitro", "0 yo'q, 1 RB bitta, 2 ikki marta (shockwave)", "verified"],
            ["c06", "a_gas", "RT 0/1 (yangi versiyada doim 1)", "verified"],
            ["c07", "reward", "qadam reward'i; FORMULA fayldan faylga farq qiladi (faqat sessiya ichida solishtiring)", "verified"],
            ["c08", "eps", "exploration ehtimoli", "verified"],
            ["c09", "loss", "oxirgi o'rganish yo'qotishi", "verified"],
            ["c10", "age_ms", "ko'rish xususiyatlari yoshi, ms", "verified"],
            ["c11", "r_speed", "tezlik komponenti (22 ustunli format)", "verified"],
            ["c12", "r_wall", "kinetik energiya yo'qotish jazosi", "verified"],
            ["c13", "r_off", "yo'ldan chiqish jazosi", "verified"],
            ["c14", "r_steer", "rul silliqligi jazosi", "verified"],
            ["c15", "r_prog", "progress (DIST%) komponenti", "verified"],
            ["c16", "dist", "HUD DIST%, -1 = o'qilmadi", "verified"],
            ["c17", "road_ahead", "oldinda yo'l ulushi (tayyor segmentatsiya, 0..1)", "verified"],
            ["c18", "offroad_frac", "yo'ldan tashqari ulushi (tayyor segmentatsiya)", "verified"],
            ["c19", "nitro_fill", "nitro chizig'i to'liqligi 0..1 (rang bo'yicha)", "verified"],
            ["c20", "evt_active", "hodisa ustunida yorqin piksel ulushi (osmonga sezgir, ishonchsiz)", "verified"],
            ["c21", "coll_pred", "kodlovchi to'qnashuv bashorati (kodlovchi bo'lmasa 0)", "verified"],
            ["c22", "ww", "WRONG WAY belgisi (faqat yangi APK'da)", "verified"],
        ],
    },
    "rec_bin": "yozuv: int32 step, int64 t(=now_ms), kadr 48x96x3 RGB uint8, hodisa ustuni 40x128 kul, nitro chizig'i 8x96x3 (Vision/data.py REC)",
    "trajectory.csv": {
        "episode_id": "sessiya_e<n>", "step_id": "agent qadami (c00)", "timestamp_ms": "c01", "dt_ms": "oldingi qadamdan, epizod boshida bo'sh",
        "rgb_path": "kadr (96x48 RGB); rec.bin bo'lmagan sessiyada bo'sh", "seg_path/depth_path": "BO'SH: segmentatsiya/chuqurlik massivlari saqlanmagan, faqat vizual rasm (evt_*.png) bor",
        "speed": "c02", "action_id": "a_steer*9 + a_drift*3 + a_nitro (0..44); gaz kodga kirmaydi", "next_step_id": "shu epizoddagi keyingi qadam",
        "transition_valid": "keyingi qadam bor va 150<=dt<=800 ms",
    },
    "outcomes.csv": {
        "delta_speed_mph": "speed(next)-speed",
        "speed_drop_1s_mph": "keyingi 4 TO'G'RI o'tish ichida eng katta tezlik yo'qotishi (aniq hisob, 'to'qnashuv' degan talqin EMAS)",
        "collision/offtrack/recovery": "BO'SH: manbada haqiqiy yorliq yo'q (tezlik tushishi to'qnashuv bilan teng emas: agent o'zi gaz/drift qilganda ham tushadi)",
        "wrong_way": "o'yin belgisi (qizil-pushti X) rang detektori bilan kadrdan; manba: frame_color_detector; rec.bin bo'lmasa bo'sh",
        "progress_dist_pct": "DIST% o'sishi, faqat 0..3 oralig'ida (o'qish xatolari chiqarib tashlanadi), aks holda bo'sh",
        "reward": "c07; reward_source=agent.csv:c07 (formula sessiyaga xos)",
    },
    "limitations": [
        "seg/depth tasvirlari faqat vizualizatsiya (tayyor modellar chiqishi, HUD yashirilgan kirishdan); klass ID massivlari va fizik chuqurlik saqlanmagan.",
        "evt_*.png kompozitlarida vaqt belgisi yo'q (perception chiqishlari orasidagi vaqt noma'lum, ~0.7-1.4 s).",
        "eski agent_*.csv fayllarida reward formulasi va harakatlar to'plami boshqacha bo'lishi mumkin; sessiyalarni aralashtirib reward'ni solishtirma.",
    ],
}


def episodes(now, step):
    brk = np.r_[True, (np.diff(step) != 1) | (np.diff(now) > GAP_MS) | (np.diff(now) < 0)]
    return np.cumsum(brk) - 1


def load_csv(path):
    a = np.genfromtxt(path, delimiter=",")
    return a[None] if a.ndim == 1 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.data, "dataset")
    os.makedirs(out, exist_ok=True)
    traj, outc = [], []
    stats = {}
    for cp in sorted(glob.glob(os.path.join(a.data, "agent_*.csv"))):
        tag = os.path.basename(cp)[6:-4]
        A = load_csv(cp)
        ncol = A.shape[1]
        step, now, sp = A[:, 0].astype(np.int64), A[:, 1].astype(np.int64), A[:, 2]
        col = lambda i, d=np.nan: A[:, i] if ncol > i else np.full(len(A), d)
        steer = np.nan_to_num(col(3), nan=-1).astype(int); drift = np.nan_to_num(col(4), nan=-1).astype(int); nit = np.nan_to_num(col(5), nan=-1).astype(int)
        ep = episodes(now, step)
        frames, ww = None, None
        rp = os.path.join(a.data, f"rec_{tag}.bin")
        if os.path.exists(rp):
            d = dt.load_session(rp, cp)
            frames = {int(s): i for i, s in enumerate(d["step"])}
            fr = d["frame"]
            wwv = lb.wrongway(fr)
            ww = {int(s): bool(wwv[i]) for s, i in frames.items()}
        n = len(A)
        valid_t = 0
        for i in range(n):
            e = f"{tag}_e{ep[i]}"
            nxt = i + 1 if (i + 1 < n and ep[i + 1] == ep[i]) else None
            dtm = int(now[i] - now[i - 1]) if (i > 0 and ep[i - 1] == ep[i]) else None
            ndt = int(now[nxt] - now[i]) if nxt is not None else None
            ok = nxt is not None and DT_MIN <= ndt <= DT_MAX
            valid_t += ok
            rgb = ""
            if frames is not None and int(step[i]) in frames:
                rgb = f"rgb/{e}/{int(step[i])}.png"
                if not a.no_png:
                    p = os.path.join(out, rgb)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    Image.fromarray(fr[frames[int(step[i])]]).save(p)
            aid = int(steer[i] * 9 + drift[i] * 3 + nit[i]) if min(steer[i], drift[i], nit[i]) >= 0 else ""
            traj.append([e, int(step[i]), int(now[i]), dtm if dtm is not None else "", rgb, "", "", float(sp[i]), aid,
                         steer[i] if steer[i] >= 0 else "", drift[i] if drift[i] >= 0 else "", nit[i] if nit[i] >= 0 else "",
                         int(step[nxt]) if nxt is not None else "", int(ok), os.path.basename(cp), i])
            if nxt is None:
                continue
            drops = [sp[i] - sp[j] for j in range(i + 1, min(n, i + 5)) if ep[j] == ep[i]]
            dist = col(16)
            pr = ""
            if ncol > 16 and dist[i] >= 0 and dist[nxt] >= 0 and 0 <= dist[nxt] - dist[i] <= 3:
                pr = float(dist[nxt] - dist[i])
            w = ww.get(int(step[i]), "") if ww is not None else ""
            outc.append([e, int(step[i]), int(step[nxt]), float(sp[nxt] - sp[i]), float(max(0.0, max(drops))) if (ok and drops) else "",
                         "", "", "", int(w) if w != "" else "", pr, float(A[i, 7]) if ncol > 7 else "", "agent.csv:c07", os.path.basename(cp)])
        stats[tag] = {"qatorlar": n, "epizodlar": int(ep.max()) + 1, "yaroqli_o'tishlar": int(valid_t),
                      "dt_median_ms": float(np.median(np.diff(now))), "dt_max_ms": int(np.max(np.diff(now))), "rec_bin": os.path.exists(rp)}
    with open(os.path.join(out, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "step_id", "timestamp_ms", "dt_ms", "rgb_path", "seg_path", "depth_path", "speed", "action_id", "a_steer", "a_drift",
                    "a_nitro", "next_step_id", "transition_valid", "source_file", "source_row"])
        w.writerows(traj)
    with open(os.path.join(out, "outcomes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "step_id", "next_step_id", "delta_speed_mph", "speed_drop_1s_mph", "collision", "offtrack", "recovery", "wrong_way",
                    "progress_dist_pct", "reward", "reward_source", "source_file"])
        w.writerows(outc)
    SCHEMA["sessions"] = stats
    json.dump(SCHEMA, open(os.path.join(out, "schema.json"), "w"), indent=1, ensure_ascii=False)
    for k, v in stats.items():
        print(k, v)
    print("yozildi:", out)


if __name__ == "__main__":
    main()
