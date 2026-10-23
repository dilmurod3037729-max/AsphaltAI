"""Barcha xom yozuvlardan (data/rec_<tag>.bin + data/agent_<tag>.csv) yagona dataset: data/ds2/.
    py Vision/build_ds2.py [--data data] [--rebuild]
Chiqish: frames.u8 (N,48,96,3) / evt.u8 (N,40,128) / nitro.u8 (N,8,96,3) memmap, steps.csv, episodes.csv, schema.json, manifest.json.
Reward YOZILMAYDI — faqat xom signallar va hosila yorliqlar; bosqich reward'lari Vision/rewards.py da shu jadvaldan hisoblanadi.
Qayta ishga tushirish: xom fayllar o'zgarmagan bo'lsa faqat yangi taglar oxiriga qo'shiladi; biror eski tag o'zgargan bo'lsa (yoki --rebuild) to'liq qayta quriladi.
Xom fayllar faqat o'qiladi."""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

import labels as lb
from data import REC

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_MS = 250
CONT_MAX_MS = 2 * STEP_MS          # bundan uzun oraliq = segment chegarasi (HUD uzilishi / takedown sahnasi / pauza)
RACE_GAP_MS = 8000                 # App/Agent.cs RaceGapMs: HUD shundan uzoq yo'qolsa yangi poyga
MIN_RACE_S = 20
START_MPH = 80                     # haqiqiy start ~0 mph dan; bundan tez boshlangan bo'lak = poyga o'rtasi (uzoq HUD uzilishi / agent kech ulangan)
DQ_S = 170                         # poyga 3:00 da tugaydi; HUD start sanog'idan keyin chiqadi
GLITCH_MPH = 80                    # App/Agent.cs GlitchMph
H = 4                              # yorliq gorizonti: 4 qadam = 1 s
COLL_FROM, COLL_DROP = 60, 30      # urilish: >60 mph dan 1 s ichida 30+ mph tushish (loyiha "crash" metrikasi)
KNOWN_BAD = {"PC1": "100 mph dan past tezlik o'qilmagan (PC o'qigichining eski xatosi, 2026-10-22 tuzatilgan) — urilishlar yo'qolgan"}
STEER_V = [-1.0, -0.5, 0.0, 0.5, 1.0]

COLS = ["global_idx", "episode_id", "seg_id", "tag", "source", "step", "t_ms", "t_race_ms", "dt_ms", "frame_age_ms",
        "speed", "speed_glitch", "dist", "nitro_fill", "ww", "evt_cls", "evt_active",
        "a_steer", "a_drift", "a_nitro", "a_gas", "steer_x", "prev_steer", "prev_drift", "prev_nitro", "expl", "eps", "coll_pred",
        "is_first", "cont", "terminal", "finish", "gap_after_ms",
        "valid_h", "coll", "d_speed_1s", "drop_1s", "nitro_gain_1s", "prog_1s", "frame_mean"]


def source_of(tag):
    # PC*: qo'lda poygalar; N<oykun>_*: tsikl (tools/overnight.py) poygalari — ikkalasi ham PC (ghost oyna, PC HUD)
    return "pc" if tag.startswith("PC") or re.match(r"N\d{4}_", tag) else "emulator"


def load_tag(data, tag):
    cp = os.path.join(data, f"agent_{tag}.csv"); rp = os.path.join(data, f"rec_{tag}.bin")
    A = np.genfromtxt(cp, delimiter=",")
    A = A[None] if A.ndim == 1 else A
    n = os.path.getsize(rp) // REC.itemsize
    R = np.memmap(rp, REC, "r", shape=(n,))
    rs = np.asarray(R["step"]).astype(np.int64)
    pos = {int(s): i for i, s in enumerate(rs)}
    keep = np.array([int(s) in pos for s in A[:, 0]])
    A = A[keep]
    A = A[np.argsort(A[:, 0], kind="stable")]
    ri = np.array([pos[int(s)] for s in A[:, 0]], np.int64)
    return A, R, ri, int((~keep).sum())


def col(A, i, default=np.nan):
    return A[:, i].astype(np.float64) if A.shape[1] > i else np.full(len(A), default)


def segment_races(t, step, dist):
    """seg: uzluksiz bo'lak (dt <= 500 ms, qadam +1). race: bo'laklar RACE_GAP_MS dan qisqa uzilish bilan birlashadi;
    DIST keskin qaytsa (>40 -> <10) uzilish qisqa bo'lsa ham yangi poyga."""
    n = len(t)
    dtn = np.r_[0, np.diff(t)]
    new_seg = np.r_[True, (np.diff(step) != 1) | (dtn[1:] > CONT_MAX_MS) | (dtn[1:] <= 0)]
    new_race = np.zeros(n, bool); new_race[0] = True
    last_d = -1
    for i in range(1, n):
        if dist[i - 1] >= 0:
            last_d = dist[i - 1]
        if new_seg[i] and (dtn[i] > RACE_GAP_MS or dtn[i] <= 0):
            new_race[i] = True
        elif dist[i] >= 0 and last_d > 40 and dist[i] < 10:
            new_race[i] = True
    new_seg |= new_race
    return np.cumsum(new_seg) - 1, np.cumsum(new_race) - 1


def pc_outcomes(data, tag):
    p = os.path.join(data, "pc_races.csv")
    if not os.path.exists(p):
        return []
    out = []
    for r in csv.reader(open(p)):
        if r and r[0] == tag:
            out.append({"finish": r[2] == "True", "time": float(r[3]) if r[3] else None, "steps": int(r[9])})
    return out


def process_tag(data, tag, centers, g0, ep_rows, w):
    A, R, ri, dropped = load_tag(data, tag)
    src = source_of(tag)
    n = len(A)
    step = A[:, 0].astype(np.int64); t = A[:, 1].astype(np.int64)
    sp = col(A, 2); dist = col(A, 16, -1); bonus = col(A, 23, 0)
    seg, race = segment_races(t, step, dist)
    frames = np.asarray(R["frame"][ri]); evt = np.asarray(R["evt"][ri]); nit = np.asarray(R["nitro"][ri])
    frame_age = t - np.asarray(R["t"][ri]).astype(np.int64)
    fmean = frames.reshape(n, -1).mean(1)
    if src == "emulator":
        fill = lb.nitro_fill(nit)
        cls, _ = lb.event_classes(evt, centers)
    else:                                                    # PC HUD joylashuvi boshqa: nitro chizig'i/hodisa ustuni shu nisbiy qutilarda emas
        fill = np.full(n, -1.0); cls = np.full(n, -1)
    ww = lb.wrongway(frames).astype(int)
    evt_act = col(A, 20)

    dtn = np.r_[0, np.diff(t)]
    is_first = np.r_[True, seg[1:] != seg[:-1]]
    last_in_seg = np.r_[seg[1:] != seg[:-1], True]
    gap_after = np.where(last_in_seg, np.r_[np.diff(t), 0], 0)
    gap_after[-1] = 0
    glitch = np.zeros(n, int)
    glitch[1:] = ((np.abs(np.diff(sp)) >= GLITCH_MPH) & ~is_first[1:]).astype(int)

    steer = col(A, 3, 2).astype(int); drift = col(A, 4, 0).astype(int); nitro_a = col(A, 5, 0).astype(int); gas = col(A, 6, 1)
    sx = col(A, 25)                                                  # PC agent 2026-10-22 dan: joystik qiymati (uzluksiz)
    sx = np.where(np.isfinite(sx), sx, np.array(STEER_V)[np.clip(steer, 0, 4)])
    prev = lambda a, d: np.where(is_first, d, np.r_[d, a[:-1]])

    valid_h = np.zeros(n, int); coll = np.zeros(n, int)
    d1 = np.full(n, np.nan); drop = np.full(n, np.nan); ng = np.full(n, np.nan); prog = np.full(n, np.nan)
    seg_end = np.zeros(n, np.int64)
    e = n
    for i in range(n - 1, -1, -1):
        if last_in_seg[i]:
            e = i + 1
        seg_end[i] = e
    for i in range(n):
        if i + H < seg_end[i]:
            valid_h[i] = 1
            fut = sp[i + 1:i + H + 1]
            drop[i] = max(0.0, sp[i] - fut.min())
            d1[i] = sp[i + H] - sp[i]
            coll[i] = int(sp[i] > COLL_FROM and drop[i] >= COLL_DROP)
            if fill[i] >= 0:
                ng[i] = max(0.0, fill[i + H] - fill[i])
            dd = np.diff(dist[i:i + H + 1])
            if (dist[i:i + H + 1] >= 0).all() and (dd >= 0).all() and (dd <= 3).all():
                prog[i] = dist[i + H] - dist[i]

    terminal = np.zeros(n, int); finish = np.zeros(n, int)
    t_race = np.zeros(n, np.int64)
    pco = pc_outcomes(data, tag) if src == "pc" else []
    pc_i = 0
    episodes = []
    for r in np.unique(race):
        idx = np.where(race == r)[0]
        s, e = idx[0], idx[-1]
        t_race[idx] = t[idx] - t[s]
        dur = (t[e] - t[s]) / 1000
        fin_bonus = np.where(bonus[idx] >= 20)[0]
        fin_dist = np.where((dist[idx] >= 99) & (t_race[idx] > 20000))[0]
        fin_idx, fin_src, res_time = None, "", None
        if len(fin_bonus):
            fin_idx, fin_src = idx[fin_bonus[0]], "bonus"
        elif len(fin_dist):
            fin_idx, fin_src = idx[fin_dist[0]], "dist99"
        if src == "pc":
            j = next((k for k in range(pc_i, len(pco)) if abs(pco[k]["steps"] - len(idx)) <= 5), None)   # sinov poygalari yozuvi o'chirilgan bo'lishi mumkin
            if j is not None:
                o = pco[j]; pc_i = j + 1
                if o["finish"]:
                    fin_idx, fin_src, res_time = e, "pc_results", o["time"]
        finished = fin_idx is not None
        dq = (not finished) and dur >= DQ_S
        reason = ""
        if tag in KNOWN_BAD:
            reason = KNOWN_BAD[tag]
        elif dur < MIN_RACE_S:
            reason = f"qisqa ({dur:.0f} s)"
        elif sp[s] > START_MPH:
            reason = f"o'rtasidan boshlangan (birinchi tezlik {sp[s]:.0f} mph)"
        post = 0
        if finished and not reason:
            terminal[fin_idx] = 1; finish[fin_idx] = 1
            post = int((idx > fin_idx).sum())                # marradan keyingi "sayr" (HUD hali turibdi) epizodga kirmaydi
            idx = idx[idx <= fin_idx]
            dur = (t[idx[-1]] - t[s]) / 1000
        if dq and not reason:
            terminal[e] = 1
        ftime = res_time if res_time is not None else ((t[fin_idx] - t[s]) / 1000 if finished else None)
        episodes.append({"episode_id": f"{tag}_r{r:03d}", "tag": tag, "source": src, "policy": tag, "n": len(idx), "post_finish_dropped": post, "dur_s": round(dur, 1),
                         "finished": int(finished), "finish_src": fin_src, "finish_time_s": round(ftime, 3) if ftime else "", "dq": int(dq),
                         "end": "finish" if finished else ("dq_timeout" if dq else "to'xtatilgan/noma'lum"),
                         "avg_speed": round(float(sp[idx].mean()), 1), "max_dist": int(dist[idx].max()),
                         "dist_read": round(float((dist[idx] >= 0).mean()), 3), "segments": int(len(np.unique(seg[idx]))),
                         "coll_per_min": round(float((np.diff(np.r_[0, coll[idx]]) == 1).sum()) / max(dur / 60, 1e-6), 2), "excluded": reason, "_idx": idx})

    keep = np.zeros(n, bool)
    for ep in episodes:
        if not ep["excluded"]:
            keep[ep["_idx"]] = True
    order = np.where(keep)[0]
    gidx = g0 + np.arange(len(order))
    gmap = dict(zip(order.tolist(), gidx.tolist()))
    cont = (~last_in_seg).astype(int)
    for ep in episodes:
        idx = ep.pop("_idx")
        cont[idx[-1]] = 0
        if ep["excluded"]:
            ep["start_idx"] = ep["end_idx"] = -1
        else:
            ep["start_idx"] = gmap[int(idx[0])]; ep["end_idx"] = gmap[int(idx[-1])] + 1
        ep_rows.append(ep)
    race_id = {r: f"{tag}_r{r:03d}" for r in np.unique(race)}
    expl, eps, cpred = col(A, 24), col(A, 8), col(A, 21)
    psteer, pdrift, pnit = prev(steer, 2), prev(drift, 0), prev(nitro_a, 0)
    for i in order:
        w.writerow([gmap[int(i)], race_id[race[i]], f"{tag}_s{seg[i]:04d}", tag, src, int(step[i]), int(t[i]), int(t_race[i]),
                    int(dtn[i]) if not is_first[i] else "", int(frame_age[i]),
                    int(sp[i]), glitch[i], int(dist[i]), round(float(fill[i]), 3), ww[i], int(cls[i]),
                    "" if np.isnan(evt_act[i]) else round(float(evt_act[i]), 3),
                    steer[i], drift[i], nitro_a[i], int(gas[i]) if not np.isnan(gas[i]) else "", round(float(sx[i]), 3),
                    psteer[i], pdrift[i], pnit[i],
                    "" if np.isnan(expl[i]) else int(expl[i]), eps[i], "" if np.isnan(cpred[i]) else round(float(cpred[i]), 3),
                    int(is_first[i]), cont[i], terminal[i], finish[i], int(gap_after[i]),
                    valid_h[i], coll[i], "" if np.isnan(d1[i]) else int(d1[i]), "" if np.isnan(drop[i]) else int(drop[i]),
                    "" if np.isnan(ng[i]) else round(float(ng[i]), 3), "" if np.isnan(prog[i]) else int(prog[i]), round(float(fmean[i]), 1)])
    return frames[order], evt[order], nit[order], {"rows_csv": n + dropped, "rows_joined": n, "csv_without_frame": dropped, "kept": int(len(order))}


def file_sig(data, tag):
    return [os.path.getsize(os.path.join(data, f"agent_{tag}.csv")), os.path.getsize(os.path.join(data, f"rec_{tag}.bin"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.data, "ds2")
    os.makedirs(out, exist_ok=True)
    tags = sorted(os.path.basename(p)[6:-4] for p in glob.glob(os.path.join(a.data, "agent_*.csv")))
    no_rec = [t for t in tags if not os.path.exists(os.path.join(a.data, f"rec_{t}.bin"))]
    tags = [t for t in tags if t not in no_rec]
    mf_path = os.path.join(out, "manifest.json")
    mf = json.load(open(mf_path)) if os.path.exists(mf_path) and not a.rebuild else {"tags": {}, "n": 0}
    if any(t not in tags or file_sig(a.data, t) != v["sig"] for t, v in mf["tags"].items()):
        print("eski tag o'zgargan — to'liq qayta qurish", flush=True)
        mf = {"tags": {}, "n": 0}
    fresh = mf["n"] == 0
    def usable(t):                                   # 0 poygali aylanish bo'sh rec/csv qoldiradi — butun yig'ishni to'xtatmasin
        rp, cp = os.path.join(a.data, f"rec_{t}.bin"), os.path.join(a.data, f"agent_{t}.csv")
        return os.path.exists(rp) and os.path.getsize(rp) > 0 and os.path.exists(cp) and os.path.getsize(cp) > 0
    skipped = [t for t in tags if t not in mf["tags"] and not usable(t)]
    if skipped:
        print("bo'sh yozuvlar o'tkazib yuborildi:", ", ".join(skipped), flush=True)
    todo = [t for t in tags if t not in mf["tags"] and usable(t)]
    cpath = os.path.join(out, "evt_centers.npy")
    if fresh or not os.path.exists(cpath):
        sample = []
        rs = np.random.RandomState(0)
        for t in tags:
            if source_of(t) != "emulator":
                continue
            n = os.path.getsize(os.path.join(a.data, f"rec_{t}.bin")) // REC.itemsize
            R = np.memmap(os.path.join(a.data, f"rec_{t}.bin"), REC, "r", shape=(n,))
            sample.append(np.asarray(R["evt"][np.sort(rs.choice(n, min(n, 2000), replace=False))]))
        _, centers = lb.event_classes(np.concatenate(sample))
        np.save(cpath, centers)
    centers = np.load(cpath)
    mode = "w" if fresh else "a"
    files = {k: open(os.path.join(out, f"{k}.u8"), mode + "b") for k in ("frames", "evt", "nitro")}
    sf = open(os.path.join(out, "steps.csv"), mode, newline="")
    w = csv.writer(sf)
    if fresh:
        w.writerow(COLS)
    ep_path = os.path.join(out, "episodes.csv")
    ep_rows = [] if fresh else list(csv.DictReader(open(ep_path)))
    g = mf["n"]
    for t in todo:
        fr, ev, ni, st = process_tag(a.data, t, centers, g, ep_rows, w)
        files["frames"].write(fr.tobytes()); files["evt"].write(ev.tobytes()); files["nitro"].write(ni.tobytes())
        g += len(fr)
        mf["tags"][t] = {"sig": file_sig(a.data, t), **st}
        mf["n"] = g
        print(f"{t}: {st}", flush=True)
    for f in files.values():
        f.close()
    sf.close()
    keys = ["episode_id", "tag", "source", "policy", "start_idx", "end_idx", "n", "post_finish_dropped", "dur_s", "segments", "finished", "finish_src", "finish_time_s",
            "dq", "end", "avg_speed", "max_dist", "dist_read", "coll_per_min", "excluded"]
    for t in no_rec:
        if not any(r["tag"] == t for r in ep_rows):
            ep_rows.append({"episode_id": f"{t}_all", "tag": t, "source": source_of(t), "policy": t, "start_idx": -1, "end_idx": -1,
                            "excluded": "rec.bin yo'q (kadrsiz, 11 ustunli eski format)"})
    with open(ep_path, "w", newline="") as f:
        dw = csv.DictWriter(f, keys, extrasaction="ignore")
        dw.writeheader(); dw.writerows(ep_rows)
    json.dump(mf, open(mf_path, "w"), indent=1)
    print("jami qadam", g, flush=True)


def load_steps(path=None):
    """steps.csv -> {ustun: np.ndarray}. Bo'sh katak = nan."""
    path = path or os.path.join(HERE, "..", "data", "ds2", "steps.csv")
    with open(path) as f:
        rd = csv.reader(f)
        head = next(rd)
        rows = list(rd)
    S = {}
    txt = {"episode_id", "seg_id", "tag", "source"}
    for j, k in enumerate(head):
        v = [r[j] for r in rows]
        S[k] = np.array(v) if k in txt else np.array([float(x) if x != "" else np.nan for x in v])
    return S


if __name__ == "__main__":
    main()
