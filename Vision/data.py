"""APK yozgan sessiyalarni (rec.bin + agent.csv) o'qish va epizodlarga ajratish.
rec.bin yozuvi: int32 step, int64 t(ms), kadr 48x96x3 RGB, hodisa ustuni 40x128 kul, nitro chizig'i 8x96x3 RGB."""
import glob
import os

import numpy as np

REC = np.dtype([("step", "<i4"), ("t", "<i8"), ("frame", "u1", (48, 96, 3)), ("evt", "u1", (40, 128)), ("nitro", "u1", (8, 96, 3))])
COLS = ["step", "now", "speed", "a_steer", "a_drift", "a_nitro", "a_gas", "r", "eps", "loss", "age", "r_speed", "r_wall", "r_off",
        "r_steer", "r_prog", "dist", "ra", "off", "nitro", "evt", "coll_pred", "ww", "bonus", "expl"]
GAP_MS = 1500


def load_session(rec_path, csv_path):
    buf = open(rec_path, "rb").read()
    n = len(buf) // REC.itemsize
    rec = np.frombuffer(buf[: n * REC.itemsize], dtype=REC)
    a = np.genfromtxt(csv_path, delimiter=",")
    if a.ndim == 1:
        a = a[None]
    if a.shape[1] < len(COLS):                       # eski format: yetishmagan ustunlar NaN
        a = np.hstack([a, np.full((len(a), len(COLS) - a.shape[1]), np.nan)])
    by_step = {int(s): i for i, s in enumerate(a[:, 0])}
    rows = [(i, by_step[int(r["step"])]) for i, r in enumerate(rec) if int(r["step"]) in by_step]
    if not rows:
        return None
    ri = np.array([x[0] for x in rows]); ai = np.array([x[1] for x in rows])
    order = np.argsort(rec["step"][ri], kind="stable")
    ri, ai = ri[order], ai[order]
    d = {"frame": rec["frame"][ri], "evt_px": rec["evt"][ri], "nitro_px": rec["nitro"][ri], "t": rec["t"][ri]}
    for k, name in enumerate(COLS):
        d[name] = a[ai, k]
    step = d["step"].astype(np.int64); t = d["t"]
    brk = np.r_[True, (np.diff(step) != 1) | (np.diff(t) > GAP_MS)]
    d["ep"] = np.cumsum(brk) - 1
    return d


def load_all(data_dir, tags=None):
    """data/ ichidagi rec_<tag>.bin + agent_<tag>.csv juftliklarini birlashtiradi."""
    parts = []
    for rp in sorted(glob.glob(os.path.join(data_dir, "rec_*.bin"))):
        tag = os.path.basename(rp)[4:-4]
        if tags and tag not in tags:
            continue
        cp = os.path.join(data_dir, f"agent_{tag}.csv")
        if not os.path.exists(cp):
            continue
        d = load_session(rp, cp)
        if d is not None:
            d["session"] = np.full(len(d["step"]), len(parts))
            parts.append(d)
    if not parts:
        raise SystemExit("sessiya topilmadi: data/rec_<tag>.bin + agent_<tag>.csv kerak")
    out, off = {}, 0
    for p in parts:
        p["ep"] = p["ep"] + off
        off = int(p["ep"].max()) + 1
    for k in parts[0]:
        out[k] = np.concatenate([p[k] for p in parts])
    return out


def episode_bounds(ep):
    """Har bir namuna uchun epizod boshi/oxiri indekslari."""
    n = len(ep)
    start = np.zeros(n, np.int64); end = np.zeros(n, np.int64)
    i = 0
    while i < n:
        j = i
        while j < n and ep[j] == ep[i]:
            j += 1
        start[i:j] = i; end[i:j] = j
        i = j
    return start, end
