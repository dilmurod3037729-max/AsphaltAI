"""Yordamchi yorliqlar (o'yinning o'zidan, qo'lda belgilashsiz): to'qnashuv, tezlik o'zgarishi, nitro o'sishi, hodisa sinfi, kelajak reward."""
import numpy as np

from data import episode_bounds

H = 4              # kelajak gorizonti (qadam, 250 ms) = 1 soniya
COLL_MPH = 20      # H qadamda shuncha tezlik yo'qotish = to'qnashuv deb olinadi
K_EVT = 8          # hodisa sinflari soni (0 = hodisa yo'q)
GAMMA = 0.97


def nitro_fill(px):
    """APK'dagi hisob bilan bir xil: ustun to'lgan = yorqin va to'yingan."""
    a = px[:, 2:6].astype(np.float32).mean(1)            # [N,96,3]
    mx = a.max(-1); mn = a.min(-1)
    return ((mx > 120) & (mx - mn > 80)).mean(1)


def evt_thumbs(evt):
    # yozuv = qora yorliq ichidagi oq matn: yorqin piksel VA 7x7 atrofida qora piksel bor (osmon/yorug' fon chiqarib tashlanadi)
    from scipy.ndimage import minimum_filter
    mn = minimum_filter(evt, size=(1, 7, 7))
    m = ((evt > 200) & (mn < 70)).astype(np.float32)      # [N,40,128]
    n = len(m)
    return m.reshape(n, 10, 4, 32, 4).mean((2, 4)).reshape(n, -1)


def kmeans(x, k, iters=40, seed=0):
    rs = np.random.RandomState(seed)
    c = [x[rs.randint(len(x))]]
    for _ in range(k - 1):
        d = np.min([((x - ci) ** 2).sum(1) for ci in c], 0)
        c.append(x[rs.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rs.randint(len(x))])
    c = np.array(c)
    for _ in range(iters):
        a = ((x[:, None] - c[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = x[a == j].mean(0)
    return c


def event_classes(evt, centers=None):
    """0 = hodisa yozuvi yo'q; 1..K-1 = klasterlar (nomlari events_sheet.png ga qarab qo'yiladi)."""
    th = evt_thumbs(evt)
    active = th.mean(1) > 0.01
    if centers is None:
        centers = kmeans(th[active], K_EVT - 1) if active.sum() >= K_EVT * 5 else np.zeros((K_EVT - 1, th.shape[1]))
    cls = np.zeros(len(th), np.int64)
    if active.any():
        cls[active] = 1 + ((th[active][:, None] - centers[None]) ** 2).sum(2).argmin(1)
    return cls, centers


def wrongway(frames):
    """O'yinning "WRONG WAY!" belgisi (qizil-pushti X) — APK'dagi hisob bilan bir xil. frames: [N,48,96,3] uint8 RGB."""
    F = frames.astype(np.int16)
    R, G, B = F[..., 0], F[..., 1], F[..., 2]
    pink = (R > 200) & (G < 110) & (B > 40) & (B < 170) & (R - G > 110)
    return pink[:, 8:34, 28:68].sum((1, 2)) >= 30


def build(D, centers=None):
    n = len(D["speed"])
    start, end = episode_bounds(D["ep"])
    v = D["speed"].astype(np.float32)
    fill = nitro_fill(D["nitro_px"])
    r = np.nan_to_num(D["r"]).astype(np.float32)
    drop = np.zeros(n, np.float32); gain = np.zeros(n, np.float32); ngain = np.zeros(n, np.float32)
    valid = np.zeros(n, bool); G = np.zeros(n, np.float32)
    ww = wrongway(D["frame"])
    ww_next = np.zeros(n, bool)
    for i in range(n):
        e = end[i]
        if i + H < e:
            valid[i] = True
            ww_next[i] = ww[i + 1: i + H + 1].any() and not ww[i]
            drop[i] = max(0.0, v[i] - v[i + 1: i + H + 1].min())
            gain[i] = v[i + H] - v[i]
            ngain[i] = max(0.0, fill[i + H] - fill[i])
    for s in sorted(set(start)):
        e = end[s]
        acc = 0.0
        for i in range(e - 1, s - 1, -1):
            acc = r[i] + GAMMA * acc
            G[i] = acc
    cls, centers = event_classes(D["evt_px"], centers)
    return {
        "valid": valid, "coll": (drop >= COLL_MPH) & valid, "drop": np.clip(drop / 50, 0, 1.5), "gain": np.clip(gain / 50, -1.5, 1.5),
        "ngain": np.clip(ngain * 5, 0, 1.5), "evt": cls, "ww": ww, "ww_next": ww_next,
        "steer": np.nan_to_num(D["a_steer"], nan=2).astype(np.int64).clip(0, 4), "drift": np.nan_to_num(D["a_drift"], nan=0).astype(np.int64).clip(0, 2),
        "nitro_a": np.nan_to_num(D["a_nitro"], nan=0).astype(np.int64).clip(0, 2), "val": np.clip(G / 10, -3, 3), "fill": fill, "start": start,
    }, centers
