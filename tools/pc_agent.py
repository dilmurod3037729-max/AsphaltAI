"""PC Asphalt'da bizning agent (App/Agent.cs porti): kodlovchi ONNX + siyosat Q-MLP + bo'sh-tomon qoidasi, arvoh rejimda.
Yozuvlar APK formatida: data/rec_<tag>.bin + data/agent_<tag>.csv (Vision/ skriptlari o'zgarishsiz o'qiydi).
    py tools/pc_agent.py --tag PC1 --races 10"""
import argparse
import gc
import json
import os
import re
import struct
import time

import numpy as np
import onnxruntime as ort
import vgamepad as vg

from pc_env import Game, blocked, ocr, small
from pc_digits import SpeedReader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

STEP = 0.25
STALE = 6.0   # kamera/takedown sahnalarida HUD bir necha soniya yo'qoladi
STEER_V = [-1.0, -0.5, 0.0, 0.5, 1.0]
OFF, CNT = [0, 5, 8], [5, 3, 3]
MARGIN = [0.30, 0.25, 0.10]
SAFE_T, SAFE_GAP = 0.5, 0.3
DREAM_T, DREAM_GAP = 0.5, 0.2   # Dreamer: siyosat rulining 1 s xavfi >= T va eng xavfsiz variantdan GAP ko'p bo'lsa almashtiriladi
EPS = 0.08
FINISH_BONUS = 20
UP_W, STABLE_W, STABLE_BAND, GLITCH = 0.10, 0.5, 1.5, 80
WALL_W, STEER_W, ZIG_W, EXTREME_W, PROG_W, WRONG_W = 2.0, 0.05, 0.10, 0.10, 2.0, 3.0
R_MIN, R_MAX = -8, 5

# HUD qutilari (mijoz sohasiga nisbatan; 3146x1711 kadrda o'lchangan)
SPEED_BOX = (.058, .132, .873, .943)
SPEED_WIDE = (.053, .140, .845, .949)
DIST_BOX = (.111, .143, .173, .216)
B = vg.XUSB_BUTTON
DIGITS = SpeedReader()
FALSE_UNTIL = 0.0       # soxta startdan keyin natija ekrani raqamlari tezlik bo'lib ko'rinadi — shu vaqtgacha tezlik o'qishga ishonilmaydi
PRIVATE_NEVADA = False   # --private: bosh menyudan Private Room (Nevada, The Narrows) ga — emulator ma'lumotining asosiy trassasi


def crop(f, box):
    H, W = f.shape[:2]
    return f[int(box[0] * H):int(box[1] * H), int(box[2] * W):int(box[3] * W)]


def binar(c, thr):
    """Oq HUD yozuvi -> oq fonda qora (winocr rangli fonda o'qiy olmaydi)."""
    b = np.where(c.astype(np.int16).mean(2) > thr, 0, 255).astype(np.uint8)
    b = np.pad(b, 30, constant_values=255)
    return np.stack([b] * 3, 2)


def read_speed(f):
    v, w = DIGITS(f)
    if v >= 0 and w < 0.12:
        return v
    for t in ocr(binar(crop(f, SPEED_BOX), 160), 2.0):
        d = re.sub(r"\D", "", t["text"])
        if d and int(d) <= 450:
            return int(d)
    for t in ocr(binar(crop(f, SPEED_WIDE), 160), 2.0):   # 2 xonali tezlik tor qutida o'qilmaydi ("MPH" xolos)
        m = re.search(r"(\d{1,3})$", t["text"].strip())
        if m:
            return int(m.group(1))
    return -1


def read_dist(f):
    c = crop(f, DIST_BOX)
    for thr in (190, 130):
        for t in ocr(binar(c, thr), 3.0):
            s = t["text"].replace(" ", "")
            m = re.match(r"^(\d{1,3})(%|00|0/0|0/6)?$", s)   # kursiv '%' ko'pincha '00' bo'lib o'qiladi
            if m and int(m.group(1)) <= 100:
                return int(m.group(1))
    return -1


class Mlp:
    def __init__(self, path):
        buf = open(path, "rb").read()
        self.steps = struct.unpack_from("<q", buf, 0)[0]
        n = struct.unpack_from("<i", buf, 8)[0]
        sz = struct.unpack_from(f"<{n}i", buf, 12)
        o = 12 + 4 * n
        self.layers = []
        for i in range(n - 1):
            w = np.frombuffer(buf, "<f4", sz[i] * sz[i + 1], o).reshape(sz[i + 1], sz[i]); o += 4 * w.size
            b = np.frombuffer(buf, "<f4", sz[i + 1], o); o += 4 * b.size
            self.layers.append((w, b))
        self.n_in = sz[0]

    def __call__(self, x):
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)
        return x


class Pad:
    """Pad faqat poyga atrofida ulanadi: ulangan pad menyuda klaviaturani yutadi; ulanish poygani pauza qiladi (RESUME bosiladi),
    natija ekranida uzish esa muammosiz."""
    p = None

    def connect(self):
        if self.p is None:
            self.p = vg.VX360Gamepad(); self.p.reset(); self.p.update()

    def close(self):
        if self.p is not None:
            self.p.reset(); self.p.update()
            self.p = None
            gc.collect()   # VX360Gamepad.__del__ virtual padni uzadi

    def apply(self, a, tick, sx=None):
        p = self.p
        p.left_joystick_float(STEER_V[a[0]] if sx is None else float(np.clip(sx, -1, 1)), 0.0)
        if a[1] == 1:
            p.press_button(B.XUSB_GAMEPAD_LEFT_SHOULDER)
        else:
            p.release_button(B.XUSB_GAMEPAD_LEFT_SHOULDER)
        # nitro: A tugmasi bir qadam bosiq (RT ham nitro — gaz emas, PC'da gaz avtomatik); 2 = shockwave (ikki marta)
        if a[2] in (1, 2):
            p.press_button(B.XUSB_GAMEPAD_A)
        else:
            p.release_button(B.XUSB_GAMEPAD_A)
        p.update()
        if a[2] == 2:
            time.sleep(0.06); p.release_button(B.XUSB_GAMEPAD_A); p.update()
            time.sleep(0.06); p.press_button(B.XUSB_GAMEPAD_A); p.update()
        if a[1] == 2:
            for _ in range(2):
                p.press_button(B.XUSB_GAMEPAD_LEFT_SHOULDER); p.update(); time.sleep(0.06)
                p.release_button(B.XUSB_GAMEPAD_LEFT_SHOULDER); p.update(); time.sleep(0.06)



class Agent:
    def __init__(self, tag, policy="", eps=EPS, dream=False, dream_only=False):
        self.eps = eps
        self.dg = None
        self.dream_only = dream_only
        self.probe = False
        self.dream_eps = 0.0
        if dream or dream_only:
            from pc_dream import DreamGuard
            self.dg = DreamGuard()
            self.dcsv = open(os.path.join(DATA, f"dream_{tag}.csv"), "a")
            print(f"Dreamer world model qadam {self.dg.ver}", flush=True)
        ver = open(os.path.join(DATA, "live", "vision.ver")).read().strip()
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        self.enc = ort.InferenceSession(os.path.join(DATA, "live", "vision.onnx"), so, providers=["CPUExecutionProvider"])
        self.net = Mlp(policy or os.path.join(DATA, "offline", f"policy_v2_{ver}.bin"))
        self.steps = self.net.steps
        self.rnd = np.random.default_rng()
        self.csv = open(os.path.join(DATA, f"agent_{tag}.csv"), "a")
        self.rec = open(os.path.join(DATA, f"rec_{tag}.bin"), "ab")
        self.summ = open(os.path.join(DATA, "pc_races.csv"), "a")
        self.tag = tag
        print(f"kodlovchi {ver}, siyosat qadam {self.steps}", flush=True)

    def encode(self, ring):
        x = np.concatenate([r.transpose(2, 0, 1) for r in ring]).astype(np.float32)[None] / 255.0
        emb, aux = self.enc.run(None, {"x": x})
        return emb[0], aux[0]

    def perturb(self, b, cur, hold):
        u = self.rnd.random()
        if b == 0:
            hold[b] = int(self.rnd.integers(1, 4))
            if u < .10:
                return 0 if self.rnd.integers(2) == 0 else 4
            return int(np.clip(cur[b] + (-1 if u < .55 else 0 if u < .75 else 1), 0, 4))
        if b == 1:
            if u < .05:
                hold[b] = int(self.rnd.integers(2, 6)); return 1
            return 2 if u < .06 else 0
        return 1 if u < .05 else 2 if u < .06 else 0

    def race(self, g, pad):
        """Bitta poyga: HUD tezligi yo'qolib STALE dan oshguncha. Qaytaradi xulosa dict."""
        t0 = time.time(); last_fresh = t0
        ring = []; hold = [0] * 3; cur = [2, 0, 0]; hold_expl = [False] * 3
        prev_sp = prev_dist = -1; act = [2, 0, 0, 1]; last_steer = 2; last_dsteer = 0
        last_in = [0.0, 0.0, 1.0]
        speeds, overrides, finished, max_dist, n, miss = [], 0, False, -1, 0, 0
        next_t = time.time()
        probe_t0 = None
        if self.dg:
            self.dg.reset()
        try:
            while True:
                next_t += STEP
                time.sleep(max(0.0, next_t - time.time()))
                now = time.time()
                if blocked():
                    pad.p.reset(); pad.p.update()            # qulf ekrani padni tinglaydi: hech qanday buyruq yo'q
                    print("  kompyuter qulflangan — poyga to'xtatildi", flush=True)
                    break
                f = g.frame()
                sp = read_speed(f)
                if sp < 0:
                    miss += 1
                    if miss % 3 == 0:                            # pad ulanganda o'yin pauza qiladi ("CONTROLLER CONNECTED")
                        tx = g.texts()
                        U = {t.upper() for t, _, _ in tx}
                        if any("RACE RESULTS" in u for u in U):
                            break
                        if U & {"RESUME", "PAUSE", "CONTROLLER CONNECTED"} and now - t0 < 60:
                            g.tap("RESUME", texts=tx)
                            last_fresh = time.time(); next_t = time.time() + 1.5; time.sleep(1.5)
                            continue
                    if now - last_fresh > STALE:
                        break
                    continue
                miss = 0
                last_fresh = now
                dist = read_dist(f) if n % 2 == 0 else -1
                if dist >= 0 and prev_dist >= 0 and not (prev_dist <= dist <= prev_dist + 5):
                    dist = -1                                   # noto'g'ri o'qish raqam tashlab ketadi
                fr = small(f)
                ring.append(fr); ring = ring[-3:]
                emb, aux = self.encode([ring[0], ring[min(1, len(ring) - 1)], ring[-1]])
                dS = 0 if prev_sp < 0 else sp - prev_sp
                s = np.zeros(self.net.n_in, np.float32)
                s[:128] = emb; s[128:146] = aux
                s[146] = sp / 200; s[147] = np.clip(dS / 20, -2, 2)
                s[148:151] = last_in
                s[151] = np.clip((now - t0) / 180, 0, 1)
                d_dist = dist - prev_dist if dist >= 0 and prev_dist >= 0 else 0
                bonus = 0.0
                if not finished and dist >= 99 and now - t0 > 20:
                    finished = True; bonus = FINISH_BONUS
                rw = self.reward(act[0], prev_sp, sp, last_steer, last_dsteer, d_dist) if prev_sp >= 0 and abs(dS) < GLITCH else (0, 0, 0, 0, 0)
                if act[0] != last_steer:
                    last_dsteer = act[0] - last_steer
                last_steer = act[0]
                q = self.net(s)
                a = [0, 0, 0, 1]; expl = 0
                for b in range(3):
                    if hold[b] > 0:
                        hold[b] -= 1
                        if hold_expl[b]:
                            expl |= 1 << b
                        a[b] = cur[b]; continue
                    if self.rnd.random() < self.eps:
                        v = self.perturb(b, cur, hold); expl |= 1 << b
                    else:
                        qs = q[OFF[b]:OFF[b] + CNT[b]]; v = int(np.argmax(qs)); d = 2 if b == 0 else 0
                        if qs[v] - qs[d] < MARGIN[b]:
                            v = d
                    a[b] = cur[b] = v; hold_expl[b] = bool(expl & (1 << b))
                if self.dream_only:                                      # siyosat O'CHIQ: rul va nitroni faqat Dreamer rejalashtiradi
                    (sx, nit_d, dr_d), sc, Rk, Sp = self.dg.plan(fr, eps=self.dream_eps)
                    ns = len(Rk) // 4; si = int(round((sx + 1) / 2 * (ns - 1))); blk = (dr_d * 2 + nit_d) * ns
                    pol = a[0]; a = [int(np.argmin(np.abs(np.array(STEER_V) - sx))), dr_d, nit_d, 1]; expl = 32; ov = True
                    R = np.interp(STEER_V, np.linspace(-1, 1, ns), Rk[blk:blk + ns])   # video uchun 5 ustun
                    self.dcsv.write(f"{self.steps + 1},{int(now * 1000)},{sp},{Sp[blk + si]:.1f},{pol},{a[0]},{sx:.3f}," + ",".join(f"{r:.3f}" for r in Rk) + "," + ",".join(f"{v:.1f}" for v in Sp) + f",{nit_d},{dr_d}" + chr(10))
                    try:
                        with open(os.path.join(DATA, "dream_live.json.tmp"), "w") as jf:
                            json.dump({"risk": [float(r) for r in R], "policy": pol, "final": a[0], "override": False, "speed": sp, "only": True, "nitro": nit_d, "drift": dr_d, "sx": sx}, jf)
                        os.replace(os.path.join(DATA, "dream_live.json.tmp"), os.path.join(DATA, "dream_live.json"))
                    except OSError:
                        pass
                elif self.dg:                                            # Dreamer: 5 rul varianti uchun tasavvur qilingan 1 s xavfi
                    R, spd_p = self.dg.step(fr)
                    pol = a[0]; best = int(np.argmin(R)); ov = R[pol] >= DREAM_T and R[pol] - R[best] >= DREAM_GAP
                    if ov:
                        a[0] = best; expl |= 16; overrides += 1
                    self.dcsv.write(f"{self.steps + 1},{int(now * 1000)},{sp},{spd_p:.1f},{pol},{a[0]}," + ",".join(f"{r:.3f}" for r in R) + chr(10))
                    try:
                        with open(os.path.join(DATA, "dream_live.json.tmp"), "w") as jf:
                            json.dump({"risk": [float(r) for r in R], "policy": pol, "final": a[0], "override": bool(ov), "speed": sp}, jf)
                        os.replace(os.path.join(DATA, "dream_live.json.tmp"), os.path.join(DATA, "dream_live.json"))
                    except OSError:
                        pass                                             # video jarayoni shu payt o'qiyotgan bo'lishi mumkin
                else:
                    best = int(np.argmin(aux[12:17])); tk = aux[12 + a[0]]
                    if tk >= SAFE_T and tk - aux[12 + best] >= SAFE_GAP:     # bo'sh tomon: yo'l yo'q joy = to'siq
                        a[0] = best; expl |= 8; overrides += 1
                if not self.dream_only:
                    sx = STEER_V[a[0]]
                if self.probe and probe_t0 is None and sp > 80:
                    probe_t0 = now                                     # pad ulanishidagi pauzadan keyin, haqiqiy haydash paytida
                if self.probe and probe_t0 is not None and now - probe_t0 < 6.5:   # input sinovi: o'yin padni sezyaptimi
                    tt = now - probe_t0; sx = -1.0 if tt < 1.5 else 0.0 if tt < 2.5 else 1.0 if tt < 4.0 else 0.0
                    a = [int(np.argmin(np.abs(np.array(STEER_V) - sx))), 0, 1 if 4.5 < tt < 5.0 else 0, 1]
                pad.apply(a, n, sx)
                if self.dg:
                    self.dg.taken(a[0], a[1], a[2], sx)
                last_in = [STEER_V[a[0]], 1.0 if a[1] == 1 else 0.0, 1.0]
                self.steps += 1; n += 1
                ms = int(now * 1000)
                evt = crop(f, (.33, .55, .66, 1.0)); evt = evt[(np.arange(40) * evt.shape[0] // 40)][:, (np.arange(128) * evt.shape[1] // 128)]
                evt = (evt[..., 2].astype(np.int32) * 77 + evt[..., 1] * 150 + evt[..., 0] * 29 >> 8).astype(np.uint8)
                nit = small(crop(f, (.02, .09, .30, .68)), 96, 8)
                self.csv.write(f"{self.steps},{ms},{sp},{a[0]},{a[1]},{a[2]},{a[3]},{rw[0]:.3f},{self.eps:.2f},0,0,{rw[1]:.3f},{rw[2]:.3f},0,{rw[3]:.3f},{rw[4]:.3f},"
                               f"{dist},0.5,0,0,{(evt > 200).mean():.3f},{aux[0]:.3f},0,{bonus:.2f},{expl},{sx:.3f}\n")
                self.rec.write(struct.pack("<iq", self.steps, ms) + fr.tobytes() + evt.tobytes() + nit.tobytes())
                speeds.append(sp); act = a; prev_sp = sp
                if dist >= 0:
                    prev_dist = dist; max_dist = max(max_dist, dist)
                if n % 40 == 0:
                    g.hide(); self.csv.flush(); self.rec.flush()
                    if self.dg:
                        self.dcsv.flush()
                    print(f"  {now - t0:5.1f}s tezlik {sp} dist {prev_dist} bo'sh-tomon {overrides}", flush=True)
        finally:
            pad.p.reset(); pad.p.update()
            self.csv.flush(); self.rec.flush()
        sp = np.array(speeds)
        crashes, i = 0, 4
        while i < len(sp):
            if sp[i - 4] > 60 and sp[i - 4] - sp[i] > 30:
                crashes += 1; i += 4
            i += 1
        return {"dur": round(time.time() - t0, 1), "steps": n, "speed": round(float(sp.mean()), 1) if len(sp) else 0,
                "crash_min": round(crashes / max(1e-6, n * STEP / 60), 2), "dist": max_dist, "overrides": overrides}

    @staticmethod
    def reward(steer, prev, sp, last_steer, last_dsteer, d_dist):
        r_speed = 0.0
        if sp - prev > 0:
            r_speed += UP_W * 0.75 * (sp - prev)          # ra=0.5 (perc yo'q): sf = sp*0.75
        if abs(sp - prev) <= STABLE_BAND:
            r_speed += STABLE_W * sp * 0.75 / 150
        loss = prev * prev - sp * sp
        r_wall = -WALL_W * loss / 10000 if loss > 0 else 0.0
        ds = steer - last_steer
        zig = ds != 0 and last_dsteer != 0 and np.sign(ds) != np.sign(last_dsteer)
        r_steer = -STEER_W * abs(ds) - (ZIG_W if zig else 0) - (EXTREME_W if steer in (0, 4) else 0)
        r_prog = PROG_W * d_dist if d_dist > 0 else WRONG_W * d_dist
        tot = float(np.clip(r_speed + r_wall + r_steer + r_prog, R_MIN, R_MAX))
        return tot, r_speed, r_wall, r_steer, r_prog


MENU_TAP = ("RESUME", "CLAIM NOW", "CHECK NOW!", "PLAY CAREER", "MISS OUT", "NO THANKS", "NEXT", "CONTINUE", "PLAY", "RETRY", "OK", "SKIP", "CLAIM", "COLLECT", "TAP TO CONTINUE")


def _find(tx, word, ymin=0.0, ymax=1.0):
    for t, x, y in tx:
        if word in t.upper() and ymin <= y <= ymax:
            return x, y
    return None


def goto_private_nevada(g):
    """Bosh menyu -> MULTIPLAYER -> PRIVATE ROOM -> CREATE MATCH -> (OK) -> RACE SETTINGS -> MAP: NEVADA -> TRACK: THE NARROWS -> SAVE.
    2026-10-23 qo'lda o'lchangan ketma-ketlik (ro'yxatlar strelka bilan suriladi). Muvaffaqiyat: xona ekrani (ROOM ID) va NEVADA."""
    def step(click=None, key=None, wait=3.0):
        if click:
            g.click(*click)
        if key:
            g.key(key)
        time.sleep(wait)
        return g.texts()
    tx = g.texts()
    if not _find(tx, "ROOM ID"):
        tx = step(click=_find(tx, "MULTIPLAYER", 0.88) or (0.79, 0.93))
        for _ in range(4):                                                       # Multiplayer'da mukofot oynalari (SERIES SCORE / CLAIM REWARD) chiqadi
            if _find(tx, "PRIVATE") and _find(tx, "ROOM"):
                break
            c = _find(tx, "CLAIM REWARD") or _find(tx, "CLAIM") or _find(tx, "NEXT") or _find(tx, "CONTINUE")
            tx = step(click=c) if c else step(key="ESC")
        tx = step(click=_find(tx, "PRIVATE") or (0.83, 0.63))
        tx = step(click=_find(tx, "CREATE") or (0.57, 0.73), wait=4)
        if _find(tx, "CROSS-PLAY"):
            tx = step(click=(0.50, 0.71), wait=5)
    for _ in range(6):                                                           # xona yuklanishi sekin bo'lishi mumkin
        if _find(tx, "ROOM ID"):
            break
        tx = step(wait=2)
    if not _find(tx, "ROOM ID"):
        return False
    if _find(tx, "NEVADA", 0, 0.3) and _find(tx, "NARROWS", 0, 0.3):
        return True
    tx = step(click=_find(tx, "RACE SETTINGS") or (0.50, 0.86))
    tx = step(click=(0.42, 0.21))                                              # MAP
    for _ in range(30):                                                          # NEVADA ko'ringuncha o'ngga
        pos = _find(tx, "NEVADA", 0.6, 0.75)
        if pos:
            break
        tx = step(key="RIGHT", wait=0.8)
    else:
        return False
    tx = step(click=(pos[0], pos[1] - 0.07), wait=2)
    tx = step(click=(0.59, 0.21))                                              # TRACK
    for _ in range(20):
        pos = _find(tx, "NARROWS", 0.6, 0.75)
        if pos:
            break
        tx = step(key="RIGHT", wait=0.8)
    else:
        return False
    tx = step(click=(pos[0], pos[1] - 0.07), wait=2)
    tx = step(click=(0.93, 0.99), wait=3)                                      # SAVE
    return bool(_find(tx, "ROOM ID")) and bool(_find(tx, "NARROWS"))


def to_race(g, pad, timeout=240):
    """Menyulardan poyga boshlanguncha. Pad yuklanish ekranida ulanadi."""
    t_end = time.time() + timeout; idle = 0
    while time.time() < t_end:
        if blocked():
            pad.close(); time.sleep(5); continue
        g.hide()
        if time.time() > FALSE_UNTIL and read_speed(g.frame()) >= 0:
            pad.connect()                                             # qulfdan keyin pad uzilgan bo'lishi mumkin
            return True
        tx = g.texts()
        U = {t.upper().strip() for t, _, _ in tx}
        if any("YOUR TIME" in u or "RACE RESULTS" in u for u in U):          # natija ekranida ham "POS." bor — avval NEXT
            nx = [(x, y) for t, x, y in tx if t.upper().strip() == "NEXT"]
            if nx:
                g.click(*nx[0]); print("  menyu: NEXT (natija)", flush=True)
            time.sleep(3); continue
        if U & {"DIST.", "POS."}:
            pad.connect()
            return True                                               # poyga HUD'i (startda tezlik 0 o'qilmasligi mumkin)
        if any("ROOM ID" in u for u in U):                              # Private Room (Nevada, The Narrows): READY bosiladi, ESC xonadan chiqarib yuboradi
            if PRIVATE_NEVADA and not any("NARROWS" in u for u in U):
                print("  menyu: xonada boshqa trassa -> Nevada/The Narrows", goto_private_nevada(g), flush=True); continue
            rd = [(x, y) for t, x, y in tx if t.upper().strip() == "READY"]
            if rd:
                g.click(*rd[0]); print("  menyu: READY (private room)", flush=True)
            time.sleep(4); continue
        if any("QUIT THE GAME" in u for u in U):
            g.click(0.326, 0.686); print("  menyu: NO (chiqish oynasi)", flush=True); time.sleep(3); continue
        again = [(x, y) for t, x, y in tx if t.upper().replace(" ", "").startswith("PLAYA") and len(t.replace(" ", "")) <= 11 and t.upper().replace(" ", "").endswith(("WAY", "WAV"))]   # PLAYANYWAY / OCR "PLAYAWWAV"; "PLAY AGAIN", "PLAY AND WATCH AN AD" (reklama!) emas   # "PLAY ANYWAY" (bayroqlar yig'ilgan) — trassa A/B uchun o'zgarmasin
        if again:
            g.click(*again[0]); print("  menyu: PLAY ANYWAY", flush=True); time.sleep(4); continue
        if {"CAREER", "MULTIPLAYER"} <= U and PRIVATE_NEVADA:          # umumiy tugmalardan OLDIN: aks holda "PLAY CAREER" bosilib ketadi
            print("  menyu: private room / Nevada / The Narrows ->", goto_private_nevada(g), flush=True); continue
        hit = g.tap(*[w for w in MENU_TAP if not (PRIVATE_NEVADA and w == "PLAY CAREER")], texts=tx)
        if hit:
            print("  menyu:", hit, flush=True); idle = 0; time.sleep(4); continue
        if any("RACE GO" in u for u in U):
            pad.connect(); time.sleep(2); continue                     # poyga oldi yuklanish: pad shu yerda ulanadi
        if not tx or U <= {"ASPHALT - V53.5.3A - D3D12", "ASPHALT"}:
            time.sleep(2); continue                                    # yuklanish
        if {"CAREER", "MULTIPLAYER"} <= U:
            print("  menyu:", g.tap("CAREER", texts=tx), flush=True)  # bosh menyu: boshqa bo'lim tanlangan bo'lishi mumkin -> Career bo'limi, keyin PLAY CAREER
        elif any("COLLECTED" in u for u in U):
            g.key("ENTER"); print("  menyu: ENTER", flush=True)       # Career xaritasi: joriy bosqich fokusda
        elif "BACK" in U and any("RECOMMENDED RANK" in u for u in U):
            g.click(0.13, 0.45); print("  menyu: 1-mashina", flush=True)
        elif {"CAMERA", "CONTROLS"} <= U and "CLOSE" in U:                  # poyga ichidagi sozlamalar oynasi (pad uzilganda ochiladi): ESC-belgi uni qayta ochadi
            print("  menyu: sozlamalar oynasi -> CLOSE", g.tap("CLOSE", texts=tx), flush=True)
        elif "BACK" in U and "EXIT GAME" not in U:
            g.click(0.905, 0.982); print("  menyu: ESC-belgi", sorted(U)[:6], flush=True)   # tanimagan oyna: "BACK" yozuvi bosilmaydi, chapdagi ESC belgisi bosiladi
        elif "EXIT GAME" not in U:
            g.key("ESC"); print("  menyu: ESC", sorted(U)[:6], flush=True)
        idle += 1
        time.sleep(4)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="PC1")
    ap.add_argument("--races", type=int, default=10)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--policy", default="", help="siyosat fayli (bo'sh = jonli policy_v2_<ver>.bin)")
    ap.add_argument("--dream", action="store_true", help="Dreamer xavfsizlik qatlami (K14 qoidasi o'rniga)")
    ap.add_argument("--dream-only", action="store_true", help="siyosat o'chiq, faqat Dreamer rejalashtiradi")
    ap.add_argument("--dream-eps", type=float, default=0.0, help="faqat-Dreamer: tasodifiy (top-4) reja ulushi")
    ap.add_argument("--private", action="store_true", help="Private Room: Nevada / The Narrows")
    ap.add_argument("--probe", action="store_true", help="poyga boshida 6 s input sinovi (chap/o'ng/nitro)")
    ap.add_argument("--video", type=int, default=0, help="shu raqamli poygani video qilish (tools/pc_video.py data/video_go ni kutadi)")
    a = ap.parse_args()
    import subprocess, sys
    subprocess.Popen([sys.executable, os.path.join(HERE, "keep_awake.py"), str(os.getpid())], creationflags=0x08000000)   # o'zi qulflanib qolmasin (qulfda input o'yinga yetmaydi)
    g = Game()
    ag = Agent(a.tag, a.policy, 0.0 if a.dream_only else a.eps, a.dream, a.dream_only)
    ag.probe = a.probe
    global PRIVATE_NEVADA
    PRIVATE_NEVADA = a.private
    ag.dream_eps = a.dream_eps
    go = os.path.join(DATA, "video_go")
    pad = Pad()
    i, tries = 0, 0
    while i < a.races and tries < 3 * a.races:
        tries += 1
        while blocked():
            pad.close(); time.sleep(5)
        if not to_race(g, pad):
            print("poyga boshlanmadi — to'xtadim", flush=True); break
        print(f"poyga {i + 1} boshlandi", flush=True)
        if a.video == i + 1:
            open(go, "w").close()
        r = ag.race(g, pad)
        if r["steps"] < 20:                                         # soxta start (natija ekrani poyga deb olingan) — hisoblanmaydi
            global FALSE_UNTIL
            FALSE_UNTIL = time.time() + 20
            print(f"  soxta start ({r['steps']} qadam) — hisoblanmadi; ekran: {sorted(t.upper() for t, _, _ in g.texts())[:8]}", flush=True); pad.close(); time.sleep(3); continue
        if os.path.exists(go):
            os.remove(go)
        time.sleep(2)
        pad.close()
        res = " ".join(t for t, _, _ in g.texts())
        m = re.search(r"(\d\d):(\d\d)[.,](\d{3})", res)
        r["finish"] = "RACE RESULTS" in res.upper() or r["dist"] >= 97
        r["time"] = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000 if m and r["finish"] else ""
        print(f"poyga {i + 1}: {r}", flush=True)
        ag.summ.write(f"{a.tag},{time.strftime('%Y-%m-%d %H:%M:%S')},{r['finish']},{r['time']},{r['dur']},{r['speed']},{r['crash_min']},{r['dist']},{r['overrides']},{r['steps']}\n")
        ag.summ.flush()
        i += 1


if __name__ == "__main__":
    main()
