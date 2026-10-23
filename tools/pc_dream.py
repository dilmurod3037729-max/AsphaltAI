"""Dreamer world model jonli poygada: har qadamda holatni yangilaydi va 5 rul varianti uchun 1 s kelajakdagi urilish xavfini tasavvur qiladi.
pc_agent.py --dream bilan ishlatiladi; K14 bo'sh-tomon qoidasi o'rniga."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Vision"))
import dreamer as DR  # noqa: E402
import gpuguard  # noqa: E402

K, HOR = 8, 8


SX = np.linspace(-1, 1, 9)                       # reja tuzuvchi sinaydigan joystik qiymatlari


def onehot(steer, drift, nitro):
    a = np.zeros(DR.NA, np.float32); a[steer] = 1; a[5 + drift] = 1; a[8 + nitro] = 1
    return a


def act_vec(sx, drift, nitro):
    """Uzluksiz joystik qiymati bilan harakat vektori (rul 5 bin bo'yicha interpolatsiya)."""
    a = np.zeros(DR.NA, np.float32); a[:5] = DR.steer_soft(sx); a[5 + drift] = 1; a[8 + nitro] = 1
    return a


class DreamGuard:
    def __init__(self):
        gpuguard.cap_vram()
        self.dev = "cuda"
        self.wm = DR.WorldModel().to(self.dev)
        ck = torch.load(DR.CKPT, map_location="cpu")
        self.wm.load_state_dict(ck["wm"]); self.wm.eval()
        self.ver = ck["step"]
        self.plans = torch.stack([torch.from_numpy(np.stack([onehot(s, 0, 0)] * 4 + [onehot(2, 0, 0)] * (HOR - 4))) for s in range(5)]).to(self.dev)
        # to'liq boshqaruv rejalari: drift {yo'q, ushla 1 s} x nitro {yo'q, bir marta} x 9 joystik qiymati; qiymat 4 qadam (1 s), keyin to'g'ri
        P = []
        for dr in (0, 1):
            for nit in (0, 1):
                for sx in SX:
                    P.append(np.stack([act_vec(sx, dr, nit)] + [act_vec(sx, dr, 0)] * 3 + [act_vec(0.0, 0, 0)] * (HOR - 4)))
        self.plans_full = torch.from_numpy(np.stack(P)).to(self.dev)
        self.reset()

    @torch.no_grad()
    def plan(self, frame, risk_w=2.0, eps=0.0):
        """Siyosatsiz: holatni yangilab, 18 rejadan eng yaxshisini tanlaydi. Baho = 2 s dagi o'rtacha tezlik/100 - risk_w * max(1 s urilish, 2 s tiqilish).
        Qaytaradi: (joystik qiymati, nitro, drift), baholar [36], xavf [36], tezlik [36]."""
        wm = self.wm
        x = torch.from_numpy(frame).to(self.dev).permute(2, 0, 1)[None].float().div(255).sub(0.5)
        e = wm.enc(x)
        self.h, _ = wm.img_step(self.h, self.z, self.a_prev)
        self.z = wm.sample(wm.dist(wm.post(torch.cat([self.h, e], -1))))
        n = len(self.plans_full)
        h = self.h.repeat(n * K, 1); z = self.z.repeat(n * K, 1)
        A = self.plans_full.repeat_interleave(K, 0)
        risk, spd = [], []
        for t in range(HOR):
            h, p = wm.img_step(h, z, A[:, t]); z = wm.sample(p)
            f = torch.cat([h, z], -1)
            r = torch.sigmoid(wm.collh(f)).squeeze(-1)
            if hasattr(wm, "stuckh"):
                r = torch.maximum(r, torch.sigmoid(wm.stuckh(f)).squeeze(-1))
            risk.append(r); spd.append(wm.spdh(f).squeeze(-1))
        Rk = torch.stack(risk, 1).view(n, K, HOR)[:, :, :4].amax(2).mean(1)
        Sp = torch.stack(spd, 1).view(n, K, HOR).mean((1, 2))
        score = Sp - risk_w * Rk
        i = int(torch.argmax(score))
        if eps > 0 and np.random.rand() < eps:                       # tajriba: eng yaxshi 4 reja ichidan tasodifiy (o'zi o'rganishi uchun yangi holatlar)
            i = int(np.random.choice(torch.topk(score, 4).indices.cpu().numpy()))
        ns = len(SX)
        return (float(SX[i % ns]), (i // ns) % 2, i // (2 * ns)), score.cpu().numpy(), Rk.cpu().numpy(), (Sp * 100).cpu().numpy()

    def reset(self):
        self.h, self.z = self.wm.initial(1, self.dev)
        self.a_prev = torch.zeros(1, DR.NA, device=self.dev)

    @torch.no_grad()
    def step(self, frame):
        """frame: 48x96x3 uint8 RGB (rec formati). Qaytaradi: xavf [5] (keyingi 1 s ichida urilish ehtimoli), tezlik bashorati mph."""
        wm = self.wm
        x = torch.from_numpy(frame).to(self.dev).permute(2, 0, 1)[None].float().div(255).sub(0.5)
        e = wm.enc(x)
        self.h, _ = wm.img_step(self.h, self.z, self.a_prev)
        self.z = wm.sample(wm.dist(wm.post(torch.cat([self.h, e], -1))))
        h = self.h.repeat(5 * K, 1); z = self.z.repeat(5 * K, 1)
        A = self.plans.repeat_interleave(K, 0)
        risk = []
        for t in range(4):
            h, p = wm.img_step(h, z, A[:, t]); z = wm.sample(p)
            risk.append(torch.sigmoid(wm.collh(torch.cat([h, z], -1))).squeeze(-1))
        R = torch.stack(risk, 1).view(5, K, 4).amax(2).mean(1)
        spd = float(wm.spdh(torch.cat([self.h, self.z], -1))) * 100
        return R.cpu().numpy(), spd

    def taken(self, steer, drift, nitro, sx=None):
        a = act_vec(sx, min(drift, 2), min(nitro, 2)) if sx is not None else onehot(steer, min(drift, 2), min(nitro, 2))
        self.a_prev = torch.from_numpy(a)[None].to(self.dev)
