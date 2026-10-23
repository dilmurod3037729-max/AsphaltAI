"""DreamerV3 uslubidagi world model (RSSM) — data/ds2 ustida oflayn, fp32, GPU himoyasi bilan (gpuguard).
Kirish: kadr 48x96 RGB, harakat one-hot (rul 5 + drift 3 + nitro 3), is_first (segment boshi -> holat nolga).
Chiqish boshlari: kadr rekonstruksiyasi, har curriculum bosqichi reward'i (rewards.STAGES, symlog), continue (1 - terminal).
Reward'lar xom signallardan qayta hisoblanadi — bosqich almashsa faqat reward boshi qayta o'rgatiladi.
    py tools/safe_run.py 7200 -- py Vision/dreamer.py --steps 20000
    py Vision/dreamer.py --eval            (open-loop bashorat sifati, nusxa-bazaga nisbatan)
Aktyor-kritik (tasavvurda o'rganish) — keyingi bosqich, bu fayl faqat world model."""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import build_ds2 as B
import gpuguard
import rewards as RW

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "data", "ds2")
CKPT = os.path.join(HERE, "out", "dreamer_wm.pt")
NA = 11
STAGES = [0, 6]                       # to'liq yozilgan reward'lar; qolganlari skelet


STEER_V = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


def steer_soft(x):
    """Joystik qiymati [-1,1] -> 5 bin bo'yicha chiziqli interpolatsiya (0.3 = 0.4*[0] + 0.6*[0.5]); diskret qiymatlarda one-hot bilan bir xil."""
    x = np.clip(np.asarray(x, np.float64), -1, 1)
    f = (x + 1) * 2; i0 = np.clip(np.floor(f).astype(int), 0, 3); w = f - i0
    out = np.zeros(x.shape + (5,), np.float32)
    np.put_along_axis(out, i0[..., None], (1 - w)[..., None], -1)
    np.add.at(out.reshape(-1, 5), (np.arange(out.size // 5), (i0 + 1).ravel()), w.ravel())
    return out


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


class Data:
    def __init__(self, T, source, exclude=(), pc_frac=0.0):
        S = B.load_steps(os.path.join(DS, "steps.csv"))
        self.n = len(S["speed"])
        self.fr = np.memmap(os.path.join(DS, "frames.u8"), np.uint8, "r", shape=(self.n, 48, 96, 3))
        A = np.zeros((self.n, NA), np.float32)
        sx = np.where(np.isfinite(S["steer_x"]), S["steer_x"], STEER_V[S["a_steer"].astype(int).clip(0, 4)])
        A[:, :5] = steer_soft(sx)
        A[np.arange(self.n), 5 + S["a_drift"].astype(int).clip(0, 2)] = 1
        A[np.arange(self.n), 8 + S["a_nitro"].astype(int).clip(0, 2)] = 1
        self.act = A
        self.first = (S["is_first"] == 1).astype(np.float32)
        self.cont = (1 - S["terminal"]).astype(np.float32)
        self.rew = np.stack([RW.STAGES[k](S) for k in STAGES], 1).astype(np.float32)
        self.cmask = np.isfinite(S["coll"]).astype(np.float32)
        self.coll = np.nan_to_num(S["coll"]).astype(np.float32)
        self.spd = (S["speed"] / 100).astype(np.float32)
        # tiqilish: poyga o'rtasida keyingi 2 s (8 qadam) tezlik < 30 (segment ichida)
        v, seg, tr = S["speed"], S["seg_id"], S["t_race_ms"]
        st_ = np.zeros(self.n, np.float32)
        for i in range(self.n - 8):
            if tr[i] > 8000 and seg[i + 8] == seg[i] and (v[i:i + 9] < 30).all():
                st_[i] = 1
        self.stuck = st_
        self.spw = np.where(S["speed"] < 40, 4.0, 1.0).astype(np.float32)   # past tezlik: avval eng ko'p adashgan joy
        eps = [e for e in csv.DictReader(open(os.path.join(DS, "episodes.csv"))) if not e["excluded"] and e["start_idx"] != "-1"
               and e["source"] in source and e["tag"] not in exclude and int(e["end_idx"]) - int(e["start_idx"]) > T]
        rs = np.random.RandomState(0)
        ids = sorted({e["episode_id"] for e in eps}); rs.shuffle(ids)
        hold = set(ids[:max(1, len(ids) // 10)])
        self.tr = [(int(e["start_idx"]), int(e["end_idx"])) for e in eps if e["episode_id"] not in hold]
        self.va = [(int(e["start_idx"]), int(e["end_idx"])) for e in eps if e["episode_id"] in hold]
        self.T = T
        self.rs = np.random.RandomState(1)
        ep_src = {(int(e["start_idx"]), int(e["end_idx"])): e["source"] for e in eps}
        self.tr_pc = [x for x in self.tr if ep_src.get(x) == "pc"]
        self.pc_frac = pc_frac if self.tr_pc else 0.0

    def batch(self, bs, val=False):
        eps = self.va if val else self.tr
        idx = []
        for _ in range(bs):
            pool = self.tr_pc if (not val and self.pc_frac and self.rs.rand() < self.pc_frac) else eps
            a, b = pool[self.rs.randint(len(pool))]
            s = self.rs.randint(a, b - self.T)
            idx.append(np.arange(s, s + self.T))
        idx = np.stack(idx)
        first = self.first[idx].copy(); first[:, 0] = 1           # har oyna boshida holat nolga
        return {"obs": torch.from_numpy(self.fr[idx.reshape(-1)].reshape(bs, self.T, 48, 96, 3).copy()),
                "act": torch.from_numpy(self.act[idx]), "first": torch.from_numpy(first),
                "rew": torch.from_numpy(self.rew[idx]), "cont": torch.from_numpy(self.cont[idx]),
                "coll": torch.from_numpy(self.coll[idx]), "cmask": torch.from_numpy(self.cmask[idx]), "spd": torch.from_numpy(self.spd[idx]),
                "stuck": torch.from_numpy(self.stuck[idx]), "spw": torch.from_numpy(self.spw[idx])}


class Encoder(nn.Module):
    def __init__(self, d=48):
        super().__init__()
        ch = [3, d, 2 * d, 4 * d, 8 * d]
        self.c = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1) for i in range(4)])
        self.n = nn.ModuleList([nn.GroupNorm(1, ch[i + 1]) for i in range(4)])
        self.out = 8 * d * 3 * 6

    def forward(self, x):                                  # [N,3,48,96] -> [N,out]
        for c, n in zip(self.c, self.n):
            x = F.silu(n(c(x)))
        return x.flatten(1)


class Decoder(nn.Module):
    def __init__(self, feat, d=48):
        super().__init__()
        self.d = d
        self.fc = nn.Linear(feat, 8 * d * 3 * 6)
        ch = [8 * d, 4 * d, 2 * d, d]
        self.c = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i + 1], 4, 2, 1) for i in range(3)])
        self.n = nn.ModuleList([nn.GroupNorm(1, ch[i + 1]) for i in range(3)])
        self.last = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    def forward(self, f):
        x = self.fc(f).view(-1, 8 * self.d, 3, 6)
        for c, n in zip(self.c, self.n):
            x = F.silu(n(c(x)))
        return self.last(x)                                # [N,3,48,96], kadr - 0.5


def mlp(i, o, h=512, n=2):
    L = []
    for _ in range(n):
        L += [nn.Linear(i, h), nn.LayerNorm(h), nn.SiLU()]; i = h
    return nn.Sequential(*L, nn.Linear(i, o))


class WorldModel(nn.Module):
    def __init__(self, deter=1024, stoch=32, classes=32, d=48, hidden=512):
        super().__init__()
        self.deter, self.stoch, self.classes = deter, stoch, classes
        self.enc = Encoder(d)
        z = stoch * classes
        self.img_in = nn.Sequential(nn.Linear(z + NA, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, deter)
        self.prior = mlp(deter, z, hidden, 1)
        self.post = mlp(deter + self.enc.out, z, hidden, 1)
        feat = deter + z
        self.dec = Decoder(feat, d)
        self.rew = mlp(feat, len(STAGES), hidden)
        self.cont = mlp(feat, 1, hidden)
        self.collh = mlp(feat, 1, hidden)                            # 1 s ichida urilish (logit)
        self.spdh = mlp(feat, 1, hidden)                             # tezlik / 100 mph
        self.stuckh = mlp(feat, 1, hidden)                           # keyingi 2 s tiqilish (logit)

    def dist(self, logits):
        lg = logits.view(*logits.shape[:-1], self.stoch, self.classes)
        p = 0.99 * F.softmax(lg, -1) + 0.01 / self.classes          # unimix 1%
        return p

    def sample(self, p):
        idx = torch.multinomial(p.reshape(-1, self.classes), 1).view(*p.shape[:-1])
        oh = F.one_hot(idx, self.classes).float()
        return (oh + p - p.detach()).flatten(-2)                     # straight-through

    def initial(self, n, dev):
        return torch.zeros(n, self.deter, device=dev), torch.zeros(n, self.stoch * self.classes, device=dev)

    def img_step(self, h, z, a):
        h = self.gru(self.img_in(torch.cat([z, a], -1)), h)
        return h, self.dist(self.prior(h))

    def observe(self, obs, act, first):
        """obs [B,T,48,96,3] uint8, act [B,T,NA] (a_t = shu qadamda tanlangan harakat), first [B,T]."""
        Bn, T = act.shape[:2]
        x = obs.permute(0, 1, 4, 2, 3).float().div(255).sub(0.5)
        e = self.enc(x.reshape(Bn * T, 3, 48, 96)).view(Bn, T, -1)
        h, z = self.initial(Bn, act.device)
        a_prev = torch.zeros(Bn, NA, device=act.device)
        H, Z, PP, PQ = [], [], [], []
        for t in range(T):
            keep = (1 - first[:, t]).unsqueeze(-1)
            h, z, a_prev = h * keep, z * keep, a_prev * keep
            h, pp = self.img_step(h, z, a_prev)
            pq = self.dist(self.post(torch.cat([h, e[:, t]], -1)))
            z = self.sample(pq)
            H.append(h); Z.append(z); PP.append(pp); PQ.append(pq)
            a_prev = act[:, t]
        return torch.stack(H, 1), torch.stack(Z, 1), torch.stack(PP, 1), torch.stack(PQ, 1), x

    def loss(self, b):
        H, Z, PP, PQ, x = self.observe(b["obs"], b["act"], b["first"])
        f = torch.cat([H, Z], -1)
        Bn, T = H.shape[:2]
        rec = self.dec(f.reshape(Bn * T, -1)).view(Bn, T, 3, 48, 96)
        l_img = (rec - x).pow(2).sum((2, 3, 4)).mean()
        l_rew = (self.rew(f) - symlog(b["rew"])).pow(2).sum(-1).mean()
        l_cont = F.binary_cross_entropy_with_logits(self.cont(f).squeeze(-1), b["cont"])
        kl = lambda p, q: (p * (p.log() - q.log())).sum(-1).sum(-1)
        l_dyn = torch.clamp(kl(PQ.detach(), PP), min=1.0).mean()          # free bits 1 nat
        l_rep = torch.clamp(kl(PQ, PP.detach()), min=1.0).mean()
        lc = F.binary_cross_entropy_with_logits(self.collh(f).squeeze(-1), b["coll"], reduction="none", pos_weight=torch.tensor(5.0, device=f.device))
        l_coll = (lc * b["cmask"]).sum() / b["cmask"].sum().clamp(min=1)
        l_spd = ((self.spdh(f).squeeze(-1) - b["spd"]).pow(2) * b["spw"]).mean()
        l_stk = F.binary_cross_entropy_with_logits(self.stuckh(f).squeeze(-1), b["stuck"], pos_weight=torch.tensor(5.0, device=f.device))
        tot = l_img + l_rew + l_cont + 0.5 * l_dyn + 0.1 * l_rep + 10 * l_coll + 50 * l_spd + 10 * l_stk
        return tot, {"img": l_img.item(), "rew": l_rew.item(), "coll": l_coll.item(), "spd": l_spd.item(), "stuck": l_stk.item(), "kl": kl(PQ, PP).mean().item()}

    @torch.no_grad()
    def open_loop(self, b, ctx=16):
        """ctx qadam kuzatib, qolganini faqat haqiqiy harakatlar bilan tasavvur qiladi. Qaytaradi: kadr MSE (model, nusxa-baza), reward korrelyatsiyasi."""
        obs, act = b["obs"], b["act"]
        H, Z, _, _, x = self.observe(obs[:, :ctx], act[:, :ctx], b["first"][:, :ctx])
        h, z = H[:, -1], Z[:, -1]
        preds, rws = [], []
        for t in range(ctx, act.shape[1]):
            h, p = self.img_step(h, z, act[:, t - 1])
            z = self.sample(p)
            f = torch.cat([h, z], -1)
            preds.append(self.dec(f)); rws.append(symexp(self.rew(f)))
        P = torch.stack(preds, 1)
        tgt = obs[:, ctx:].permute(0, 1, 4, 2, 3).float().div(255).sub(0.5)
        last = x[:, -1:].expand_as(tgt)
        R = torch.stack(rws, 1); Rt = b["rew"][:, ctx:]
        corr = [float(np.corrcoef(R[..., k].flatten().cpu(), Rt[..., k].flatten().cpu())[0, 1]) for k in range(len(STAGES))]
        return (P - tgt).pow(2).mean().item(), (last - tgt).pow(2).mean().item(), corr


def to(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--source", default="emulator,pc")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exclude", default="", help="treningga qo'shilmaydigan taglar (tekshiruv uchun), vergul bilan")
    ap.add_argument("--pc-frac", type=float, default=0.0, help="batch'dagi PC epizod ulushi")
    a = ap.parse_args()
    gpuguard.preflight()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpuguard.cap_vram()
    torch.set_num_threads(3)
    D = Data(a.T, a.source.split(","), tuple(t for t in a.exclude.split(",") if t), a.pc_frac)
    wm = WorldModel().to(dev)
    print(f"parametrlar {sum(p.numel() for p in wm.parameters()) / 1e6:.1f}M, epizodlar o'rgatish {len(D.tr)} tekshiruv {len(D.va)}", flush=True)
    opt = torch.optim.Adam(wm.parameters(), a.lr, eps=1e-8)
    step = 0
    if a.resume or a.eval:
        ck = torch.load(CKPT, map_location="cpu")            # GPU ga to'g'ridan yuklash VRAM cho'qqisini beradi
        miss = wm.load_state_dict(ck["wm"], strict=False).missing_keys      # yangi boshlar (stuckh) noldan
        if not miss:
            opt.load_state_dict(ck["opt"])
        step = ck["step"]
    if a.eval:
        wm.eval()
        r = [wm.open_loop(to(D.batch(8, val=True), dev)) for _ in range(10)]
        print(f"open-loop 48 qadam (12 s): kadr MSE model {np.mean([x[0] for x in r]):.4f}  nusxa-baza {np.mean([x[1] for x in r]):.4f}; "
              f"reward korr {dict(zip(STAGES, np.round(np.nanmean([x[2] for x in r], 0), 2)))}", flush=True)
        return
    th = gpuguard.Throttle()
    t0 = time.time()
    while step < a.steps:
        wm.train()
        tot, info = wm.loss(to(D.batch(a.bs), dev))
        opt.zero_grad(); tot.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1000.0)
        opt.step(); th.step(); step += 1
        if step % 100 == 0:
            print(f"qadam {step} " + " ".join(f"{k} {v:.3f}" for k, v in info.items()) + f"  {time.time() - t0:.0f} s", flush=True)
        if step % 500 == 0 or step == a.steps:                         # tez-tez saqlash: himoya o'ldirsa ko'p yo'qolmasin
            os.makedirs(os.path.dirname(CKPT), exist_ok=True)
            torch.save({"wm": wm.state_dict(), "opt": opt.state_dict(), "step": step}, CKPT)
            wm.eval()
            r = [wm.open_loop(to(D.batch(8, val=True), dev)) for _ in range(4)]
            print(f"  open-loop: model {np.mean([x[0] for x in r]):.4f} nusxa-baza {np.mean([x[1] for x in r]):.4f} reward korr "
                  f"{np.round(np.nanmean([x[2] for x in r], 0), 2)}", flush=True)


if __name__ == "__main__":
    main()
