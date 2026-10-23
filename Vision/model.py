"""Ko'rish kodlovchisi: 3 ketma-ket kadr (9 kanal, 48x96) -> embedding(128) + HARAKATGA SHARTLANGAN bosh'lar.
Bosh'lar embedding + shu qadamda tanlangan harakat (rul 5, drift 3, nitro 3 = 11 bir-issiq) dan "keyingi 1 s da nima bo'ladi"ni bashorat qiladi:
to'qnashuv/tezlik yo'qotishi, tezlik o'zgarishi, nitro o'sishi, WRONG WAY. Hodisa sinfi va kelajak reward harakatsiz (kadr holatidan)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB = 128
K_EVT = 8
AUX = 18          # APK: Vision.Aux bilan bir xil bo'lishi shart
N_ACT = 11


def onehot(steer, drift, nitro):
    return torch.cat([F.one_hot(steer, 5), F.one_hot(drift, 3), F.one_hot(nitro, 3)], 1).float()


class Net(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(9, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(), nn.Conv2d(96, 128, 3, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 3 * 6, EMB), nn.LayerNorm(EMB), nn.ReLU(), nn.Dropout(dropout))
        self.ctx = nn.Sequential(nn.Linear(EMB + N_ACT, 64), nn.ReLU())
        self.coll = nn.Linear(64, 1)
        self.reg = nn.Linear(64, 3)          # tezlik yo'qotishi, tezlik o'zgarishi, nitro o'sishi
        self.ww = nn.Linear(64, 1)           # WRONG WAY
        self.evt = nn.Linear(EMB, K_EVT)
        self.val = nn.Linear(EMB, 1)
        self.register_buffer("neutral", onehot(torch.tensor([2]), torch.tensor([0]), torch.tensor([0])))

    def forward(self, x, act=None):
        e = self.enc(x)
        if act is None:
            act = self.neutral.expand(len(e), -1)
        h = self.ctx(torch.cat([e, act], 1))
        return e, self.coll(h)[:, 0], self.reg(h), self.evt(e), self.val(e)[:, 0], self.ww(h)[:, 0]


class Export(nn.Module):
    """ONNX: emb + aux(18) = [to'qnashuv(neytral), yo'qotish, o'zgarish, nitro, hodisa softmax(8), to'qnashuv har rul uchun(5), WRONG WAY]."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        z = torch.zeros(5, dtype=torch.long)
        self.register_buffer("steer_acts", onehot(torch.arange(5), z, z))

    def forward(self, x):
        n = self.net
        e = n.enc(x)
        h = n.ctx(torch.cat([e, n.neutral.expand(len(e), -1)], 1))
        coll = torch.sigmoid(n.coll(h))
        reg = n.reg(h)
        ev = torch.softmax(n.evt(e), 1)
        ww = torch.sigmoid(n.ww(h))
        by_steer = []
        for s in range(5):
            hs = n.ctx(torch.cat([e, self.steer_acts[s:s + 1].expand(len(e), -1)], 1))
            by_steer.append(torch.sigmoid(n.coll(hs)))
        return e, torch.cat([coll, reg, ev] + by_steer + [ww], 1)
