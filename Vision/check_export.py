"""Eksport tekshiruvi: ONNX (APK ishlatadigan) va PyTorch modeli bir xil chiqish beradimi? Embedding o'lik emasmi?
    py check_export.py [--tags L1]        chiqish kodi 0 = OK, 1 = BUZILGAN
Tekshiruvlar: (1) ONNX vs PyTorch (emb, aux) farq < 1e-3; (2) embedding dispersiyasi > 0 (hamma kadrga bir xil emas);
(3) aux ichida NaN yo'q va ehtimollar [0,1] da; (4) kirish tartibi APK bilan bir xil (3 kadr x RGB, [0,1])."""
import argparse
import os
import sys

import numpy as np
import torch

import data as dt
from model import Export, Net

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    import onnxruntime as ort
    D = dt.load_all(a.data, [t for t in a.tags.split(",") if t] or None)
    n = len(D["frame"])
    rs = np.random.RandomState(0)
    idx = rs.choice(np.arange(2, n), 64, replace=False)
    fr = D["frame"][np.stack([idx - 2, idx - 1, idx], 1)]                       # [B,3,48,96,3]
    x = torch.from_numpy(fr).float().div(255).permute(0, 1, 4, 2, 3).reshape(64, 9, 48, 96)   # APK bilan bir xil tartib
    net = Net()
    net.load_state_dict(torch.load(os.path.join(HERE, "out", "vision.pt"), map_location="cpu"))
    net.eval()
    with torch.no_grad():
        e_t, a_t = Export(net)(x)
    sess = ort.InferenceSession(os.path.join(HERE, "..", "App", "Assets", "vision.onnx"), providers=["CPUExecutionProvider"])
    outs = [sess.run(None, {"x": x[i:i + 1].numpy()}) for i in range(len(x))]      # APK batch=1 bilan ishlaydi
    e_o = np.concatenate([o[0] for o in outs]); a_o = np.concatenate([o[1] for o in outs])
    de, da = float(np.abs(e_o - e_t.numpy()).max()), float(np.abs(a_o - a_t.numpy()).max())
    ok = True
    print(f"1. ONNX vs PyTorch: emb farqi {de:.2e}, aux farqi {da:.2e}   {'OK' if max(de, da) < 1e-3 else 'BUZILGAN'}")
    ok &= max(de, da) < 1e-3
    sd = float(e_o.std(0).mean())
    print(f"2. embedding dispersiyasi (kadrlar bo'yicha) {sd:.3f}   {'OK' if sd > 1e-3 else 'O`LIK (hamma kadrga bir xil)'}")
    ok &= sd > 1e-3
    probs = a_o[:, [0] + list(range(12, 18))]
    fin = bool(np.isfinite(a_o).all())
    rng = bool((probs >= 0).all() and (probs <= 1).all())
    print(f"3. aux: NaN yo'q {fin}, ehtimollar [0,1] da {rng}   {'OK' if fin and rng else 'BUZILGAN'}   aux o'lchami {a_o.shape[1]} (APK kutadi: 18)")
    ok &= fin and rng and a_o.shape[1] == 18 and e_o.shape[1] == 128
    print(f"   neytral to'qnashuv ehtimoli: o'rtacha {a_o[:, 0].mean():.3f} (min {a_o[:, 0].min():.3f}, max {a_o[:, 0].max():.3f}); WRONG WAY: o'rtacha {a_o[:, 17].mean():.3f}")
    print("NATIJA:", "OK" if ok else "BUZILGAN")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
