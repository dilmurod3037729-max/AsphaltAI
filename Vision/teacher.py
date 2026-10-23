"""O'qituvchi: V-JEPA 2 (ViT-L, Meta) — har lahza t uchun klip [t-2 .. t+5] (8 kadr, ~2 s, KELAJAKNI HAM ko'radi) embeddingi.
O'quvchi (bizning kichik kodlovchi) faqat [t-2, t-1, t] ni ko'radi va shu embeddingni bashorat qiladi -> oldindan ko'rish.
Manbalar: bizning poygalar (rec_*.bin, 4 Hz), YouTube (data/yt/*.rgb, 4 kadr/s), GT4 (1 kadr/s, alohida).
    py tools/safe_run.py 14400 -- py Vision/teacher.py --src ours|yt [--stride 2]     (emulator YOPIQ)
Natija: data/teacher/<src>.npz : idx (lahza indekslari), emb (float16, 1024)"""
import argparse
import glob
import os
import time

import numpy as np
import torch

import data as dt
import gpuguard

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = 8          # kadr: t-2 .. t+5
SIDE = 256
MASK = (slice(15, 31), slice(78, 96))   # YouTube qo'l kamerasi oynasi (96x48 da)


def load_teacher(dev):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(os.path.join(HERE, "..", "data", "hf", "vjepa2"), torch_dtype=torch.float16)
    return m.to(dev).eval()


def clips(frames, idx):
    """frames [N,H,W,3] uint8 (96x48 yoki boshqa) -> [B, T, 3, 256, 256] float16, ImageNet normallashtirish."""
    x = torch.from_numpy(frames[np.stack([idx + k for k in range(-2, CLIP - 2)], 1)])   # B,T,H,W,3
    x = x.permute(0, 1, 4, 2, 3).float().div(255)
    b, t = x.shape[:2]
    x = torch.nn.functional.interpolate(x.reshape(b * t, 3, *x.shape[-2:]), size=(SIDE, SIDE), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).reshape(b, t, 3, SIDE, SIDE).half()


def run(model, frames, valid_idx, dev, th, bs=4):
    out = np.zeros((len(valid_idx), 1024), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(valid_idx), bs):
            idx = valid_idx[i:i + bs]
            h = model(pixel_values_videos=clips(frames, idx).to(dev)).last_hidden_state
            out[i:i + len(idx)] = h.float().mean(1).cpu().numpy().astype(np.float16)
            th.step()
            if i % (bs * 500) == 0:
                print(f"  {i}/{len(valid_idx)}  {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--src", default="ours", choices=["ours", "yt"])
    ap.add_argument("--stride", type=int, default=2, help="har nechta lahzadan bittasi")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    gpuguard.preflight()
    dev = "cuda"
    gpuguard.cap_vram()
    th = gpuguard.Throttle()
    model = load_teacher(dev)
    os.makedirs(os.path.join(a.data, "teacher"), exist_ok=True)
    if a.src == "ours":
        D = dt.load_all(a.data, None)
        ep, n = D["ep"], len(D["frame"])
        ok = np.zeros(n, bool)
        ok[2:n - 5] = (ep[:n - 7] == ep[7:])            # butun klip bitta epizod ichida
        idx = np.where(ok)[0][::a.stride]
        if a.limit: idx = idx[:a.limit]
        emb = run(model, D["frame"], idx, dev, th)
        np.savez(os.path.join(a.data, "teacher", "ours.npz"), idx=idx, emb=emb)
    else:
        for f in sorted(glob.glob(os.path.join(a.data, "yt", "*.rgb"))):
            name = os.path.basename(f)[:-4]
            outp = os.path.join(a.data, "teacher", f"yt_{name}.npz")
            if os.path.exists(outp):
                continue
            fr = np.fromfile(f, np.uint8).reshape(-1, 48, 96, 3).copy()
            fr[:, MASK[0], MASK[1]] = 90
            idx = np.arange(2, len(fr) - 5)[::a.stride]
            if a.limit: idx = idx[:a.limit]
            np.savez(outp, idx=idx, emb=run(model, fr, idx, dev, th))
            print(name, len(idx), flush=True)
    print("tayyor")


if __name__ == "__main__":
    main()
