"""Depth Anything V2 Small bilan o'yin kadrlaridan chuqurlik: sifat va tezlik. py depth_probe.py <chiqish.png> <kadr1> <kadr2> ..."""
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

NAME = "depth-anything/Depth-Anything-V2-Small-hf"
dev = "cuda"
proc = AutoImageProcessor.from_pretrained(NAME)
model = AutoModelForDepthEstimation.from_pretrained(NAME, torch_dtype=torch.float16).to(dev).eval()


def depth(img: Image.Image, size=(252, 518)):
    x = proc(images=img, return_tensors="pt", size={"height": size[0], "width": size[1]}, do_resize=True,
             keep_aspect_ratio=False)["pixel_values"].to(dev).half()
    with torch.no_grad():
        d = model(pixel_values=x).predicted_depth[0].float().cpu().numpy()
    return d


if __name__ == "__main__":
    out, files = sys.argv[1], sys.argv[2:]
    rows = []
    depth(Image.open(files[0]).convert("RGB"))
    for f in files:
        im = Image.open(f).convert("RGB")
        torch.cuda.synchronize(); t = time.time()
        d = depth(im)
        torch.cuda.synchronize(); ms = (time.time() - t) * 1000
        n = (d - d.min()) / (d.max() - d.min() + 1e-9)
        col = (255 * np.stack([n, n, n], -1)).astype(np.uint8)   # yorqin = yaqin
        dm = Image.fromarray(col).resize((520, 240))
        rows.append(np.hstack([np.asarray(im.resize((520, 240))), np.asarray(dm)]))
        print(f, f"{ms:.0f} ms", d.shape)
    Image.fromarray(np.vstack(rows)).save(out)
