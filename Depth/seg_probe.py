"""SegFormer-B0 (Cityscapes) o'yin kadrlarida: yo'l / devor / mashina ajratish sifati. py seg_probe.py <chiqish.png> <kadr...>"""
import sys
import time

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

sys.path.insert(0, ".")
from perceive import clean  # noqa: E402

NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
proc = AutoImageProcessor.from_pretrained(NAME)
model = AutoModelForSemanticSegmentation.from_pretrained(NAME).to("cuda").half().eval()
PAL = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30],
                [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142],
                [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], np.uint8)
NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "tlight", "tsign", "vegetation", "terrain", "sky", "person",
         "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def seg(bgr, size=(288, 640)):
    rgb = cv2.cvtColor(clean(bgr), cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (size[1], size[0]))
    x = (x.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    t = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to("cuda").half()
    with torch.no_grad():
        lg = model(pixel_values=t).logits[0].float()
    lg = torch.nn.functional.interpolate(lg[None], size=size, mode="bilinear")[0]
    return lg.argmax(0).cpu().numpy()


if __name__ == "__main__":
    out, files = sys.argv[1], sys.argv[2:]
    rows = []
    seg(cv2.imread(files[0]))
    for f in files:
        bgr = cv2.imread(f)
        torch.cuda.synchronize(); t = time.time()
        m = seg(bgr)
        torch.cuda.synchronize(); ms = (time.time() - t) * 1000
        h, w = bgr.shape[:2]
        cnt = {NAMES[i]: round(float((m == i).mean()), 2) for i in range(19) if (m == i).mean() > .03}
        print(f, f"{ms:.0f} ms", cnt)
        col = cv2.resize(PAL[m][:, :, ::-1], (w, h), interpolation=cv2.INTER_NEAREST)
        ov = cv2.addWeighted(bgr, .45, col, .55, 0)
        rows.append(np.hstack([cv2.resize(bgr, (520, 240)), cv2.resize(ov, (520, 240))]))
    cv2.imwrite(out, np.vstack(rows))
