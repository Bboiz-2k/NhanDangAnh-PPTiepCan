"""B3: kiểm tra tiền xử lý bằng mắt và bằng số (ảnh Test ngẫu nhiên, seed cố định).
Chạy: python check_preprocess.py   ->   outputs/figures/check_preprocess.png
"""
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from common import CFG, FIGURES, class_names, read_manifest
from hog_svm import H, hog_features
from resnet18_tl import EVAL_TF, MEAN, STD

N = 6


def crop_box(w, h):
    """Vùng ảnh gốc còn lại sau Resize(256) + CenterCrop(224) (xấp xỉ, để nhìn)."""
    s = 256 / min(w, h)
    left, top = (round(w * s) - 224) / 2, (round(h * s) - 224) / 2
    return left / s, top / s, (left + 224) / s, (top + 224) / s


def main():
    classes = class_names()
    rows = random.Random(CFG["seed"]).sample(read_manifest("test"), N)
    fig, ax = plt.subplots(N, 3, figsize=(9, 3 * N))
    for i, r in enumerate(rows):
        with Image.open(r["path"]) as im:
            raw = im.copy()
        rgb = raw.convert("RGB")                      # ResNet: RGB
        gray = np.asarray(rgb.convert("L").resize((H["size"], H["size"])))  # HOG: xám 128x128
        x = EVAL_TF(rgb)                              # tensor 3x224x224 đã chuẩn hóa
        vec = hog_features(raw)
        print(f"{r['path'].name:<26} gốc={raw.size} mode={raw.mode:<4} | HOG: {len(vec)} chiều "
              f"| ResNet: {tuple(x.shape)} min={float(x.min()):.2f} max={float(x.max()):.2f}")

        shown = rgb.copy()
        ImageDraw.Draw(shown).rectangle(crop_box(*rgb.size), outline="red", width=4)
        back = x.numpy() * np.array(STD)[:, None, None] + np.array(MEAN)[:, None, None]  # bỏ chuẩn hóa để xem
        ax[i, 0].imshow(shown); ax[i, 0].set_title(f"Gốc: {classes[r['label_id']]} (khung đỏ = phần ResNet giữ)", fontsize=8)
        ax[i, 1].imshow(gray, cmap="gray"); ax[i, 1].set_title("HOG nhận: xám 128x128", fontsize=8)
        ax[i, 2].imshow(back.transpose(1, 2, 0).clip(0, 1)); ax[i, 2].set_title("ResNet nhận: crop 224 (màu RGB)", fontsize=8)
        for a in ax[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "check_preprocess.png", dpi=130)
    print("Đã lưu", FIGURES / "check_preprocess.png")


if __name__ == "__main__":
    main()
