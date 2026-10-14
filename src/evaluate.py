"""B7: đánh giá trên Test một lần, lưu số liệu + biểu đồ + thời gian.
Chạy: python src/evaluate.py   (sau khi đã train các hệ thống)
"""
import json
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import CFG, FIGURES, MODELS, RESULTS, class_names, read_manifest


def load_systems():
    systems = {}
    if (MODELS / "hog_svm.joblib").exists():
        from hog_svm import HogSvm
        systems["hog_svm"] = (HogSvm(), MODELS / "hog_svm.joblib")
    if (MODELS / "resnet18_head.pt").exists():
        from resnet18_tl import ResNet18
        systems["resnet18"] = (ResNet18(), MODELS / "resnet18_head.pt")
    return systems


def plot_cm(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Dự đoán (cột)")
    ax.set_ylabel("Nhãn thật (hàng)")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def time_ms(system, images):
    t = CFG["timing"]
    sample = (images * (t["warmup"] + t["n_images"]))[: t["warmup"] + t["n_images"]]
    for img in sample[: t["warmup"]]:
        system.predict(img)
    ms = []
    for img in sample[t["warmup"]:]:
        t0 = time.perf_counter()
        system.predict(img)  # gồm tiền xử lý + (HOG) + suy luận, ảnh đã giải mã sẵn
        ms.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": round(statistics.mean(ms), 2), "median_ms": round(statistics.median(ms), 2),
            "n": len(ms), "warmup": t["warmup"], "batch": 1, "device": "CPU"}


def main():
    classes = class_names()
    rows = read_manifest("test")
    imgs = []
    for r in rows:
        with Image.open(r["path"]) as im:
            imgs.append(im.convert("RGB").copy())
    y_true = [r["label_id"] for r in rows]

    for key, (system, mfile) in load_systems().items():
        y_pred = [system.predict(im)[0] for im in imgs]
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
        rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
        wrong = [{"path": str(r["path"].relative_to(r["path"].parents[1])), "true": classes[t], "pred": classes[p]}
                 for r, t, p in zip(rows, y_true, y_pred) if t != p][:12]
        out = {"system": system.name, "split": "test", "n_images": len(rows),
               "accuracy": round(accuracy_score(y_true, y_pred), 4),
               "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
               "per_class": {c: {k: round(rep[c][k], 4) if k != "support" else int(rep[c][k])
                                 for k in ("precision", "recall", "f1-score", "support")} for c in classes},
               "confusion_matrix": cm.tolist(), "timing": time_ms(system, imgs[:CFG["timing"]["n_images"]]),
               "model_size_mb": round(mfile.stat().st_size / 1e6, 2), "wrong_examples": wrong}
        (RESULTS / f"{key}_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_cm(cm, classes, f"{system.name} - Test", FIGURES / f"cm_{key}.png")
        print(f"{system.name}: acc={out['accuracy']}  macro-F1={out['macro_f1']}  "
              f"{out['timing']['median_ms']} ms/ảnh  {out['model_size_mb']} MB")

    h = RESULTS / "resnet18_train.json"
    if h.exists():
        hist = json.loads(h.read_text())["history"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot([e["epoch"] for e in hist], [e["train_loss"] for e in hist], label="Train loss")
        ax.plot([e["epoch"] for e in hist], [e["val_loss"] for e in hist], label="Val loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); fig.tight_layout()
        fig.savefig(FIGURES / "resnet18_history.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
