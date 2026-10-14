"""Hệ thống 1: ảnh -> xám -> resize -> HOG -> StandardScaler -> LinearSVC.
Chạy huấn luyện: python src/hog_svm.py
"""
import json
import time

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import CFG, MODELS, RESULTS, class_names, read_manifest

H = CFG["hog"]
MODEL_FILE = MODELS / "hog_svm.joblib"


def hog_features(img):
    """PIL Image (bất kỳ mode) -> vector HOG 1 chiều."""
    g = np.asarray(img.convert("L").resize((H["size"], H["size"])))
    return hog(g, orientations=H["orientations"], pixels_per_cell=tuple(H["pixels_per_cell"]),
               cells_per_block=tuple(H["cells_per_block"]), block_norm=H["block_norm"])


def extract(rows):
    X, y = [], []
    for r in rows:
        with Image.open(r["path"]) as im:
            X.append(hog_features(im))
        y.append(r["label_id"])
    return np.array(X), np.array(y)


def train():
    Xtr, ytr = extract(read_manifest("train"))
    Xva, yva = extract(read_manifest("val"))
    print("Chiều vector HOG:", Xtr.shape[1])  # kỳ vọng 8100 với cấu hình mặc định

    table, best = [], None
    for C in CFG["svm"]["C_grid"]:
        pipe = make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=CFG["svm"]["max_iter"],
                                                         random_state=CFG["seed"]))
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)  # scaler chỉ fit trên Train
        fit_s = time.perf_counter() - t0
        f1 = f1_score(yva, pipe.predict(Xva), average="macro")
        table.append({"C": C, "val_macro_f1": round(float(f1), 4), "fit_seconds": round(fit_s, 2)})
        print(f"C={C:<5} val macro-F1={f1:.4f}  fit={fit_s:.1f}s")
        if best is None or f1 > best[0]:
            best = (f1, C, pipe, fit_s)

    _, C, pipe, fit_s = best
    joblib.dump(pipe, MODEL_FILE)
    (RESULTS / "hog_svm_train.json").write_text(json.dumps(
        {"hog": H, "vector_dim": int(Xtr.shape[1]), "chosen_C": C, "fit_seconds_chosen": round(fit_s, 2),
         "grid": table, "note": "thời gian fit không gồm thời gian trích HOG"}, indent=2), encoding="utf-8")
    print("Đã lưu", MODEL_FILE, "| C chọn theo Validation =", C)


class HogSvm:
    name = "HOG + SVM"

    def __init__(self):
        self.pipe = joblib.load(MODEL_FILE)
        self.classes = class_names()

    def predict(self, img):
        """PIL Image -> (id lớp, decision score từng lớp). Không phải xác suất."""
        s = self.pipe.decision_function(hog_features(img).reshape(1, -1))[0]
        return int(np.argmax(s)), s.tolist()


if __name__ == "__main__":
    train()
