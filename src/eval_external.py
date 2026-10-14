"""B9: kiểm tra thêm bằng ảnh ngoài nguồn (tự chụp), TÁCH RIÊNG khỏi Test.
Đặt ảnh vào data/external/<tên lớp>/*.jpg  (tên thư mục trùng class_names.json)
Chạy: python eval_external.py   ->   outputs/results/external_<hệ thống>.json
"""
import json

from PIL import Image, ImageOps

from common import ROOT, RESULTS, class_names
from evaluate import load_systems

EXT = ROOT / "data" / "external"


def main():
    classes = class_names()
    items = [(p, cid) for cid, c in enumerate(classes) if (EXT / c).is_dir()
             for p in sorted((EXT / c).glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not items:
        print(f"Chưa có ảnh trong {EXT}. Tạo thư mục con theo tên lớp: {classes}")
        return
    imgs = []
    for p, _ in items:
        with Image.open(p) as im:
            imgs.append(ImageOps.exif_transpose(im).convert("RGB"))  # sửa xoay ảnh điện thoại
    y_true = [t for _, t in items]

    for key, (system, _) in load_systems().items():
        y_pred = [system.predict(im)[0] for im in imgs]
        per_class = {c: [sum(1 for t, q in zip(y_true, y_pred) if t == k and q == k), y_true.count(k)]
                     for k, c in enumerate(classes)}
        correct = sum(v[0] for v in per_class.values())
        out = {"system": system.name, "n_images": len(items), "correct": correct,
               "accuracy": round(correct / len(items), 4),
               "per_class": {c: f"{a}/{n}" for c, (a, n) in per_class.items()},
               "wrong": [{"file": p.name, "true": classes[t], "pred": classes[q]}
                         for (p, t), q in zip(items, y_pred) if t != q],
               "note": "Ảnh tự chụp, số lượng nhỏ nên chỉ mang tính tham khảo; không gộp vào Test."}
        (RESULTS / f"external_{key}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{system.name}: đúng {correct}/{len(items)} (accuracy {out['accuracy']})  {out['per_class']}")


if __name__ == "__main__":
    main()
