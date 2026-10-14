"""B1 + B2: kiểm tra dữ liệu, loại ảnh hỏng/trùng, chia Train/Val/Test theo lớp và theo nhóm.
Chạy: python src/prepare_data.py
"""
import csv
import hashlib
import json
import random
from collections import defaultdict

import numpy as np
from PIL import Image

from common import CFG, CLASS_FILE, MANIFEST, RAW, RESULTS


def dhash(img, n=8):
    """Băm 64 bit theo gradient; ảnh giống hệt/gần giống thường cho cùng mã."""
    g = np.asarray(img.convert("L").resize((n + 1, n))).flatten().tolist()
    bits = ""
    for r in range(n):
        for c in range(n):
            bits += "1" if g[r * (n + 1) + c] > g[r * (n + 1) + c + 1] else "0"
    return format(int(bits, 2), "016x")


def main():
    classes = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    assert len(classes) == 5, f"Cần 5 thư mục lớp trong {RAW}, đang thấy: {classes}"
    CLASS_FILE.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {"classes": classes, "removed_unreadable": [], "removed_exact_dup": []}
    seen_md5, per_class = {}, defaultdict(list)
    for cid, cname in enumerate(classes):
        for p in sorted((RAW / cname).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                with Image.open(p) as im:
                    im.load()
                    g = dhash(im)
            except Exception:  # ảnh hỏng
                report["removed_unreadable"].append(str(p.relative_to(RAW)))
                continue
            md5 = hashlib.md5(p.read_bytes()).hexdigest()
            if md5 in seen_md5:  # trùng hệt
                report["removed_exact_dup"].append([str(p.relative_to(RAW)), seen_md5[md5]])
                continue
            seen_md5[md5] = str(p.relative_to(RAW))
            per_class[cid].append((str(p.relative_to(RAW)).replace("\\", "/"), g))

    # chia theo từng lớp; ảnh cùng mã dhash (gần giống) nằm cùng một split
    rng = random.Random(CFG["seed"])
    ratio = CFG["split"]
    rows = []
    for cid, items in per_class.items():
        groups = defaultdict(list)
        for rel, g in items:
            groups[g].append(rel)
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(items)
        quota_train, quota_val = ratio["train"] * n, (ratio["train"] + ratio["val"]) * n
        count = 0
        for k in keys:
            split = "train" if count < quota_train else ("val" if count < quota_val else "test")
            for rel in groups[k]:
                rows.append([rel, classes[cid], cid, split, k])
            count += len(groups[k])

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "label_id", "split", "group"])
        w.writerows(rows)

    stat = defaultdict(lambda: defaultdict(int))
    for _, label, _, split, _ in rows:
        stat[label][split] += 1
    report["counts"] = {k: dict(v) for k, v in stat.items()}
    report["total"] = len(rows)
    (RESULTS / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Tổng ảnh giữ lại:", len(rows))
    print("Loại ảnh hỏng:", len(report["removed_unreadable"]), "| loại ảnh trùng hệt:", len(report["removed_exact_dup"]))
    for label, d in report["counts"].items():
        print(f"  {label:<11} train={d.get('train',0):<4} val={d.get('val',0):<4} test={d.get('test',0)}")


if __name__ == "__main__":
    main()
