"""Đường dẫn, cấu hình, đọc manifest - dùng chung cho mọi script."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8"))

RAW = ROOT / CFG["raw_dir"]
MANIFEST = ROOT / CFG["manifest"]
CLASS_FILE = ROOT / CFG["class_names"]
MODELS = ROOT / CFG["out_dir"] / "models"
RESULTS = ROOT / CFG["out_dir"] / "results"
FIGURES = ROOT / CFG["out_dir"] / "figures"
for _d in (MODELS, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


def class_names():
    """Danh sách tên lớp theo thứ tự id (0..4)."""
    return json.loads(CLASS_FILE.read_text(encoding="utf-8"))


def read_manifest(split=None):
    """Trả về list dict: path (tuyệt đối), label_id, split, group."""
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if split is None or r["split"] == split:
                rows.append({"path": RAW / r["path"], "label_id": int(r["label_id"]),
                             "split": r["split"], "group": r["group"]})
    return rows
