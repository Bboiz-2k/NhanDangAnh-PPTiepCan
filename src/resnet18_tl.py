"""Hệ thống 2: ResNet18 pretrained ImageNet, đóng băng backbone, chỉ học đầu phân loại (fc) 5 lớp.
Chạy huấn luyện: python src/resnet18_tl.py
"""
import json
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import CFG, MODELS, RESULTS, class_names, read_manifest

R = CFG["resnet"]
MODEL_FILE = MODELS / "resnet18_head.pt"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Validation/Test/demo: cố định, không ngẫu nhiên (giống transforms của weights IMAGENET1K_V1)
EVAL_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
# Train: thêm lật ngang + crop ngẫu nhiên
TRAIN_TF = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Flowers(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            x = self.tf(im.convert("RGB"))  # ép RGB
        return x, r["label_id"]


def build_model(n_classes, pretrained):
    w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=w)
    for p in m.parameters():
        p.requires_grad = False  # đóng băng backbone
    m.fc = nn.Linear(m.fc.in_features, n_classes)  # đầu mới: requires_grad=True mặc định
    return m


def set_train_mode(m):
    """fc ở train, phần còn lại eval để BatchNorm không cập nhật thống kê."""
    m.train()
    for name, child in m.named_children():
        if name != "fc":
            child.eval()


def run_epoch(m, loader, loss_fn, device, opt=None):
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(opt is not None):
            loss = loss_fn(m(x), y)  # logits đưa thẳng vào CrossEntropyLoss
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def train():
    torch.manual_seed(CFG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = class_names()
    m = build_model(len(classes), pretrained=True).to(device)
    tr = DataLoader(Flowers(read_manifest("train"), TRAIN_TF), batch_size=R["batch_size"], shuffle=True)
    va = DataLoader(Flowers(read_manifest("val"), EVAL_TF), batch_size=R["batch_size"])
    opt = torch.optim.Adam(m.fc.parameters(), lr=R["lr"])
    loss_fn = nn.CrossEntropyLoss()

    hist, best, bad, t0 = [], float("inf"), 0, time.perf_counter()
    for ep in range(1, R["epochs"] + 1):
        set_train_mode(m)
        tl = run_epoch(m, tr, loss_fn, device, opt)
        m.eval()
        vl = run_epoch(m, va, loss_fn, device)
        hist.append({"epoch": ep, "train_loss": round(tl, 4), "val_loss": round(vl, 4)})
        print(f"epoch {ep:>2}  train_loss={tl:.4f}  val_loss={vl:.4f}")
        if vl < best:
            best, bad = vl, 0
            torch.save({"state_dict": m.state_dict(), "classes": classes}, MODEL_FILE)
        else:
            bad += 1
            if bad >= R["patience"]:
                print("Early stopping")
                break
    (RESULTS / "resnet18_train.json").write_text(json.dumps(
        {"config": R, "device": device, "best_val_loss": round(best, 4), "history": hist,
         "train_seconds": round(time.perf_counter() - t0, 1),
         "note": "không gồm giai đoạn pretrained ImageNet; backbone đóng băng"}, indent=2), encoding="utf-8")
    print("Đã lưu checkpoint tốt nhất:", MODEL_FILE)


class ResNet18:
    name = "ResNet18 (transfer learning)"

    def __init__(self):
        ck = torch.load(MODEL_FILE, map_location="cpu")
        self.classes = ck["classes"]
        self.model = build_model(len(self.classes), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()  # bắt buộc khi suy luận

    def predict(self, img):
        """PIL Image -> (id lớp, softmax từng lớp)."""
        x = EVAL_TF(img.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            p = torch.softmax(self.model(x), dim=1)[0]
        return int(p.argmax()), p.tolist()


if __name__ == "__main__":
    train()
