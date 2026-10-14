# NhanDangAnh-PPTiepCan

Đồ án môn Xử lý ảnh số: tổng quan bài toán nhận dạng ảnh và các phương pháp tiếp cận.
Phần thực nghiệm: phân loại 5 loài hoa (flower_photos) bằng hai hệ thống
**HOG + LinearSVC** và **ResNet18 pretrained ImageNet (chỉ học đầu phân loại)**.

## Cấu trúc
```
configs/config.json     # mọi tham số (seed, tỉ lệ chia, HOG, C, lr...)
data/raw/               # ảnh gốc, KHÔNG đẩy lên git (tải theo hướng dẫn)
data/manifest.csv       # split dùng chung cho hai hệ thống
data/class_names.json   # mapping id <-> tên lớp
src/                    # prepare_data, hog_svm, resnet18_tl, evaluate, app
outputs/models/         # model đã học
outputs/results/        # số liệu JSON
outputs/figures/        # confusion matrix, đường loss
docs/code_explanation.md
```

## Cách chạy
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Tải https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
# giải nén sao cho có data/raw/flower_photos/daisy, dandelion, roses, sunflowers, tulips
cd src
python prepare_data.py     # kiểm tra + chia dữ liệu
python hog_svm.py          # huấn luyện HOG + SVM
python resnet18_tl.py      # huấn luyện ResNet18 (head)
python evaluate.py         # đánh giá Test, lưu kết quả
cd .. && streamlit run src/app.py
```
Ghi chú: số liệu trong `outputs/results/` là kết quả thật của lần chạy; không sửa tay.
