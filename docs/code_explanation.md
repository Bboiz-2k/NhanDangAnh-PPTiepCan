# Bảng giải thích hàm và tham số (B10)

Điền theo code THẬT sau khi chạy. Mỗi dòng: hàm làm gì → vào/ra → vì sao dùng → tham số đã chọn → đổi thì sao → cách kiểm tra.
Chỉ ghi "đã thử" khi bạn thật sự chạy và có số trong `outputs/results/`.

| File::hàm | Nhiệm vụ | Input → Output | Vì sao dùng | Tham số đã chọn | Đổi tham số thì sao | Cách kiểm tra |
|---|---|---|---|---|---|---|
| prepare_data.py::dhash | Tạo mã băm 64 bit để gom ảnh gần giống | PIL Image → chuỗi hex | Ảnh gần giống nằm khác split gây rò rỉ | n=8 | n lớn → ít ảnh bị gom hơn | Đếm số nhóm trong manifest |
| prepare_data.py::main | Loại ảnh hỏng/trùng, chia 70/15/15 theo lớp | thư mục ảnh → manifest.csv | Hai hệ thống dùng chung split | seed=42, tỉ lệ ở config | Đổi seed → ảnh mỗi tập đổi | Đếm ảnh/lớp/split, so với data_report.json |
| hog_svm.py::hog_features | Xám → resize → HOG | PIL Image → vector 8100 | Baseline truyền thống | size=128, 9 hướng, cell 8×8, block 2×2, L2-Hys | Đổi cell/orientations → đổi chiều vector, phải train lại | In `len(vector)` |
| hog_svm.py::train | Chọn C bằng Validation, fit scaler+LinearSVC | manifest → hog_svm.joblib | Scaler chỉ học Train để tránh rò rỉ | C ∈ {0,1; 1; 10} | C lớn → phạt vi phạm mạnh hơn, chưa chắc Test tốt hơn | Xem `hog_svm_train.json` |
| resnet18_tl.py::build_model | Nạp ResNet18, đóng băng, thay fc 5 đầu ra | n_classes → model | Transfer learning | weights IMAGENET1K_V1 | Đầu ra phải khớp số lớp | In số tham số requires_grad |
| resnet18_tl.py::set_train_mode | fc train, backbone eval | model → model | Tránh BatchNorm cập nhật thống kê | — | Bỏ → thống kê BN bị đổi dù đã đóng băng | In `training` từng module |
| resnet18_tl.py::run_epoch | Một lượt qua dữ liệu | loader → loss trung bình | Dùng chung train/val | CrossEntropyLoss trên logits | Thêm softmax trước loss là sai | So loss với tay cho 1 batch nhỏ |
| resnet18_tl.py::train | Vòng học + early stopping | manifest → checkpoint tốt nhất | Chọn theo Val loss | Adam, lr=0,001, batch 32, ≤15 epoch, patience 3 | lr lớn → loss dao động | Đọc `resnet18_train.json` |
| resnet18_tl.py::ResNet18.predict | Suy luận 1 ảnh | PIL → (id, softmax) | Dùng chung evaluate/demo | model.eval + inference_mode | Quên eval → kết quả lệch evaluate | Cùng 1 ảnh: evaluate và app phải ra cùng nhãn |
| evaluate.py::main | Đánh giá Test một lần | models + manifest → *_test.json, PNG | Số liệu cho demo | warmup 10, đo 100 ảnh, batch 1 | Đổi phạm vi đo → tốc độ không so được | Mở JSON, đối chiếu confusion matrix |
| app.py | Demo upload + kết quả Test | ảnh → nhãn, thời gian | Không train lại lúc demo | — | Đổi checkpoint mà giữ cache → sai phiên bản | Khởi động lại app sau khi đổi model |

## Các thay đổi nhỏ đã thử trên Validation (điền số thật)
1. ...
2. ...
