"""B8: demo Streamlit. Chạy: streamlit run src/app.py"""
import json
import time

import streamlit as st
from PIL import Image

from common import FIGURES, MODELS, RAW, RESULTS, class_names

st.set_page_config(page_title="Nhận dạng ảnh hoa", layout="wide")
st.title("Nhận dạng ảnh: HOG + SVM so với ResNet18")
classes = class_names()


@st.cache_resource
def get_systems():
    systems = {}
    if (MODELS / "hog_svm.joblib").exists():
        from hog_svm import HogSvm
        systems["hog_svm"] = HogSvm()
    if (MODELS / "resnet18_head.pt").exists():
        from resnet18_tl import ResNet18
        systems["resnet18"] = ResNet18()
    return systems


systems = get_systems()
tab1, tab2 = st.tabs(["Dự đoán một ảnh", "Đánh giá tập Test (kết quả đã lưu)"])

with tab1:
    st.caption("Chỉ dự đoán ảnh được tải lên. Demo chỉ biết 5 lớp: " + ", ".join(classes))
    up = st.file_uploader("Tải ảnh hoa", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, width=320)
        cols = st.columns(len(systems) or 1)
        for col, (key, s) in zip(cols, systems.items()):
            t0 = time.perf_counter()
            idx, scores = s.predict(img)
            ms = (time.perf_counter() - t0) * 1000
            kind = "decision score (không phải xác suất)" if key == "hog_svm" else "softmax"
            col.subheader(s.name)
            col.write(f"**Nhãn:** {classes[idx]}")
            col.write(f"Thời gian: {ms:.1f} ms (1 ảnh, gồm tiền xử lý)")
            col.caption(kind)
            col.bar_chart({c: v for c, v in zip(classes, scores)})

with tab2:
    for key in ("hog_svm", "resnet18"):
        f = RESULTS / f"{key}_test.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        st.subheader(r["system"])
        st.write(f"Tập {r['split']} ({r['n_images']} ảnh) | Accuracy **{r['accuracy']}** | "
                 f"macro-F1 **{r['macro_f1']}** | {r['timing']['median_ms']} ms/ảnh (trung vị, "
                 f"{r['timing']['device']}, batch {r['timing']['batch']}) | model {r['model_size_mb']} MB")
        st.dataframe(r["per_class"])
        st.image(str(FIGURES / f"cm_{key}.png"), caption="Hàng = nhãn thật, cột = dự đoán (số ảnh)", width=420)
        if r["wrong_examples"]:
            st.write("Một số ảnh dự đoán sai:")
            ex = st.columns(4)
            for i, w in enumerate(r["wrong_examples"][:4]):
                p = RAW / w["path"]
                if p.exists():
                    ex[i].image(str(p), caption=f"thật: {w['true']} | đoán: {w['pred']}")
    hist = FIGURES / "resnet18_history.png"
    if hist.exists():
        st.image(str(hist), caption="Lịch sử học ResNet18", width=420)
