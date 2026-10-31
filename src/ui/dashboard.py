"""Streamlit real-time industrial defect inspection dashboard."""

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.api.config import settings
from src.models.benchmark import ONNXDefectDetector
from src.vision.annotate import draw_detections
from src.vision.image_io import check_upload_size, decode_image

# Same payload budget the API enforces, so both surfaces reject alike.
MAX_UPLOAD_BYTES = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024

# Page layout and configuration
st.set_page_config(
    page_title="Industrial Defect Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom industrial CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-pass {
        color: #16A34A;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .status-fail {
        color: #DC2626;
        font-weight: 700;
        font-size: 1.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_detector(model_path: str, conf: float, iou: float) -> ONNXDefectDetector | None:
    p = Path(model_path)
    if not p.exists():
        return None
    return ONNXDefectDetector(p, confidence_threshold=conf, iou_threshold=iou)


# Sidebar controls
st.sidebar.title("⚙️ Inspection Controls")

model_file = st.sidebar.text_input("ONNX Model Path", value=settings.MODEL_PATH)
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
iou_thresh = st.sidebar.slider("IoU (NMS) Threshold", 0.1, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Defect Classes")
# Sidebar swatches derive from the same config palette the renderer uses
# (BGR -> hex), so legend and boxes can never disagree.
def _bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


palette = {
    name: _bgr_to_hex(settings.CLASS_COLORS[i])
    for i, name in enumerate(settings.CLASS_NAMES)
}
for name, color in palette.items():
    st.sidebar.markdown(f"<span style='color:{color}; font-weight:600;'>■</span> {name}", unsafe_allow_html=True)


# Main header
st.markdown('<div class="main-header">🏭 Industrial Defect Detection Portal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Edge-optimized Automated Optical Inspection (AOI) powered by YOLOv8 & ONNX Runtime</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔬 Single Part Inspection", "🔄 Conveyor Simulation", "📊 Batch Quality Audit"])

detector = get_detector(model_file, conf_thresh, iou_thresh)


def annotate_image(img_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Render detections via the shared annotator (one drawing implementation)."""
    return draw_detections(img_bgr, detections, settings.CLASS_NAMES, settings.CLASS_COLORS)


# Tab 1: Single Part Inspection
with tab1:
    uploaded_file = st.file_uploader(
        "Upload product surface image",
        type=["jpg", "jpeg", "png", "bmp"],
        max_upload_size=settings.MAX_IMAGE_SIZE_MB,
    )
    if uploaded_file is not None:
        # Byte cap first, then the header-first dimension guard: a rejected
        # upload never becomes a file buffer or a pixel buffer in the worker.
        img_bgr: np.ndarray | None = None
        try:
            check_upload_size(uploaded_file.size, MAX_UPLOAD_BYTES)
            img_bgr = decode_image(uploaded_file.read(), settings.MAX_IMAGE_DIMENSION)
        except ValueError as exc:
            st.error(str(exc))

        if img_bgr is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Raw Surface Image")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width="stretch")

            with col2:
                st.subheader("Inspection Diagnosis")
                if detector is not None:
                    detections, latencies = detector.predict(img_bgr)
                    annotated_bgr = annotate_image(img_bgr, detections)
                    st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), width="stretch")

                    # Quality decision
                    if len(detections) == 0:
                        st.markdown('<p class="status-pass">✅ PASS — Zero Surface Defects Detected</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="status-fail">❌ REJECT — {len(detections)} Defect(s) Identified</p>', unsafe_allow_html=True)

                    # Latency metrics cards
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Preprocess", f"{latencies['preprocess_ms']:.1f} ms")
                    m2.metric("Inference", f"{latencies['inference_ms']:.1f} ms")
                    m3.metric("NMS Postprocess", f"{latencies['postprocess_ms']:.1f} ms")
                    m4.metric("Total Latency", f"{latencies['total_ms']:.1f} ms")

                    # Detections table
                    if detections:
                        records = []
                        for idx, d in enumerate(detections, 1):
                            cls_id = d["class_id"]
                            cname = settings.CLASS_NAMES[cls_id] if 0 <= cls_id < len(settings.CLASS_NAMES) else str(cls_id)
                            box = d["box"]
                            records.append({
                                "#": idx,
                                "Defect Class": cname,
                                "Confidence": f"{d['confidence'] * 100:.1f}%",
                                "Coordinates": f"({box['x_min']}, {box['y_min']}) to ({box['x_max']}, {box['y_max']})",
                            })
                        st.table(pd.DataFrame(records))
                else:
                    st.info(f"Model file '{model_file}' not found. Please train/export the ONNX model first.")


# Tab 2: Conveyor Simulation
with tab2:
    st.markdown("Simulate camera inspection on an active conveyor belt using test dataset frames.")
    test_dir = Path("data/processed/images/test")
    if test_dir.exists():
        test_images = sorted(list(test_dir.glob("*.jpg")))
        if test_images:
            idx = st.slider("Conveyor Step (Image Index)", 0, len(test_images) - 1, 0)
            target_img_path = test_images[idx]
            sim_img = cv2.imread(str(target_img_path))
            if sim_img is None:
                st.error(f"Could not read {target_img_path.name} — the file may be corrupt.")
            else:
                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown(f"**Camera Frame:** `{target_img_path.name}`")
                    st.image(cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB), width="stretch")

                with c_right:
                    if detector is not None:
                        dets, lats = detector.predict(sim_img)
                        ann = annotate_image(sim_img, dets)
                        st.markdown(f"**Inspection Result** ({lats['total_ms']:.1f} ms)")
                        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), width="stretch")
                        if not dets:
                            st.markdown('<p class="status-pass">STATUS: PASS</p>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<p class="status-fail">STATUS: DEFECTIVE ({len(dets)} defects)</p>', unsafe_allow_html=True)
        else:
            st.info("No test images found in data/processed/images/test.")
    else:
        st.info("Dataset directory 'data/processed' not generated yet.")


# Tab 3: Batch Quality Audit
with tab3:
    st.markdown("Upload multiple product images to generate a certified inspection report.")
    batch_files = st.file_uploader(
        "Upload batch images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        max_upload_size=settings.MAX_IMAGE_SIZE_MB,
    )
    if batch_files and detector is not None:
        batch_results = []
        progress_bar = st.progress(0)
        for i, bf in enumerate(batch_files):
            # Byte cap, then header-first dimension guard (same order as tab 1).
            img: np.ndarray | None = None
            try:
                check_upload_size(bf.size, MAX_UPLOAD_BYTES)
                img = decode_image(bf.read(), settings.MAX_IMAGE_DIMENSION)
            except ValueError as exc:
                st.error(f"{bf.name}: {exc}")
            if img is not None:
                dets, lats = detector.predict(img)
                defects_found = [
                    settings.CLASS_NAMES[d["class_id"]] if 0 <= d["class_id"] < len(settings.CLASS_NAMES) else str(d["class_id"])
                    for d in dets
                ]
                status = "PASS" if len(dets) == 0 else "REJECT"
                batch_results.append({
                    "Filename": bf.name,
                    "Status": status,
                    "Defect Count": len(dets),
                    "Defects Detected": ", ".join(defects_found) if defects_found else "None",
                    "Inference (ms)": round(lats["inference_ms"], 2),
                    "Total (ms)": round(lats["total_ms"], 2),
                })
            progress_bar.progress((i + 1) / len(batch_files))

        if batch_results:
            df = pd.DataFrame(batch_results)
            st.dataframe(df, width="stretch")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Audit Report (CSV)", data=csv, file_name="inspection_audit_report.csv", mime="text/csv")
