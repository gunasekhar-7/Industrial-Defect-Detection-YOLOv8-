"""Dashboard behaviour: every inspection tab, driven through a scripted
Streamlit surface.

Streamlit's own AppTest harness cannot fill a ``file_uploader`` or pick a
conveyor frame, so these tests execute the real dashboard script with a
record-and-return stand-in for ``streamlit``. The production code under
test is untouched: only the UI primitives are faked, and each run records
what the page would have rendered.
"""

import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.models import benchmark

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "ui" / "dashboard.py"

LATENCIES = {
    "preprocess_ms": 1.5,
    "inference_ms": 12.0,
    "postprocess_ms": 0.5,
    "total_ms": 14.0,
}


class _Upload:
    """Minimal stand-in for ``streamlit.UploadedFile``."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self.size = len(data)
        self._data = data

    def read(self) -> bytes:
        return self._data


def _jpeg_bytes(size: int = 96) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def _boxes(count: int) -> list[dict]:
    return [
        {
            "class_id": i % 6,
            "confidence": 0.9 - i * 0.1,
            "box": {"x_min": 8, "y_min": 8, "x_max": 60, "y_max": 60},
        }
        for i in range(count)
    ]


def _new_state() -> dict:
    return {
        "page_config": [],
        "markdown": [],
        "sidebar_title": [],
        "sidebar_markdown": [],
        "sidebar_subheader": [],
        "subheaders": [],
        "images": [],
        "tables": [],
        "dataframes": [],
        "infos": [],
        "errors": [],
        "metrics": [],
        "progress": [],
        "downloads": [],
        "model_path": None,
        "upload": None,
        "batch": None,
    }


def _fake_streamlit(state: dict):
    """A ``streamlit`` module whose calls are recorded into ``state``."""

    def rec(name):
        def record(*args, **kwargs):
            state[name].append((args, kwargs))

        return record

    class _Container:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def metric(self, label, value):
            state["metrics"].append((label, value))

        def subheader(self, text):
            state["subheaders"].append(text)

    class _Columns(_Container):
        def __init__(self, count):
            self.columns = [_Container() for _ in range(count)]

        def __iter__(self):
            return iter(self.columns)

    class _Progress:
        def progress(self, value):
            state["progress"].append(value)

    def columns(count, **_kwargs):
        return _Columns(count)

    def tabs(labels):
        return [_Container() for _ in labels]

    def file_uploader(_label, type=None, accept_multiple_files=False, max_upload_size=None):
        return state["batch"] if accept_multiple_files else state["upload"]

    def slider(label, min_value=None, max_value=None, value=None, step=None, **_kwargs):
        if "Confidence" in label:
            return 0.25
        if "IoU" in label:
            return 0.45
        return value

    sidebar = SimpleNamespace(
        title=rec("sidebar_title"),
        text_input=lambda _label, value=None, **_kwargs: state["model_path"] or value,
        slider=slider,
        markdown=rec("sidebar_markdown"),
        subheader=rec("sidebar_subheader"),
    )

    return SimpleNamespace(
        set_page_config=rec("page_config"),
        markdown=rec("markdown"),
        sidebar=sidebar,
        tabs=tabs,
        file_uploader=file_uploader,
        columns=columns,
        subheader=rec("subheaders"),
        image=rec("images"),
        table=rec("tables"),
        dataframe=rec("dataframes"),
        info=rec("infos"),
        error=rec("errors"),
        slider=slider,
        progress=lambda _start: _Progress(),
        download_button=rec("downloads"),
        cache_resource=lambda fn=None, **_kwargs: fn if fn is not None else (lambda f: f),
    )


def _run_dashboard(tmp_path, monkeypatch, *, model_exists, detections, test_frames=(), **widgets):
    """Execute the dashboard script once and return what it rendered."""
    state = _new_state()
    state.update(widgets)

    model_file = tmp_path / "model.onnx"
    if model_exists:
        model_file.touch()
        state["model_path"] = str(model_file)

    frame_dir = tmp_path / "data" / "processed" / "images" / "test"
    if test_frames is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in test_frames:
            (frame_dir / name).write_bytes(payload)

    class FakeDetector:
        def __init__(self, path, confidence_threshold=0.25, iou_threshold=0.45):
            self.path = path
            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold

        def predict(self, img_bgr):
            return [dict(d, box=dict(d["box"])) for d in detections], dict(LATENCIES)

    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit(state))
    monkeypatch.setattr(benchmark, "ONNXDefectDetector", FakeDetector)
    monkeypatch.chdir(tmp_path)

    spec = importlib.util.spec_from_file_location("dashboard_under_test", DASHBOARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return state


def _texts(entries) -> str:
    return "\n".join(str(args[0]) for args, _kwargs in entries)


def test_dashboard_without_artifacts_explains_what_is_missing(tmp_path, monkeypatch):
    state = _run_dashboard(
        tmp_path, monkeypatch, model_exists=False, detections=[], test_frames=None
    )

    assert "Dataset directory 'data/processed' not generated yet." in _texts(state["infos"])
    assert state["images"] == []
    assert state["tables"] == []


def test_dashboard_reports_missing_model_and_empty_conveyor(tmp_path, monkeypatch):
    state = _run_dashboard(
        tmp_path,
        monkeypatch,
        model_exists=False,
        detections=[],
        upload=_Upload("part.jpg", _jpeg_bytes()),
        test_frames=[],
    )

    infos = _texts(state["infos"])
    assert "not found. Please train/export the ONNX model first." in infos
    assert "No test images found in data/processed/images/test." in infos


def test_dashboard_flags_defects_and_exports_batch_report(tmp_path, monkeypatch):
    frames = [("frame_0.jpg", _jpeg_bytes()), ("frame_1.jpg", _jpeg_bytes())]
    state = _run_dashboard(
        tmp_path,
        monkeypatch,
        model_exists=True,
        detections=_boxes(2),
        test_frames=frames,
        upload=_Upload("part.jpg", _jpeg_bytes()),
        batch=[_Upload("bad.jpg", b"not an image"), _Upload("good.jpg", _jpeg_bytes())],
    )

    # Single-part tab: reject verdict, latency cards, per-defect table.
    assert "REJECT — 2 Defect(s) Identified" in _texts(state["markdown"])
    assert [label for label, _ in state["metrics"]] == [
        "Preprocess",
        "Inference",
        "NMS Postprocess",
        "Total Latency",
    ]
    assert state["metrics"][3][1] == "14.0 ms"
    table = state["tables"][0][0][0]
    assert list(table["Defect Class"]) == ["crazing", "inclusion"]
    assert table["Confidence"].tolist() == ["90.0%", "80.0%"]

    # Conveyor tab: defective frame.
    assert "STATUS: DEFECTIVE (2 defects)" in _texts(state["markdown"])

    # Batch tab: the bad upload is reported and does not abort the audit.
    assert "bad.jpg: unreadable image" in _texts(state["errors"])
    report = state["dataframes"][0][0][0]
    assert report["Filename"].tolist() == ["good.jpg"]
    assert report["Status"].tolist() == ["REJECT"]
    assert state["progress"] == [0.5, 1.0]
    download = state["downloads"][0][1]
    assert download["file_name"] == "inspection_audit_report.csv"
    assert "good.jpg,REJECT" in download["data"].decode("utf-8")


def test_dashboard_reports_clean_parts_as_pass(tmp_path, monkeypatch):
    state = _run_dashboard(
        tmp_path,
        monkeypatch,
        model_exists=True,
        detections=[],
        test_frames=[("frame_0.jpg", _jpeg_bytes())],
        upload=_Upload("part.jpg", _jpeg_bytes()),
        batch=[_Upload("good.jpg", _jpeg_bytes())],
    )

    markdown = _texts(state["markdown"])
    assert "PASS — Zero Surface Defects Detected" in markdown
    assert "STATUS: PASS" in markdown
    assert state["tables"] == []
    report = state["dataframes"][0][0][0]
    assert report["Status"].tolist() == ["PASS"]
    assert report["Defects Detected"].tolist() == ["None"]


def test_dashboard_surfaces_unreadable_conveyor_frame(tmp_path, monkeypatch):
    state = _run_dashboard(
        tmp_path,
        monkeypatch,
        model_exists=True,
        detections=[],
        test_frames=[("broken.jpg", b"definitely not a jpeg")],
    )

    assert "broken.jpg" in _texts(state["errors"])
    assert "may be corrupt" in _texts(state["errors"])


def test_dashboard_rejects_undecodable_upload(tmp_path, monkeypatch):
    state = _run_dashboard(
        tmp_path,
        monkeypatch,
        model_exists=True,
        detections=[],
        upload=_Upload("garbage.jpg", b"definitely not a jpeg"),
        test_frames=None,
    )

    assert "unreadable image" in _texts(state["errors"])
    assert state["images"] == []
