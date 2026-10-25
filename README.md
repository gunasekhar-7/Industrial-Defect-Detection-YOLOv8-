# Industrial Surface Defect Detection (YOLOv8 → ONNX, FastAPI)

[![CI Pipeline](https://github.com/gunasekhar-7/Industrial-Defect-Detection-YOLOv8-/actions/workflows/ci.yml/badge.svg)](https://github.com/gunasekhar-7/Industrial-Defect-Detection-YOLOv8-/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00599C.svg)](https://ultralytics.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A 6-class surface-defect detector for hot-rolled steel, trained on the public
**NEU-DET** dataset and served as a hardened **FastAPI** service with an
interactive **Streamlit** inspection dashboard.

Every number on this page comes from a committed artifact —
[`reports/eval_report.json`](reports/eval_report.json) for model quality and
[`reports/benchmark.json`](reports/benchmark.json) for latency. Exact,
unrounded values are listed under [Metric provenance](#metric-provenance).

| Input | Output |
|---|---|
| ![](docs/images/example-original.jpg) | ![](docs/images/example-annotated.jpg) |

---

## Honest results

### Dataset and training

* **Dataset:** [NEU-DET](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/) —
  1800 images of six hot-rolled steel surface defect classes (300 per class).
* **Split:** stratified 70/15/15 into train/val/test
  ([`src/data/neu_det.py`](src/data/neu_det.py)); the evaluation below runs on
  the **270-image validation split** (`"split": "val"`, `"images": 270` in
  [`reports/eval_report.json`](reports/eval_report.json)).
* **Training:** `yolov8n` fine-tuned for **50 epochs** at **imgsz 320**,
  batch 16, on **CPU only** — the executed configuration is recorded in
  `reports/eval_report.json` under `"training"` and was tracked with MLflow
  (local tracking DB, experiment `industrial-defect`).

The model was trained at 320×320 because NEU-DET source images are 200×200;
upscaling to 640 adds no information while quadrupling the pixel count
(640²/320² = 4×) — on CPU-only training that cost is paid every epoch.

### Model quality (validation split, 270 images)

| Metric | Value |
|---|---|
| Precision | 0.6578 |
| Recall | 0.6982 |
| **mAP@50** | **0.7209** |
| mAP@50-95 | 0.4191 |

### Per-class mAP@50-95

| Class | mAP@50-95 |
|---|---|
| crazing | 0.1546 |
| inclusion | 0.3949 |
| patches | 0.6481 |
| pitted_surface | 0.5267 |
| rolled_in_scale | 0.2570 |
| scratches | 0.5334 |

Crazing and rolled-in scale are the hardest classes: their texture is subtle
and they overlap heavily with the other defect types. See
[Limitations](#limitations).

### Latency and throughput (in-process ONNX Runtime micro-benchmark)

Measured on `models/industrial_yolov8n.onnx`, input `1×3×320×320`,
`CPUExecutionProvider`, 200 runs after 50 warmups:

| Metric | Value |
|---|---|
| Median inference | 8.21 ms |
| Mean inference | 8.44 ms |
| p95 inference | 10.64 ms |
| p99 inference | 13.86 ms |
| Mean end-to-end (pre/infer/post + NMS) | 14.05 ms |
| Throughput | 71.2 FPS |

**Hardware:** `Windows-10-10.0.26300-SP0 | Intel64 Family 6 Model 186
Stepping 2, GenuineIntel | python 3.11.7` — an Intel x86-64 CPU on Windows,
no GPU. These are single-image, in-process numbers: they do not include
HTTP overhead.

### Metric provenance (exact, unrounded values)

Verbatim from [`reports/eval_report.json`](reports/eval_report.json):

```json
"mAP50": 0.7208593384148835,
"mAP50_95": 0.4191165798397718,
"precision": 0.6577969755601648,
"recall": 0.6982250905072714,
"per_class": {
  "crazing":         { "mAP50_95": 0.15460208198193207 },
  "inclusion":       { "mAP50_95": 0.39490997930439536 },
  "patches":         { "mAP50_95": 0.6480634660172571 },
  "pitted_surface":  { "mAP50_95": 0.5266763903671192 },
  "rolled_in_scale": { "mAP50_95": 0.2570232558010965 },
  "scratches":       { "mAP50_95": 0.533424305566831 }
}
```

Verbatim from [`reports/benchmark.json`](reports/benchmark.json):

```json
"mean_inference_ms": 8.438234499990358,
"median_inference_ms": 8.205399997677887,
"p95_inference_ms": 10.63821999450737,
"p99_inference_ms": 13.860115002316885,
"mean_e2e_ms": 14.051090499888232,
"throughput_fps": 71.16885340735328
```

---

## Architecture

```
   [Client / test harness]
            │  multipart image
            ▼
 ┌───────────────────────────────┐
 │  FastAPI service (port 8000)  │
 │  • magic-byte content sniffing│──► 415/400/413 raised before any
 │  • decompression-bomb guards  │    full buffer is allocated
 │  • 15 MB streaming cap        │
 │  • rate limit 60 req/min/IP   │
 │  • explicit CORS origins      │
 ├───────────────────────────────┤
 │  ONNX Runtime session         │──► models/industrial_yolov8n.onnx
 │  letterbox → infer → per-class│    (loaded once, thread-safe)
 │  NMS (IoU 0.45)               │
 └──────────────┬────────────────┘
                │  JSON detections  /  annotated JPEG
                ▼
      GET /health · POST /v1/detect · POST /v1/detect/annotated
      Interactive docs at /docs

 Streamlit dashboard (port 8501) runs its OWN in-process ONNX session
 for local inspection — it does not call the API. That is a deliberate
 trade-off (single-container demos, no cross-service coupling), not a
 shared inference path.
```

---

## Quickstart (tutorial)

Requires Python 3.11.

```bash
git clone https://github.com/gunasekhar-7/Industrial-Defect-Detection-YOLOv8-.git
cd Industrial-Defect-Detection-YOLOv8-

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**1. Generate the CI dataset and run the tests**

```bash
python -m src.data.download        # synthetic dataset, no download needed
pytest tests/ -q                   # 38 tests
```

The synthetic generator is what CI uses; it exercises the full
data → label → validation pipeline without network access.

**2. Run the API**

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

* Health: <http://localhost:8000/health>
* Interactive API docs: <http://localhost:8000/docs>

**3. Run the dashboard**

```bash
streamlit run src/ui/dashboard.py  # http://localhost:8501
```

**4. Or run both with Docker**

```bash
docker compose up --build
```

The API container is multi-stage, runs as a non-root user and ships without
`curl`/`libgl` — its `HEALTHCHECK` uses Python's `urllib` against `/health`
(see [`Dockerfile`](Dockerfile)).

---

## How-to: train and evaluate on real NEU-DET data

### Get the dataset (manual-download fallback built in)

The converter fetches the official archive from
<https://faculty.neu.edu.cn/songkc/en/zdylm/263265/> (a Google Drive link)
and caches it:

```bash
python -m src.data.neu_det
```

If that link ever rots — it is a university page — download `NEU-DET.zip`
yourself and point the converter at the local archive or an already-extracted
directory; no network access is needed from that point on:

```bash
python -m src.data.neu_det --archive path/to/NEU-DET.zip
python -m src.data.neu_det --archive path/to/extracted/NEU-DET
```

Both paths are covered by tests
([`tests/test_neu_det.py`](tests/test_neu_det.py)), including the failure
message that tells you where to download the dataset manually.

### Train, evaluate, export, benchmark

```bash
# Train (params/metrics land in the local MLflow tracking DB)
python -m src.models.train --data configs/dataset.yaml --epochs 50 --imgsz 320 --batch 16

# Evaluate -> writes executed config + metrics as JSON
python -m src.models.evaluate \
  --weights runs/detect/industrial_defect_yolov8n/weights/best.pt \
  --data configs/dataset.yaml --split val --imgsz 320 \
  --out reports/eval_report.json

# Export (opset 17, graph-checked) and install as the served model
python -m src.models.export \
  --weights runs/detect/industrial_defect_yolov8n/weights/best.pt --imgsz 320

# Reproduce the latency table
python -m src.models.benchmark --onnx models/industrial_yolov8n.onnx \
  --warmup 50 --runs 200 --json reports/benchmark.json
```

CI never downloads NEU-DET; it always runs against the synthetic generator.

---

## API reference

Interactive OpenAPI docs at **`/docs`** (Swagger UI). Summary:

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness + model status; returns **503** when the model is not loaded |
| `POST /v1/detect` | JSON detections with a per-stage latency breakdown |
| `POST /v1/detect/annotated` | JPEG with drawn, labelled boxes |

Input guardrails (values from [`src/api/config.py`](src/api/config.py)):

| Guard | Value | Status |
|---|---|---|
| Content types | JPEG / PNG / BMP, verified by magic bytes | 415 |
| Payload cap | 15 MB, enforced while streaming (aborts early) | 413 |
| Image dimensions | 32 … 4096 px, read from the header before allocation | 400 |
| Rate limit | 60 requests/minute per client IP | 429 |
| CORS | explicit origin list, never `*` | — |

---

## Development

```bash
ruff check src tests                 # lint
mypy src/api src/models src/vision   # types
pytest tests/ -q                     # 38 tests, with coverage
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs three jobs on
every push and pull request: **lint** (ruff + mypy), **test** (pinned dev
requirements → synthetic dataset → pytest with a coverage artifact) and
**docker** (image build smoke gate).

Layout:

```
src/api/       FastAPI app, settings, security boundaries
src/models/    train / evaluate / export / benchmark (ONNX Runtime engine)
src/data/      synthetic generator, validator, NEU-DET acquisition + conversion
src/vision/    shared detection annotation (one drawing implementation)
src/ui/        Streamlit dashboard
configs/       dataset.yaml (NEU-DET), dataset_synthetic.yaml (CI)
reports/       committed eval + benchmark evidence
```

All dependency versions are pinned exactly in `requirements.txt`,
`requirements-train.txt` and `requirements-dev.txt`.

---

## Limitations

* **Numbers are honest, not impressive.** mAP@50 of 0.7209 comes from 50 CPU
  epochs on a 1800-image dataset; crazing (0.1546) and rolled-in scale
  (0.2570) are weak. Longer or GPU training should improve these — the MLflow
  tracking setup exists to compare such runs.
* **CPU-only measurements.** No GPU was available; latency figures are
  single-image, in-process, from one x86-64 CPU (the exact hardware string is
  in [`reports/benchmark.json`](reports/benchmark.json)).
* **No authentication or TLS.** The API is meant for a trusted internal
  network; put it behind a gateway before exposing it.
* **The dashboard runs its own model session** instead of calling the API, so
  it can drift from a swapped-in API model if only one side is updated.
* **NEU-DET is not shipped.** Download it (converter or manual) to reproduce
  training; tests and CI use the synthetic generator instead.

## License

[MIT](LICENSE) — see [`LICENSE`](LICENSE).
