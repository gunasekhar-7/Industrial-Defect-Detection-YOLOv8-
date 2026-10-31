# Resume package — Industrial Surface Defect Detection (YOLOv8 → ONNX, FastAPI)

Every number below cites the file it comes from. Rounded values are what you
put on paper; the exact strings live in the cited artifact.

---

## STAR bullets

### 1. Computer vision — trained and honest evaluation

> **S:** A public 6-class steel-surface defect dataset (NEU-DET, 1800 images)
> needed a detector that could run on CPU-only infra. **T:** Deliver a
> fine-tuned YOLOv8n with evaluation anyone can re-check. **A:** Built the
> VOC→YOLO conversion with stratified 70/15/15 splitting and a dataset
> validator, trained 50 epochs at 320×320, and made the evaluator record the
> executed config (weights, split, image count, hardware, timestamp) next to
> the metrics. **R:** **mAP@50 0.7209**, mAP@50-95 0.4191 on the 270-image
> validation split, with per-class results published honestly (worst class
> crazing 0.1546) — all verbatim in
> [`reports/eval_report.json`](../reports/eval_report.json).

### 2. Performance — CPU inference budget

> **S:** The serving path had to meet an inspection budget on CPU with no
> GPU available, and an early `8.2 ms median` claim did not survive a longer
> run. **A:** Exported to ONNX (opset 17, graph-checked), letterbox-
> preprocessed in OpenCV, per-class NMS, then built a benchmark harness that
> records hardware, provider, run counts and session options with the
> numbers — which exposed ONNX Runtime's busy-wait thread pool as the cause:
> on a hybrid P-core/E-core CPU the same 200-run protocol measured 12 ms at
> the start and 245 ms sustained. Disabling `session.intra_op.allow_spinning`
> made latency stable, then a concurrent HTTP load harness measured the
> deployed service. **R:** **p50 36.08 ms / p95 45.80 ms** inference,
> **40.97 ms** end-to-end with NMS, **24.4 FPS** sustained; the service serves
> **~27 req/s at 4+ concurrent clients** with every request at HTTP 200 —
> [`reports/benchmark.json`](../reports/benchmark.json),
> [`reports/benchmark_spinning.json`](../reports/benchmark_spinning.json),
> [`reports/loadtest.json`](../reports/loadtest.json).

### 3. API security — hardened input path with adversarial tests

> **S:** An image-upload inference API is a DoS and spoofing surface.
> **A:** Implemented magic-byte content sniffing (415), decompression-bomb
> dimension checks read from the header *before* allocation (400), a 15 MB
> payload cap checked from `Content-Length` before the form is parsed and
> enforced **while streaming** (413), per-IP rate limiting at 60 req/min
> (429), explicit CORS origins (never `*`), and thread-safe one-shot model
> loading. **R:** Each guard has an adversarial test — the suite is **122
> tests**, including a spy that asserts bytes-read stay bounded, a check that
> an oversize body never reaches the app, and a check that handlers never
> block the event loop — [`tests/test_api.py`](../tests/test_api.py),
> [`src/api/security.py`](../src/api/security.py).

### 4. MLOps — reproducible pipeline from clone to container

> **S:** The original repo did not work from a fresh clone and its README
> quoted metrics no artifact supported. **A:** Pinned every dependency
> exactly (`requirements*.txt`), added ruff + mypy + pytest gates, a
> three-job GitHub Actions pipeline (lint / test with coverage artifact /
> docker build), and a true multi-stage Docker image that runs as a non-root
> user with no `curl`/`libgl` — its `HEALTHCHECK` uses stdlib `urllib`
> against `/health`. **R:** A fresh clone goes green with one command per
> gate, and `docker compose up --build` serves `/health` 200 and `/docs` —
> [`.github/workflows/ci.yml`](../.github/workflows/ci.yml),
> [`Dockerfile`](../Dockerfile).

### 5. Data engineering — resilient dataset acquisition

> **S:** The dataset lives behind a university Google Drive link that can
> rot at any time. **A:** Built an acquisition module that downloads and
> converts the official archive, **also accepts a locally downloaded archive
> or extracted directory**, and prints manual-download instructions on
> failure — plus a deterministic synthetic generator so CI never touches the
> network. **R:** Both acquisition paths and the failure message are covered
> by tests (`tests/test_neu_det.py`), and CI stays green with the synthetic
> dataset — [`src/data/neu_det.py`](../src/data/neu_det.py).

### 6. Code quality — one drawing implementation, typed core

> **S:** Box-annotation logic was duplicated between the API and the
> dashboard and could drift. **A:** Extracted `draw_detections()` into
> `src/vision/annotate.py` with the class palette moved to config as the
> single source of truth, added a mypy gate over the whole `src` package,
> and fixed every finding without suppressions. **R:**
> `grep cv2.rectangle src/` shows one *detection-annotation* site (the other
> hit is the synthetic-data generator stamping defect patches), the sidebar
> legend is
> derived from the same palette, and mypy reports zero issues —
> [`src/vision/annotate.py`](../src/vision/annotate.py),
> [`src/api/config.py`](../src/api/config.py).

---

## 30-second verbal pitch

"I built an end-to-end defect-detection service for steel surfaces: YOLOv8n
fine-tuned on the public NEU-DET dataset — mAP@50 of 0.72 on the held-out
split, and I publish the weak classes rather than hiding them. It serves
through ONNX Runtime at 36 ms median / 46 ms p95 per image on CPU, about 27
requests per second from a single worker under concurrent HTTP load, behind a
FastAPI layer that treats every upload as hostile: magic-byte checks,
decompression-bomb limits, a streaming payload cap and rate limiting, each
with an adversarial test — 122 tests at 100% statement coverage, gated by
ruff, mypy and a three-job CI pipeline, packaged in a non-root multi-stage
Docker image. When my first latency number didn't hold up under a longer run
I chased it to ORT's busy-wait thread pool, fixed it and published both
measurements — every number traces to a committed JSON artifact, because in
ML work the provenance *is* the credibility."

---

## Likely interview questions (repo-grounded answers)

**Why is mAP only 0.72?**
50 epochs on CPU over an 1800-image dataset — a compute budget, not a
ceiling. The per-class breakdown shows exactly where it's lost (crazing
0.1546, rolled-in scale 0.2570 vs patches 0.6481); those two classes overlap
visually with the rest. Runs are tracked in MLflow
(`src/models/train.py`, experiment `industrial-defect`) so longer or GPU
runs can be compared properly. Evidence: `reports/eval_report.json`.

**Why train at 320×320 instead of the usual 640?**
NEU-DET sources are 200×200, so 640 adds no information while quadrupling
pixels (640²/320² = 4) that CPU training pays for every epoch. The config is
recorded in the eval report under `"training"`.

**How do you know the published numbers aren't fabricated?**
They're generated, not typed: `python -m src.models.evaluate --out
reports/eval_report.json` and `python -m src.models.benchmark --json
reports/benchmark.json`. The README quotes rounded values and lists the
exact unrounded strings under "Metric provenance", including the hardware
string and run counts.

**What attacks does the upload path handle, and how do you know?**
Spoofed content types → magic-byte sniffing (415); decompression bombs →
dimension read from the header before allocating (400); memory exhaustion →
15 MB cap enforced while streaming, aborting early (413, with a spy test
asserting bounded reads); brute force → 60 req/min per IP (429); drive-by
origins → explicit CORS list, never `*`. Each has a test in
`tests/test_api.py`.

**Why are the endpoints `def` instead of `async def`?**
Inference is blocking CPU work; sync handlers run in the threadpool so they
can't stall the event loop — `test_detect_handlers_do_not_block_event_loop`
fails if anyone switches them back.

**Why does the dashboard not call the API?**
Deliberate trade-off: the dashboard is a local inspection tool with its own
in-process ONNX session, so a single container demos end-to-end with no
cross-service coupling. It's stated in the README architecture section as a
known trade-off (it can drift if only one side's model is swapped).

**What does Docker health checking look like when the model is missing?**
`/health` returns 503 when the model isn't loaded, so the image's
Python-`urllib` HEALTHCHECK fails and orchestration sees the container as
unhealthy; the shipped image includes the model so it starts healthy. Tested
by `test_health_503_when_model_missing`.

**Why class-aware NMS?**
Defect classes overlap heavily on steel surfaces; class-agnostic NMS would
drop a second defect type sitting on top of another — a false pass in
inspection. `test_nms_is_per_class` pins the behavior.

**CI takes the dataset from where?**
Nowhere — CI generates the synthetic dataset (`python -m src.data.generator`),
which is why the generator still exists. NEU-DET download is manual/on-demand
only.

**You published two very different latency numbers. Which is real?**
Both, and the difference is the story: ORT's busy-wait thread pool made the
same 200-run protocol report 12 ms early and 245 ms once sustained (the
numbers swing because spin-waiting threads compete with the graph on a hybrid
CPU). Disabling spinning gives a stable 36 ms median / 46 ms p95, which is
what the service actually holds and what the load test confirms (14 req/s at
one client, p50 69 ms including HTTP). The 8 ms figure is gone from the
README rather than quietly kept.

**Why not ship INT8?**
Measured, not assumed: dynamic INT8 is 3.8× smaller but 15× slower and changes
predicted classes on 12 of 24 frames here, because ORT dequantizes Conv
weights back to FP32 before each convolution. The script and the JSON are
committed so the decision can be re-checked; static QDQ with calibration is
the next attempt, not a claim.

**What would you do next?**
Longer/GPU training with MLflow run comparison (crazing is the bottleneck),
a 3-seed mean ± std on the validation split, static QDQ INT8, routing the
dashboard through the API with auth, and a `/metrics` endpoint so the
load-test percentiles can be watched in production.

---

## Reproduce every number

```bash
python -m src.models.evaluate \
  --weights runs/detect/industrial_defect_yolov8n/weights/best.pt \
  --data configs/dataset.yaml --split val --imgsz 320 --out reports/eval_report.json

python -m src.models.benchmark --onnx models/industrial_yolov8n.onnx \
  --warmup 50 --runs 200 --json reports/benchmark.json
python -m src.models.benchmark --onnx models/industrial_yolov8n.onnx \
  --warmup 50 --runs 200 --allow-spinning --json reports/benchmark_spinning.json
python -m src.models.quantize --onnx models/industrial_yolov8n.onnx \
  --compare data/processed/images/val   # -> reports/quantization_tradeoff.json
python scripts/make_gallery.py          # -> docs/failure_gallery.png

# service-level throughput (needs the API running with the rate limit raised)
python scripts/loadtest.py --image data/processed/images/val/crazing_109.jpg \
  --concurrency 8 --requests 240 --json reports/loadtest.json

ruff check src tests && mypy src && pytest tests/ -q
```
