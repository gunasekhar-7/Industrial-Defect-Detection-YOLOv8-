# Future scope

What this project would do next, ordered by value per unit of effort, with the
current state stated honestly for each item. Nothing here is claimed as done.

The roadmap is organised by the question a reviewer actually asks: *what would
you fix first, and why?*

---

## Phase 1 — raise model quality (highest value)

The model is the weakest part of the project: mAP@50 0.7209, mAP@50-95 0.4191,
with `crazing` at 0.1546 and `rolled_in_scale` at 0.2570. Everything else is
already stronger than the model. Concrete steps, in order:

| # | Change | Why it helps | Cost |
|---|---|---|---|
| 1 | **Train at 640 with `yolov8s`** | Crazing is a fine crack network; 320 px destroys it. More capacity helps texture classes | GPU hours (or ~10× CPU time) |
| 2 | **Augment for the weak classes** — copy-paste of defect patches, mosaic, small-angle rotation, brightness/contrast | 300 images per class is the binding constraint, not architecture | Hours |
| 3 | **Three seeds, report mean ± std** | A single 270-image validation run cannot distinguish a real gain from noise; without error bars no comparison is trustworthy | 3× training |
| 4 | **Per-class confidence thresholds** | `crazing` at 0.25 produces one oversized box; a class-specific operating point beats a global one | Small code change |
| 5 | **Box-merging post-process** | Fixes rolled-in scale over-segmentation visible in the [failure gallery](failure_gallery.png) | Small |
| 6 | **Second opinion on the test split** | The 270-image test split is currently untouched; quoting both splits guards against val-overfitting | Minutes |
| 7 | **Evaluate on a second dataset** (GC10-DET, Severstal) | Proves the model is not NEU-DET-specific | Days |

Success criterion: mAP@50 ≥ 0.80 and mAP@50-95 ≥ 0.50 with a 3-seed interval,
published in `reports/eval_report.json` with the same provenance discipline.

---

## Phase 2 — ship it somewhere clickable

| Item | Why | Notes |
|---|---|---|
| **Hosted demo URL** | The strongest remaining gap in the project's credibility; a reviewer can click instead of reading | A small container host runs the existing image unchanged; the graded-recall caveat must be visible on the page |
| **`curl` example in the README pointing at the live instance** | Turns "it runs" into "it runs for you" | Two lines |
| **Public dashboard** | The Streamlit tabs are the product story | Same image, different command |
| **Cold-start budget** | A sleeping container pays model-load time on first request | Measure, then either keep-warm or accept and document |

Deployment prerequisites that must not be skipped: TLS termination, an auth
gate, and a request-size cap at the proxy so the service is not the only line
of defence.

---

## Phase 3 — observability and operations

| Item | Why | Shape |
|---|---|---|
| **`/metrics` endpoint** (Prometheus text format) | The load-test percentiles exist only in a JSON file; operators need them live | Request count, latency histogram, 4xx/5xx counters, model-load state |
| **Structured JSON logs with request ids** | Abuse and failures are currently only in uvicorn's default log | Middleware + `logging` formatter |
| **Drift check** | A model served for months sees different lighting/camera conditions | Nightly script comparing class distribution to the training set |
| **Model versioning in the response** | `/health` reports `model_type` but not *which* graph | Add a hash/version to `MODEL_PATH` handling and to `DetectionResponse` |
| **CI performance job** | A regression in lazy imports or letterboxing would pass today's CI | Run the benchmark with a generous threshold on a self-hosted runner |
| **Dependency hardening** | Exact pins exist; hashes and `pip-audit` do not | Hash-pinned requirements + audit step in `lint` |
| **Multi-arch image** | Apple Silicon and ARM edge boxes need `linux/arm64` | `docker buildx` in the `docker` job |

---

## Phase 4 — API and product surface

| Item | Why it is not there yet |
|---|---|
| **Batch endpoint** (`POST /v1/detect/batch`) | Throughput gains and cleaner client code, but it changes the memory bound the guards rely on — needs a per-request budget, not a per-file one |
| **Async job queue for large batches** | Same reason, plus the service is deliberately stateless |
| **Authentication + API keys** | Deployed behind a gateway by design; adding auth means quotas, rotation and a datastore |
| **Dashboard routed through the API** | Would remove the documented drift trade-off but adds coupling and failure modes to a local tool |
| **Video / conveyor stream input** | The natural AOI next step; needs frame sampling and per-frame tracking to avoid duplicate alarms |
| **Per-defect severity scoring** | Turning boxes into accept/reject grades is a separate model or a rules layer |

---

## Phase 5 — testing depth

| Item | Gap it closes |
|---|---|
| **Branch coverage gate** | 100% statements can still leave untaken `if` arms |
| **Mutation testing** (mutmut/cosmic-ray) | Proves the suite *detects* changes, not just executes lines |
| **End-to-end test against the built container** | CI builds the image but never runs it |
| **Load/soak test in CI** | Sustained-behaviour regressions (the spinning class of bug) are invisible to unit tests |
| **Property-based tests for upload guards** | `hypothesis` over the byte/dimension space would find boundary cases faster than hand-written examples |
| **Cross-platform CI matrix** | The latency pathology was Windows-only; behaviour differences deserve a matrix |

---

## Phase 6 — efficiency experiments worth measuring

These are hypotheses, not commitments. Each needs the same treatment the INT8
experiment got: measure, publish the artifact, keep the winner.

| Idea | Expected outcome | Risk |
|---|---|---|
| **Static QDQ INT8 with calibration** | Real int8 kernels; unlike dynamic quantization, could actually be faster | Calibration set needed; accuracy drop must be measured per class |
| **`intra_op_num_threads` tuning** | Fewer threads can beat the default when serving concurrent requests | Machine-specific; needs the load harness, not the micro-benchmark |
| **ONNX `graph_optimization_level` variations** | Graph fusion differences | Usually already at max |
| **NMS in ONNX vs OpenCV** | Removes a Python post-process step (~1–3 ms) | Export complexity; ultralytics graph surgery |
| **FP16 on GPU** | Large latency win where a GPU exists | No GPU was available; unmeasured and therefore unclaimed |
| **Fixed-size 200×200 input** | Skips upscaling for NEU-DET-shaped inputs | Breaks on other resolutions unless input is padded/cropped |

---

## Explicit non-goals

Stated so they are not mistaken for oversights:

* **Safety-critical certification.** This is an AOI demo; a false PASS on a
  production line needs process-level validation, not a portfolio project.
* **Serving many models from one process.** One model per container keeps the
  memory bound and the healthcheck honest.
* **A public multi-tenant service.** Auth, quotas, isolation and abuse handling
  are a product, not a feature.
* **Training on the synthetic dataset for reported metrics.** It is a CI
  fixture and stays that way.

---

## How to pick up any of these

1. Read [Architecture](explanation/architecture.md) for where the change belongs.
2. Follow the [how-to guides](README.md#how-to-guides--task-oriented) to
   reproduce the current number first — you cannot claim an improvement without a
   baseline.
3. Add the artifact under `reports/` and update the pages that quote it, keeping
   the provenance rule: **if it is not in a committed artifact, it does not go
   in the docs.**
