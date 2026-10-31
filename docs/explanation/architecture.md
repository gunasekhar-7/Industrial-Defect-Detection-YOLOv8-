# Architecture

How the pieces fit, why the boundaries are where they are, and which trade-offs
were chosen deliberately.

---

## The short version

```
 image ──► FastAPI guard ──► letterbox ──► ONNX Runtime ──► per-class NMS ──► JSON / JPEG
              (src/api)        └────────── src/models/benchmark.py ─────────┘
```

Two surfaces share one inference core and one renderer:

* **FastAPI service** (`src/api/`) — the integrable interface, with the upload
  guards and rate limiting.
* **Streamlit dashboard** (`src/ui/`) — a local inspection tool with its *own*
  in-process ONNX session.

Everything numeric — training, evaluation, export, benchmarking, quantization —
lives under `src/models/`, and dataset acquisition/validation under
`src/data/`. Nothing in `src/` imports the dashboard, so the service can be
deployed without Streamlit.

---

## Layer by layer

| Layer | Module | Responsibility | Depends on |
|---|---|---|---|
| HTTP edge | `src/api/app.py` | Routing, model lifecycle, response shaping | security, schemas, config, models |
| Input guards | `src/api/security.py` | Magic bytes, streaming cap, dimension checks, whole-body limit | config |
| Contracts | `src/api/schemas.py` | Response models (Pydantic) | — |
| Settings | `src/api/config.py` | The only place defaults live | — |
| Vision I/O | `src/vision/image_io.py` | Header-first upload decoding for the dashboard | config-free |
| Rendering | `src/vision/annotate.py` | The single box/label drawing implementation | — |
| Inference | `src/models/benchmark.py` | `ONNXDefectDetector`: preprocess → run → NMS | onnxruntime, cv2 |
| Lifecycle tools | `src/models/{train,evaluate,export,quantize}.py` | Producing and measuring models | ultralytics/onnx (function-local imports) |
| Dataset | `src/data/{neu_det,generator,validator}.py` | Acquire, synthesize, validate | Pillow, cv2 |

Two dependency rules are enforced by review rather than tooling, and are worth
preserving:

1. **Heavy ML imports are function-local.** `train.py`, `evaluate.py`,
   `export.py` import `ultralytics`/`torch` inside the function, so the API
   image can import `src.models` without the training stack and each CLI's
   `--help` works in a runtime-only environment.
2. **Rendering is not duplicated.** `grep cv2.rectangle src/` yields the
   annotator and the synthetic generator (which stamps defects). The API, the
   dashboard and the gallery all call `draw_detections()`, so palette and label
   style cannot drift.

---

## Request path for `POST /v1/detect`

1. **`BodySizeLimitMiddleware`** (outermost after CORS) rejects an oversized
   `Content-Length` with 413 before Starlette spools the multipart body to
   disk. Bodies without a length are counted while streaming and cut off at the
   same budget.
2. **CORS** runs outermost, so even a rejected upload carries CORS headers.
3. **slowapi** applies `RATE_LIMIT_DEFAULT` per client IP → 429.
4. **`sanitize_and_read_image()`** reads in 1 MiB chunks, aborts at the byte
   cap, verifies magic bytes as soon as 16 bytes are available, then reads
   dimensions from the Pillow header (no pixel allocation) and enforces
   32–4096 px. Only then does OpenCV decode to BGR.
5. **`run_detection()`** returns detections plus a latency breakdown from the
   loaded engine. If the model was missing at startup it retries the load once,
   then answers 503 rather than inventing results.
6. **Response shaping** maps raw dicts into `DefectItem`s, resolving class names
   and preserving `unknown_<id>` for out-of-range ids.

Both detection handlers are synchronous `def` functions, so Starlette runs them
in its threadpool. That is deliberate: inference is blocking CPU work, and an
`async def` handler would stall the event loop for every other request. A test
(`test_detect_handlers_do_not_block_event_loop`) fails if that changes.

---

## Why the dashboard does not call the API

The dashboard runs its own ONNX session instead of proxying HTTP. This is a
trade-off, not an oversight:

**Gained** — one container (or one `streamlit run`) demos the entire system
with no service discovery, no second hop, no CORS/rate-limit configuration, and
no failure mode where the UI is up but the API is down. Thresholds can be
changed interactively without the API's config.

**Given up** — the two surfaces can drift if only one side's model is swapped,
and the dashboard cannot be used to exercise the API's guards. The mitigations
are that both read the same `MODEL_PATH` default and share the same annotator
and upload limits, and that the API contract is documented and tested
independently ([api reference](../reference/api.md)).

If this were a multi-user product, the dashboard would call the API (with auth)
and this trade-off would flip. It is recorded here and in the README so a
reviewer does not have to guess.

---

## Model lifecycle

```
synthetic generator ──► CI dataset (never trained for reported metrics)
NEU-DET ──► converter ──► train ──► best.pt ──► evaluate ──► reports/eval_report.json
                                        └───► export ──► .onnx ──► benchmark/quantize ──► reports/
                                                                     └──► models/ ──► API & dashboard
```

The arrow that matters is *evaluate before export*: the `.pt` is what the
metrics describe, and the `.onnx` is a graph-faithful copy of it (checked with
`onnx.checker`). Benchmarking the ONNX graph and quoting `.pt` accuracy is only
honest because the export is verified and the preprocessing is identical.

---

## Configuration surface

`src/api/config.py` is the only settings module. Classes, palette, upload
limits, rate limit, CORS origins and model paths all live there, so the API,
dashboard and scripts cannot disagree about them. Values that would weaken a
guard if made configurable (magic-byte table, multipart overhead, letterbox
padding) are constants in code instead — see
[Configuration](../reference/configuration.md#settings-that-are-deliberately-hard-coded).

---

## What is intentionally absent

| Not present | Why |
|---|---|
| Database / queue | The service is stateless; nothing to persist |
| Auth | Deployed behind a gateway by design; documented as a limitation |
| `async` inference | Blocking CPU work belongs in the threadpool |
| Microservices split | Two surfaces and one model do not justify the operational cost |
| A model registry | MLflow tracking exists locally; a registry is in [future scope](../future-scope.md) |
