# HTTP API reference

Base URL in local development: `http://127.0.0.1:8000`. Interactive OpenAPI
docs: `GET /docs`. Machine-readable schema: `GET /openapi.json`.

Three endpoints, all defined in [`src/api/app.py`](../../src/api/app.py).

---

## `GET /health`

Liveness and model status. Returns **200** when a model is loaded and **503**
when none is, so Docker `HEALTHCHECK` and orchestrators can act on it.

**200 response** (`HealthResponse`):

```json
{
  "status": "healthy",
  "app_version": "1.0.0",
  "model_loaded": true,
  "model_type": "onnx_runtime",
  "classes": ["crazing", "inclusion", "patches", "pitted_surface",
              "rolled_in_scale", "scratches"]
}
```

| Field | Type | Meaning |
|---|---|---|
| `status` | string | `healthy` or `unhealthy` |
| `app_version` | string | `settings.APP_VERSION` (default `1.0.0`) |
| `model_loaded` | bool | Whether an inference engine is available |
| `model_type` | string | `onnx_runtime`, `pytorch_yolo`, `not_initialized`, or `none` |
| `classes` | string[] | Class names in model order |

`model_type: pytorch_yolo` means the `.onnx` graph was missing or failed to
load and the `.pt` fallback was used. `not_initialized` means neither loaded;
the body still has a `200`-shaped payload, but the status code is `503`.

---

## `POST /v1/detect`

Detect defects in one uploaded image. Returns JSON detections plus a per-stage
latency breakdown.

**Request** — `multipart/form-data` with exactly one file part named `file`:

```bash
curl -X POST http://127.0.0.1:8000/v1/detect \
  -F "file=@docs/images/example-original.jpg"
```

| Constraint | Value |
|---|---|
| Accepted formats | JPEG, PNG, BMP — verified by magic bytes, not extension or `Content-Type` |
| Max bytes | 15 MB (`MAX_IMAGE_SIZE_MB`), enforced while streaming |
| Min/max dimensions | 32 … 4096 px (`MIN_IMAGE_DIMENSION` / `MAX_IMAGE_DIMENSION`) |
| Auth | none |
| Rate limit | 60 requests/minute per client IP (`RATE_LIMIT_DEFAULT`) |

**200 response** (`DetectionResponse`):

```json
{
  "filename": "example-original.jpg",
  "image_width": 200,
  "image_height": 200,
  "defect_count": 1,
  "defects": [
    {
      "class_id": 0,
      "class_name": "crazing",
      "confidence": 0.2809,
      "box": {"x_min": 1, "y_min": 0, "x_max": 199, "y_max": 120}
    }
  ],
  "latency": {
    "preprocess_ms": 15.8,
    "inference_ms": 191.4,
    "postprocess_ms": 28.0,
    "total_ms": 235.2
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `filename` | string | As uploaded; `unknown` if the client sent none |
| `image_width`, `image_height` | int | Pixels, as reported by Pillow from the header |
| `defect_count` | int | `len(defects)`; `0` is a legitimate "pass" |
| `defects[].class_id` | int | `0`–`5`, see [Configuration](configuration.md) |
| `defects[].class_name` | string | Resolved name, or `unknown_<id>` if the model emits an out-of-range id |
| `defects[].confidence` | float | 0–1, rounded to 4 decimals |
| `defects[].box` | object | `x_min`/`y_min`/`x_max`/`y_max` in original-image pixels |
| `latency.*` | float | Milliseconds for this request; `total_ms` is wall time for preprocess + inference + NMS |

Coherent boxes are guaranteed by per-class NMS at IoU 0.45, so overlapping
boxes of *different* classes both survive (a defect type on top of another is a
false pass otherwise).

**Errors**

| Status | When | Body `detail` |
|---|---|---|
| 400 | Body smaller than 16 bytes | `File is empty or too small to be a valid image.` |
| 400 | Pillow cannot parse the header | `Corrupted or invalid image structure: …` |
| 400 | Width or height over 4096 | `Image dimensions (WxH) exceed permitted limits [32 - 4096 px].` |
| 400 | Width or height under 32 | `Image dimensions (WxH) are below minimum [32 - 4096 px].` |
| 413 | Declared `Content-Length` over the request budget | `Request body is N bytes, which exceeds the 16…-byte request budget` |
| 413 | Streamed bytes exceed `MAX_IMAGE_SIZE_MB` | `Image size (N.NNMB) exceeds 15MB limit.` |
| 415 | Magic bytes match no allowed format | `Unsupported file format. Only JPEG, PNG, and BMP are accepted.` |
| 422 | FastAPI validation (e.g. missing `file` part) | Standard FastAPI error array |
| 429 | Rate limit exceeded | slowapi's `Rate limit exceeded: …` |
| 503 | No model loaded and none can be loaded | `model not loaded` |

---

## `POST /v1/detect/annotated`

Same input contract and guards as `/v1/detect`, but returns a JPEG with boxes
and labels drawn.

```bash
curl -X POST http://127.0.0.1:8000/v1/detect/annotated \
  -F "file=@docs/images/example-original.jpg" -o annotated.jpg
```

* **200** — `Content-Type: image/jpeg`, body is the encoded image.
* **500** — empty body if JPEG encoding fails (rare; would indicate a
  degenerate image buffer).
* All 4xx/503 cases above apply unchanged.

Rendering goes through the same `draw_detections()` the dashboard uses, so box
thickness, label style and the class palette cannot drift between the two
surfaces. There is no confidence for ground truth, so the gallery renderer
passes `show_confidence=False`.

---

## Cross-cutting behaviour

**CORS.** Enabled only for the origins in `ALLOWED_ORIGINS`
(`http://localhost:8501`, `http://localhost:3000` by default). Methods
`GET, POST, OPTIONS`; headers `content-type`; `allow_credentials=False`; never
a wildcard. A rejected upload still carries CORS headers because the CORS
middleware is outermost.

**Whole-request budget.** `BodySizeLimitMiddleware` runs before routing and
answers **413** from `Content-Length` for `POST`/`PUT`/`PATCH` requests over
`MAX_IMAGE_SIZE_MB * 1 MiB + 1 MiB` (16 MB by default). Bodies without a
`Content-Length` are counted as they stream and cut off at the same limit, with
no bytes reaching Starlette's form parser.

**Concurrency.** Both detection handlers are plain `def`, so Starlette runs
them in its threadpool and inference never blocks the event loop. A test pins
this (`test_detect_handlers_do_not_block_event_loop`).

**No authentication.** There is no auth, cookie or session handling, which is
why credentials are not enabled in CORS. Do not expose the service directly to
the internet — see the [security model](../explanation/security-model.md).
