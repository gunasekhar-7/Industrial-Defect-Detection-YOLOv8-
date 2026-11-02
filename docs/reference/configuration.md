# Configuration reference

Every setting lives in [`src/api/config.py`](../../src/api/config.py) as a
Pydantic `BaseSettings` subclass, so each field is also an environment
variable of the same name (case-insensitive). Values shown are the defaults.

```python
from src.api.config import settings
print(settings.MAX_IMAGE_SIZE_MB)   # 15
```

List and dict fields (`CLASS_NAMES`, `CLASS_COLORS`, `ALLOWED_ORIGINS`) accept a
JSON string in the environment, e.g. `ALLOWED_ORIGINS='["https://aoi.example"]'`.

---

## Identity and model

| Setting | Default | Effect |
|---|---|---|
| `APP_NAME` | `Industrial Defect Detection API` | OpenAPI title |
| `APP_VERSION` | `1.0.0` | Reported by `/health` |
| `DEBUG` | `False` | Debug flag for the app object |
| `MODEL_PATH` | `models/industrial_yolov8n.onnx` | Primary graph. Loaded at startup; missing file ⇒ fallback |
| `FALLBACK_PYTORCH_PATH` | `models/industrial_yolov8n.pt` | Ultralytics fallback if ONNX is unavailable |
| `CONFIDENCE_THRESHOLD` | `0.25` | Detections below this score are dropped |
| `IOU_THRESHOLD` | `0.45` | NMS overlap threshold (per class) |

Model loading order, once per process, thread-safely:

1. `MODEL_PATH` → `onnx_runtime` (the supported path).
2. `FALLBACK_PYTORCH_PATH` → `pytorch_yolo` (needs `requirements-train.txt`).
3. Neither → `not_initialized`; `/health` returns 503 and detection endpoints
   return 503. A later request retries the load once, so exporting the model
   after startup recovers without a restart.

> The detector also disables ONNX Runtime's spinning thread pool. That is a
> deliberate performance default, not a model setting — see
> [Performance](../explanation/performance.md).

---

## Classes and palette

`CLASS_NAMES` is the single source of truth for class order and is used by the
API, the dashboard legend and the annotation renderer:

| id | name |
|---|---|
| 0 | `crazing` |
| 1 | `inclusion` |
| 2 | `patches` |
| 3 | `pitted_surface` |
| 4 | `rolled_in_scale` |
| 5 | `scratches` |

`CLASS_COLORS` maps the same ids to BGR tuples (OpenCV order), because the
renderer draws in BGR:

| id | BGR | Appears as |
|---|---|---|
| 0 | `(0, 0, 255)` | red |
| 1 | `(0, 165, 255)` | orange |
| 2 | `(0, 255, 255)` | yellow |
| 3 | `(255, 0, 255)` | magenta |
| 4 | `(255, 255, 0)` | cyan |
| 5 | `(0, 255, 0)` | green |

The dashboard converts these to hex for the sidebar legend, so legend and boxes
cannot disagree.

**If you change the class list, change all of:** `CLASS_NAMES`,
`CLASS_COLORS`, `configs/dataset.yaml` `names:`, the trained model's head, and
`CLASS_MAP` in `src/data/generator.py` (the synthetic generator has its own
copy used for fixtures).

---

## Upload limits

| Setting | Default | Effect |
|---|---|---|
| `MAX_IMAGE_SIZE_MB` | `15` | Streaming byte cap per image; also the dashboard's upload cap |
| `MAX_IMAGE_DIMENSION` | `4096` | Rejected above this, read from the header before pixels are allocated |
| `MIN_IMAGE_DIMENSION` | `32` | Rejected below this; degenerate images produce meaningless boxes |
| `ALLOWED_CONTENT_TYPES` | `image/jpeg`, `image/png`, `image/bmp` | Informational; enforcement is by magic bytes |

The whole-request budget is `MAX_IMAGE_SIZE_MB` + 1 MiB of multipart overhead
(16 MB by default), enforced by `BodySizeLimitMiddleware` before the form is
parsed.

---

## Rate limiting and CORS

| Setting | Default | Effect |
|---|---|---|
| `RATE_LIMIT_DEFAULT` | `60/minute` | slowapi limit applied per client IP to both POST endpoints |
| `ALLOWED_ORIGINS` | `http://localhost:8501`, `http://localhost:3000` | Exact CORS origins; a wildcard is never used |

Rate limiting keys on the client address from `get_remote_address`, so behind a
reverse proxy every client shares the proxy's IP unless it forwards the real
address. **Load tests must raise this limit** — one load host otherwise trips
it within a second:

```bash
RATE_LIMIT_DEFAULT=100000/minute uvicorn src.api.app:app --port 8000
```

---

## Environment variables in practice

```bash
# Point at a different graph and tighten the payload cap
MODEL_PATH=/srv/models/aoi-2026-11.onnx MAX_IMAGE_SIZE_MB=8 \
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Allow a deployed frontend origin
ALLOWED_ORIGINS='["https://aoi.example.com"]' uvicorn src.api.app:app
```

`docker-compose.yml` sets `MODEL_PATH` and `MAX_IMAGE_SIZE_MB` for both
services and mounts `./models` read-only.

---

## Settings that are deliberately hard-coded

These could be settings but are constants in code, because a configurable value
would weaken a guard or change the model contract:

| Constant | Location | Why fixed |
|---|---|---|
| Multipart overhead (1 MiB) | `src/api/security.py` | Bound on framing bytes; making it configurable invites "just raise it" |
| Magic-byte table | `src/api/security.py` | Only three formats can be decoded downstream |
| NMS class-offset (10 000 px) | `src/models/benchmark.py` | Must exceed the 4096 px input cap for per-class NMS to stay exact |
| Letterbox padding value (114) | `src/models/benchmark.py` | Part of the trained preprocessing; changing it silently degrades accuracy |
| ONNX session options (spinning off) | `src/models/benchmark.py` | A measured performance decision, reproducible with `--allow-spinning` |
