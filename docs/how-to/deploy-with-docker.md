# How-to: deploy with Docker

Task: run the API and dashboard as containers, verify they are healthy, and
know what the image does and does not give you.

---

## 1. Build

```bash
docker build -t defect-detection:local .
```

The Dockerfile is two stages:

| Stage | What happens |
|---|---|
| `builder` | `python:3.11-slim`, creates `/opt/venv`, installs `requirements.txt` only |
| `runtime` | `python:3.11-slim`, copies the venv, `configs/`, `src/`, `models/`, runs as user `app` (uid 1000) |

Consequences worth knowing:

* **No PyTorch, no ultralytics, no GUI OpenCV** in the final image — the
  runtime set is FastAPI + ONNX Runtime + headless OpenCV + Streamlit.
* **No `curl`, no `libgl1`.** The healthcheck uses Python's stdlib `urllib`.
* **The model is baked in** (`COPY models/ ./models/`) so the container starts
  healthy. Compose overrides the path with a read-only mount instead.

Build time is dominated by the pip install; rebuilds reuse the layer while
`requirements.txt` is unchanged.

---

## 2. Run both services

```bash
docker compose up --build
```

| Service | Port | Notes |
|---|---|---|
| `api` | 8000 | `uvicorn src.api.app:app`, `MODEL_PATH=models/industrial_yolov8n.onnx`, models mounted read-only |
| `dashboard` | 8501 | `streamlit run src/ui/dashboard.py`, waits for `api` to be healthy |

Verify:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/detect \
  -F "file=@docs/images/example-original.jpg"
```

Open the dashboard at <http://localhost:8501>.

API docs (Swagger UI) are at <http://localhost:8000/docs>.

---

## 3. Confirm the container is actually healthy

```bash
docker inspect --format '{{.State.Health.Status}}' defect-detection-api
```

`healthy` means the `HEALTHCHECK` reached `/health` with a 2xx. If you see
`unhealthy`, read the reason:

```bash
docker logs defect-detection-api --tail 50
```

The most common cause is a missing model inside the image or a bad `MODEL_PATH`
mount — `/health` answers **503** when no model is loaded, deliberately, so an
orchestrator removes the replica from rotation instead of sending traffic to a
service that would answer `503` on every detection.

---

## 4. Swap in your own weights

The compose file mounts `./models` read-only, so you can replace the graph and
restart without rebuilding:

```bash
cp runs/detect/industrial_defect_yolov8n/weights/best.onnx \
   models/industrial_yolov8n.onnx
docker compose restart api
```

`run_detection()` retries a failed model load on the next request, so even
exporting *after* a container started can recover without a restart.

---

## 5. Scaling

One worker saturates around **27 req/s** on the machine this was measured on
(see [Performance](../explanation/performance.md)). Scale out, not up:

```bash
docker compose up --build --scale api=4
```

Put a reverse proxy in front so clients hit one address. Note that uvicorn is
started with a single worker per container on purpose — ONNX Runtime already
uses the available cores inside a process, and two workers in one container
compete for the same CPU.

---

## 6. What this deployment is not

| Missing | Implication |
|---|---|
| TLS | Terminate at a proxy; do not expose port 8000 directly to a network you do not trust |
| Authentication | Add an API gateway or reverse-proxy auth before public exposure |
| Observability | No `/metrics`, no structured logs, no tracing (see [Future scope](../future-scope.md)) |
| Persistent storage | Nothing is persisted; the service is stateless by design |
| Multi-arch build | The image targets the platform `docker build` runs on |

The service is safe to run on a trusted internal network and behind a gateway.
It is not a public-internet deployment out of the box, and the
[security model](../explanation/security-model.md) explains which threats the
upload guards do and do not cover.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `port is already allocated` | Another process on 8000/8501 | Stop it, or edit the port mapping |
| Dashboard container restarts | `depends_on: api: service_healthy` never satisfied | Check the API log first; the dashboard is not the problem |
| `permission denied` reading `models/` | The mount is read-only and the file is not readable by uid 1000 | `chmod 644` the `.onnx` file |
| Image builds but detection fails | ONNX exported with a different input size than expected | Re-export at the size the app configures |
| `docker build` fails at pip | Network/proxy blocking PyPI in the build context | Configure a build-time index mirror |
