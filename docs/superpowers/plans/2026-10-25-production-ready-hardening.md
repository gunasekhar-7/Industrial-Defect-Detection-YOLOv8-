# Production-Ready / Resume-Ready Hardening Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Execution style: **native** — implementer works in this session, test-first, frequent commits.

**Goal:** Turn this 6/10 portfolio project into a resume-honest, production-shaped ML/MLOps showcase where every README claim is backed by a committed artifact.

**Architecture:** Keep the existing layout (api / models / data / ui). Fix correctness and packaging defects first (so a fresh clone works), then swap synthetic metrics for real NEU-DET results with MLflow tracking, then add lint/type/CI gates, then a true multi-stage Docker image, and finish with an honest README plus a resume bullet document whose numbers trace to files in the repo.

**Tech Stack:** Python 3.11, FastAPI, ONNX Runtime, Ultralytics YOLOv8, MLflow (local file tracking), Streamlit, opencv-python-headless, ruff, mypy, pytest, GitHub Actions, Docker multi-stage.

**Spec:** Scoping decisions from brainstorming: role = **ML/MLOps engineer**; scope = **resume-honest MVP** (all P0/P1 review fixes, honest README, green CI, pinned deps, lint/type checks — every claim verifiable); dataset = **real NEU-DET for training/eval, synthetic generator kept for CI/tests**; deployment = **Docker + CI only, no cloud**; budget ≈ **1 week**; execution = **native**.

## Global Constraints

- Repo is now on branch **`main`** (was `master`); run `git status` before every staging step and stage only files each task names — never `git add -A`.
- No cloud accounts, no deployment, no auth, no Kubernetes — explicitly out of scope.
- CI must never download NEU-DET; CI uses the synthetic generator only.
- Every number in the final README must exist verbatim in a committed artifact (`reports/eval_report.json`, `reports/benchmark.json`).
- Runtime image and `requirements.txt` must not contain torch/ultralytics (training-only deps go to `requirements-train.txt`).
- Keep the synthetic generator working — it is the CI dataset, not dead code.
- `opencv-python` → `opencv-python-headless` everywhere (no GUI functions are used anywhere in the codebase).

## Review Focus (failure modes most likely to bite; each has its test in its task)

1. **NEU-DET download URL rots** — converter must also accept a locally downloaded archive path, with manual-download instructions as fallback. Tested with a tiny fixture archive (Task 12).
2. **`/health` changing to 503-when-unhealthy breaks the Docker HEALTHCHECK loop and existing tests** — both updated in the same task (Task 5).
3. **Anchoring `.gitignore` patterns re-routes ignores for other paths** — explicit `git check-ignore` matrix before/after (Task 1).
4. **Switching endpoints from `async def` to sync `def` changes behavior under TestClient** — full suite must pass in the same task, plus p95 measured before/after (Task 6).
5. **opencv headless swap silently breaks a cv2 function** — full suite + import smoke test (Task 3).

---

## Stage 1 — Unblock the fresh clone (P0). Day 1.

### Task 1: Fix `.gitignore` so `src/data/` is tracked

**Files:** Modify: `.gitignore`; Add to git: `src/data/download.py`, `src/data/generator.py`, `src/data/validator.py`, `src/data/__init__.py`

- [ ] Anchor ignore rules: `data/` → `/data/`, `runs/` → `/runs/`, add `/mlruns/`, add `!.gitkeep`-style exceptions only where listed below. Keep `*.pt`, `*.onnx`.
- [ ] Verify: `git check-ignore src/data/generator.py` exits 1 (not ignored); `git check-ignore data/x mlruns/x runs/x yolov8n.pt` all match.
- [ ] `git add .gitignore src/data/` and commit — then immediately `git status` and confirm no unexpected bulk additions.

### Task 2: Ship the ONNX model so `docker build` and fresh clones work

**Files:** Modify: `.gitignore`; Add: `models/industrial_yolov8n.onnx` (~12 MB)

- [ ] After `*.onnx`, add `!models/industrial_yolov8n.onnx`.
- [ ] Verify `git check-ignore models/industrial_yolov8n.onnx` exits 1, `git check-ignore yolov8n.onnx` still matches; commit the model.
- *Decision recorded:* committing the weights (vs. downloading at build) keeps builds reproducible and offline; 12 MB is acceptable for a showcase repo.

### Task 3: Split and repair requirements

**Files:** Modify: `requirements.txt`; Create: `requirements-train.txt`, `requirements-dev.txt`; Modify: any import that breaks

- [ ] `requirements.txt` (runtime): fastapi, uvicorn[standard], pydantic, **pydantic-settings** (the missing dep), slowapi, python-multipart, onnxruntime, numpy, pillow, opencv-python-headless, streamlit, pandas, httpx.
- [ ] `requirements-train.txt`: `-r requirements.txt` + torch, ultralytics, onnx, mlflow.
- [ ] `requirements-dev.txt`: `-r requirements.txt` + pytest, pytest-cov, ruff, mypy.
- [ ] Verify headless swap: `python -c "import cv2; cv2.imencode('.jpg', cv2.imread('mlruns/1/0f0152c110994c939ff536a4daf1f0e4/artifacts/labels.jpg'))"`, then full `pytest tests/ -q` → must stay at 12 passed (or better).

### Task 4: Fresh-clone gate

- [ ] `git clone . ../fresh-clone-check`, `pip install -r requirements.txt -r requirements-dev.txt` in it, `python -m src.data.download`, `pytest tests/ -q` → all green. Delete the temp clone afterwards.
- [ ] This exact gate is re-run as the final task (Task 26); getting it green now proves Stage 1.

---

## Stage 2 — P1 correctness (test-first). Days 2–3.

Each task: write failing test → run to see it fail → implement → run to pass → commit.

### Task 5: Honest health & 503 on missing model

**Files:** Modify: `src/api/app.py`, `tests/test_api.py`
**Interfaces:** `GET /health` → 200 `HealthResponse(status="healthy")` only when model loaded; 503 `{"status":"unhealthy", ...}` otherwise. `POST /v1/detect` and `/v1/detect/annotated` → 503 `detail="model not loaded"` when `model_engine is None`.
- [ ] Failing tests: monkeypatch `model_engine = None` → `/health` returns 503, `/v1/detect` returns 503; remove the fake `preprocess_ms: 1.0` latency dict entirely.
- [ ] Update the Docker HEALTHCHECK expectation comment; existing `test_health_check` updated to match (model loads in test env, so 200 path still covered).

### Task 6: Stop blocking the event loop + measure it (performance skill: before/after)

**Files:** Modify: `src/api/app.py`; Test: `tests/test_api.py`
- [ ] Change both detect endpoints from `async def` to sync `def` (FastAPI runs sync handlers in the threadpool); `sanitize_and_read_image` keeps its async signature but is called via `await` only from async context — simplest: make it a sync function `sanitize_and_read_image(file: UploadFile) -> tuple[np.ndarray, int, int]` since handlers are now sync.
- [ ] **Measure before/after** (performance skill): run `python -m src.models.benchmark --onnx models/industrial_yolov8n.onnx` and a 20-request sequential `httpx` loop against a local uvicorn before the change and after; record both in the task notes (they feed Task 15's `reports/benchmark.json`).
- [ ] Full suite passes.

### Task 7: Streaming payload cap (read-then-check → abort-early)

**Files:** Modify: `src/api/security.py`; Test: `tests/test_api.py`
**Interfaces:** `sanitize_and_read_image(file) -> tuple[np.ndarray, int, int]` reads in 1 MiB chunks with a running total; aborts with 413 the moment total exceeds `MAX_IMAGE_SIZE_MB`.
- [ ] Failing test: wrap an `UploadFile`-like object whose spy asserts total bytes read ≤ 16 MiB while posting a 16 MiB body → expect 413.
- [ ] Implement chunked loop; magic-byte check runs on the first chunk as soon as 16 bytes exist.

### Task 8: Per-class NMS

**Files:** Modify: `src/models/benchmark.py::ONNXDefectDetector.postprocess`; Test: `tests/test_onnx_inference.py`
- [ ] Failing test: two boxes, IoU > 0.45, **different** class ids → both survive; same class → one suppressed.
- [ ] Implement the standard class-offset trick (offset box coords by `class_id * 4096` before `cv2.dnn.NMSBoxes`, strip after).

### Task 9: Configurable CORS

**Files:** Modify: `src/api/config.py` (`ALLOWED_ORIGINS: List[str] = ["http://localhost:8501", "http://localhost:3000"]`), `src/api/app.py`
- [ ] Test: middleware built from `settings.ALLOWED_ORIGINS`, never `"*"`; assert configured origin is echoed in a preflight response.

### Task 10: Thread-safe one-time model load + real logging

**Files:** Modify: `src/api/app.py`
- [ ] Guard `load_model()` with a `threading.Lock` + "already loaded" short-circuit; replace all `print()` with `logging.getLogger("src.api")` at correct levels.
- [ ] Test: two concurrent `load_model()` calls result in one session construction (mock counter).

### Task 11: Honest validation messages

**Files:** Modify: `src/api/security.py`; Test: `tests/test_api.py`
- [ ] Below-minimum dimension returns detail containing `"below minimum"` (currently says "exceed"); oversized keeps `"exceed permitted limits"`; test both.

---

## Stage 3 — Real NEU-DET + honest metrics. Days 3–4.

### Task 12: NEU-DET acquisition & YOLO conversion

**Files:** Create: `src/data/neu_det.py`; Create: `configs/dataset_synthetic.yaml`; Modify: `configs/dataset.yaml`; Test: `tests/test_neu_det.py`
**Interfaces:**
- `download_neu_det(dest: str | Path = "data/raw", url: str | None = None) -> Path` — uses `url` or the documented default mirror; on failure raises with manual-download instructions (the README section it points to is written in Task 24).
- `convert_neu_det(archive_or_dir: str | Path, out: str | Path = "data/processed", splits=(0.7, 0.15, 0.15), seed: int = 42) -> None` — accepts either a downloaded archive **or an already-extracted local directory** (URL-rot fallback), emits YOLO labels, stratified split, runs `DatasetValidator` on each split and raises if invalid.
- `configs/dataset.yaml` → points at NEU-DET classes (same 6 names); `configs/dataset_synthetic.yaml` → the existing synthetic paths for CI.
- [ ] Tests use a **fixture directory of 6 tiny hand-made images + labels**, never a network call: conversion produces balanced splits, labels normalized `[0,1]`, validator passes.
- [ ] At execution time, use web research to confirm the current-working NEU-DET source URL and record it in `src/data/neu_det.py` + README fallback section (context7/web research for current availability).

### Task 13: Train & evaluate on real data

**Files:** Modify: `src/models/train.py`, `src/models/evaluate.py` defaults to accept `--data configs/dataset.yaml`; Add: `reports/eval_report.json` (committed)
- [ ] Run download → convert → `python -m src.models.train --data configs/dataset.yaml --epochs 100` (overnight CPU run; yolov8n on 1,800 images).
- [ ] `python -m src.models.evaluate --weights runs/detect/industrial_defect_yolov8n/weights/best.pt` → copy `eval_report.json` to `reports/`, re-export ONNX from the new `best.pt` into `models/industrial_yolov8n.onnx` (re-commit — Task 2's model now carries honest weights).
- [ ] **These real numbers replace every metric in the README** (Task 24).

### Task 14: MLflow experiment tracking

**Files:** Modify: `requirements-train.txt` (mlflow), `src/models/train.py`; `.gitignore` already ignores `/mlruns/` (Task 1)
- [ ] Wire Ultralytics' native MLflow integration (`MLFLOW_TRACKING_URI=file:./mlruns`, log params/metrics/artifacts); remove the currently-committed `mlruns/` files from the index (`git rm -r --cached mlruns`), keeping them local.
- [ ] Verify a training run appears: `python -c "import mlflow; print([r.name for r in mlflow.search_runs()])"` (or `mlflow ui` smoke note in README).

### Task 15: Benchmark → committed evidence

**Files:** Modify: `src/models/benchmark.py` (add `--json reports/benchmark.json` output incl. hardware string, provider, warmup/runs, mean/median/p95/p99, FPS, timestamp); Add: `reports/benchmark.json`
- [ ] Run it post-Task-6 on this machine; commit the JSON. This file is the **only** source for the README latency table.

---

## Stage 4 — Quality gates. Day 5.

### Task 16: ruff + pytest config

**Files:** Create: `pyproject.toml` (`[tool.ruff]`, `[tool.pytest.ini_options]` with `--cov=src`)
- [ ] `ruff check src tests` → fix every finding (unused `io`/`time`/`Path` imports, etc.); no `# noqa` escapes except with a written reason.

### Task 17: mypy on `src/api` and `src/models`

**Files:** Modify: `pyproject.toml` (`[tool.mypy]`, `ignore_missing_imports = true` for cv2/onnxruntime/ultralytics)
- [ ] `mypy src/api src/models` → clean.

### Task 18: Pin exact versions

**Files:** Modify: all three requirements files
- [ ] Freeze tested versions (`pkg==x.y.z` from the working environment), reinstall into a throwaway venv, full suite green.

### Task 19: Kill the vacuous test

**Files:** Modify: `tests/test_data_pipeline.py`
- [ ] `test_validator_clean_report` generates a small dataset into `tmp_path` (10 images) and asserts unconditionally — no more `if report.total_images > 0:` guard.

### Task 20: CI pipeline

**Files:** Modify: `.github/workflows/ci.yml`
- [ ] Three jobs: **lint** (ruff check + mypy), **test** (install dev reqs → `python -m src.data.download` with synthetic config → `pytest --cov` → upload coverage artifact), **docker** (`docker build .` as a smoke gate for Task 21).
- [ ] No NEU-DET download in CI (Global Constraint).

---

## Stage 5 — Real multi-stage Docker + dedup. Day 6.

### Task 21: Multi-stage Dockerfile

**Files:** Modify: `Dockerfile`
- [ ] Stage `builder`: python:3.11-slim, install runtime requirements into a venv. Stage `runtime`: slim base, non-root `app` user, copy venv + `src/` + `configs/` + `models/`, drop `libgl1`/`curl` if headless opencv + python-based healthcheck make them unnecessary (`CMD` healthcheck via `urllib` against `/health`).
- [ ] Verify: `docker build .` and `docker run --rm -p 8000:8000` → `/health` 200 (or CI docker job passes if Docker unavailable locally).

### Task 22: Compose correctness

**Files:** Modify: `docker-compose.yml`
- [ ] Remove obsolete `version:` key; `dashboard.depends_on: api: condition: service_healthy`; mount `./models:/app/models:ro`; keep both ports; `docker-compose up --build` documented path verified.

### Task 23: DRY annotation + single source of class truth

**Files:** Create: `src/vision/annotate.py` with `draw_detections(img_bgr, detections, class_names, colors) -> np.ndarray`; Modify: `src/api/app.py`, `src/ui/dashboard.py`, `src/api/config.py` (class colors live here, once)
- [ ] Test: drawing a box changes pixels at the box corner; both former copies deleted; `grep -r "cv2.rectangle" src/` shows only `annotate.py`.

---

## Stage 6 — Honest docs + resume package. Day 7.

### Task 24: README rewrite (Diátaxis: tutorial + how-to focused)

**Files:** Modify: `README.md`; Add: `docs/images/` (screenshots), `LICENSE` (MIT)
- [ ] Every metric copied **verbatim** from `reports/eval_report.json` and `reports/benchmark.json`, with hardware noted; delete the "Resume Bullet Points" section; delete/replace the `yourusername` badges; fix the mAP contradiction by sourcing numbers from the JSON only.
- [ ] Architecture diagram matches reality (state plainly: dashboard runs in-process inference for local inspection; services call the API — a documented design decision, not a diagram lie).
- [ ] Sections: what it is, honest results, quickstart (verified in Task 4/26), NEU-DET manual-download fallback, API reference (`/docs`), development (lint/mypy/test commands), limitations.
- [ ] Verify: claim audit — walk every factual claim in the README and check it against a file/line or report value; note any you cannot verify and delete them.

### Task 25: Resume deliverable

**Files:** Create: `docs/RESUME.md`
- [ ] 4–6 STAR bullets whose every number cites its artifact, e.g. mAP from `reports/eval_report.json`, p95/FPS from `reports/benchmark.json`, "6-class NEU-DET benchmark", "security-hardened FastAPI (magic-byte, decompression-bomb, rate-limit) with adversarial input tests", "multi-stage non-root Docker image", "ruff+mypy+pytest CI with coverage". Plus 30-second verbal pitch and likely interview questions with the repo-grounded answers.

### Task 26: Final verification gate (verification-before-completion skill)

- [ ] Fresh-clone gate (as Task 4): clone → install → synthetic dataset → `ruff check` → `mypy` → `pytest` → all green.
- [ ] `docker build .` passes; `docker-compose up --build` serves `/health` 200 and `/docs`.
- [ ] README claim audit completed and noted in the final report.
- [ ] Report: before/after rating against the original review dimensions with evidence per dimension.

**Deferred (explicitly not in this plan):** live cloud deployment, authentication, load testing, routing the dashboard through the API, FP16 quantization claims, Kubernetes.
