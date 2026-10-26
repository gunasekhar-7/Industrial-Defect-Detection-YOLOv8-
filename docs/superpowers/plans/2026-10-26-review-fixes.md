# Review-Fix Hardening Plan (Production/Resume Polish)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Execution style: **native**; delivery: **local commits only** (no push, no PR — user's choice).

**Goal:** Fix all five issues from the code review with test-first changes, keeping every README claim and committed artifact untouched and resume-honest.

**Architecture:** Keep the existing layout. Four targeted fixes (train-CLI pinning test, NMS class offset, header-first upload guard, deprecated 413 constant) plus one consolidation (delete `src/data/download.py`, make `python -m src.data.generator` the synthetic entrypoint), then a full quality gate with an evidence-preserving performance check.

**Tech Stack:** Python 3.11, FastAPI/Starlette, opencv-python-headless, Pillow, ONNX Runtime, Streamlit, pytest, ruff, mypy, GitHub Actions (config edit only).

**Spec:** The user's code review (5 items): (1) forward `--imgsz` in `src/models/train.py`; (2) bump NMS class-offset from 4096 to 10000 in `src/models/benchmark.py`; (3) ≤4096px guard before `cv2.imdecode` on Streamlit uploads in `src/ui/dashboard.py`; (4) `HTTP_413_REQUEST_ENTITY_TOO_LARGE` → `HTTP_413_CONTENT_TOO_LARGE` in `src/api/security.py`; (5) consolidate `src/data/download.py`. **Verified against HEAD `8d299fc` (clean tree):** item 1 was already implemented in commit `ea12873` (the reviewer saw the pre-hardening state) → covered by a regression test only; items 2–5 are open and confirmed by direct code inspection.

## Global Constraints

- **Local commits only** — never push, never open a PR. Run `git status --short --branch` before every staging step and stage only the files each task names — never `git add -A`.
- CI must never download NEU-DET; the synthetic generator remains the CI dataset.
- Do not modify committed evidence (`reports/eval_report.json`, `reports/benchmark.json`), the shipped model (`models/industrial_yolov8n.onnx`), or any metric in `README.md` / `docs/RESUME.md` beyond the command line Task 6 names.
- `python -m src.data.generator` must produce the same dataset the old entrypoint did: 240/50/50 images at `data/processed` (defaults already match).
- Finish with `ruff check src tests`, `mypy src/api src/models`, and the full pytest suite green. No `# noqa`, `# type: ignore`, or warning filters added to make checks pass.
- The archived plan `docs/superpowers/plans/2026-10-25-production-ready-hardening.md` keeps its historical references to `src.data.download` — do not edit archived documents.

## Review Focus (failure modes most likely to bite; each has its test in its task)

1. **The dimension guard sits after decode** — then it is cosmetic (the OOM it should prevent already happened). Task 4's test monkeypatches `cv2.imdecode` and asserts **zero** calls for an oversize image.
2. **The NMS offset change regresses per-class behavior** — existing `test_nms_is_per_class` must stay green in the same task, plus the full suite.
3. **Deleting `download.py` breaks CI/README/RESUME** — Task 6 greps every reference out (archived plan excluded) and smoke-runs the new entrypoint before committing.
4. **The 413 constant swap changes semantics or misses one usage** — existing 413 tests must still see status 413, and the new test records warnings to prove the deprecation is gone.
5. **Refactoring train's CLI into `main(argv)` changes behavior** — `python -m src.models.train --help` must still work and the forwarding test must capture the exact kwargs.

---

## Stage 0 — Archive the plan

### Task 1: Commit the plan document

**Files:** Create: `docs/superpowers/plans/2026-10-26-review-fixes.md`

- [ ] Save this plan verbatim to `docs/superpowers/plans/2026-10-26-review-fixes.md`.
- [ ] `git add docs/superpowers/plans/2026-10-26-review-fixes.md && git commit -m "docs: plan for review-fix hardening pass"` — then `git status --short` must be clean.

---

## Stage 1 — Functional fixes (review items 1–3)

### Task 2: Pin `--imgsz` forwarding with a regression test (item 1 — already fixed in code)

**Files:** Create: `tests/test_train_cli.py`; Modify: `src/models/train.py` (extract `main(argv)`)

**Interfaces:**
- Produces: `src.models.train.main(argv: Sequence[str] | None = None) -> Any` — parses CLI args and calls module-global `train_model(**kwargs)`; `if __name__ == "__main__": main()` (mirrors `src/data/neu_det.py:main`).

- [ ] **Step 1: Write the failing test** — `tests/test_train_cli.py`:

```python
"""CLI forwarding: user-specified --imgsz must reach train_model (review item 1)."""
import src.models.train as train_mod


def test_cli_forwards_imgsz_and_training_args(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_mod, "train_model", lambda **kw: captured.update(kw))
    train_mod.main(["--imgsz", "320", "--epochs", "2", "--batch", "4"])
    assert captured["imgsz"] == 320
    assert captured["epochs"] == 2
    assert captured["batch"] == 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_train_cli.py -q` → FAIL with `AttributeError: module 'src.models.train' has no attribute 'main'`.

- [ ] **Step 3: Extract `main(argv)` in `src/models/train.py`** — move the existing `argparse` block under `def main(argv: Sequence[str] | None = None) -> Any:` using `parser.parse_args(argv)`, call `main()` from `__main__`. No change to the arguments or to what is passed to `train_model` (imgsz forwarding already exists).

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_train_cli.py -q` → PASS; `python -m src.models.train --help` still prints the flags.

- [ ] **Step 5: Commit**

```bash
git add tests/test_train_cli.py src/models/train.py
git commit -m "test: pin train CLI forwarding of --imgsz (review item 1)"
```

### Task 3: NMS class-offset 4096 → 10000 (item 2)

**Files:** Modify: `src/models/benchmark.py:110-116`; Test: `tests/test_onnx_inference.py`

**Interfaces:**
- Consumes: `ONNXDefectDetector.postprocess(outputs, ratio, pad, orig_shape)` — signature unchanged.

- [ ] **Step 1: Write the failing test** — append to `tests/test_onnx_inference.py`, reusing its `MockDetector`:

```python
def test_wide_full_width_boxes_of_different_classes_survive_nms():
    """Class offset must exceed the widest accepted box (review item 2):
    at 4096 the shifted box overlaps an 11000px full-width box with
    IoU ~0.457 > 0.45 and wrongly suppresses the other class."""
    detector = MockDetector()
    preds = np.zeros((1, 10, 2), dtype=np.float32)
    for anchor in (0, 1):
        preds[0, 0, anchor] = 5500.0   # xc
        preds[0, 1, anchor] = 5500.0   # yc
        preds[0, 2, anchor] = 11000.0  # w (full width)
        preds[0, 3, anchor] = 200.0    # h
    preds[0, 4, 0] = 0.90              # class 0 score
    preds[0, 5, 1] = 0.80              # class 1 score

    detections = detector.postprocess(
        preds, ratio=1.0, pad=(0, 0), orig_shape=(11000, 11000)
    )
    classes = sorted(d["class_id"] for d in detections)
    assert classes == [0, 1], f"cross-class suppression returned {classes}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_onnx_inference.py::test_wide_full_width_boxes_of_different_classes_survive_nms -q` → FAIL with `got [0]` (IoU 0.457 crosses the 0.45 threshold under the 4096 offset).

- [ ] **Step 3: Change the offset in `src/models/benchmark.py`** — `[x + class_id * 10000, y, w, h]`, and rewrite the stale comment (it claims "max box width 640"): the offset must exceed the widest box the detector can produce; 10000 gives headroom over the API's 4096px input cap so cross-class IoU stays exactly 0.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_onnx_inference.py -q` → all PASS, including existing `test_nms_is_per_class`.

- [ ] **Step 5: Commit**

```bash
git add src/models/benchmark.py tests/test_onnx_inference.py
git commit -m "fix: raise NMS class offset to 10000 so wide boxes cannot suppress other classes"
```

### Task 4: Header-first dimension guard for dashboard uploads (item 3)

**Files:** Create: `src/vision/image_io.py`; Modify: `src/ui/dashboard.py` (tab 1 ≈ lines 112-115, tab 3 ≈ lines 196-202); Test: `tests/test_image_io.py`

**Interfaces:**
- Produces: `decode_image(data: bytes, max_dim: int) -> np.ndarray` — reads only the image header via Pillow first; raises `ValueError(f"image dimensions {w}x{h} exceed maximum {max_dim}px")` **before** any pixel decode when `max(w, h) > max_dim`; raises `ValueError` for unreadable/undecodable bytes; otherwise returns a BGR `np.ndarray`.
- Consumes in dashboard: `settings.MAX_IMAGE_DIMENSION` (4096) as `max_dim`; on `ValueError` the dashboard shows `st.error(str(exc))` and skips that image.

- [ ] **Step 1: Write the failing tests** — `tests/test_image_io.py`:

```python
"""Header-first upload guard (review item 3): oversize is rejected before decode."""
import io

import pytest
from PIL import Image

from src.vision.image_io import decode_image


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def test_oversize_rejected_before_pixel_decode(monkeypatch):
    import src.vision.image_io as mod

    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(mod.cv2, "imdecode", spy)
    with pytest.raises(ValueError, match="exceed maximum 4096px"):
        decode_image(_jpeg(5000, 5000), 4096)
    assert calls["n"] == 0, "guard must reject before cv2.imdecode runs"


def test_valid_image_decodes_to_bgr():
    img = decode_image(_jpeg(640, 480), 4096)
    assert img.shape == (480, 640, 3)


def test_corrupt_bytes_rejected():
    with pytest.raises(ValueError):
        decode_image(b"\xff\xd8\xffgarbage", 4096)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_image_io.py -q` → FAIL with `ModuleNotFoundError: No module named 'src.vision.image_io'`.

- [ ] **Step 3: Create `src/vision/image_io.py`** with `decode_image(data: bytes, max_dim: int) -> np.ndarray`: empty-bytes check → `Image.open(io.BytesIO(data))` header read inside a `with` (catch `Exception`, raise `ValueError("unreadable image: ...")` from it) → `max_dim` check raising the exact message above → `cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)` with `None` → `ValueError("failed to decode image data")`.

- [ ] **Step 4: Wire the dashboard** — tab 1: replace the bare `imdecode` with `try: img_bgr = decode_image(uploaded_file.read(), settings.MAX_IMAGE_DIMENSION)` / `except ValueError as exc: st.error(str(exc)); img_bgr = None`, keeping the existing `if img_bgr is not None:` structure. Tab 3 batch loop: same pattern (`img = None` + `st.error`), keeping the existing `if img is not None:` branch.

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/test_image_io.py -q` → 3 PASS; `ruff check src tests` clean.

- [ ] **Step 6: Commit**

```bash
git add src/vision/image_io.py src/ui/dashboard.py tests/test_image_io.py
git commit -m "fix: reject oversize dashboard uploads before pixel decode (review item 3)"
```

---

## Stage 2 — Cleanup (review items 4–5)

### Task 5: Replace the deprecated 413 constant (item 4)

**Files:** Modify: `src/api/security.py:55`; Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_api.py`, reusing its `_SpyUpload`:

```python
def test_oversize_413_raises_no_deprecation_warning():
    """Review item 4: the 413 path must use HTTP_413_CONTENT_TOO_LARGE —
    the deprecated REQUEST_ENTITY name emits StarletteDeprecationWarning."""
    import warnings

    from fastapi import HTTPException

    from src.api.security import sanitize_and_read_image

    payload = b"\xff\xd8\xff" + b"x" * (16 * 1024 * 1024)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(HTTPException) as exc:
            sanitize_and_read_image(_SpyUpload(payload))
    assert exc.value.status_code == 413
    stale = [w for w in caught if "HTTP_413_REQUEST_ENTITY_TOO_LARGE" in str(w.message)]
    assert not stale, "deprecated Starlette status constant still referenced"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_api.py::test_oversize_413_raises_no_deprecation_warning -q` → FAIL with the stale-constant message.

- [ ] **Step 3: Swap the constant** at `src/api/security.py:55` → `status.HTTP_413_CONTENT_TOO_LARGE` (verified present in the installed Starlette, value 413; it is the file's only usage).

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_api.py -q` → all PASS (the two existing 413 assertions still see status 413), and no `StarletteDeprecationWarning` anywhere in the output.

- [ ] **Step 5: Commit**

```bash
git add src/api/security.py tests/test_api.py
git commit -m "fix: use non-deprecated HTTP_413_CONTENT_TOO_LARGE constant"
```

### Task 6: Fold `download.py` into `generator.py` (item 5 — user chose "fold into generator.py")

**Files:** Delete: `src/data/download.py`; Modify: `src/data/generator.py` (`__main__` block + `argparse`/`Sequence` imports), `.github/workflows/ci.yml:53`, `README.md:166`, `docs/RESUME.md:154`, `configs/dataset_synthetic.yaml:2`

**Interfaces:**
- Produces: `src.data.generator.main(argv: Sequence[str] | None = None) -> None` with `--out` (default `data/processed`), called from `__main__`; `generate_dataset` itself unchanged.

- [ ] **Step 1: Replace the entrypoint** — in `generator.py` replace the bare `__main__` block with `main(argv)` using `argparse` (`--out`, default `data/processed`) calling `generate_dataset(args.out)`. Defaults reproduce the old `setup_dataset` output exactly (240/50/50, seed 42). The old module's generate-plus-validate print was cosmetic — validation is covered by `tests/test_data_pipeline.py`, so it is not re-implemented.
- [ ] **Step 2: Update every reference** — `.github/workflows/ci.yml:53` → `run: python -m src.data.generator`; `README.md:166` → `python -m src.data.generator   # synthetic dataset, no download needed`; `docs/RESUME.md:154` → same substitution; `configs/dataset_synthetic.yaml:2` comment → `# Generated by: python -m src.data.generator`.
- [ ] **Step 3: Delete the module** — `git rm src/data/download.py`.
- [ ] **Step 4: Verify no dangling references and smoke-run**

Run: `grep -rn "src\.data\.download\|src/data/download" src tests .github README.md docs/RESUME.md configs` → empty (the archived plan under `docs/superpowers/plans/2026-10-25-*.md` is excluded by the Global Constraint), **and** `python -m src.data.generator --out data/processed` prints `Generated dataset at data/processed: train=240, val=50, test=50`.
- [ ] **Step 5: Commit** — stage explicitly, never `git add -A`:

```bash
git add src/data/generator.py .github/workflows/ci.yml README.md docs/RESUME.md configs/dataset_synthetic.yaml
git commit -m "refactor: fold synthetic dataset entrypoint into generator, drop download.py"
```

*(the `git rm` from Step 3 is already staged.)*

---

## Stage 3 — Verification gate

### Task 7: Full gate (verification-before-completion skill)

- [ ] `ruff check src tests` → clean; `mypy src/api src/models` → clean.
- [ ] `pytest tests/ -q` → all green: 38 existing + 6 new = **44 passed**, with no `StarletteDeprecationWarning` in the output.
- [ ] `python -m src.data.generator --out data/processed` succeeds (Task 6 evidence re-run in the gate).
- [ ] **Performance check (performance skill — evidence rule):** run `python -m src.models.benchmark --onnx models/industrial_yolov8n.onnx --warmup 50 --runs 200` **without** `--json` so the committed `reports/benchmark.json` stays untouched. The offset change is one integer multiply and the guard is a header read on upload only — hypothesized no measurable change; compare printed p95 against the committed 10.64 ms and flag any regression beyond ~10% in the final report instead of editing evidence.
- [ ] `docker build .` → exit 0 (CI's docker job only builds; the deleted module lives inside the copied `src/`).
- [ ] Docs spot-audit: the only README/RESUME lines changed are the four in Task 6 — `git diff <first-task-commit>~1 --stat` shows only task-named files, and no metric text changed.
- [ ] `git status --short` clean; `git log --oneline -7` shows one commit per task.

**Deferred (explicitly not in this plan):** push/PR (user chose local commits), regenerating benchmark/eval evidence, mypy coverage for `src/vision`, editing the archived 2026-10-25 plan doc, and any work beyond the five review items.
