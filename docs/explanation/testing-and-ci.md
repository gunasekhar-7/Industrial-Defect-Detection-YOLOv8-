# Testing and CI

How the suite is organised, why coverage is gated at 100%, and what the pipeline
does and does not prove.

---

## The shape of the suite

122 tests, 100% statement coverage of `src/`, no network access in CI.

| Area | File | What it pins |
|---|---|---|
| API behaviour | `tests/test_api.py` | Every status code, guard, CORS, rate limit, 503 paths, threadpool handlers |
| Upload decoding | `tests/test_image_io.py` | Header-first rejection, corrupt/empty input, boundary conditions |
| Annotation | `tests/test_annotate.py` | Box rendering, palette fallback, label placement, GT labels without confidence |
| Inference math | `tests/test_onnx_inference.py` | Preprocessing shape/normalisation, per-class NMS, benchmark report schema, session options |
| Dataset pipeline | `tests/test_data_pipeline.py` | Generator output, validator rules, split integrity |
| NEU-DET acquisition | `tests/test_neu_det.py`, `tests/test_neu_det_download.py` | Conversion, stratification, zip handling, failure messages |
| Model workflows | `tests/test_model_workflows.py`, `tests/test_evaluate.py` | Train/evaluate/export/quantize with fake Ultralytics and ORT modules |
| CLIs | `tests/test_cli_entrypoints.py` | Every `python -m …` guard and flag wiring, run via `runpy` as `__main__` |
| Dashboard | `tests/test_dashboard.py`, `tests/test_dashboard_flow.py` | The Streamlit script executes end-to-end through `AppTest` |

Three deliberate choices:

1. **Fakes at the boundary, real code inside.** Ultralytics and ONNX Runtime
   are replaced with small stubs where the point is orchestration; the actual
   preprocessing, NMS and guard logic always runs.
2. **CLIs are executed, not imported.** `runpy.run_path(..., run_name="__main__")`
   means an argument parser wired to the wrong function fails the suite.
3. **The dashboard is run, not imported.** `AppTest.from_file(...).run()`
   executes the page: widgets built, tabs rendered, exceptions surfaced.

---

## Why 100% statement coverage

The coverage floor (`--cov-fail-under=100` in CI, default `addopts` locally) is
a **regression gate**, not a quality metric. Its real benefit is that it makes
dead code visible: a branch even the author cannot reach is either dead or a
missing test, and both deserve attention.

That rule has already paid for itself. Two examples from this repository:

* The image-stream reader verified magic bytes as soon as 16 bytes arrived, so
  the later "deferred verification" branch could never execute. Rather than
  excluding it from coverage, the redundant branch was deleted.
* A benchmark path with no `--json` argument had an unreachable error handler;
  the handler went away, the flag kept its test.

What 100% statement coverage does **not** mean: that every branch, every
combination of inputs, or every concurrency interleaving is tested. Branch
coverage and mutation testing are listed in
[future scope](../future-scope.md) as the next honest step. Some `if` arms share
a statement, so 100% statements can coexist with an untaken branch.

---

## CI pipeline

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) — three jobs on
every branch push, every PR to `main`/`master`, and on demand. A newer push
cancels the run in flight.

| Job | Steps | Fails when |
|---|---|---|
| `lint` | `ruff check src tests`, `mypy src` | Any style/lint finding, any type error |
| `test` | `python -m src.data.generator` → `pytest tests/ -v --cov-fail-under=100` → upload `coverage.xml` | Any test fails, or coverage drops below 100% |
| `docker` | `docker build .` | The multi-stage image cannot be built |

CI deliberately **never downloads NEU-DET**: it is a manual, licensed fetch and
a flaky dependency. The synthetic generator provides a deterministic dataset
instead. Tests do not depend on a trained model either — they use fakes or
committed tiny fixtures, so the pipeline has no model-artifact dependency.

### What CI does not prove

| Not covered | Consequence |
|---|---|
| Accuracy | CI never trains; `reports/eval_report.json` is a human-run artifact |
| Performance | No benchmark job; latency numbers are produced manually and committed |
| Container runtime behaviour | `docker build` only — no `docker run`, no healthcheck assertion |
| Dashboard interaction depth | One render per tab (AppTest), not click-through flows |
| Multi-platform | `ubuntu-latest` only; Windows/macOS differences (path separators, ONNX thread behaviour) are not exercised |

The last row is not theoretical: the spinning-thread latency pathology showed
up only on the author's Windows machine, and CI's Linux runner would never have
caught it.

---

## Reproducing checks locally

```bash
ruff check src tests                 # lint (add scripts/ to include the tooling)
mypy src                             # types over the whole package
pytest tests/ -q                     # 122 tests, 100% statement coverage floor
```

Add `--cov-report=term-missing` (the default) to see any uncovered lines with
their line numbers.

## Adding a test that fits

* **New endpoint or guard** → `tests/test_api.py`, asserting the status code
  *and* the `detail` substring.
* **New pure function** → nearest domain file; assert behaviour, not
  implementation details.
* **New CLI flag** → `tests/test_cli_entrypoints.py` via `_run_module_as_main`,
  so the parser wiring is exercised.
* **New dashboard widget** → extend `tests/test_dashboard_flow.py`; a rendered
  page with `at.exception` empty is the minimum bar.

If a new test needs a network call or a real model download, it belongs behind
a fixture instead — CI must stay offline and deterministic.
