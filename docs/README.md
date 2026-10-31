# Project documentation

Documentation for the **Industrial Surface Defect Detection** service
(YOLOv8 → ONNX Runtime → FastAPI + Streamlit). It follows the
[Diátaxis](https://diataxis.fr/) framework: each page serves one purpose, and
you should not have to read a tutorial to look up a config key.

New here? Read the [getting-started tutorial](tutorials/getting-started.md),
then skim [Architecture](explanation/architecture.md).

## Tutorials — learning-oriented

A guided lesson that takes a newcomer from a fresh clone to a running
inspection.

| Page | What you get |
|---|---|
| [Getting started](tutorials/getting-started.md) | Generate a dataset, run the API and dashboard, get your first detection |

## How-to guides — task-oriented

Recipes for a specific job. Assumes you already completed the tutorial.

| Page | Task |
|---|---|
| [Train and evaluate on NEU-DET](how-to/train-and-evaluate.md) | Acquire the real dataset, train, evaluate, export, install the served model |
| [Reproduce the benchmarks](how-to/reproduce-benchmarks.md) | Latency table, FP32-vs-INT8 trade-off, HTTP load test, failure gallery |
| [Deploy with Docker](how-to/deploy-with-docker.md) | Build the image, run API + dashboard with Compose, verify health |

## Reference — information-oriented

Exact descriptions of the machinery. No narrative, no teaching.

| Page | Covers |
|---|---|
| [HTTP API](reference/api.md) | Endpoints, request/response schemas, every status code |
| [Configuration](reference/configuration.md) | All settings, defaults, environment variables |
| [Command-line interfaces](reference/cli.md) | Every `python -m …` module and script, with flags |
| [Artifacts and formats](reference/artifacts.md) | Report JSON schemas, dataset layout, label format, model files |

## Explanation — understanding-oriented

Background, trade-offs and reasoning.

| Page | Discusses |
|---|---|
| [Architecture](explanation/architecture.md) | Components, request path, module boundaries, deliberate trade-offs |
| [Model and results](explanation/model-and-results.md) | Dataset, training choices, what each metric means, where the model fails |
| [Performance](explanation/performance.md) | Why latency numbers are hard, the spinning-thread finding, capacity |
| [Security model](explanation/security-model.md) | The upload threat model and how each guard answers it |
| [Testing and CI](explanation/testing-and-ci.md) | Test strategy, the 100% coverage floor, what CI does and does not prove |

## Roadmap

| Page | Contents |
|---|---|
| [Future scope](future-scope.md) | Prioritised next steps: model quality, deployment, observability, hardening |

## Other documents in this folder

| File | Purpose |
|---|---|
| [`RESUME.md`](RESUME.md) | STAR-format summaries, verbal pitch, likely interview questions |
| [`failure_gallery.png`](failure_gallery.png) | Ground truth vs predictions for the two weakest classes |
| [`images/`](images/) | Example input and annotated output used in the root README |
| [`superpowers/plans/`](superpowers/plans/) | Historical implementation plan from the hardening pass |

## Conventions used in these docs

* Commands are run from the repository root.
* `source .venv/bin/activate` on macOS/Linux; `.venv\Scripts\activate` on
  Windows.
* Numbers quoted here come from the committed JSON artifacts in
  [`reports/`](../reports/); when a figure and an artifact disagree, the
  artifact wins.
