# Command-line reference

Every entry point is a module with an `if __name__ == "__main__"` guard (or a
script under `scripts/`), run from the repository root with the virtualenv
active. Defaults below are the code's defaults, not suggestions.

---

## Data

### `python -m src.data.generator`

Generate the synthetic dataset used by CI and local tests.

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `data/processed` | Output root; writes `images/{train,val,test}` and `labels/{train,val,test}` |

Fixed sizes: 240 train / 50 val / 50 test at 640 px, seed 42. Output is
deterministic for a given seed, which is what makes it usable as a CI fixture.
**Never quote accuracy numbers from a model trained on this data.**

### `python -m src.data.neu_det`

Download (or accept) NEU-DET and convert Pascal VOC XML to YOLO labels.

| Flag | Default | Meaning |
|---|---|---|
| `--dest` | `data/raw` | Where `NEU-DET.zip` is cached |
| `--out` | `data/processed` | YOLO output root |
| `--archive` | `None` | Use a local `.zip` or extracted directory instead of downloading |
| `--seed` | `42` | Stratified split seed |

Splits are 70/15/15, stratified by class. Exits non-zero with manual download
instructions when the Drive fetch fails; raises if the converted dataset fails
`DatasetValidator` on any split.

---

## Models

### `python -m src.models.train`

Fine-tune YOLOv8. Requires `requirements-train.txt`.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `yolov8n.pt` | Base weights |
| `--data` | `configs/dataset.yaml` | Dataset YAML |
| `--epochs` | `15` | Epochs (the committed run used 50) |
| `--batch` | `16` | Batch size |
| `--device` | auto | `cpu`, `0`, …; CUDA when available |
| `--imgsz` | `640` | Input size (committed run used 320) |

Writes `runs/detect/industrial_defect_yolov8n/weights/best.pt` and logs to
MLflow experiment `industrial-defect` when MLflow is installed.

### `python -m src.models.evaluate`

Evaluate a checkpoint and write the provenance report.

| Flag | Default | Meaning |
|---|---|---|
| `--weights` | required | Path to `best.pt` |
| `--data` | `configs/dataset.yaml` | Dataset YAML |
| `--split` | `val` | `val` or `test` |
| `--imgsz` | `640` | Evaluation size (committed run used 320) |
| `--out` | `eval_report.json` next to weights | Report path |

### `python -m src.models.export`

Export to ONNX and verify the graph with `onnx.checker`.

| Flag | Default | Meaning |
|---|---|---|
| `--weights` | required | Path to `best.pt` |
| `--imgsz` | `640` | Export input size; must match training |
| `--opset` | `17` | ONNX opset |
| `--half` | off | FP16 weights |
| `--dynamic` | off | Dynamic batch/axes |

### `python -m src.models.benchmark`

Micro-benchmark the ONNX graph in-process.

| Flag | Default | Meaning |
|---|---|---|
| `--onnx` | required | Path to the `.onnx` graph |
| `--warmup` | `50` | Warmup iterations (not counted) |
| `--runs` | `200` | Timed iterations |
| `--json` | none | Write the evidence report here |
| `--allow-spinning` | off | Use ORT's busy-wait thread pool (reproduces the unstable case) |

Prints median/mean/p95/p99 inference, mean end-to-end and throughput.

### `python -m src.models.quantize`

Produce an INT8 graph and, optionally, measure it against FP32.

| Flag | Default | Meaning |
|---|---|---|
| `--onnx` | required | FP32 graph to quantize |
| `--out` | `<stem>.int8.onnx` | Output graph; must differ from the input |
| `--per-channel` | off | Per-channel weight quantization |
| `--compare` | none | Directory of `.jpg` frames for the trade-off measurement (max 24) |
| `--report` | `reports/quantization_tradeoff.json` | Where to write the comparison |

Dynamic INT8 does not need calibration data. On this CPU it is 15× *slower*
than FP32 and changes predicted classes on half the frames, which is why the
API ships the FP32 graph — see
[Performance](../explanation/performance.md).

---

## Scripts

### `python scripts/loadtest.py`

Concurrent HTTP load test against `/v1/detect`.

| Flag | Default | Meaning |
|---|---|---|
| `--url` | `http://127.0.0.1:8000/v1/detect` | Target endpoint |
| `--image` | required | JPEG sent with every request |
| `--concurrency` | `8` | Worker threads |
| `--requests` | `200` | Total requests (`requests // concurrency` per worker) |
| `--warmup` | `10` | Sequential warmup requests before timing |
| `--json` | none | Write the report here |

Prints throughput plus p50/p95/p99/max, and aborts if a warmup request is not
200. Raise `RATE_LIMIT_DEFAULT` on the server first.

### `python scripts/make_gallery.py`

Render ground truth vs predictions for the weakest classes.

| Flag | Default | Meaning |
|---|---|---|
| `--onnx` | `models/industrial_yolov8n.onnx` | Model to inspect |
| `--images` | `data/processed/images/val` | Frames |
| `--labels` | `data/processed/labels/val` | YOLO labels for ground truth |
| `--out` | `docs/failure_gallery.png` | Output image |
| `--classes` | two lowest mAP@50-95 from `reports/eval_report.json` | Explicit class names instead |

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Uncaught exception (missing file, validation failure, unknown class) — the message names the path or value |

There is no `--quiet`/`--verbose` layer: these tools print a human summary and,
where applicable, write JSON for machines.
