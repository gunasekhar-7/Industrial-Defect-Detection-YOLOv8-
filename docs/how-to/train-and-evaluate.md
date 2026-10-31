# How-to: train and evaluate on NEU-DET

Task: produce your own weights, a metrics report that matches the README's
provenance rules, and a served ONNX model. Assumes the
[tutorial](../tutorials/getting-started.md) is done.

Real training needs the training stack, which is intentionally separate from
the runtime image:

```bash
pip install -r requirements-train.txt   # torch, ultralytics, onnx, mlflow
```

---

## 1. Acquire NEU-DET

The converter downloads the official archive and converts it to YOLO layout:

```bash
python -m src.data.neu_det
```

```
Converted NEU-DET -> data/processed: train=1260, val=270, test=270; classes=6; images=1800
```

If the download fails (Drive quota, rotated link, offline machine) the script
prints manual instructions and exits. Do the manual fetch, then point the
converter at the file or an extracted directory:

```bash
python -m src.data.neu_det --archive data/raw/NEU-DET.zip
python -m src.data.neu_det --archive path/to/extracted/NEU-DET
```

Useful flags: `--dest` (download cache, default `data/raw`), `--out` (YOLO
output root, default `data/processed`), `--seed` (split seed, default `42`).

What the conversion guarantees:

* **Stratified split** on the first object's class, 70/15/15, seeded.
* Labels clamped into `[0,1]` with a 1e-4 minimum box size.
* Every split validated afterwards with `DatasetValidator`; the command
  **raises** on corrupted images, out-of-bounds boxes, orphaned labels or
  class ids outside 0–5.
* No partial writes: everything is parsed before `data/processed` is touched.

> Building the synthetic dataset (`python -m src.data.generator`) afterwards
> replaces this conversion — they share `data/processed/`. Regenerate the real
> dataset before training if you did that.

---

## 2. Train

```bash
python -m src.models.train \
  --data configs/dataset.yaml \
  --epochs 50 --imgsz 320 --batch 16 --device cpu
```

Flags: `--model` (base weights, default `yolov8n.pt`), `--data`, `--epochs`
(default `15`), `--batch` (16), `--device` (`cpu`, `0`, …; auto-detected when
omitted), `--imgsz` (640).

Weights land in `runs/detect/industrial_defect_yolov8n/weights/best.pt`
(`project=""` + `name=industrial_defect_yolov8n` is the default layout that
evaluate and export assume).

Two deliberate choices worth knowing before you change them:

* **320 px, not 640.** NEU-DET sources are 200×200. Upscaling to 640 adds no
  information and quadruples the pixels paid for every epoch on CPU.
* **CPU by default.** `--device` is auto-detected: CUDA when available,
  otherwise CPU. A 50-epoch CPU run is the configuration behind the committed
  metrics.

MLflow is wired through Ultralytics' callback: if `mlflow` is importable, a run
is created in the local store under experiment `industrial-defect`, logging
params, per-epoch metrics and weight artifacts. `mlflow.db` / `mlruns/` are
gitignored.

---

## 3. Evaluate (this is what writes the quotable numbers)

```bash
python -m src.models.evaluate \
  --weights runs/detect/industrial_defect_yolov8n/weights/best.pt \
  --data configs/dataset.yaml --split val --imgsz 320 \
  --out reports/eval_report.json
```

The report is not just metrics — it records the **executed** configuration so
a number can be traced to the run that produced it:

```json
{
  "model": "best.pt", "data": "configs/dataset.yaml", "split": "val",
  "imgsz": 320, "images": 270,
  "hardware": "Windows-10-10.0.26300-SP0 | … | python 3.11.7",
  "timestamp": "2026-10-25T10:44:34+00:00",
  "mAP50": 0.7208593384148835, "mAP50_95": 0.4191165798397718,
  "precision": 0.6577969755601648, "recall": 0.6982250905072714,
  "training": {"model": "yolov8n.pt", "epochs": 50, "batch": 16,
               "imgsz": 320, "device": "cpu", "optimizer": "auto"},
  "per_class": {"crazing": {"mAP50_95": 0.15460208198193207}, "…": {}}
}
```

`images` and `training` are read from the dataset YAML and the run's
`args.yaml`, not typed by hand; a missing file degrades to `0` / `{}` instead
of failing. Omit `--out` to write `eval_report.json` next to the weights.

**Always evaluate on a split you did not train on.** `--split val` is the
default; `--split test` gives a second opinion, but pick one and quote it
consistently.

---

## 4. Export and install the served model

```bash
python -m src.models.export \
  --weights runs/detect/industrial_defect_yolov8n/weights/best.pt --imgsz 320
```

Ultralytics writes `…/weights/best.onnx` and the exporter then loads it with
`onnx.checker.check_model`, so a corrupt graph fails here rather than in the
API. Flags: `--opset` (17), `--half` (FP16), `--dynamic` (dynamic batch/axes).

Copy it to the path the API reads:

```bash
cp runs/detect/industrial_defect_yolov8n/weights/best.onnx \
   models/industrial_yolov8n.onnx
```

Restart the API (or just call an endpoint — `run_detection` retries a failed
load once) and confirm `/health` reports `model_type: onnx_runtime`.

> **Keep the ONNX input size and `--imgsz` in sync.** The detector letterboxes
> to the shape baked into the graph; exporting at 320 and serving a 640 graph
> silently changes accuracy.

---

## 5. Record the evidence

Numbers quoted in the README, `docs/`, or a résumé must exist in a committed
artifact:

| Claim | Regenerate with |
|---|---|
| Accuracy | `python -m src.models.evaluate … --out reports/eval_report.json` |
| Latency | `python -m src.models.benchmark --onnx models/industrial_yolov8n.onnx --json reports/benchmark.json` |
| INT8 trade-off | `python -m src.models.quantize --onnx models/industrial_yolov8n.onnx --compare data/processed/images/val` |
| Load test | `python scripts/loadtest.py … --json reports/loadtest.json` |

See [Reproduce the benchmarks](reproduce-benchmarks.md) for the full commands
and [Artifacts and formats](../reference/artifacts.md) for the schemas.

---

## Verifying a training run end to end

```bash
python -m src.data.neu_det --archive data/raw/NEU-DET.zip
python -m src.models.train --data configs/dataset.yaml --epochs 50 --imgsz 320 --batch 16
python -m src.models.evaluate --weights runs/detect/industrial_defect_yolov8n/weights/best.pt \
  --data configs/dataset.yaml --split val --imgsz 320 --out reports/eval_report.json
python -m src.models.export --weights runs/detect/industrial_defect_yolov8n/weights/best.pt --imgsz 320
cp runs/detect/industrial_defect_yolov8n/weights/best.onnx models/industrial_yolov8n.onnx
python -m src.models.benchmark --onnx models/industrial_yolov8n.onnx --json reports/benchmark.json
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Dataset config not found` | Wrong `--data` path | Run from the repo root |
| Training ends with 0 validation images | `data/processed` holds the synthetic build | Re-run `python -m src.data.neu_det` |
| `unknown class 'rolled-in_scale'` | Non-NEU-DET XML with a new spelling | Extend `normalize_class_name()` in `src/data/neu_det.py` |
| Evaluation mAP is ~0 on `val` | Evaluating the synthetic-trained model on real data (or vice versa) | Retrain on the same distribution you evaluate |
| `onnx.checker` failure after export | Unsupported opset for the torch version | Try `--opset 17` and update torch/ultralytics |
