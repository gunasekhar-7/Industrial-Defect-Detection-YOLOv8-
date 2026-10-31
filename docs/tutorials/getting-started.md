# Tutorial: your first inspection

This lesson takes you from a fresh clone to a working defect detector on your
own machine. It uses a **synthetic** dataset, so no download, no account, and
no GPU. Budget 10–15 minutes, most of it dependency installation.

By the end you will have:

1. a generated dataset in `data/processed/`,
2. the test suite green,
3. the API answering `/health` and detecting a defect,
4. the dashboard open in a browser.

You do **not** need a trained model for steps 1 and 2. Steps 3–4 need
`models/industrial_yolov8n.onnx`; if your clone has it (the repository ships
one), you are fine. The [training guide](../how-to/train-and-evaluate.md) shows
how to produce your own.

---

## 1. Install

Requires **Python 3.11**.

```bash
git clone https://github.com/gunasekhar-7/Industrial-Defect-Detection-YOLOv8-.git
cd Industrial-Defect-Detection-YOLOv8-

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

`requirements.txt` is the runtime dependency set (API + dashboard, no
PyTorch). `requirements-dev.txt` adds pytest, coverage, ruff and mypy.

Check the install:

```bash
python -c "import fastapi, onnxruntime, cv2, streamlit; print('runtime ok')"
```

If that line prints `runtime ok`, everything you need for this tutorial is
present. `cv2` here is the headless OpenCV build — it decodes images but never
opens a window, which is what lets the Docker image drop GUI libraries.

---

## 2. Generate a dataset

```bash
python -m src.data.generator
```

Expected output:

```
Generated dataset at data/processed: train=240, val=50, test=50
```

You now have the YOLO layout the rest of the project expects:

```
data/processed/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt
```

The generator draws brushed-metal textures and stamps six procedural defect
types with exact bounding boxes, so labels are perfect by construction. That
makes it ideal for CI — and wrong for reporting accuracy, because a model
trained on it learns the generator, not steel. See
[Model and results](../explanation/model-and-results.md).

> **Careful:** `data/processed/` holds exactly one dataset at a time. Building
> the synthetic dataset replaces a previously converted NEU-DET build, and
> vice versa.

---

## 3. Run the tests

```bash
pytest tests/ -q
```

Expected tail:

```
122 passed in ~40s
Required test coverage of 100% reached. Total coverage: 100.00%
```

The coverage floor is the default (`addopts` in `pyproject.toml`), not an
extra flag. If you added a line of production code without a test, this
command fails — that is the intended behaviour.

---

## 4. Start the API

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status":"healthy","app_version":"1.0.0","model_loaded":true,
 "model_type":"onnx_runtime",
 "classes":["crazing","inclusion","patches","pitted_surface",
            "rolled_in_scale","scratches"]}
```

`model_loaded: true` and `model_type: onnx_runtime` mean the ONNX graph was
found and loaded. If you instead see `503` with `model_type: not_initialized`,
the path in `MODEL_PATH` does not exist — see
[Configuration](../reference/configuration.md).

Now send an image. Any of the generated test frames works:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/detect \
  -F "file=@data/processed/images/test/$(ls data/processed/images/test | head -1)"
```

```json
{
  "filename": "sample_test_0000.jpg",
  "image_width": 640,
  "image_height": 640,
  "defect_count": 2,
  "defects": [
    {"class_id": 4, "class_name": "rolled_in_scale", "confidence": 0.41,
     "box": {"x_min": 118, "y_min": 74, "x_max": 233, "y_max": 190}}
  ],
  "latency": {"preprocess_ms": 2.1, "inference_ms": 34.6,
              "postprocess_ms": 1.4, "total_ms": 38.1}
}
```

Your classes and confidences will differ — the model in the repository was
trained on real NEU-DET images, not on the synthetic frames, so it may find
nothing at all on them. That is a valid result, not a broken install; for a
visual check use a real steel image (see
[`docs/images/example-original.jpg`](../images/example-original.jpg)).

Two things to notice in the response:

* **`latency` is measured per request**, split into preprocess, inference and
  NMS. Trust that breakdown over any README table when debugging slowness.
* **Detections are per-class**, so a crazing region overlapping a patch region
  returns both boxes instead of one.

### Annotated output

To get a JPEG with boxes drawn:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/detect/annotated \
  -F "file=@docs/images/example-original.jpg" -o /tmp/annotated.jpg
```

Open `/tmp/annotated.jpg` to see the result.

---

## 5. Open the dashboard

```bash
streamlit run src/ui/dashboard.py
```

Streamlit prints a local URL (default <http://localhost:8501>). The page has
three tabs:

* **Single Part Inspection** — upload one image, get PASS/REJECT, latency
  cards and a detection table.
* **Conveyor Simulation** — steps through `data/processed/images/test/*.jpg`
  like a moving belt, inspecting each frame.
* **Batch Quality Audit** — upload many images, get a table plus a CSV
  download.

The sidebar sets the model path and the confidence/IoU thresholds. The
dashboard runs its **own** in-process ONNX session; it does not call the API
(see [Architecture](../explanation/architecture.md) for why).

---

## 6. Stop everything

`Ctrl+C` in each terminal. Nothing is written outside `data/`, `reports/`,
`runs/` and the Streamlit cache.

---

## Where to go next

* Want real metrics? [Train and evaluate on NEU-DET](../how-to/train-and-evaluate.md).
* Want to know what the numbers mean? [Model and results](../explanation/model-and-results.md).
* Curious why the README quotes two latency figures?
  [Performance](../explanation/performance.md).
* Shipping it? [Deploy with Docker](../how-to/deploy-with-docker.md).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/health` returns 503 | `MODEL_PATH` missing or unreadable | Export a model, or point `MODEL_PATH` at an existing `.onnx` |
| `422 Unprocessable Entity` on upload | Form field is not named `file` | `-F "file=@…"` |
| `415 Unsupported Media Type` | Bytes are not JPEG/PNG/BMP | Re-encode the image; the check reads magic bytes, not the filename |
| `413` on a small image | Request body over 16 MB (15 MB image cap + 1 MB multipart overhead) | Downscale the image |
| `429 Too Many Requests` | More than 60 requests/minute from one IP | Wait, or raise `RATE_LIMIT_DEFAULT` |
| Dashboard says model not found | Sidebar path wrong | Paste an absolute path to a `.onnx` file |
