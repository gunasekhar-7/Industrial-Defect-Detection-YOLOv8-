# How-to: reproduce the benchmarks

Task: rebuild every performance number in the README on your own hardware.
Each section is independent.

Two rules that make the results comparable:

1. **Run from the repository root** — model paths are relative.
2. **Close other CPU-heavy work.** These are CPU measurements on a shared
   machine; a video call or a second training run changes the answer.

---

## 1. In-process latency (FP32, the served graph)

```bash
python -m src.models.benchmark --onnx models/industrial_yolov8n.onnx \
  --warmup 50 --runs 200 --json reports/benchmark.json
```

```
 Pure Inference Median Latency : 36.08 ms
 Pure Inference Mean Latency   : 32.78 ms
 95th Percentile (p95) Latency : 45.80 ms
 99th Percentile (p99) Latency : 59.16 ms
 End-to-End Latency (w/ NMS)   : 40.97 ms
 Throughput (FPS)              : 24.4 FPS
```

Flags: `--warmup` (50), `--runs` (200), `--json` (report path),
`--allow-spinning` (see below).

The report records the run so the number is auditable:

```json
{ "model": "industrial_yolov8n.onnx", "model_input": [1, 3, 320, 320],
  "provider": "CPUExecutionProvider", "hardware": "…",
  "warmup_runs": 50, "benchmark_runs": 200, "allow_spinning": false,
  "timestamp": "…", "median_inference_ms": 36.08, "…": 0 }
```

### The spinning comparison

```bash
python -m src.models.benchmark --onnx models/industrial_yolov8n.onnx \
  --warmup 50 --runs 200 --allow-spinning --json reports/benchmark_spinning.json
```

```
 Pure Inference Median Latency : 244.99 ms
 95th Percentile (p95) Latency : 296.06 ms
```

Same machine, same model, same protocol; the only difference is
`session.intra_op.allow_spinning`. ONNX Runtime's default busy-wait thread pool
fights the graph for cores on hybrid CPUs, and a 200-run protocol is long
enough to expose it. The detector therefore ships with spinning disabled. If
your numbers are wildly unstable (variance larger than the median), re-run with
and without the flag before blaming the model — see
[Performance](../explanation/performance.md).

> A 20-run "quick check" will happily report 11 ms. It is measuring the boost
> window, not the service.

---

## 2. FP32 vs INT8 trade-off

```bash
python -m src.models.quantize --onnx models/industrial_yolov8n.onnx \
  --compare data/processed/images/val
```

This writes `<stem>.int8.onnx` next to the FP32 graph (3.4 MB vs 12.7 MB), then
runs both models back-to-back over up to 24 validation frames and writes
`reports/quantization_tradeoff.json`.

```json
"models": {
  "fp32": {"size_mb": 12.65, "median_ms": 19.46, "p95_ms": 36.91},
  "int8": {"size_mb": 3.36,  "median_ms": 291.6, "p95_ms": 417.01}
},
"detection_agreement": {"frames_compared": 24, "identical_classes": 12},
"int8_speedup_median": 0.07
```

Interpretation: `0.07` means INT8 took ~14× longer; agreement `12/24` means it
changed the predicted classes on half the frames. Both models must be measured
in the **same process** — absolute latency drifts between sessions, only the
ratio is meaningful.

Flags: `--out` (output graph), `--per-channel` (slower to prepare, often more
accurate), `--report`.

---

## 3. Service-level HTTP load test

The load generator hammers `/v1/detect` with real multipart uploads from N
concurrent clients — this is the number an operator cares about.

```bash
RATE_LIMIT_DEFAULT=100000/minute uvicorn src.api.app:app --port 8000 &
python scripts/loadtest.py \
  --url http://127.0.0.1:8000/v1/detect \
  --image data/processed/images/val/crazing_109.jpg \
  --concurrency 8 --requests 240 --json reports/loadtest.json
```

Why the rate-limit override: the API limits 60 requests/minute **per client
IP**, and one load host looks like one abusive client. Raising it for the run
keeps the test measuring inference, not the limiter. Say so when you quote the
result.

```json
{ "concurrency": 8, "requests": 240, "throughput_rps": 25.8,
  "ok": 240, "failures": {},
  "latency_ms": {"mean": 305.8, "p50": 297.8, "p95": 460.6, "p99": 555.7} }
```

Sweep the concurrency to find the saturation point (the committed
`reports/loadtest.json` uses 1/2/4/8):

| Clients | Throughput | p50 | p95 |
|---|---|---|---|
| 1 | 14.0 req/s | 69.1 ms | 89.5 ms |
| 2 | 20.8 req/s | 91.3 ms | 115.2 ms |
| 4 | 26.7 req/s | 135.3 ms | 247.5 ms |
| 8 | 25.8 req/s | 297.8 ms | 460.6 ms |

Flags: `--url`, `--image`, `--concurrency`, `--requests`, `--warmup` (10),
`--json`. The harness fails fast if a warmup request is not `200`, so a broken
server is reported instead of producing a beautiful latency table for errors.

---

## 4. Failure gallery

```bash
python scripts/make_gallery.py
```

Writes `docs/failure_gallery.png`: the two lowest-scoring classes from
`reports/eval_report.json`, ground truth on the left and predictions on the
right. Use `--classes crazing patches` to pick classes explicitly, or
`--onnx` / `--images` / `--labels` if your paths differ.

This is the fastest way to see *how* the model is wrong: merged regions for
crazing, over-segmentation for rolled-in scale, both visible without reading a
confusion matrix.

---

## What a valid result looks like

| Check | Expectation |
|---|---|
| Report files exist and have a fresh `timestamp` | Yes — a stale timestamp means you ran the wrong thing |
| `hardware` string matches your machine | Yes |
| p95 ≥ median | Always; if not, the run is broken |
| p95 / median ratio | Under ~2× for the spinning-off benchmark; much worse with spinning |
| Load test `failures` | `{}` — anything else needs investigation before quoting a rate |

Absolute milliseconds are hardware-specific. If your laptop reports 60 ms where
this README says 36 ms, the finding is "different machine", not "wrong code" —
quote the ratio and the shape of the curve, not the raw number.
