# Performance

Why this project quotes two latency numbers, what the service can actually
serve, and how the numbers were obtained. Sources:
[`reports/benchmark.json`](../../reports/benchmark.json),
[`reports/benchmark_spinning.json`](../../reports/benchmark_spinning.json),
[`reports/loadtest.json`](../../reports/loadtest.json),
[`reports/quantization_tradeoff.json`](../../reports/quantization_tradeoff.json).

---

## The correction that matters

An earlier version of this project reported **8.2 ms median inference**. That
number was real, and it was misleading: it only ever held as a short burst.

While building the HTTP load test, the API reported ~190 ms per request for the
same graph. Measuring in-process reproduced it: the *same 200-run protocol* in
one process produced 12 ms early and 240 ms once the run continued. Nothing in
the application code differed between those measurements.

The cause is ONNX Runtime's default **busy-wait intra-op thread pool**. Threads
spin between tasks to cut wake-up latency; on a hybrid CPU (fast P-cores plus
slow E-cores) the spinning threads compete with the graph for exactly the cores
running the convolutions. Whether you measure the boost window or the steady
state decides whether the model looks 8 ms or 240 ms.

Switching off spinning (`session.intra_op.allow_spinning = "0"`) makes latency
stable at the cost of the burst best case:

| Metric | Spinning off (shipped) | Spinning on (ORT default) |
|---|---|---|
| Median inference | **36.08 ms** | 244.99 ms |
| p95 | **45.80 ms** | 296.06 ms |
| p99 | 59.16 ms | 334.81 ms |
| Mean end-to-end (pre/infer/NMS) | 40.97 ms | 242.00 ms |
| Throughput | **24.4 FPS** | 4.1 FPS |

Same machine, same model, same 50 warmups + 200 runs; `--allow-spinning`
reproduces the second column. The README quotes the stable column because the
stable column is what a service holds.

**Generalisable lesson:** for a service, the tail and the variance are the
product. A micro-benchmark shorter than the throttling/contention timescale
measures the machine's enthusiasm, not the deployed latency. That is why this
repo ships a burst number *and* an HTTP load test, and why both are artifacts.

---

## Service-level behaviour

[`scripts/loadtest.py`](../../scripts/loadtest.py) sends real 21 KB multipart
JPEGs to `/v1/detect` from concurrent clients, single uvicorn worker:

| Clients | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| 1 | 14.0 req/s | 69.1 ms | 89.5 ms | 93.2 ms |
| 2 | 20.8 req/s | 91.3 ms | 115.2 ms | 122.4 ms |
| 4 | 26.7 req/s | 135.3 ms | 247.5 ms | 269.9 ms |
| 8 | 25.8 req/s | 297.8 ms | 460.6 ms | 555.7 ms |

All 460 requests returned 200.

What to take from it:

* **The worker saturates near 27 req/s.** Beyond four clients, added
  concurrency queues; throughput is flat while p95 triples. More clients do not
  buy more throughput.
* **Per-request cost at concurrency 1 is ~69 ms**, well above the 41 ms
  in-process end-to-end. The difference is HTTP: multipart parsing, form
  spooling, pydantic validation, JSON serialisation, socket handling. This is
  why the in-process number is not the number to quote for capacity planning.
* **Scale out, not up.** ONNX Runtime already uses all cores inside the
  process; a second uvicorn worker in the same container competes for the same
  CPU. Replicas behind a proxy are the scaling axis.

---

## Case study: INT8 is not automatically a win

Dynamic INT8 quantization is the standard "make it smaller and faster" move.
Measured on this machine, it is a clear loss:

| Model | Size | Median | p95 | Same classes |
|---|---|---|---|---|
| FP32 | 12.65 MB | 19.46 ms | 36.91 ms | — |
| INT8 dynamic | 3.36 MB | 291.60 ms | 417.01 ms | 12 / 24 frames |

It is 3.8× smaller, ~15× slower, and changes the predicted defect classes on
half the sampled frames. ONNX Runtime's *dynamic* path dequantizes convolution
weights back to FP32 before each convolution, so a convolution-heavy detector
pays conversion overhead and gets no integer kernels in return. The size win is
real; the speed win is not.

Therefore the API serves FP32 and the INT8 graph stays in `models/` as
evidence. The follow-up that could genuinely win is **static QDQ quantization
with a calibration set**, which lets kernels consume int8 directly — listed in
[future scope](../future-scope.md), not claimed here.

(The FP32 median in that table is lower than the 36 ms headline because the
comparison uses 24 real 200×200 frames over 50 iterations, not a 640-px
synthetic frame over 200. Only the same-session ratio is meaningful.)

---

## What is not measured

| Missing | Consequence |
|---|---|
| Sustained multi-hour run | Thermal/steady-state behaviour beyond a 200-run window is uncharacterised |
| GPU numbers | No GPU was available; DirectML/CUDA providers are selected automatically if present but never benchmarked here |
| Concurrency beyond 8 clients | The curve is flat by then, but no queueing model is validated |
| Memory footprint under load | Not instrumented; a batch endpoint would need it |
| Cold start | Model load time is not measured (the container healthcheck has a 15 s start period) |

Absolute milliseconds are hardware-specific. The portable part of this page is
the *method*: measure both regimes, measure over HTTP, and publish the
comparison artifacts.
