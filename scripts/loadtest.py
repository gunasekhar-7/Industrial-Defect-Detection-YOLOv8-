"""Service-level load test for /v1/detect.

Micro-benchmarks measure one request in isolation; this harness measures the
deployed surface: real HTTP, real multipart uploads, a fixed number of
concurrent clients, and the percentiles an operator is actually paged on.

    python scripts/loadtest.py --image data/processed/images/val/patches_1.jpg \
        --concurrency 8 --requests 200 --json reports/loadtest.json

The API rate-limits per client IP (60/minute by default), which a single
load-generating host would trip immediately. Run the service with the limit
raised for the duration of the test, e.g.

    RATE_LIMIT_DEFAULT=100000/minute uvicorn src.api.app:app --port 8000

and record that override next to the numbers it produced.
"""

import argparse
import json
import platform
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _multipart(image_bytes: bytes, field: str = "file", filename: str = "frame.jpg") -> Tuple[bytes, str]:
    boundary = "----defectloadtestboundary"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            image_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _one_request(url: str, body: bytes, content_type: str, timeout: float) -> Tuple[int, float]:
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    request.add_header("Connection", "keep-alive")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        exc.read()
        status = exc.code
    return status, (time.perf_counter() - start) * 1000.0


def _percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(quantile / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def load_test(
    url: str,
    image_bytes: bytes,
    concurrency: int = 8,
    requests: int = 200,
    warmup: int = 10,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Fire ``requests`` POSTs across ``concurrency`` clients and summarise them."""
    body, content_type = _multipart(image_bytes)

    for _ in range(warmup):
        status, _ = _one_request(url, body, content_type, timeout)
        if status != 200:
            raise RuntimeError(f"warmup request failed with HTTP {status}")

    def worker(count: int) -> List[Tuple[int, float]]:
        return [_one_request(url, body, content_type, timeout) for _ in range(count)]

    per_worker = requests // concurrency
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = [r for batch in pool.map(worker, [per_worker] * concurrency) for r in batch]
    elapsed = time.perf_counter() - started

    latencies = [latency for status, latency in results if status == 200]
    failures = {str(status): sum(1 for s, _ in results if s == status) for status in {s for s, _ in results} if status != 200}
    return {
        "url": url,
        "hardware": f"{platform.platform()} | {platform.processor() or platform.machine()}",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "concurrency": concurrency,
        "requests": len(results),
        "wall_seconds": round(elapsed, 2),
        "throughput_rps": round(len(results) / elapsed, 1),
        "image_kb": round(len(image_bytes) / 1024, 1),
        "ok": len(latencies),
        "failures": failures,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1),
            "p50": round(_percentile(latencies, 50), 1),
            "p95": round(_percentile(latencies, 95), 1),
            "p99": round(_percentile(latencies, 99), 1),
            "max": round(max(latencies), 1),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Concurrent load test for the detection API")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/detect")
    parser.add_argument("--image", required=True, help="JPEG used for every request")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json", default=None, help="Write the report to this path")
    args = parser.parse_args(argv)

    report = load_test(
        url=args.url,
        image_bytes=Path(args.image).read_bytes(),
        concurrency=args.concurrency,
        requests=args.requests,
        warmup=args.warmup,
    )
    print(json.dumps(report, indent=2))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
