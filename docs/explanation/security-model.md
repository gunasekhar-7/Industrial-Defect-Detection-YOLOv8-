# Security model

An image-upload inference endpoint accepts bytes from the network and hands
them to a decoder. That is the whole threat model: **the file is hostile, the
metadata is hostile, and the volume is hostile.** This page explains what each
guard stops, what it does not, and where the remaining risk sits.

Implementation: [`src/api/security.py`](../../src/api/security.py),
[`src/vision/image_io.py`](../../src/vision/image_io.py). Tests:
`tests/test_api.py`, `tests/test_image_io.py`.

---

## Threats and answers

| Threat | Without a guard | Guard | Evidence |
|---|---|---|---|
| Content spoofing (`.exe` renamed `.jpg`, wrong `Content-Type`) | Decoder crash or unexpected parser path | Magic-byte check on the first 16 bytes → **415** | `verify_image_magic_bytes`, spoofing test |
| Decompression bomb (tiny file, gigapixel image) | Memory exhaustion / OOM kill | Dimensions read from the Pillow header **before** pixel allocation, 32–4096 px → **400** | `test_oversize_rejected_before_pixel_decode` asserts `cv2.imdecode` is never called |
| Oversized upload | Multi-GB temp file spooled to disk by the form parser | `Content-Length` gate → **413** before parsing; 1 MiB-chunk streaming cap → **413** during read | streaming spy test asserts bounded bytes read |
| Request smuggling via chunked bodies | Unbounded body with no declared length | Middleware counts streamed bytes to the same budget and cuts the connection | `BodySizeLimitMiddleware` |
| Denial of service by request rate | One client starves everyone | slowapi 60 req/min per IP → **429** | rate-limit test |
| Cross-origin data theft from a browser | Any site reads responses | Explicit CORS allow-list, no wildcard, no credentials | CORS config test |
| Event-loop starvation | Inference blocks all other requests | Sync handlers in the threadpool | `test_detect_handlers_do_not_block_event_loop` |
| Silent wrong answers when the model is absent | 200 with empty detections, indistinguishable from "no defects" | **503** `model not loaded` | health/detection 503 tests |
| Zip-slip during dataset conversion | Writing outside the target directory | `_safe_extract` validates member paths | `tests/test_neu_det_download.py` |

The ordering matters: `Content-Length` → magic bytes → dimensions → decode.
Work is done in increasing cost order, so a malicious upload is rejected by the
cheapest possible check.

---

## Why magic bytes, not `Content-Type`

The client controls `Content-Type` and the filename. Both are hints, and both
are trivially forged. The first 16 bytes of the payload are also
attacker-controlled, but they must *be* a real JPEG/PNG/BMP signature for the
downstream decoder to do anything useful — checking them constrains the input
to the three decoders that are actually supported.

Signature table:

| Magic | MIME |
|---|---|
| `FF D8 FF` | `image/jpeg` |
| `89 50 4E 47 0D 0A 1A 0A` | `image/png` |
| `42 4D` (`BM`) | `image/bmp` |

A file can still be a valid header followed by garbage; that is caught by the
Pillow header parse (400) and by `cv2.imdecode` returning `None` (400).

---

## Dimensions from the header, not the pixels

Pillow's `Image.open()` is lazy: it parses the header and exposes `.size`
without allocating the pixel buffer. Comparing `width`/`height` there costs
microseconds. Decoding first and checking after would mean allocating exactly
the memory the attack wants you to allocate — the check would be pointless.

The declared dimension cap is 4096 px, which bounds a decoded RGB buffer at
4096 × 4096 × 3 ≈ 50 MB per request. Combined with the 15 MB byte cap and the
threadpool, that bounds worst-case concurrent memory for the decode stage.

---

## The whole-request budget

`sanitize_and_read_image()` caps the **image**, but Starlette parses the
multipart form *before* the endpoint runs and spools the body to a temporary
file. A 5 GB body would therefore reach disk before any endpoint-level check.

`BodySizeLimitMiddleware` closes that gap:

* declared `Content-Length` over `MAX_IMAGE_SIZE_MB + 1 MiB` → 413 immediately,
  no body bytes read;
* otherwise the body is counted as it streams and the connection is cut at the
  same limit, with a synthetic `http.disconnect` preventing partial chunks from
  reaching the form parser.

The 1 MiB allowance covers multipart boundaries, part headers and sibling form
fields — real framing bytes, not image data.

---

## What this model does **not** cover

| Gap | Risk | Mitigation |
|---|---|---|
| No authentication | Anyone who can reach the port can run inference | Deploy behind a gateway; the service is intended for a trusted network |
| No TLS | Credentials/data exposed on the wire | Terminate TLS at the proxy |
| Rate limiting is per-IP | A distributed client pool can exceed the limit; a shared proxy IP throttles everyone | Put per-key quotas at the gateway |
| No antivirus / content policy | A well-formed image can still contain arbitrary metadata (EXIF) | Metadata is not parsed or persisted; add scanning if uploads are stored |
| No request logging / audit | Abuse is hard to reconstruct | Add structured logs and `/metrics` — see [future scope](../future-scope.md) |
| Model is not adversarially robust | Adversarial patches can force false PASS | Out of scope for an AOI demo; note it before any safety-critical use |
| Denial by compute | Slow, valid requests still consume CPU | Concurrency and rate limits bound it; add a queue if latency SLOs must hold |
| Dependency supply chain | A malicious pinned package | Exact pins + CI; add hash-pinned requirements and `pip-audit` as hardening |

The last row is deliberately honest: "hardened input path" describes the
upload surface, not a full security posture. A production deployment also needs
auth, TLS, audit logging and dependency scanning, none of which are pretended
to exist here.

---

## Failure behaviour is part of the model

* **No model** → 503 forever, never a 200 with zero defects. A false PASS is
  worse than an error, because an inspection line acts on it.
* **Oversized body** → 413, with the limit named in the message.
* **Unknown class id from the model** → `unknown_<id>` in JSON, not a crash.
* **JPEG encode failure** on `/v1/detect/annotated` → 500 with an empty body.

Each is pinned by a test so a refactor cannot quietly turn an error into a
plausible-looking empty result.
