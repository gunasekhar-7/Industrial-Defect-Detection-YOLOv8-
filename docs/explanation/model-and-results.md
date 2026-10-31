# Model and results

What the model is, what the numbers mean, and where it is weak. Every figure
here traces to [`reports/eval_report.json`](../../reports/eval_report.json).

---

## Dataset

**NEU-DET** — 1,800 grayscale 200×200 images of hot-rolled steel, six surface
defect classes, 300 images each, VOC XML annotations. Converted to YOLO layout
with a **seeded, class-stratified 70/15/15 split** (seed 42), which yields
1,260 train / 270 validation / 270 test images.

Two properties of this dataset shape everything downstream:

* **It is small.** 1,260 training images for six fine-grained texture classes
  is a few hundred examples per class. Capacity is not the binding constraint;
  data is.
* **Classes overlap visually.** Crazing is a fine crack network, rolled-in
  scale is a pressed-in oxide band — both sit *inside* the texture rather than
  on top of it, so they do not look like discrete objects the way `patches`
  does. This is why the per-class spread is so wide.

---

## Training configuration (as executed)

| Setting | Value |
|---|---|
| Base weights | `yolov8n.pt` (3.2 M params) |
| Epochs | 50 |
| Input size | 320 |
| Batch | 16 |
| Device | CPU |
| Optimizer | auto (Ultralytics default) |
| Tracking | MLflow, experiment `industrial-defect` |

**Why 320 and not 640?** The source images are 200×200. Training at 640
upscales them 3.2× — no new information, 4× the pixels per epoch, paid every
epoch on a CPU. The same reasoning applies at inference: the graph's input is
`1×3×320×320`, so serving does not pay for pixels the data never had.

**Why yolov8n?** It is the smallest YOLOv8 variant and the only one that trains
50 epochs to completion on a CPU in a reasonable time. The honest framing is
"small model, small data, CPU budget" — not "state of the art".

---

## Results (validation split, 270 images)

| Metric | Value | Reads as |
|---|---|---|
| Precision | 0.6578 | ~34% of reported boxes are false positives |
| Recall | 0.6982 | ~30% of real defects are missed |
| **mAP@50** | **0.7209** | Detections are roughly in the right place when IoU ≥ 0.5 |
| mAP@50-95 | 0.4191 | Boxes are loose — accuracy collapses as the IoU bar rises |

Per class (mAP@50-95):

| Class | Score | Interpretation |
|---|---|---|
| `patches` | 0.6481 | Large, high-contrast regions — the easy case |
| `scratches` | 0.5334 | Elongated, distinct from texture |
| `pitted_surface` | 0.5267 | Localized dark spots |
| `inclusion` | 0.3949 | Embedded material, variable shape |
| `rolled_in_scale` | 0.2570 | Diffuse banding; over-segmented into many boxes |
| `crazing` | 0.1546 | Fine crack network; merged into one oversized box |

The gap between mAP@50 (0.72) and mAP@50-95 (0.42) is the single most
informative number here: the model **finds** defects far better than it
**localizes** them. That is expected for texture-like classes where "where does
the crack end" is genuinely ambiguous, and it is visible in
[`failure_gallery.png`](../failure_gallery.png): crazing ground truth has three
tight boxes, the prediction has one huge one.

---

## How to read these numbers without fooling yourself

* **270 images is a small validation set.** A single run has no error bars; a
  ±2-point difference between two models on this split could be noise. The
  honest fix is 3 seeds with mean ± std — listed in
  [future scope](../future-scope.md).
* **mAP@50-95 is the metric that resists box inflation.** Reporting only
  mAP@50 would flatter a model that draws generous boxes.
* **Never quote the synthetic dataset.** The generator stamps defects with
  exact boxes; a model trained on it scores near 1.0 on synthetic data and is
  meaningless on steel. CI uses it as a fixture precisely because it is easy.
* **Precision/recall here are at IoU 0.5 and the configured confidence
  threshold (0.25).** Changing the threshold moves both — for an inspection
  line, recall is usually the one you trade for.

---

## Failure modes seen in practice

| Mode | Symptom | Likely cause |
|---|---|---|
| Merging | One box covers a whole crazing region | Small receptive field at 320 px; crack pixels are thin |
| Over-segmentation | A single rolled-in scale band becomes 4 boxes | NMS IoU 0.45 splits adjacent fragments; no post-hoc merging |
| Class confusion | `inclusion` ↔ `pitted_surface` | Both are small dark regions in grey texture |
| Truncation | Boxes clipped at image edges | Letterboxing + int rounding at the boundary |

These are data- and resolution-limited, not code bugs, which is why the roadmap
puts a larger input size and augmentation first, and post-processing (box
merging, per-class thresholds) second.

---

## Why this is published as-is

The temptation in portfolio ML work is to train until a number looks good, then
publish only that number. The alternative — publish the run you actually did,
with the weak classes named, the per-class breakdown, and a gallery of failures
— is more useful to a reviewer because it is checkable. `reports/eval_report.json`
records the split, image count, hardware and executed training arguments, so
"is this real?" has a one-command answer:

```bash
python -m src.models.evaluate --weights runs/detect/industrial_defect_yolov8n/weights/best.pt \
  --data configs/dataset.yaml --split val --imgsz 320 --out reports/eval_report.json
```

The roadmap in [future scope](../future-scope.md) describes what would actually
move these numbers, in the order that maximises gain per hour.
