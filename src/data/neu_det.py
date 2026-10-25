"""NEU-DET acquisition and Pascal-VOC -> YOLO conversion.

The official NEU Surface Defect Database detection archive (NEU-DET,
1,800 grayscale 200x200 images, 6 classes, VOC XML annotations) is
distributed via the authors' Google Drive / Baidu-Pan links:

    https://faculty.neu.edu.cn/songkc/en/zdylm/263265/

Google Drive serves a virus-scan interstitial for this file; the download
flow below follows that two-step form. When automation fails (quota, link
rotation, offline CI), download manually from the official page, place the
archive at data/raw/NEU-DET.zip, and run conversion — convert_neu_det
accepts a local archive or an already-extracted directory.

This module is NEVER used by CI: tests use tiny hand-made fixtures and
the synthetic generator (src.data.generator) provides the CI dataset.
"""

import argparse
import random
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

from src.data.generator import CLASS_MAP
from src.data.validator import DatasetValidator

# Canonical class names (configs/dataset.yaml order). Raw XML names are
# normalized with normalize_class_name() before lookup — NEU-DET ships
# "rolled-in_scale" (hyphen) and similar variants.
CLASS_NAMES: Tuple[str, ...] = tuple(CLASS_MAP[i] for i in range(len(CLASS_MAP)))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

OFFICIAL_PAGE = "https://faculty.neu.edu.cn/songkc/en/zdylm/263265/"
# Google Drive file id of NEU-DET.zip as linked from OFFICIAL_PAGE (verified 2026-10-25).
GDRIVE_FILE_ID = "1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k"
GDRIVE_UC_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
GDRIVE_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={GDRIVE_FILE_ID}&export=download&confirm=t"
)
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) industrial-defect-dataset-fetcher"

_UUID_RE = re.compile(rb'name="uuid" value="([^"]+)"')

MANUAL_INSTRUCTIONS = """\
Automatic NEU-DET download failed ({reason}).

Manual download (takes ~1 minute):
  1. Open the official page: {page}
  2. Follow "For NEU-DET: Google-Drive" (or Baidu-Pan, code: pmqx)
  3. Save the archive as: {dest}/NEU-DET.zip
  4. Re-run: python -m src.data.neu_det

convert_neu_det() also accepts an extracted NEU-DET directory directly.
"""


def normalize_class_name(name: str) -> str:
    """Fold a raw annotation class name onto the canonical class names.

    Lowercases, maps any run of non-alphanumerics to a single underscore,
    and strips edge underscores: "rolled-in_scale" -> "rolled_in_scale".
    """
    folded = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return folded.strip("_")


# class id lookup keyed by normalized name
CLASS_TO_ID: Dict[str, int] = {
    normalize_class_name(name): cls_id for cls_id, name in CLASS_MAP.items()
}


def _manual_error(reason: str, dest: Path) -> RuntimeError:
    return RuntimeError(
        MANUAL_INSTRUCTIONS.format(reason=reason, dest=dest, page=OFFICIAL_PAGE)
    )


def download_neu_det(dest: str | Path = "data/raw", url: str | None = None) -> Path:
    """Download NEU-DET.zip into ``dest`` and return the archive path.

    Uses ``url`` when given (any direct-zip URL), otherwise the official
    Google Drive link. If ``dest/NEU-DET.zip`` already exists and is a
    valid zip, no network call is made. All failures raise RuntimeError
    carrying manual-download instructions; partial files are removed.
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    target = dest_path / "NEU-DET.zip"

    if target.exists() and zipfile.is_zipfile(target):
        return target

    effective_url = url or GDRIVE_UC_URL

    try:
        if url is None:
            _fetch_gdrive(target)
        else:
            _fetch_direct(effective_url, target)
    except Exception as exc:  # noqa: BLE001 - normalized into manual instructions
        target.unlink(missing_ok=True)
        (dest_path / "NEU-DET.zip.part").unlink(missing_ok=True)
        raise _manual_error(f"{type(exc).__name__}: {exc}", dest_path) from exc

    if not zipfile.is_zipfile(target):
        target.unlink(missing_ok=True)
        raise _manual_error("downloaded file is not a ZIP archive", dest_path)

    return target


def _fetch_direct(url: str, target: Path) -> None:
    """Stream a direct-zip URL to ``target`` (via a .part file)."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    part = target.with_suffix(".zip.part")
    with urllib.request.urlopen(request, timeout=120) as resp, open(part, "wb") as out:  # noqa: S310 (trusted operator-supplied URL)
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    part.replace(target)


def _fetch_gdrive(target: Path) -> None:
    """Two-step Google Drive download: clear the virus-scan interstitial,
    then stream the file from drive.usercontent.google.com."""
    request = urllib.request.Request(GDRIVE_UC_URL, headers={"User-Agent": _USER_AGENT})
    part = target.with_suffix(".zip.part")
    cookie_header = ""

    with urllib.request.urlopen(request, timeout=120) as resp:  # noqa: S310 (fixed official host)
        content_type = resp.headers.get("Content-Type", "")
        set_cookies = resp.headers.get_all("Set-Cookie") or []
        if set_cookies:
            cookie_header = "; ".join(c.split(";", 1)[0] for c in set_cookies)
        if "html" not in content_type.lower():
            # Small enough that Drive served the file directly.
            with open(part, "wb") as out:
                shutil.copyfileobj(resp, out, length=1024 * 1024)
            part.replace(target)
            return
        page = resp.read()

    match = _UUID_RE.search(page)
    if match is None:
        title = re.search(rb"<title>([^<]*)</title>", page)
        shown = title.group(1).decode("utf-8", "replace") if title else "unknown page"
        raise RuntimeError(f"unexpected Google Drive interstitial ({shown})")

    uuid = match.group(1).decode("ascii")
    headers = {"User-Agent": _USER_AGENT}
    if cookie_header:
        headers["Cookie"] = cookie_header
    follow = urllib.request.Request(  # noqa: S310 (fixed official host)
        f"{GDRIVE_DOWNLOAD_URL}&uuid={uuid}", headers=headers
    )
    with urllib.request.urlopen(follow, timeout=300) as resp, open(part, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    part.replace(target)


def _safe_extract(zf: zipfile.ZipFile, into: Path) -> None:
    """Extract with a zip-slip guard (members must stay under ``into``)."""
    root = into.resolve()
    for member in zf.infolist():
        name = member.filename
        if not name:
            continue
        dest = (into / name).resolve()
        if not str(dest).startswith(str(root)):
            raise ValueError(f"unsafe path in archive: {name!r}")
    zf.extractall(into)


def _parse_voc_objects(xml_path: Path) -> List[Tuple[str, float, float, float, float]]:
    """Return [(raw_class, xmin, ymin, xmax, ymax), ...] from a VOC XML."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    objects = []
    for obj in tree.getroot().findall("object"):
        name_el = obj.find("name")
        box = obj.find("bndbox")
        if name_el is None or box is None or name_el.text is None:
            raise ValueError(f"malformed object in {xml_path}")
        objects.append(
            (
                name_el.text.strip(),
                float(box.findtext("xmin", "0")),
                float(box.findtext("ymin", "0")),
                float(box.findtext("xmax", "0")),
                float(box.findtext("ymax", "0")),
            )
        )
    if not objects:
        raise ValueError(f"no objects in {xml_path}")
    return objects


def _allocate_split_counts(n: int, splits: Sequence[float]) -> List[int]:
    """Per-item split counts summing to n, each >=0; caller guarantees
    len(splits)>=1 and sum(splits)==1."""
    raw = [n * f for f in splits]
    counts = [int(r) for r in raw]
    remainder = n - sum(counts)
    # Largest-remainder: hand the leftover slots to the biggest fractions.
    by_frac = sorted(range(len(splits)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in by_frac[:remainder]:
        counts[i] += 1
    return counts


def convert_neu_det(
    archive_or_dir: str | Path,
    out: str | Path = "data/processed",
    splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> None:
    """Convert a NEU-DET archive (or extracted directory) to YOLO layout.

    Writes ``out/images/{train,val,test}`` and ``out/labels/{train,val,test}``
    with a seeded, class-stratified split, then validates every split with
    DatasetValidator and raises if any split is invalid. Existing
    images/labels under ``out`` are replaced (after a successful parse),
    so a previous synthetic build cannot mix into the real dataset.
    """
    if abs(sum(splits) - 1.0) > 1e-9:
        raise ValueError(f"splits must sum to 1.0, got {splits}")
    if len(splits) < 2:
        raise ValueError("need at least two splits")

    out_path = Path(out)
    src_path = Path(archive_or_dir)
    split_names = ["train", "val", "test"][: len(splits)]
    if len(split_names) < len(splits):
        split_names = [f"split{i}" for i in range(len(splits))]

    tmp_extract: tempfile.TemporaryDirectory[str] | None = None
    try:
        if src_path.is_file():
            if not zipfile.is_zipfile(src_path):
                raise ValueError(f"not a ZIP archive: {src_path}")
            tmp_extract = tempfile.TemporaryDirectory(prefix="neu_det_")
            with zipfile.ZipFile(src_path) as zf:
                _safe_extract(zf, Path(tmp_extract.name))
            scan_root = Path(tmp_extract.name)
        elif src_path.is_dir():
            scan_root = src_path
        else:
            raise FileNotFoundError(f"NEU-DET source not found: {src_path}")

        images = sorted(
            p for p in scan_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        xml_index: Dict[str, Path] = {}
        for xml in scan_root.rglob("*.xml"):
            if xml.stem in xml_index:
                raise ValueError(f"duplicate annotation stem: {xml.stem}")
            xml_index[xml.stem] = xml
        if not images:
            raise ValueError(f"no images found under {scan_root}")
        if not xml_index:
            raise ValueError(
                f"no VOC XML annotations found under {scan_root} — expected NEU-DET layout"
            )

        # ---- Parse everything BEFORE touching out/ (fail fast, no partial writes)
        records = []  # (stem, image_path, stratum_class_id, yolo_lines)
        missing_xml = []
        for img in images:
            xml = xml_index.get(img.stem)
            if xml is None:
                missing_xml.append(img.stem)
                continue
            objects = _parse_voc_objects(xml)

            class_ids = []
            lines = []
            for raw_name, xmin, ymin, xmax, ymax in objects:
                canonical = normalize_class_name(raw_name)
                cls_id = CLASS_TO_ID.get(canonical)
                if cls_id is None:
                    raise ValueError(
                        f"unknown class {raw_name!r} in {xml.name} "
                        f"(normalized: {canonical!r}); known: {sorted(CLASS_TO_ID)}"
                    )
                with Image.open(img) as im:
                    width, height = im.size
                xc = min(max((xmin + xmax) / 2.0 / width, 0.0), 1.0)
                yc = min(max((ymin + ymax) / 2.0 / height, 0.0), 1.0)
                bw = (xmax - xmin) / width
                bh = (ymax - ymin) / height
                bw = min(max(bw, 1e-4), 1.0)
                bh = min(max(bh, 1e-4), 1.0)
                xc = min(max(xc, bw / 2.0), 1.0 - bw / 2.0)
                yc = min(max(yc, bh / 2.0), 1.0 - bh / 2.0)
                class_ids.append(cls_id)
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            stratum = class_ids[0]
            records.append((img.stem, img, stratum, lines))

        if missing_xml:
            preview = ", ".join(missing_xml[:5])
            raise ValueError(
                f"{len(missing_xml)} image(s) have no annotation XML "
                f"(e.g. {preview}) — incomplete NEU-DET source"
            )

        # ---- Seeded, class-stratified allocation
        rng = random.Random(seed)
        by_class: Dict[int, List[int]] = {}
        for idx, rec in enumerate(records):
            by_class.setdefault(rec[2], []).append(idx)

        assignment: Dict[int, int] = {}  # record idx -> split idx
        totals = [0] * len(splits)
        for cls_id in sorted(by_class):
            group = by_class[cls_id]
            rng.shuffle(group)
            counts = _allocate_split_counts(len(group), splits)
            cursor = 0
            for split_idx, count in enumerate(counts):
                for rec_idx in group[cursor : cursor + count]:
                    assignment[rec_idx] = split_idx
                    totals[split_idx] += 1
                cursor += count

        # Tiny datasets (e.g. 6-image fixtures): per-class allocation can
        # leave a split empty — rebalance from the largest split so every
        # split is non-empty whenever there are enough images.
        if len(splits) >= 2:
            targets = [len(records) * f for f in splits]
            for empty_idx, total in enumerate(totals):
                if total > 0:
                    continue
                donor = max(range(len(splits)), key=lambda i: (totals[i] - targets[i], totals[i]))
                if totals[donor] <= 1:
                    continue
                movable = [i for i, s in assignment.items() if s == donor]
                rec_idx = movable[-1]
                assignment[rec_idx] = empty_idx
                totals[donor] -= 1
                totals[empty_idx] += 1

        # ---- Write (replaces any previous build under out/)
        for split_name in split_names:
            (out_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (out_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        for stale_root in ("images", "labels"):
            for split_name in split_names:
                for old in (out_path / stale_root / split_name).iterdir():
                    old.unlink()

        for rec_idx, (stem, img, _stratum, lines) in enumerate(records):
            split_name = split_names[assignment[rec_idx]]
            shutil.copy2(img, out_path / "images" / split_name / f"{stem}{img.suffix.lower()}")
            label_path = out_path / "labels" / split_name / f"{stem}.txt"
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        if tmp_extract is not None:
            tmp_extract.cleanup()

    # ---- Validate every split; raise on any integrity failure
    validator = DatasetValidator(out_path)
    for split_name in split_names:
        report = validator.validate_split(split_name)
        if not report.is_valid:
            raise RuntimeError(
                f"converted dataset failed validation for split {split_name!r}: "
                f"corrupted={len(report.corrupted_images)} "
                f"oob={len(report.out_of_bounds_boxes)} "
                f"orphaned={len(report.orphaned_labels)}"
            )

    distribution: Dict[int, int] = {}
    for _stem, _img, stratum, _lines in records:
        distribution[stratum] = distribution.get(stratum, 0) + 1
    print(
        f"Converted NEU-DET -> {out_path}: "
        + ", ".join(f"{n}={t}" for n, t in zip(split_names, totals))
        + f"; classes={len(distribution)}; images={len(records)}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download (official Google Drive) and convert NEU-DET to YOLO format"
    )
    parser.add_argument("--dest", default="data/raw", help="where to cache NEU-DET.zip")
    parser.add_argument("--out", default="data/processed", help="YOLO dataset output root")
    parser.add_argument("--archive", default=None, help="use a local archive/dir instead of downloading")
    parser.add_argument("--seed", type=int, default=42, help="split seed")
    args = parser.parse_args(argv)

    source = Path(args.archive) if args.archive else download_neu_det(args.dest)
    convert_neu_det(source, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
