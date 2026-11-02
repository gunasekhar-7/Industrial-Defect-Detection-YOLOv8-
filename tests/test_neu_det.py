"""Unit tests for NEU-DET acquisition and YOLO conversion.

No network access: fixtures are hand-made images + VOC XML written into
tmp_path; download failures are simulated by monkeypatching urlopen.
"""

import zipfile
from pathlib import Path

import pytest
from PIL import Image

from src.data.neu_det import convert_neu_det, download_neu_det, normalize_class_name

# Raw class names exactly as they appear in official NEU-DET XML files
# (note the hyphen in rolled-in_scale).
RAW_CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def make_fixture(root: Path) -> Path:
    """Build a miniature NEU-DET tree: IMAGES/ + ANNOTATIONS/ with one
    64x64 image + one single-object VOC XML per class."""
    img_dir = root / "NEU-DET" / "IMAGES"
    ann_dir = root / "NEU-DET" / "ANNOTATIONS"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    for i, cls in enumerate(RAW_CLASS_NAMES):
        stem = f"{cls}_{i}"
        Image.new("L", (64, 64), color=120).save(img_dir / f"{stem}.jpg")
        xml = f"""<annotation>
\t<folder>fixture</folder>
\t<filename>{stem}.jpg</filename>
\t<size><width>64</width><height>64</height><depth>1</depth></size>
\t<object>
\t\t<name>{cls}</name>
\t\t<bndbox><xmin>8</xmin><ymin>8</ymin><xmax>40</xmax><ymax>40</ymax></bndbox>
\t</object>
</annotation>
"""
        (ann_dir / f"{stem}.xml").write_text(xml, encoding="utf-8")
    return root


def test_normalize_class_name():
    assert normalize_class_name("rolled-in_scale") == "rolled_in_scale"
    assert normalize_class_name("pitted surface") == "pitted_surface"
    assert normalize_class_name("Crazing") == "crazing"


def test_convert_from_directory(tmp_path: Path):
    src = make_fixture(tmp_path / "src")
    out = tmp_path / "out"

    convert_neu_det(src, out)

    # Balanced splits for 6 images at (0.7, 0.15, 0.15): 4 / 1 / 1
    counts = {}
    all_stems = set()
    for split in ("train", "val", "test"):
        imgs = list((out / "images" / split).glob("*.jpg"))
        lbls = list((out / "labels" / split).glob("*.txt"))
        counts[split] = len(imgs)
        assert len(imgs) == len(lbls), f"{split}: images/labels mismatch"
        all_stems.update(p.stem for p in imgs)

        # Labels must be YOLO-normalized to [0, 1] with a valid class id
        for lbl in lbls:
            for line in lbl.read_text().splitlines():
                parts = line.split()
                assert len(parts) == 5, f"bad label line in {lbl}: {line!r}"
                cls_id = int(parts[0])
                assert 0 <= cls_id <= 5
                xc, yc, w, h = map(float, parts[1:])
                assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
                assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0

    assert counts == {"train": 4, "val": 1, "test": 1}, counts
    assert len(all_stems) == 6, f"every fixture image must be assigned once: {all_stems}"

    # The hyphenated raw class must land on id 4 (rolled_in_scale)
    labels_dir = out / "labels"
    all_lines = "\n".join(
        p.read_text() for p in labels_dir.rglob("*.txt")
    )
    assert all_lines.count("4 ") >= 1, "rolled-in_scale must map to class 4"

    # converter validates each split with DatasetValidator and raises on
    # invalid output; reaching here means all three splits validated.
    # Re-run explicitly to prove the validator agrees with the files on disk.
    from src.data.validator import DatasetValidator

    validator = DatasetValidator(out)
    for split in ("train", "val", "test"):
        report = validator.validate_split(split)
        assert report.is_valid, f"{split} invalid: {report}"
        assert report.total_images > 0


def test_convert_from_archive(tmp_path: Path):
    src = make_fixture(tmp_path / "src")
    archive = tmp_path / "NEU-DET.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src).as_posix())

    out = tmp_path / "out"
    convert_neu_det(archive, out)

    total = sum(
        len(list((out / "images" / split).glob("*.jpg")))
        for split in ("train", "val", "test")
    )
    assert total == 6


def test_convert_rejects_unknown_class(tmp_path: Path):
    src = make_fixture(tmp_path / "src")
    bad_xml = src / "NEU-DET" / "ANNOTATIONS" / "crazing_0.xml"
    bad_xml.write_text(
        bad_xml.read_text().replace("<name>crazing</name>", "<name>unobtainium</name>"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unobtainium"):
        convert_neu_det(src, tmp_path / "out")


def test_download_failure_raises_manual_instructions(tmp_path: Path, monkeypatch):
    import urllib.error
    import urllib.request

    def boom(*args, **kwargs):
        raise urllib.error.URLError("simulated network failure")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(RuntimeError) as exc:
        download_neu_det(tmp_path / "raw")

    msg = str(exc.value)
    # Manual-download fallback must point at the official page + local path.
    assert "faculty.neu.edu.cn" in msg, msg
    assert "NEU-DET.zip" in msg, msg


def test_zip_slip_member_rejected(tmp_path: Path):
    """A member escaping the destination is refused before extraction."""
    from src.data.neu_det import _safe_extract

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escaped.txt", "owned")

    into = tmp_path / "NEU-DET"
    into.mkdir()
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ValueError, match="unsafe path"):
            _safe_extract(zf, into)

    assert not (tmp_path.parent / "escaped.txt").exists()


def test_zip_slip_sibling_prefix_rejected(tmp_path: Path):
    """A sibling that shares the destination's *prefix* must be rejected.

    ``NEU-DET-evil/`` is not inside ``NEU-DET/``, but ``str(dest).startswith(
    str(root))`` said it was — the bypass this guard now closes.
    """
    from src.data.neu_det import _safe_extract

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../NEU-DET-evil/pwned.txt", "owned")

    into = tmp_path / "NEU-DET"
    into.mkdir()
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ValueError, match="unsafe path"):
            _safe_extract(zf, into)

    assert not (tmp_path / "NEU-DET-evil" / "pwned.txt").exists()


def test_safe_extract_allows_nested_members(tmp_path: Path):
    """The guard must not reject legitimate nested entries."""
    from src.data.neu_det import _safe_extract

    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("NEU-DET/IMAGES/a.jpg", "jpeg")
        zf.writestr("NEU-DET/ANNOTATIONS/a.xml", "<xml/>")

    into = tmp_path / "out"
    into.mkdir()
    with zipfile.ZipFile(archive) as zf:
        _safe_extract(zf, into)

    assert (into / "NEU-DET" / "IMAGES" / "a.jpg").is_file()
    assert (into / "NEU-DET" / "ANNOTATIONS" / "a.xml").is_file()


def test_download_reuses_existing_archive_without_network(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "NEU-DET.zip"
    with zipfile.ZipFile(existing, "w") as zf:
        zf.writestr("placeholder.txt", "hi")

    def boom(*args, **kwargs):
        raise AssertionError("network must not be touched when archive exists")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    result = download_neu_det(raw)
    assert result == existing
