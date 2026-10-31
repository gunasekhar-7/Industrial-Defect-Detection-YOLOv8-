"""Deterministic tests for downloader and malformed-conversion branches."""

import io
import urllib.request
import zipfile

import pytest
from PIL import Image

import src.data.neu_det as neu
from tests.test_neu_det import make_fixture


def _zip_payload() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("fixture.txt", "valid zip")
    return stream.getvalue()


class _Headers:
    def __init__(self, content_type="application/zip", cookies=None):
        self.content_type = content_type
        self.cookies = cookies or []

    def get(self, name, default=""):
        return self.content_type if name.lower() == "content-type" else default

    def get_all(self, name):
        return self.cookies if name.lower() == "set-cookie" else None


class _Response:
    def __init__(self, body: bytes, content_type="application/zip", cookies=None):
        self.body = body
        self.headers = _Headers(content_type, cookies)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        if size is None or size < 0:
            chunk, self.body = self.body, b""
            return chunk
        chunk, self.body = self.body[:size], self.body[size:]
        return chunk

    def readinto(self, buffer):
        n = min(len(buffer), len(self.body))
        buffer[:n] = self.body[:n]
        self.body = self.body[n:]
        return n


def test_download_direct_zip_and_reuses_validated_target(tmp_path, monkeypatch):
    payload = _zip_payload()
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = neu.download_neu_det(tmp_path, url="https://example.test/data.zip")

    assert result == tmp_path / "NEU-DET.zip"
    assert zipfile.is_zipfile(result)
    assert requests[0][1] == 120
    assert requests[0][0].full_url == "https://example.test/data.zip"
    assert neu.download_neu_det(tmp_path, url="https://ignored.test/archive.zip") == result
    assert len(requests) == 1


def test_download_gdrive_interstitial_carries_cookie_and_uuid(tmp_path, monkeypatch):
    payload = _zip_payload()
    responses = [
        _Response(
            b'<html><input name="uuid" value="abc-123"></html>',
            "text/html",
            ["download_warning=token; Path=/", "other=value; Path=/"],
        ),
        _Response(payload),
    ]
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return responses.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    target = neu.download_neu_det(tmp_path)

    assert zipfile.is_zipfile(target)
    assert len(requests) == 2
    assert requests[1][1] == 300
    assert "uuid=abc-123" in requests[1][0].full_url
    assert requests[1][0].get_header("Cookie") == "download_warning=token; other=value"


def test_download_gdrive_direct_and_bad_interstitial_pages(tmp_path, monkeypatch):
    payload = _zip_payload()
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_a, **_k: _Response(payload, "application/zip")
    )
    assert zipfile.is_zipfile(neu.download_neu_det(tmp_path / "direct"))

    for html, title in ((b"<html>no token</html>", "unexpected Google Drive interstitial (unknown page)"),
                        (b"<title>Quota exceeded</title>", "Quota exceeded")):
        target_dir = tmp_path / f"bad-{len(html)}"
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *_a, _html=html, **_k: _Response(_html, "text/html")
        )
        with pytest.raises(RuntimeError, match="Automatic NEU-DET download failed") as error:
            neu.download_neu_det(target_dir)
        assert title in str(error.value)
        assert not (target_dir / "NEU-DET.zip").exists()
        assert not (target_dir / "NEU-DET.zip.part").exists()


def test_download_rejects_non_zip_and_cleans_partial_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_a, **_k: _Response(b"not a zip")
    )
    with pytest.raises(RuntimeError, match="not a ZIP archive"):
        neu.download_neu_det(tmp_path / "bad", url="https://example.test/bad")
    assert not (tmp_path / "bad" / "NEU-DET.zip").exists()

    def write_partial_then_fail(_request, timeout):
        (tmp_path / "partial" / "NEU-DET.zip.part").write_bytes(b"partial")
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", write_partial_then_fail)
    with pytest.raises(RuntimeError, match="OSError: offline"):
        neu.download_neu_det(tmp_path / "partial", url="https://example.test/data")
    assert not (tmp_path / "partial" / "NEU-DET.zip.part").exists()


def test_convert_rejects_invalid_sources_splits_and_empty_archives(tmp_path):
    with pytest.raises(ValueError, match="splits must sum"):
        neu.convert_neu_det(tmp_path, tmp_path / "out", splits=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="at least two"):
        neu.convert_neu_det(tmp_path, tmp_path / "out", splits=(1.0,))
    with pytest.raises(FileNotFoundError, match="source not found"):
        neu.convert_neu_det(tmp_path / "absent", tmp_path / "out")

    non_zip = tmp_path / "bad.zip"
    non_zip.write_bytes(b"no")
    with pytest.raises(ValueError, match="not a ZIP"):
        neu.convert_neu_det(non_zip, tmp_path / "out")

    no_images = tmp_path / "no-images"
    no_images.mkdir()
    with pytest.raises(ValueError, match="no images found"):
        neu.convert_neu_det(no_images, tmp_path / "out")

    no_xml = tmp_path / "no-xml"
    no_xml.mkdir()
    Image.new("RGB", (16, 16)).save(no_xml / "sample.jpg")
    with pytest.raises(ValueError, match="no VOC XML"):
        neu.convert_neu_det(no_xml, tmp_path / "out")


def test_convert_rejects_duplicate_missing_and_malformed_annotations(tmp_path):
    duplicate = make_fixture(tmp_path / "duplicate")
    original = duplicate / "NEU-DET" / "ANNOTATIONS" / "crazing_0.xml"
    extra = duplicate / "NEU-DET" / "OTHER"
    extra.mkdir()
    (extra / original.name).write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate annotation stem"):
        neu.convert_neu_det(duplicate, tmp_path / "out")

    missing = make_fixture(tmp_path / "missing")
    (missing / "NEU-DET" / "ANNOTATIONS" / "scratches_5.xml").unlink()
    with pytest.raises(ValueError, match="no annotation XML"):
        neu.convert_neu_det(missing, tmp_path / "out")

    malformed = make_fixture(tmp_path / "malformed")
    xml = malformed / "NEU-DET" / "ANNOTATIONS" / "crazing_0.xml"
    xml.write_text("<annotation><object><name>crazing</name></object></annotation>", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed object"):
        neu.convert_neu_det(malformed, tmp_path / "out")

    empty_objects = make_fixture(tmp_path / "empty-objects")
    xml = empty_objects / "NEU-DET" / "ANNOTATIONS" / "crazing_0.xml"
    xml.write_text("<annotation />", encoding="utf-8")
    with pytest.raises(ValueError, match="no objects"):
        neu.convert_neu_det(empty_objects, tmp_path / "out")


def test_convert_supports_extra_splits_clamps_boxes_and_replaces_stale_files(tmp_path):
    source = make_fixture(tmp_path / "source")
    xml = source / "NEU-DET" / "ANNOTATIONS" / "crazing_0.xml"
    xml.write_text(
        "<annotation><object><name>crazing</name><bndbox>"
        "<xmin>-10</xmin><ymin>-10</ymin><xmax>1000</xmax><ymax>1000</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    stale_image_dir = out / "images" / "split0"
    stale_label_dir = out / "labels" / "split0"
    stale_image_dir.mkdir(parents=True)
    stale_label_dir.mkdir(parents=True)
    (stale_image_dir / "stale.jpg").write_bytes(b"stale")
    (stale_label_dir / "stale.txt").write_text("stale", encoding="utf-8")

    neu.convert_neu_det(source, out, splits=(0.4, 0.2, 0.2, 0.2), seed=7)

    assert not (stale_image_dir / "stale.jpg").exists()
    assert not (stale_label_dir / "stale.txt").exists()
    assert all((out / "images" / f"split{i}").is_dir() for i in range(4))
    label_files = list((out / "labels").rglob("*.txt"))
    assert any("0 0.500000 0.500000 1.000000 1.000000" in p.read_text(encoding="utf-8") for p in label_files)


def test_download_helper_direct_fetch_uses_part_then_renames(tmp_path, monkeypatch):
    payload = _zip_payload()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Response(payload))
    target = tmp_path / "target.zip"
    neu._fetch_direct("https://example.test/archive.zip", target)
    assert zipfile.is_zipfile(target)
    assert not target.with_suffix(".zip.part").exists()


def test_parse_voc_rejects_absent_name_and_absent_box(tmp_path):
    for content in (
        "<annotation><object><bndbox /></object></annotation>",
        "<annotation><object><name>crazing</name></object></annotation>",
    ):
        xml = tmp_path / "object.xml"
        xml.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="malformed object"):
            neu._parse_voc_objects(xml)


def test_neu_cli_parses_local_archive_and_seed(tmp_path, monkeypatch):
    archive = tmp_path / "source"
    out = tmp_path / "out"
    calls = []
    monkeypatch.setattr(neu, "convert_neu_det", lambda *args, **kwargs: calls.append((args, kwargs)))

    neu.main(["--archive", str(archive), "--out", str(out), "--seed", "11"])

    assert calls == [((archive, str(out)), {"seed": 11})]


def test_safe_extract_skips_unnamed_members(tmp_path):
    """An archive member with an empty name is skipped, not resolved to the
    destination root (which would extract over the output directory)."""
    from types import SimpleNamespace

    extracted = []

    class _Archive:
        def infolist(self):
            return [
                SimpleNamespace(filename=""),
                SimpleNamespace(filename="NEU-DET/IMAGES/a.jpg"),
            ]

        def extractall(self, into):
            extracted.append(into)

    target = tmp_path / "out"
    target.mkdir()

    neu._safe_extract(_Archive(), target)

    assert extracted == [target]


def test_convert_reports_split_that_cannot_be_populated(tmp_path):
    """A dataset too small to fill three splits must fail validation with the
    empty split named, instead of shipping a train-only dataset."""
    images = tmp_path / "tiny" / "NEU-DET" / "IMAGES"
    xmls = tmp_path / "tiny" / "NEU-DET" / "ANNOTATIONS"
    images.mkdir(parents=True)
    xmls.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images / "only.jpg")
    (xmls / "only.xml").write_text(
        "<annotation><object><name>crazing</name><bndbox>"
        "<xmin>4</xmin><ymin>4</ymin><xmax>20</xmax><ymax>20</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="failed validation for split 'val'"):
        neu.convert_neu_det(tmp_path / "tiny", tmp_path / "out")


def test_neu_cli_downloads_when_archive_not_given(tmp_path, monkeypatch):
    source = tmp_path / "download.zip"
    calls = []
    monkeypatch.setattr(neu, "download_neu_det", lambda dest: calls.append(dest) or source)
    monkeypatch.setattr(neu, "convert_neu_det", lambda *args, **kwargs: calls.append((args, kwargs)))

    neu.main(["--dest", str(tmp_path / "raw"), "--out", str(tmp_path / "out")])

    assert calls[0] == str(tmp_path / "raw")
    assert calls[1] == ((source, str(tmp_path / "out")), {"seed": 42})
