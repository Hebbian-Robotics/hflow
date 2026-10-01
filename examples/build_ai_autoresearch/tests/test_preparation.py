"""Public teacher data produces deduplicated, auditable frame-level splits."""

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import duckdb
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest
from examples.build_ai_autoresearch.prepare import PUBLISHED_SOURCES, SourceFile, prepare_dataset
from PIL import Image


def _image_bytes(color: str, compression: int = 6) -> bytes:
    image = Image.new("RGB", (8, 6), color)
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=compression)
    return output.getvalue()


def _write_source(path: Path, rows: list[tuple[bytes, int]]) -> None:
    table = arrow.table(
        {
            "image": [{"bytes": image_bytes, "path": None} for image_bytes, _ in rows],
            "hand_count": [label for _, label in rows],
            "frame_id": [f"frame-{index}" for index in range(len(rows))],
        }
    )
    parquet.write_table(table, path)


def _prepare_inputs(directory: Path) -> list[SourceFile]:
    first = directory / "first.parquet"
    second = directory / "second.parquet"
    _write_source(
        first,
        [
            (_image_bytes("red"), 0),
            (_image_bytes("green"), 1),
            (_image_bytes("blue"), 2),
            (_image_bytes("yellow"), 0),
        ],
    )
    _write_source(
        second,
        [
            (_image_bytes("red"), 0),
            (_image_bytes("green", 0), 1),
            (_image_bytes("blue"), 1),
            (_image_bytes("purple"), 2),
        ],
    )
    return [
        SourceFile(replace(PUBLISHED_SOURCES[0], expected_sha256=None), first),
        SourceFile(replace(PUBLISHED_SOURCES[3], expected_sha256=None), second),
    ]


def test_mix_deduplicate_conflicts_and_freeze_evidence(tmp_path: Path) -> None:
    sources = _prepare_inputs(tmp_path)
    output = tmp_path / "prepared"
    report = prepare_dataset(sources, output)
    assert (
        report.input_rows,
        report.unique_images,
        report.retained_images,
        report.conflicting_images,
    ) == (8, 5, 4, 1)
    receipt = json.loads((output / "preparation.json").read_text())
    assert receipt["independence_scope"] == "exact-pixel-frame-only"
    assert receipt["label_source"] == "published-gemini-teacher-labels"
    assert all(
        source["verification"] == "local-file-revision-unverified" for source in receipt["sources"]
    )
    assert (
        receipt["manifest_sha256"]
        == hashlib.sha256((output / "samples.parquet").read_bytes()).hexdigest()
    )
    assert (
        receipt["split_receipt_sha256"]
        == hashlib.sha256((output / "splits/receipt.json").read_bytes()).hexdigest()
    )
    conflicts = json.loads((output / "conflicts.json").read_text())
    assert {reference["hand_count"] for reference in conflicts[0]["references"]} == {1, 2}
    assert not (output / "images" / f"{conflicts[0]['pixel_sha256']}.image").exists()
    with duckdb.connect() as connection:
        samples = connection.read_parquet(str(output / "samples.parquet")).fetchall()
        assert len(samples) == 4
        reference_counts = []
        for (
            sample_id,
            pixel_digest,
            image_path,
            encoded_digest,
            width,
            height,
            label,
            references_json,
        ) in samples:
            assert sample_id == pixel_digest
            assert (width, height) == (8, 6)
            assert hashlib.sha256((output / image_path).read_bytes()).hexdigest() == encoded_digest
            references = json.loads(references_json)
            assert {reference["hand_count"] for reference in references} == {label}
            reference_counts.append(len(references))
        assert sorted(reference_counts) == [1, 1, 2, 2]
        assignments = connection.read_parquet(str(output / "splits/assignments.parquet")).fetchall()
        assert len({sample_id for sample_id, _, _ in assignments}) == 4
        assert {partition for _, partition, _ in assignments} == {"train", "development", "test"}
    with pytest.raises(FileExistsError):
        prepare_dataset(sources, output)
    assert json.loads((output / "preparation.json").read_text()) == receipt


def test_source_order_does_not_change_selected_images_or_splits(tmp_path: Path) -> None:
    sources = _prepare_inputs(tmp_path)
    prepare_dataset(sources, tmp_path / "forward", seed=7)
    prepare_dataset(list(reversed(sources)), tmp_path / "reverse", seed=7)
    with duckdb.connect() as connection:
        for name in ("samples.parquet", "splits/assignments.parquet"):
            assert (
                connection.read_parquet(str(tmp_path / "forward" / name)).fetchall()
                == connection.read_parquet(str(tmp_path / "reverse" / name)).fetchall()
            )


@pytest.mark.parametrize("label", [3, -1])
def test_invalid_teacher_labels_leave_no_completed_dataset(tmp_path: Path, label: int) -> None:
    source = tmp_path / "invalid.parquet"
    _write_source(source, [(_image_bytes("red"), label)])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="hand_count"):
        prepare_dataset(
            [SourceFile(replace(PUBLISHED_SOURCES[0], expected_sha256=None), source)], output
        )
    assert not output.exists()


def test_too_few_unique_frames_cannot_create_holdouts(tmp_path: Path) -> None:
    sources = _prepare_inputs(tmp_path)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="not enough independent groups"):
        prepare_dataset(sources, output, row_limit_per_source=1)
    assert not output.exists()


def test_cache_files_must_match_pinned_source_content(tmp_path: Path) -> None:
    sources = _prepare_inputs(tmp_path)
    first = sources[0]
    mismatched = SourceFile(PUBLISHED_SOURCES[0], first.path)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="does not match the pinned revision"):
        prepare_dataset([mismatched, sources[1]], output)
    assert not output.exists()
