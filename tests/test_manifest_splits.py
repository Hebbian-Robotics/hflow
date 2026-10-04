"""Frozen dataset partitions keep declared relationships out of holdouts."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from hflow import ManifestPartition, ManifestSplitSettings, split_manifest


def _write_manifest(path: Path, rows: Sequence[tuple[str, str | None, str, int]]) -> None:
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE samples (sample_id VARCHAR, recording_id VARCHAR, image_digest VARCHAR, label INTEGER)"
        )
        if rows:
            connection.executemany("INSERT INTO samples VALUES (?, ?, ?, ?)", rows)
        connection.sql(
            "SELECT *, [label, label + 1] AS label_history, CAST(sample_id AS BLOB) AS image FROM samples"
        ).write_parquet(str(path))


def _assignments(directory: Path) -> list[tuple[str, str, str]]:
    with duckdb.connect() as connection:
        return connection.read_parquet(str(directory / "assignments.parquet")).fetchall()


def test_bridge_duplicates_join_recordings_and_preserve_all_samples(tmp_path: Path) -> None:
    rows = [
        ("a", "recording-a", "image-a", 0),
        ("b", "recording-a", "shared-image", 1),
        ("c", "recording-b", "shared-image", 1),
        ("d", "recording-b", "image-d", 2),
        *[
            (f"sample-{index:02}", f"recording-{index}", f"image-{index}", index % 3)
            for index in range(19)
        ],
    ]
    source = tmp_path / "source.parquet"
    _write_manifest(source, rows)
    settings = ManifestSplitSettings("sample_id", ("recording_id", "image_digest"), seed=42)
    report = split_manifest(source, tmp_path / "split", settings=settings)
    assignments = {
        sample_id: (partition, group)
        for sample_id, partition, group in _assignments(report.output_directory)
    }
    assert len({assignments[sample_id] for sample_id in ("a", "b", "c", "d")}) == 1
    assert sum(partition.row_count for partition in report.partitions) == len(rows)
    assert [partition.group_count for partition in report.partitions] == [16, 2, 2]
    with duckdb.connect() as connection:
        expected_rows = connection.read_parquet(str(source)).order("sample_id").fetchall()
        reconstructed_rows = (
            connection.read_parquet(
                [
                    str(report.output_directory / f"{partition.name}.parquet")
                    for partition in report.partitions
                ]
            )
            .order("sample_id")
            .fetchall()
        )
        assert reconstructed_rows == expected_rows
        for column in ("recording_id", "image_digest"):
            grouped_partitions = connection.execute(
                f"SELECT source.{column}, count(DISTINCT assignments.partition) FROM read_parquet(?) source JOIN read_parquet(?) assignments USING (sample_id) GROUP BY source.{column}",
                [str(source), str(report.output_directory / "assignments.parquet")],
            ).fetchall()
            assert all(count == 1 for _, count in grouped_partitions)
    receipt = json.loads((report.output_directory / "receipt.json").read_text())
    assert receipt["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["settings"]["seed"] == 42
    assert receipt["settings"]["group_columns"] == ["recording_id", "image_digest"]
    assert (
        receipt["assignments_sha256"]
        == hashlib.sha256(
            (report.output_directory / "assignments.parquet").read_bytes()
        ).hexdigest()
    )
    for partition in receipt["partitions"]:
        assert (
            partition["sha256"]
            == hashlib.sha256(
                (report.output_directory / f"{partition['name']}.parquet").read_bytes()
            ).hexdigest()
        )
    with pytest.raises(FileExistsError):
        split_manifest(source, report.output_directory, settings=settings)
    assert json.loads((report.output_directory / "receipt.json").read_text()) == receipt


def test_assignments_ignore_row_order_and_support_custom_partitions(tmp_path: Path) -> None:
    rows = [(str(index), f"r-{index}", f"i-{index}", index % 3) for index in range(30)]
    settings = ManifestSplitSettings(
        "sample_id",
        ("recording_id", "image_digest"),
        (ManifestPartition("search", 0.7), ManifestPartition("confirmation", 0.3)),
        seed=7,
    )
    for name, ordered_rows in (("forward", rows), ("reverse", list(reversed(rows)))):
        source = tmp_path / f"{name}.parquet"
        _write_manifest(source, ordered_rows)
        report = split_manifest(source, tmp_path / name, settings=settings)
        assert [partition.group_count for partition in report.partitions] == [21, 9]
    assert _assignments(tmp_path / "forward") == _assignments(tmp_path / "reverse")
    source = tmp_path / "forward.parquet"
    split_manifest(
        source,
        tmp_path / "other-seed",
        settings=ManifestSplitSettings(
            "sample_id", ("recording_id", "image_digest"), settings.partitions, seed=8
        ),
    )
    assert _assignments(tmp_path / "forward") != _assignments(tmp_path / "other-seed")


def test_relationship_namespaces_and_small_holdouts(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write_manifest(
        source, [("a", "same", "other", 0), ("b", "other", "same", 1), ("c", "third", "third", 2)]
    )
    report = split_manifest(
        source,
        tmp_path / "split",
        settings=ManifestSplitSettings("sample_id", ("recording_id", "image_digest")),
    )
    assert [partition.group_count for partition in report.partitions] == [1, 1, 1]
    assert [partition.row_count for partition in report.partitions] == [1, 1, 1]


@pytest.mark.parametrize(
    "rows, groups, expected_error",
    [
        ([], ("recording_id",), "must contain samples"),
        ([("a", "r", "i", 0), ("a", "s", "j", 1)], ("recording_id",), "occurs more than once"),
        ([("a", None, "i", 0)], ("recording_id",), "null or empty"),
        ([("a", "", "i", 0)], ("recording_id",), "null or empty"),
        ([("a", "r", "i", 0)], ("missing",), "missing columns"),
        ([("a", "r", "i", 0)], ("label",), "string type"),
        (
            [("a", "r", "i", 0), ("b", "r", "j", 1), ("c", "r", "k", 2)],
            ("recording_id",),
            "not enough independent groups",
        ),
    ],
)
def test_invalid_identities_never_publish_a_split(
    tmp_path: Path,
    rows: list[tuple[str, str | None, str, int]],
    groups: tuple[str, ...],
    expected_error: str,
) -> None:
    source = tmp_path / "source.parquet"
    _write_manifest(source, rows)
    output = tmp_path / "split"
    with pytest.raises(ValueError, match=expected_error):
        split_manifest(source, output, settings=ManifestSplitSettings("sample_id", groups))
    assert not output.exists()


@pytest.mark.parametrize(
    "name, fraction",
    [
        ("../escape", 0.5),
        ("assignments", 0.5),
        ("train", float("nan")),
        ("train", True),
        ("train", 0),
    ],
)
def test_partition_values_reject_unsafe_names_and_invalid_fractions(
    name: str, fraction: float
) -> None:
    with pytest.raises(ValueError):
        ManifestPartition(name, fraction)


def test_policy_rejects_ambiguous_relationships_and_fractions() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ManifestSplitSettings("sample_id", ("recording", "recording"))
    with pytest.raises(ValueError, match="distinct ignoring case"):
        ManifestSplitSettings("sample_id", ("recording", "RECORDING"))
    with pytest.raises(ValueError, match="sum to one"):
        ManifestSplitSettings(
            "sample_id",
            ("recording",),
            (ManifestPartition("train", 0.7), ManifestPartition("test", 0.2)),
        )
    with pytest.raises(ValueError, match="names must be distinct"):
        ManifestSplitSettings(
            "sample_id",
            ("recording",),
            (ManifestPartition("train", 0.5), ManifestPartition("train", 0.5)),
        )


def test_case_colliding_source_columns_are_refused_not_renamed(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.parquet"
    parquet.write_table(
        arrow.table(
            {
                "sample_id": ["a", "b", "c"],
                "recording_id": ["r1", "r2", "r3"],
                "metadata": ["m1", "m2", "m3"],
                "METADATA": ["M1", "M2", "M3"],
            }
        ),
        source,
    )
    with pytest.raises(ValueError, match="distinct ignoring case"):
        split_manifest(
            source,
            tmp_path / "split",
            settings=ManifestSplitSettings("sample_id", ("recording_id",)),
        )
    assert not (tmp_path / "split").exists()
