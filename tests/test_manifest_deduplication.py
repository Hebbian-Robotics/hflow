"""Manifest curation preserves duplicate provenance and exposes disagreement."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from hflow import ManifestDeduplicationSettings, deduplicate_manifest


def _write_manifest(
    path: Path, rows: Sequence[tuple[str | None, str | None, str, int | None]]
) -> None:
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE samples (sample_id VARCHAR, digest VARCHAR, recording VARCHAR, label INTEGER)"
        )
        if rows:
            connection.executemany("INSERT INTO samples VALUES (?, ?, ?, ?)", rows)
        connection.sql(
            "SELECT *, [label] AS label_history, sample_id::BLOB AS payload FROM samples"
        ).write_parquet(str(path))


def test_duplicates_keep_all_provenance_and_report_conflicts_without_excluding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.parquet"
    _write_manifest(
        source,
        [
            ("b", "same", "recording-b", 1),
            ("a", "same", "recording-a", 1),
            ("c", "different-labels", "recording-c", 0),
            ("d", "different-labels", "recording-d", None),
            ("e", "singleton", "recording-e", 2),
        ],
    )
    report = deduplicate_manifest(
        source,
        tmp_path / "deduplicated",
        settings=ManifestDeduplicationSettings(
            "sample_id", ("digest",), ("label", "label_history")
        ),
    )
    assert (report.input_rows, report.unique_samples, report.conflicting_samples) == (5, 3, 1)
    with duckdb.connect() as connection:
        samples = connection.read_parquet(str(report.output_directory / "samples.parquet"))
        assert samples.project("sample_id, deduplication_conflicts").fetchall() == [
            ("a", []),
            ("c", ["label", "label_history"]),
            ("e", []),
        ]
        members = connection.read_parquet(str(report.output_directory / "members.parquet"))
        assert members.project("sample_id, retained_sample_id").fetchall() == [
            ("a", "a"),
            ("b", "a"),
            ("c", "c"),
            ("d", "c"),
            ("e", "e"),
        ]
        original = connection.read_parquet(str(source)).order("sample_id")
        assert members.project("* EXCLUDE (retained_sample_id)").fetchall() == original.fetchall()
    receipt = json.loads((report.output_directory / "receipt.json").read_text())
    for filename, digest_name in (
        ("samples.parquet", "samples_sha256"),
        ("members.parquet", "members_sha256"),
    ):
        assert (
            receipt[digest_name]
            == hashlib.sha256((report.output_directory / filename).read_bytes()).hexdigest()
        )
    assert receipt["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        deduplicate_manifest(
            source,
            report.output_directory,
            settings=ManifestDeduplicationSettings("sample_id", ("digest",)),
        )
    assert json.loads((report.output_directory / "receipt.json").read_text()) == receipt


def test_composite_identity_requires_every_column_and_ignores_row_order(tmp_path: Path) -> None:
    rows = [("b", "same", "r1", 1), ("a", "same", "r1", 1), ("c", "same", "r2", 1)]
    settings = ManifestDeduplicationSettings("sample_id", ("digest", "recording"))
    for name, source_rows in (("forward", rows), ("reverse", list(reversed(rows)))):
        source = tmp_path / f"{name}.parquet"
        _write_manifest(source, source_rows)
        report = deduplicate_manifest(source, tmp_path / name, settings=settings)
        assert report.unique_samples == 2
        assert report.conflicting_samples == 0
    with duckdb.connect() as connection:
        for filename in ("samples.parquet", "members.parquet"):
            assert (
                connection.read_parquet(str(tmp_path / "forward" / filename)).fetchall()
                == connection.read_parquet(str(tmp_path / "reverse" / filename)).fetchall()
            )


@pytest.mark.parametrize(
    "rows, error",
    [
        ([], "must contain samples"),
        ([(None, "d", "r", 0)], "null or empty"),
        ([("", "d", "r", 0)], "null or empty"),
        ([("a", None, "r", 0)], "null or empty"),
        ([("a", "", "r", 0)], "null or empty"),
        ([("a", "d", "r", 0), ("a", "e", "s", 1)], "must be unique"),
    ],
)
def test_invalid_identities_publish_no_output(
    tmp_path: Path, rows: Sequence[tuple[str | None, str | None, str, int | None]], error: str
) -> None:
    source = tmp_path / "source.parquet"
    _write_manifest(source, rows)
    with pytest.raises(ValueError, match=error):
        deduplicate_manifest(
            source,
            tmp_path / "output",
            settings=ManifestDeduplicationSettings("sample_id", ("digest",)),
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "settings, error",
    [
        (ManifestDeduplicationSettings("sample_id", ("missing",)), "missing column"),
        (ManifestDeduplicationSettings("sample_id", ("label",)), "string type"),
        (ManifestDeduplicationSettings("sample_id", ("digest",), ("missing",)), "missing column"),
    ],
)
def test_invalid_column_selection_leaves_no_output(
    tmp_path: Path, settings: ManifestDeduplicationSettings, error: str
) -> None:
    source = tmp_path / "source.parquet"
    _write_manifest(source, [("a", "digest", "recording", 0)])
    with pytest.raises(ValueError, match=error):
        deduplicate_manifest(source, tmp_path / "output", settings=settings)
    assert not (tmp_path / "output").exists()


def test_quoted_columns_and_null_conflict_values_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    with duckdb.connect() as connection:
        connection.sql(
            "SELECT * FROM (VALUES ('a', 'same', NULL::INTEGER), ('b', 'same', NULL::INTEGER)) AS samples(\"sample id\", \"pixel\"\"identity\", \"teacher's label\")"
        ).write_parquet(str(source))
    report = deduplicate_manifest(
        source,
        tmp_path / "output",
        settings=ManifestDeduplicationSettings(
            "sample id", ('pixel"identity',), ("teacher's label",)
        ),
    )
    assert (report.unique_samples, report.conflicting_samples) == (1, 0)


def test_reserved_metadata_names_cannot_overwrite_source_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    with duckdb.connect() as connection:
        connection.sql(
            "SELECT 'a' AS sample_id, 'same' AS digest, 'original' AS retained_sample_id"
        ).write_parquet(str(source))
    with pytest.raises(ValueError, match="reserved"):
        deduplicate_manifest(
            source,
            tmp_path / "output",
            settings=ManifestDeduplicationSettings("sample_id", ("digest",)),
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "identities, conflicts",
    [
        ((), ()),
        (("digest", "DIGEST"), ()),
        (("digest",), ("label", "LABEL")),
        (("retained_sample_id",), ()),
    ],
)
def test_invalid_settings_reject_ambiguous_or_reserved_columns(
    identities: tuple[str, ...], conflicts: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        ManifestDeduplicationSettings("sample_id", identities, conflicts)


def test_ambiguous_source_names_cannot_be_silently_renamed(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    parquet.write_table(
        arrow.table(
            {
                "sample_id": ["a"],
                "digest": ["d"],
                "source": ["first"],
                "SOURCE": ["second"],
            }
        ),
        source,
    )
    with pytest.raises(ValueError, match="distinct ignoring case"):
        deduplicate_manifest(
            source,
            tmp_path / "output",
            settings=ManifestDeduplicationSettings("sample_id", ("digest",)),
        )
    assert not (tmp_path / "output").exists()
