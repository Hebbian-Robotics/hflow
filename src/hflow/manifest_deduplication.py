"""Dataset curation by declared identities, independent of media and episodes."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from hflow._manifest_parquet_schema import _reject_ambiguous_column_names
from hflow._version import __version__

_RESERVED_COLUMNS = {"deduplication_conflicts", "retained_sample_id"}


@dataclass(frozen=True)
class ManifestDeduplicationSettings:
    """Group by the complete identity tuple and report disagreeing columns.

    Identities must already be discovered by the caller. Conflict columns are
    evidence to inspect, not an automatic exclusion or label-resolution policy.
    """

    sample_id_column: str
    identity_columns: tuple[str, ...]
    conflict_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id_column, str) or not self.sample_id_column:
            raise ValueError("sample_id_column must be a nonempty string")
        for name, columns in (
            ("identity_columns", self.identity_columns),
            ("conflict_columns", self.conflict_columns),
        ):
            if not isinstance(columns, tuple) or any(
                not isinstance(column, str) or not column for column in columns
            ):
                raise ValueError(f"{name} must be a tuple of nonempty column names")
            if len({column.casefold() for column in columns}) != len(columns):
                raise ValueError(f"{name} must contain distinct column names")
        if not self.identity_columns:
            raise ValueError("identity_columns must be nonempty")
        selected_columns = (self.sample_id_column, *self.identity_columns, *self.conflict_columns)
        if any(column.casefold() in _RESERVED_COLUMNS for column in selected_columns):
            raise ValueError("deduplication output column names are reserved")


@dataclass(frozen=True)
class ManifestDeduplicationReport:
    output_directory: Path
    input_sha256: str
    samples_sha256: str
    members_sha256: str
    input_rows: int
    unique_samples: int
    conflicting_samples: int


def _quoted_column(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _validate_manifest(
    connection: duckdb.DuckDBPyConnection, settings: ManifestDeduplicationSettings
) -> None:
    column_types = {
        row[0]: row[1] for row in connection.execute("DESCRIBE source_manifest").fetchall()
    }
    if {column.casefold() for column in column_types} & _RESERVED_COLUMNS:
        raise ValueError("manifest contains reserved deduplication output columns")
    for column in (
        settings.sample_id_column,
        *settings.identity_columns,
        *settings.conflict_columns,
    ):
        if column not in column_types:
            raise ValueError(f"manifest is missing column {column!r}")
    for column in {settings.sample_id_column, *settings.identity_columns}:
        if column_types[column] != "VARCHAR":
            raise ValueError(f"identity column {column!r} must have string type")
        identifier = _quoted_column(column)
        if connection.execute(
            f"SELECT count(*) FROM source_manifest WHERE {identifier} IS NULL OR {identifier} = ''"
        ).fetchall()[0][0]:
            raise ValueError(f"identity column {column!r} contains a null or empty value")
    sample_id = _quoted_column(settings.sample_id_column)
    row_count, unique_ids = connection.execute(
        f"SELECT count(*), count(DISTINCT {sample_id}) FROM source_manifest"
    ).fetchall()[0]
    if not row_count:
        raise ValueError("manifest must contain samples")
    if row_count != unique_ids:
        raise ValueError("sample IDs must be unique; give each source occurrence its own identity")


def deduplicate_manifest(
    source_manifest: Path,
    output_directory: Path,
    *,
    settings: ManifestDeduplicationSettings,
) -> ManifestDeduplicationReport:
    """Retain a deterministic representative and preserve every original row.

    Rows are equivalent only when all identity columns match. The smallest
    sample ID is retained, independent of input order. ``samples.parquet``
    preserves representative columns and adds ``deduplication_conflicts``:
    the configured column names with differing values (null is a value).
    ``members.parquet`` preserves all original rows and adds
    ``retained_sample_id``. No group is dropped, including conflicting groups.

    The local output directory must not exist. A private input snapshot binds
    hashes to processed bytes. The receipt is written last; interrupted outputs
    without a receipt are incomplete. Ordinary failures remove the output.
    """
    if output_directory.exists():
        raise FileExistsError(f"deduplication output already exists: {output_directory}")
    with tempfile.TemporaryDirectory(prefix="hflow-manifest-deduplication-") as temporary_directory:
        input_snapshot = Path(temporary_directory) / "input.parquet"
        shutil.copyfile(source_manifest, input_snapshot)
        input_sha256 = _file_sha256(input_snapshot)
        with duckdb.connect() as connection:
            _reject_ambiguous_column_names(connection, input_snapshot)
            connection.read_parquet(str(input_snapshot)).create_view("source_manifest")
            _validate_manifest(connection, settings)
            sample_id = _quoted_column(settings.sample_id_column)
            identity_columns = ", ".join(
                _quoted_column(column) for column in settings.identity_columns
            )
            # Wrapping a value in a struct includes null in the distinct count.
            conflict_expressions = [
                f"CASE WHEN count(DISTINCT struct_pack(value := {_quoted_column(column)})) > 1 THEN ? ELSE NULL END"
                for column in settings.conflict_columns
            ]
            conflicts_sql = (
                f"list_filter([{', '.join(conflict_expressions)}], value -> value IS NOT NULL)"
                if conflict_expressions
                else "[]::VARCHAR[]"
            )
            connection.execute(
                f"CREATE TEMP TABLE deduplication_groups AS SELECT {identity_columns}, min({sample_id}) AS retained_sample_id, {conflicts_sql} AS deduplication_conflicts FROM source_manifest GROUP BY {identity_columns}",
                list(settings.conflict_columns),
            )
            join_condition = " AND ".join(
                f"source_manifest.{_quoted_column(column)} = deduplication_groups.{_quoted_column(column)}"
                for column in settings.identity_columns
            )
            input_rows = connection.execute("SELECT count(*) FROM source_manifest").fetchall()[0][0]
            unique_samples, conflicting_samples = connection.execute(
                "SELECT count(*), count(*) FILTER (WHERE len(deduplication_conflicts) > 0) FROM deduplication_groups"
            ).fetchall()[0]
            output_directory.mkdir(parents=True, exist_ok=False)
            try:
                connection.sql(
                    f"SELECT source_manifest.*, deduplication_groups.deduplication_conflicts FROM source_manifest JOIN deduplication_groups ON source_manifest.{sample_id} = deduplication_groups.retained_sample_id ORDER BY source_manifest.{sample_id}"
                ).write_parquet(str(output_directory / "samples.parquet"))
                connection.sql(
                    f"SELECT source_manifest.*, deduplication_groups.retained_sample_id FROM source_manifest JOIN deduplication_groups ON {join_condition} ORDER BY source_manifest.{sample_id}"
                ).write_parquet(str(output_directory / "members.parquet"))
                report = ManifestDeduplicationReport(
                    output_directory,
                    input_sha256,
                    _file_sha256(output_directory / "samples.parquet"),
                    _file_sha256(output_directory / "members.parquet"),
                    input_rows,
                    unique_samples,
                    conflicting_samples,
                )
                receipt = {
                    "schema_version": 1,
                    "algorithm": "declared-identities-smallest-sample-id-v1",
                    "settings": asdict(settings),
                    **{
                        name: value
                        for name, value in asdict(report).items()
                        if name != "output_directory"
                    },
                    "hflow_version": __version__,
                    "duckdb_version": duckdb.__version__,
                }
                with (output_directory / "receipt.json").open("x") as receipt_file:
                    json.dump(receipt, receipt_file, indent=2, ensure_ascii=False, allow_nan=False)
                    receipt_file.write("\n")
                return report
            except Exception:
                shutil.rmtree(output_directory)
                raise
