"""Reproducible partitions of a local Parquet manifest by connected groups.

This module consumes declared relationships, not media: callers discover
duplicates and recording identities before splitting. It does not deduplicate,
infer missing identities, stratify labels, or read the HFlow episode catalog.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from hflow._manifest_parquet_schema import _reject_ambiguous_column_names
from hflow._version import __version__

MANIFEST_SPLIT_VERSION = 1


@dataclass(frozen=True)
class ManifestPartition:
    """A portable partition name and its requested fraction of connected groups."""

    name: str
    fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", self.name):
            raise ValueError("partition name must match [a-z][a-z0-9_-]*")
        if self.name == "assignments":
            raise ValueError("partition name 'assignments' is reserved")
        if (
            isinstance(self.fraction, bool)
            or not isinstance(self.fraction, (int, float))
            or not math.isfinite(self.fraction)
            or not 0 < self.fraction <= 1
        ):
            raise ValueError("partition fraction must be finite and in (0, 1]")


DEFAULT_MANIFEST_PARTITIONS = (
    ManifestPartition("train", 0.8),
    ManifestPartition("development", 0.1),
    ManifestPartition("test", 0.1),
)


@dataclass(frozen=True)
class ManifestSplitSettings:
    """Column names, partitions, and seed defining a complete splitting policy.

    Identity and relationship columns must contain nonempty strings. Values
    are namespaced by column: matching a recording ID to a duplicate ID does
    not establish a relationship. A missing relationship must be resolved by
    the caller; a unique placeholder means independence is only assumed.
    """

    sample_id_column: str
    group_columns: tuple[str, ...]
    partitions: tuple[ManifestPartition, ...] = DEFAULT_MANIFEST_PARTITIONS
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id_column, str) or not self.sample_id_column:
            raise ValueError("sample_id_column must be a nonempty string")
        if not isinstance(self.group_columns, tuple) or not self.group_columns:
            raise ValueError("group_columns must be a nonempty tuple")
        if any(not isinstance(column, str) or not column for column in self.group_columns):
            raise ValueError("group columns must be nonempty strings")
        if len({column.casefold() for column in self.group_columns}) != len(self.group_columns):
            raise ValueError("group_columns must be distinct ignoring case")
        if (
            not isinstance(self.partitions, tuple)
            or len(self.partitions) < 2
            or any(not isinstance(partition, ManifestPartition) for partition in self.partitions)
        ):
            raise ValueError("partitions must be a tuple of at least two ManifestPartition values")
        if len({partition.name for partition in self.partitions}) != len(self.partitions):
            raise ValueError("partition names must be distinct")
        if not math.isclose(
            sum(partition.fraction for partition in self.partitions), 1, abs_tol=1e-12, rel_tol=0
        ):
            raise ValueError("partition fractions must sum to one")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class ManifestPartitionReport:
    name: str
    fraction: float
    row_count: int
    group_count: int
    sha256: str


@dataclass(frozen=True)
class ManifestSplitReport:
    output_directory: Path
    input_sha256: str
    assignments_sha256: str
    partitions: tuple[ManifestPartitionReport, ...]


@dataclass(frozen=True)
class _SampleIdentity:
    sample_id: str
    relationships: tuple[str, ...]


@dataclass(frozen=True)
class _Assignment:
    sample_id: str
    partition: str
    group_id: str


@dataclass(frozen=True)
class _ManifestSplitReceipt:
    schema_version: int
    algorithm: str
    input_sha256: str
    assignments_sha256: str
    settings: ManifestSplitSettings
    row_count: int
    partitions: tuple[ManifestPartitionReport, ...]
    hflow_version: str
    duckdb_version: str


def _quoted_identifier(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_sample_identities(
    connection: duckdb.DuckDBPyConnection, settings: ManifestSplitSettings
) -> list[_SampleIdentity]:
    selected_columns = (settings.sample_id_column, *settings.group_columns)
    column_types = {
        row[0]: row[1] for row in connection.execute("DESCRIBE source_manifest").fetchall()
    }
    missing_columns = set(selected_columns) - column_types.keys()
    if missing_columns:
        raise ValueError(f"manifest is missing columns: {sorted(missing_columns)}")
    for column in set(selected_columns):
        if column_types[column] != "VARCHAR":
            raise ValueError(f"identity column {column!r} must have string type")
    selected_sql = ", ".join(_quoted_identifier(column) for column in selected_columns)
    cursor = connection.execute(f"SELECT {selected_sql} FROM source_manifest")
    sample_identities: list[_SampleIdentity] = []
    seen_sample_ids: set[str] = set()
    while rows := cursor.fetchmany(4096):
        for row in rows:
            values: list[str] = []
            for column, value in zip(selected_columns, row, strict=True):
                if not isinstance(value, str) or not value:
                    raise ValueError(f"identity column {column!r} contains a null or empty value")
                values.append(value)
            sample_id = values[0]
            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"sample ID {sample_id!r} occurs more than once; deduplicate first"
                )
            seen_sample_ids.add(sample_id)
            sample_identities.append(_SampleIdentity(sample_id, tuple(values[1:])))
    if not sample_identities:
        raise ValueError("manifest must contain samples")
    return sample_identities


def _connected_groups(samples: Sequence[_SampleIdentity]) -> list[tuple[str, ...]]:
    parents = list(range(len(samples)))

    def find_component_root(sample_index: int) -> int:
        while parents[sample_index] != sample_index:
            parents[sample_index] = parents[parents[sample_index]]
            sample_index = parents[sample_index]
        return sample_index

    relationship_owners: dict[tuple[int, str], int] = {}
    for sample_index, sample in enumerate(samples):
        for column_index, relationship in enumerate(sample.relationships):
            relationship_key = (column_index, relationship)
            owner_index = relationship_owners.setdefault(relationship_key, sample_index)
            parents[find_component_root(sample_index)] = find_component_root(owner_index)
    grouped_ids: dict[int, list[str]] = {}
    for sample_index, sample in enumerate(samples):
        grouped_ids.setdefault(find_component_root(sample_index), []).append(sample.sample_id)
    return [tuple(sorted(sample_ids)) for sample_ids in grouped_ids.values()]


def _group_quotas(group_count: int, partitions: tuple[ManifestPartition, ...]) -> list[int]:
    if group_count < len(partitions):
        raise ValueError("not enough independent groups to populate every partition")
    targets = [group_count * partition.fraction for partition in partitions]
    quotas = [math.floor(target) for target in targets]
    remainder_order = sorted(
        range(len(partitions)),
        key=lambda index: (-(targets[index] - quotas[index]), partitions[index].name),
    )
    for index in remainder_order[: group_count - sum(quotas)]:
        quotas[index] += 1
    # Tiny corpora still need a real holdout. Move whole groups rather than
    # splitting a connected component to meet a numerical target.
    for index, quota in enumerate(quotas):
        if quota == 0:
            donors = [donor for donor, count in enumerate(quotas) if count > 1]
            donor = max(
                donors,
                key=lambda candidate: (
                    quotas[candidate] - targets[candidate],
                    partitions[candidate].name,
                ),
            )
            quotas[donor] -= 1
            quotas[index] += 1
    return quotas


def _plan_assignments(
    samples: Sequence[_SampleIdentity], settings: ManifestSplitSettings
) -> tuple[list[_Assignment], list[int]]:
    groups = _connected_groups(samples)
    quotas = _group_quotas(len(groups), settings.partitions)
    identified_groups = [
        (
            hashlib.sha256(
                json.dumps(group, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest(),
            group,
        )
        for group in groups
    ]
    ordered_groups = sorted(
        identified_groups,
        key=lambda group: (
            hashlib.sha256(f"{settings.seed}:{group[0]}".encode()).digest(),
            group[0],
        ),
    )
    assignments: list[_Assignment] = []
    group_offset = 0
    for partition, quota in zip(settings.partitions, quotas, strict=True):
        for group_id, sample_ids in ordered_groups[group_offset : group_offset + quota]:
            assignments.extend(
                _Assignment(sample_id, partition.name, group_id) for sample_id in sample_ids
            )
        group_offset += quota
    return assignments, quotas


def split_manifest(
    source_manifest: Path,
    output_directory: Path,
    *,
    settings: ManifestSplitSettings,
) -> ManifestSplitReport:
    """Partition a local Parquet file, preserving every row and column.

    Fractions target connected-group counts, not row counts. Largest-remainder
    allocation ensures each partition gets at least one group. Group ordering
    uses SHA-256 over seed and sorted member identities, independent of input
    row order. Adding samples can change assignments: freeze the snapshot.

    The output directory must not exist. A private input copy binds the receipt
    hash to the bytes actually read. The receipt is written last; publication
    is local and exclusive, but the directory is not atomically visible.
    """
    if output_directory.exists():
        raise FileExistsError(f"split output already exists: {output_directory}")
    with tempfile.TemporaryDirectory(prefix="hflow-manifest-split-") as temporary_directory:
        input_snapshot = Path(temporary_directory) / "input.parquet"
        shutil.copyfile(source_manifest, input_snapshot)
        input_sha256 = _file_sha256(input_snapshot)
        with duckdb.connect() as connection:
            _reject_ambiguous_column_names(connection, input_snapshot)
            connection.read_parquet(str(input_snapshot)).create_view("source_manifest")
            samples = _read_sample_identities(connection, settings)
            assignments, group_quotas = _plan_assignments(samples, settings)
            connection.execute(
                "CREATE TABLE split_assignments (sample_id VARCHAR, partition VARCHAR, group_id VARCHAR)"
            )
            connection.executemany(
                "INSERT INTO split_assignments VALUES (?, ?, ?)",
                [
                    (assignment.sample_id, assignment.partition, assignment.group_id)
                    for assignment in assignments
                ],
            )
            output_directory.mkdir(parents=True, exist_ok=False)
            try:
                connection.sql("SELECT * FROM split_assignments ORDER BY sample_id").write_parquet(
                    str(output_directory / "assignments.parquet")
                )
                partition_reports: list[ManifestPartitionReport] = []
                sample_id_sql = _quoted_identifier(settings.sample_id_column)
                for partition, group_count in zip(settings.partitions, group_quotas, strict=True):
                    output_path = output_directory / f"{partition.name}.parquet"
                    connection.sql(
                        f"SELECT source_manifest.* FROM source_manifest JOIN split_assignments ON source_manifest.{sample_id_sql} = split_assignments.sample_id WHERE split_assignments.partition = ? ORDER BY source_manifest.{sample_id_sql}",
                        params=[partition.name],
                    ).write_parquet(str(output_path))
                    partition_reports.append(
                        ManifestPartitionReport(
                            partition.name,
                            partition.fraction,
                            sum(
                                assignment.partition == partition.name for assignment in assignments
                            ),
                            group_count,
                            _file_sha256(output_path),
                        )
                    )
                report = ManifestSplitReport(
                    output_directory,
                    input_sha256,
                    _file_sha256(output_directory / "assignments.parquet"),
                    tuple(partition_reports),
                )
                receipt = _ManifestSplitReceipt(
                    schema_version=MANIFEST_SPLIT_VERSION,
                    algorithm="connected-groups-sha256-largest-remainder-v1",
                    input_sha256=report.input_sha256,
                    assignments_sha256=report.assignments_sha256,
                    settings=settings,
                    row_count=len(samples),
                    partitions=report.partitions,
                    hflow_version=__version__,
                    duckdb_version=duckdb.__version__,
                )
                with (output_directory / "receipt.json").open("x") as receipt_file:
                    json.dump(
                        asdict(receipt), receipt_file, indent=2, ensure_ascii=False, allow_nan=False
                    )
                    receipt_file.write("\n")
                return report
            except Exception:
                shutil.rmtree(output_directory)
                raise
