"""Shared Parquet manifest schema helpers for deduplication and splits."""

from __future__ import annotations

from pathlib import Path

import duckdb

# DuckDB compares quoted identifiers case-insensitively for ASCII letters only.
_ASCII_IDENTIFIER_FOLD = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _reject_ambiguous_column_names(
    connection: duckdb.DuckDBPyConnection, input_snapshot: Path
) -> None:
    # DuckDB renames ASCII case-colliding top-level columns when it opens the file, so
    # inspect the original names before the view exists.
    source_columns: list[str] = []
    nested_fields_remaining = 0
    schema_fields = connection.execute(
        "SELECT name, num_children FROM parquet_schema(?)", [str(input_snapshot)]
    ).fetchall()
    for column_name, child_count in schema_fields[1:]:
        if nested_fields_remaining:
            nested_fields_remaining += (child_count or 0) - 1
        else:
            source_columns.append(column_name)
            nested_fields_remaining = child_count or 0
    if len({column.translate(_ASCII_IDENTIFIER_FOLD) for column in source_columns}) != len(
        source_columns
    ):
        raise ValueError("manifest column names must be distinct ignoring case")
