"""Validation for URIs relative to a runtime data root."""

from pathlib import PureWindowsPath
from posixpath import normpath
from typing import NewType

DataRootRelativeUri = NewType("DataRootRelativeUri", str)


def parse_data_root_relative_uri(uri: str) -> DataRootRelativeUri:
    """Trim and validate one URI before a runtime can resolve it."""
    if not isinstance(uri, str):
        raise ValueError("ingest URI must be a string")

    candidate = uri.strip()
    if not candidate:
        raise ValueError("every URI must be a non-empty string")
    if candidate.startswith("/") or PureWindowsPath(candidate).is_absolute():
        raise ValueError(f"{candidate!r} is not relative to the data root")

    normalized = normpath(candidate)
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"{candidate!r} is not relative to the data root")

    return DataRootRelativeUri(candidate)
