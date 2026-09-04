"""Validation for episode URIs relative to a runtime data root."""

from pathlib import PureWindowsPath
from posixpath import normpath
from typing import NewType

DataRootRelativeUri = NewType("DataRootRelativeUri", str)
DataRootRelativeUri.__module__ = __name__


def parse_data_root_relative_uri(uri: str) -> DataRootRelativeUri:
    """Trim and validate one URI before a runtime can resolve it.

    The returned value keeps safe internal segments such as ``a/../b``
    unchanged. Only surrounding whitespace is normalized; the parser rejects
    paths that are empty, anchored, use Windows separators, or escape above
    the data root.
    """
    if not isinstance(uri, str):
        raise ValueError("ingest URI must be a string")

    candidate = uri.strip()
    if not candidate:
        raise ValueError("every uri must be a non-empty string")
    if "\\" in candidate:
        raise ValueError("ingest URIs must use '/' as a separator")

    windows_path = PureWindowsPath(candidate)
    if candidate.startswith("/") or windows_path.anchor:
        raise ValueError(f"{candidate!r} is not relative to the data root")

    # Preserve the candidate itself; normalize only this containment check.
    normalized = normpath(candidate)
    if normalized == "." or normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"{candidate!r} is not relative to the data root")

    return DataRootRelativeUri(candidate)
