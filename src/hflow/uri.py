"""Validation for episode URIs relative to a runtime data root."""

from pathlib import Path, PurePath, PureWindowsPath
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


def resolve_local_data_root_path(data_root: Path, relative_reference: str | PurePath) -> Path:
    """The real file a data-root-relative reference names on a local root.

    ``parse_data_root_relative_uri`` can only judge a spelling. Resolution
    follows symlinks, so ``link/../x`` or a file symlink can leave the root
    even though the spelling never does. The returned real path is the one
    object callers may open: re-joining the original spelling later would
    resolve it again.
    """
    real_data_root = data_root.resolve()
    try:
        real_path = (real_data_root / relative_reference).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"cannot resolve data-root reference {str(relative_reference)!r}"
        ) from error
    if not real_path.is_relative_to(real_data_root):
        raise ValueError(
            f"{str(relative_reference)!r} resolves outside the data root {real_data_root}"
        )
    return real_path
