"""Curation sidecar state: ``<data_root>/curation/state.json``, owned here.

Workspace convention: curation persists exactly two kinds of durable state --
saved queries and the pinned-manifest registry -- in ONE JSON sidecar file,
``<data_root>/curation/state.json``. Together with the manifest files under
``<data_root>/manifests/``, that sidecar is the ONLY thing this server ever
writes into a workspace.

It sits under ``curation/`` rather than under any client's name because the
content is the operator's, not a browser's: saved queries and pinned
manifests belong to the workspace, and a second client (another UI, a
script) reads the same file.

Two rules hold at this boundary:

- Every write is atomic: the payload lands in a temp file beside the target
  and is moved into place with ``os.replace`` (via ``Path.replace``), so a
  crash never leaves a torn file. Concurrent writers are last-writer-wins,
  which a single-operator local tool accepts.
- Every read parses loudly: a payload that is not JSON, carries a
  ``state_version`` this build does not speak, or holds a malformed entry is
  refused with an error NAMING THE FILE -- never silently coerced, dropped,
  or rewritten (the state is the user's curation record).

The stored entries ARE the published contract models
(:class:`hflow_server._contract.SavedQueryEntry` and
:class:`~hflow_server._contract.PinnedManifestEntry`): the file a user can read
with ``jq`` and the payload the API serves are one shape with one owner, so
they cannot drift apart. Changing either therefore changes this file's
format, which ``STATE_VERSION`` guards.
"""

import json
from dataclasses import dataclass

from fastapi import HTTPException

from hflow.storage import parse_storage_root
from hflow_server._contract import CheckCoverageEntry, PinnedManifestEntry, SavedQueryEntry

STATE_VERSION = 1
SIDECAR_DIRECTORY_NAME = "curation"
SIDECAR_FILE_NAME = "state.json"


class SidecarError(HTTPException):
    """One refusal to read or write the sidecar.

    It IS the HTTP refusal rather than something the router converts into one:
    raised from anywhere under a route, FastAPI renders it with the status and
    detail given here.
    """


@dataclass(frozen=True)
class SidecarState:
    """The whole parsed sidecar; a missing file reads as this default."""

    saved_queries: tuple[SavedQueryEntry, ...] = ()
    manifests: tuple[PinnedManifestEntry, ...] = ()


SIDECAR_STATE_KEY = f"{SIDECAR_DIRECTORY_NAME}/{SIDECAR_FILE_NAME}"


def sidecar_state_description(data_root: str) -> str:
    """How the sidecar is named in refusals: the root as given, plus the key.

    Deliberately the root's own string rather than a resolved URI, so an error
    names the workspace the operator launched against.
    """
    return f"{parse_storage_root(data_root)}/{SIDECAR_STATE_KEY}"


def load_sidecar_state(data_root: str) -> SidecarState:
    """The parsed sidecar; empty when never written, loud on anything corrupt."""
    state_description = sidecar_state_description(data_root)
    try:
        raw_payload = parse_storage_root(data_root).read_bytes(SIDECAR_STATE_KEY).decode("utf-8")
    except FileNotFoundError:
        return SidecarState()
    except (OSError, UnicodeDecodeError) as error:
        raise SidecarError(
            500, f"cannot read curation state file {state_description}: {error}"
        ) from error
    return _parsed_state(raw_payload, state_description)


def store_sidecar_state(data_root: str, state: SidecarState) -> None:
    """Replace the sidecar with ``state``.

    Written through the storage root, so a bucket-backed workspace keeps saved
    queries and pinned manifests like any other. Both backends replace the
    object in one step -- a local temp-file rename, a single bucket PUT -- so a
    reader sees the old complete state or the new one, never a mix.
    """
    # by_alias: the stored keys are the published ones ("id", not "query_id").
    payload = json.dumps(
        {
            "state_version": STATE_VERSION,
            "saved_queries": [entry.model_dump(by_alias=True) for entry in state.saved_queries],
            "manifests": [entry.model_dump(by_alias=True) for entry in state.manifests],
        },
        indent=2,
    )
    try:
        parse_storage_root(data_root).write_bytes(SIDECAR_STATE_KEY, (payload + "\n").encode())
    except OSError as error:
        raise SidecarError(
            500,
            f"cannot write curation state file {sidecar_state_description(data_root)}: {error}",
        ) from error


def _refused(state_file: str, problem: str) -> SidecarError:
    return SidecarError(
        500, f"corrupt curation state file {state_file}: {problem}; fix or remove the file"
    )


def _parsed_state(raw_payload: str, state_file: str) -> SidecarState:
    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError as error:
        raise _refused(state_file, f"not valid JSON ({error})") from error
    if not isinstance(parsed, dict):
        raise _refused(state_file, "expected a JSON object")
    found_version = parsed.get("state_version")
    if found_version != STATE_VERSION:
        # 409, not 500: the same mapping _connections gives a catalog written
        # in a format version this build cannot read -- the state is there and
        # intact, this build just cannot speak to it, which is a conflict with
        # the workspace rather than a fault of this server. A file that is
        # corrupt (rather than merely newer) keeps the 500 _refused gives it.
        raise SidecarError(
            409,
            f"curation state file {state_file} has state_version {found_version!r}; "
            f"this build reads version {STATE_VERSION!r}",
        )
    saved_queries = tuple(
        _parsed_saved_query(entry, state_file)
        for entry in _entry_list(parsed, "saved_queries", state_file)
    )
    manifests = tuple(
        _parsed_manifest(entry, state_file)
        for entry in _entry_list(parsed, "manifests", state_file)
    )
    return SidecarState(saved_queries=saved_queries, manifests=manifests)


def _entry_list(parsed: dict[str, object], key: str, state_file: str) -> list[dict[str, object]]:
    entries = parsed.get(key, [])
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise _refused(state_file, f"{key!r} must be a list of objects")
    return entries


def _string_field(entry: dict[str, object], key: str, state_file: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str):
        raise _refused(state_file, f"entry field {key!r} must be a string, got {value!r}")
    return value


def _int_field(entry: dict[str, object], key: str, state_file: str) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _refused(state_file, f"entry field {key!r} must be an integer, got {value!r}")
    return value


def _float_field(entry: dict[str, object], key: str, state_file: str) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _refused(state_file, f"entry field {key!r} must be a number, got {value!r}")
    return float(value)


def _parsed_saved_query(entry: dict[str, object], state_file: str) -> SavedQueryEntry:
    # Field-by-field on purpose: a model_validate refusal would name pydantic's
    # own error shape, not this file and the fix for it.
    return SavedQueryEntry(
        query_id=_string_field(entry, "id", state_file),
        name=_string_field(entry, "name", state_file),
        sql=_string_field(entry, "sql", state_file),
        updated_at=_string_field(entry, "updated_at", state_file),
    )


def _parsed_manifest(entry: dict[str, object], state_file: str) -> PinnedManifestEntry:
    raw_coverage = entry.get("coverage", [])
    if not isinstance(raw_coverage, list) or not all(
        isinstance(coverage_entry, dict) for coverage_entry in raw_coverage
    ):
        raise _refused(state_file, "'coverage' must be a list of objects")
    coverage = [
        CheckCoverageEntry(
            check_name=_string_field(coverage_entry, "check_name", state_file),
            episodes_ran=_int_field(coverage_entry, "episodes_ran", state_file),
            total_episodes=_int_field(coverage_entry, "total_episodes", state_file),
            fraction=_float_field(coverage_entry, "fraction", state_file),
        )
        for coverage_entry in raw_coverage
    ]
    return PinnedManifestEntry(
        manifest_id=_string_field(entry, "id", state_file),
        name=_string_field(entry, "name", state_file),
        description=_string_field(entry, "description", state_file),
        sql=_string_field(entry, "sql", state_file),
        manifest_path=_string_field(entry, "manifest_path", state_file),
        row_count=_int_field(entry, "row_count", state_file),
        total_episodes=_int_field(entry, "total_episodes", state_file),
        coverage=coverage,
        created_at=_string_field(entry, "created_at", state_file),
    )
