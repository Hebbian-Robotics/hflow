"""Launch configuration for the workspace UI server, and what it refuses.

The settings own the launch-wide facts the routers keep asking about --
``read_only``, the port a launch may bind, and whether the data root is a
local directory -- so each fact is derived here once, beside the field, and
the refusal it maps to lives next to it rather than being hand-written per
router.
"""

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from hflow.storage import LocalStorageRoot, parse_storage_root

DEFAULT_HOST = "127.0.0.1"
# "HFLO" on a phone keypad; mirrored by the core CLI's DEFAULT_UI_PORT.
DEFAULT_PORT = 4356

# The TCP ports a launch may ask for, the same range (and the same reason for
# excluding 0) that ``hflow.runtime``'s RuntimeConfig enforces for the
# bundle's api port: 0 means "any free port" to bind(2), but this value is
# interpolated into the URL `serve` prints and hands to the browser, and
# http://127.0.0.1:0 is not dialable.
MIN_PORT = 1
MAX_PORT = 65535


@dataclass(frozen=True)
class UiSettings:
    """One ``hflow ui`` launch, fully parsed.

    ``data_root`` stays a string: it may be a local path or a bucket URL, and
    ``hflow.workspace.Workspace.parse`` owns that distinction. ``assets_dir``
    overrides where the built SPA is served from (tests and frontend dev).

    Nothing here is a credential: the server authenticates nobody, so ``host``
    is the whole access-control story (see docs/UI.md, "Trust posture").
    """

    data_root: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    assets_dir: Path | None = None
    open_browser: bool = True
    # When true, every mutating endpoint (manifest pinning, saved-query
    # writes, ingest triggering) answers 403 and /api/v1/config reports it
    # (CLI flag: --read-only).
    read_only: bool = False
    # ``path/to/pipeline.py[:app]`` (CLI flag: --pipeline). The server
    # imports -- EXECUTES -- this file exactly once at startup to serve
    # /api/v1/pipeline; ``None`` leaves that capability off.
    pipeline: str | None = None

    def __post_init__(self) -> None:
        # A range invariant of the field, checked where the field is set, so a
        # library caller building UiSettings directly gets the same answer as
        # the command line. Left to bind(2) instead, an out-of-range port
        # surfaces as an OverflowError from inside the port probe, and port 0
        # binds fine while printing a URL nobody can open.
        if not MIN_PORT <= self.port <= MAX_PORT:
            raise ValueError(f"port {self.port!r} is not in {MIN_PORT}-{MAX_PORT}")


def local_data_root_or_none(data_root: str) -> Path | None:
    """The data root as a local directory, or ``None`` for a bucket URL.

    The ONE derivation of "this workspace's files are reachable as paths" --
    the precondition media serving, the sidecar, and pinned manifest files all
    share. Each caller decides what to do without one (``_media`` and
    ``_sidecar`` refuse 501 in their own error type; /api/v1/config turns it
    into capability flags the frontend can hide affordances behind), but none
    of them re-derives the predicate.
    """
    parsed_root = parse_storage_root(data_root)
    return parsed_root.path if isinstance(parsed_root, LocalStorageRoot) else None


def refuse_when_read_only(settings: UiSettings, *, disabled_actions: str) -> None:
    """The 403 every mutating route owes a read-only launch.

    One owner for the status and the sentence, shared by the curation studio
    and the runs monitor; only the named actions differ, so a third mutating
    router cannot invent a third wording or a different code.
    ``disabled_actions`` carries its own agreeing verb ("... is" / "... are")
    because the routes name one action or several.
    """
    if settings.read_only:
        raise HTTPException(
            status_code=403,
            detail=f"this workspace UI is running read-only; {disabled_actions} disabled",
        )
