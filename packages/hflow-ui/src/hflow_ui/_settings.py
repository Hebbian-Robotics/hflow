"""Launch configuration for the workspace UI server."""

import secrets
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
# "HFLO" on a phone keypad; mirrored by the core CLI's DEFAULT_UI_PORT.
DEFAULT_PORT = 4356


def new_session_token() -> str:
    """Mint one launch's browser session token (URL-safe, 256 bits)."""
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class UiSettings:
    """One ``hflow ui`` launch, fully parsed.

    ``data_root`` stays a string: it may be a local path or a bucket URL, and
    ``hflow.workspace.Workspace.parse`` owns that distinction. ``token=None``
    disables session auth entirely (trusted loopback only); ``assets_dir``
    overrides where the built SPA is served from (tests and frontend dev).
    """

    data_root: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: str | None = None
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
