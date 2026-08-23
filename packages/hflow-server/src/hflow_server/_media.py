"""Media byte-serving: catalog URIs resolved and contained under the data root.

The browser never chooses a filesystem path. It addresses bytes as
(episode_id, artifact name); the URI comes out of the catalog, and the
strictly-resolved file must land inside the strictly-resolved local data
root -- anything else is refused, and a refusal never echoes the offending
path (only the containment fact appears in errors).
"""

import mimetypes
from pathlib import Path

from fastapi import HTTPException
from starlette.responses import FileResponse

from hflow.storage import is_bucket_url
from hflow.workspace import (
    CATALOG_DIRECTORY_NAME,
    EPISODES_DIRECTORY_NAME,
    TEST_RUNS_DIRECTORY_NAME,
)
from hflow_server._settings import local_data_root_or_none

# The layout directories a workspace's own files live under, owned by
# hflow.workspace. Used to recognise a path recorded from another vantage of
# this workspace (a container mount) and re-anchor it here.
_WORKSPACE_LAYOUT_DIRECTORY_NAMES = frozenset(
    {EPISODES_DIRECTORY_NAME, CATALOG_DIRECTORY_NAME, TEST_RUNS_DIRECTORY_NAME}
)

# Media types inert enough to render inline in the browser: raster images and
# common audio/video containers. Deliberately excludes text/html,
# image/svg+xml, and application/xhtml+xml -- an active document served from
# the UI's own origin runs script same-origin with this workspace's API and
# could drive every endpoint it exposes (read the catalog, pin manifests,
# trigger runs), so anything not on this list is forced to download.
_INLINE_SERVABLE_MEDIA_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/apng",
        "image/avif",
        "image/x-icon",
        "video/mp4",
        "video/webm",
        "video/ogg",
        "video/quicktime",
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/aac",
        "audio/mp4",
        "audio/flac",
    }
)

# Every byte-serving response carries these: no sniffing an octet-stream back
# into an active type, and a policy that denies script/resource loads even if
# a viewer opens the bytes directly.
_HARDENING_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
}


class MediaResolutionError(HTTPException):
    """One refusal to serve a catalog URI.

    It IS the HTTP refusal rather than something a caller converts into one:
    raised from anywhere under a route, FastAPI renders it with the status and
    detail given here. It stays its own type so
    :func:`is_uri_servable` can catch media refusals specifically, without also
    swallowing an unrelated refusal raised nearby.
    """


def _resolved_local_data_root(data_root: str) -> Path:
    local_root = local_data_root_or_none(data_root)
    if local_root is None:
        raise MediaResolutionError(
            501,
            "media serving requires a local data root; bucket-backed workspaces are not served yet",
        )
    return local_root.resolve()


def _strictly_resolved(candidate: Path) -> Path | None:
    """The real file behind a path, or ``None`` when it cannot be reached."""
    try:
        return candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None


def _rebased_onto_this_workspace(recorded_path: Path, resolved_data_root: Path) -> Path | None:
    """The same workspace-relative file, as THIS host addresses it.

    One workspace is reachable from several vantages: the Compose runtime
    mounts the data root inside its containers, so a run executed there
    catalogs ``/opt/airflow/data/episodes/<run>/media/x.jpg`` while the very
    same bytes sit at ``<data root>/episodes/<run>/media/x.jpg`` here. A path
    recorded from another vantage is not foreign data -- it is this
    workspace, named differently -- so it is re-anchored at the first
    workspace layout directory in it and re-checked exactly like any other
    candidate. Containment is still enforced afterwards, so this only ever
    resolves to files already inside the data root; it never widens what may
    be served.
    """
    for index, component in enumerate(recorded_path.parts):
        if component in _WORKSPACE_LAYOUT_DIRECTORY_NAMES:
            return resolved_data_root.joinpath(*recorded_path.parts[index:])
    return None


def resolve_served_file(uri: str, *, data_root: str) -> Path:
    """The real file a catalog URI may be served from, or a typed refusal.

    Resolution is strict (symlinks followed, missing components refused), and
    the result must be contained in the resolved data root -- a symlink that
    points out of the workspace is refused exactly like a foreign path. A URI
    recorded from another vantage of this same workspace (see
    :func:`_rebased_onto_this_workspace`) is retried against this host's data
    root under the identical containment rule.
    """
    if is_bucket_url(uri):
        raise MediaResolutionError(
            501, "this file lives in an object store; bucket media serving is not implemented yet"
        )
    resolved_data_root = _resolved_local_data_root(data_root)
    recorded_path = Path(uri.removeprefix("file://"))

    resolved_file = _strictly_resolved(recorded_path)
    escapes_workspace = resolved_file is not None and not resolved_file.is_relative_to(
        resolved_data_root
    )
    if resolved_file is None or escapes_workspace:
        rebased_path = _rebased_onto_this_workspace(recorded_path, resolved_data_root)
        rebased_file = None if rebased_path is None else _strictly_resolved(rebased_path)
        if rebased_file is not None and rebased_file.is_relative_to(resolved_data_root):
            resolved_file = rebased_file
        elif escapes_workspace:
            raise MediaResolutionError(
                403, "the cataloged URI resolves outside the workspace data root"
            )
        else:
            raise MediaResolutionError(404, "the cataloged file does not exist on this machine")

    if not resolved_file.is_file():
        raise MediaResolutionError(404, "the cataloged URI does not name a regular file")
    return resolved_file


def is_uri_servable(uri: str, *, data_root: str) -> bool:
    """Whether a GET for this URI would serve bytes (containment + existence)."""
    try:
        resolve_served_file(uri, data_root=data_root)
    except MediaResolutionError:
        return False
    return True


def served_file_response(
    resolved_file: Path, *, attachment_filename: str | None = None
) -> FileResponse:
    """Bytes served safely: an allowlisted inert media type renders inline;
    anything else (or an explicit ``attachment_filename``) is downloaded as
    opaque ``application/octet-stream``. Every response carries ``nosniff``
    and a locked-down CSP, so a workspace file whose name ends in .html/.svg
    can never execute as an active document on the UI's own origin.

    Starlette's FileResponse handles Range requests where it can, and a plain
    GET always works. Passing ``attachment_filename`` names the download
    (manifest exports)."""
    if attachment_filename is not None:
        return FileResponse(
            resolved_file,
            media_type="application/octet-stream",
            filename=attachment_filename,
            headers=dict(_HARDENING_HEADERS),
        )
    guessed_type, _ = mimetypes.guess_type(resolved_file.name)
    if guessed_type in _INLINE_SERVABLE_MEDIA_TYPES:
        return FileResponse(
            resolved_file, media_type=guessed_type, headers=dict(_HARDENING_HEADERS)
        )
    # Not a known-inert type: force a download so an .html/.svg/unknown file
    # is never rendered as an active document on this origin.
    return FileResponse(
        resolved_file,
        media_type="application/octet-stream",
        filename=resolved_file.name,
        headers=dict(_HARDENING_HEADERS),
    )
