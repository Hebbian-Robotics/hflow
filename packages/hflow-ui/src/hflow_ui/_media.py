"""Media byte-serving: catalog URIs resolved and contained under the data root.

The browser never chooses a filesystem path. It addresses bytes as
(episode_id, artifact name); the URI comes out of the catalog, and the
strictly-resolved file must land inside the strictly-resolved local data
root -- anything else is refused, and a refusal never echoes the offending
path (only the containment fact appears in errors).
"""

import mimetypes
from pathlib import Path

from starlette.responses import FileResponse

from hflow.storage import LocalStorageRoot, is_bucket_url, parse_storage_root

# Media types inert enough to render inline in the browser: raster images and
# common audio/video containers. Deliberately excludes text/html,
# image/svg+xml, and application/xhtml+xml -- an active document served from
# the UI's own origin could read the session token and issue authenticated
# same-origin calls, so anything not on this list is forced to download.
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


class MediaResolutionError(Exception):
    """One refusal to serve a catalog URI, carrying its HTTP mapping."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _resolved_local_data_root(data_root: str) -> Path:
    workspace_root = parse_storage_root(data_root)
    if not isinstance(workspace_root, LocalStorageRoot):
        raise MediaResolutionError(
            501,
            "media serving requires a local data root; "
            "bucket-backed workspaces are not served in M0",
        )
    return workspace_root.path.resolve()


def resolve_served_file(uri: str, *, data_root: str) -> Path:
    """The real file a catalog URI may be served from, or a typed refusal.

    Resolution is strict (symlinks followed, missing components refused), and
    the result must be contained in the resolved data root -- a symlink that
    points out of the workspace is refused exactly like a foreign path.
    """
    if is_bucket_url(uri):
        raise MediaResolutionError(
            501, "this file lives in an object store; bucket media serving is not implemented in M0"
        )
    resolved_data_root = _resolved_local_data_root(data_root)
    try:
        resolved_file = Path(uri.removeprefix("file://")).resolve(strict=True)
    except FileNotFoundError:
        raise MediaResolutionError(
            404, "the cataloged file does not exist on this machine"
        ) from None
    except OSError:
        raise MediaResolutionError(404, "the cataloged file is unreachable") from None
    if not resolved_file.is_relative_to(resolved_data_root):
        raise MediaResolutionError(
            403, "the cataloged URI resolves outside the workspace data root"
        )
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
