"""Storage roots: the local-directory-or-bucket seam every data root crosses.

A data root names where episodes, artifacts, and the catalog live. It is
either a directory on the local filesystem or an object-store prefix
(``gs://``, ``s3://``, ``az://``...). :func:`parse_storage_root` parses the
user's value once, at the boundary; everything downstream holds a
:data:`StorageRoot` and never re-inspects strings.

Local roots stay pure-stdlib. Bucket roots are backed by
`obstore <https://github.com/developmentseed/obstore>`_ (the Rust
``object_store`` crate as a dependency-free wheel) behind the optional
``hflow[bucket]`` extra; the import happens lazily on first bucket I/O, so
local-only installs never need it. Credentials come from the provider's
standard environment (``GOOGLE_APPLICATION_CREDENTIALS`` or the GCE/GKE
metadata server for GCS, ``AWS_ACCESS_KEY_ID``/... for S3,
``AZURE_STORAGE_ACCOUNT_NAME``/... for Azure).

Processing model for bucket roots -- **spool through a mirror**: every bucket
root owns a local mirror directory laid out one-to-one with the bucket prefix
(``~/.cache/hflow/mirrors/<url hash>``, override the base with
``HFLOW_MIRROR_DIR``). Sources download into the mirror, the pipeline runs
on local files exactly as it does for a local root, and results upload back
to the same relative keys. Two conventions make the mirror a correct cache
rather than a consistency problem:

- Catalog files are append-only and content-named (create-if-absent), so a
  file that exists locally is final -- syncing means downloading only what
  the mirror lacks.
- Every other fetched file keeps an etag sidecar (``<name>.remote-etag``);
  :meth:`BucketStorageRoot.fetch` re-downloads only when the remote etag
  changed, so a stale mirror on one worker can never mask a re-synced
  episode uploaded by another.
"""

import errno
import fcntl
import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

# Schemes obstore's from_url() accepts that name an object store we support.
# file:// maps to a LocalStorageRoot instead (same semantics, no obstore);
# http(s) is excluded on purpose -- a bare web server has no create-if-absent
# or listing contract, so the catalog's durability idioms cannot hold there.
BUCKET_URL_SCHEMES = frozenset({"s3", "gs", "az", "abfs", "abfss", "azure"})

_ETAG_SIDECAR_SUFFIX = ".remote-etag"

_OBSTORE_INSTALL_HINT = (
    "bucket storage roots need the optional obstore backend -- install the "
    'extra: pip install "hflow[bucket]" (uv: uv add "hflow[bucket]")'
)


# Credential variables an environment can plausibly hand us EMPTY (compose
# `${VAR}` passthrough resolves unset variables to empty strings). An empty
# credential is never meaningful, but object_store treats present-but-empty
# values as credentials and breaks the metadata-server/instance-role
# fallback -- so they are dropped at the boundary where obstore first reads
# the environment.
_CREDENTIAL_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_ALLOW_HTTP",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_SERVICE_ACCOUNT",
    "GOOGLE_SERVICE_ACCOUNT_KEY",
    "AZURE_STORAGE_ACCOUNT_NAME",
    "AZURE_STORAGE_ACCOUNT_KEY",
    "AZURE_STORAGE_SAS_KEY",
    "AZURE_STORAGE_TOKEN",
)


def _load_obstore() -> Any:
    """Import obstore lazily so local-path roots never require the extra."""
    for variable_name in _CREDENTIAL_ENV_VARS:
        if os.environ.get(variable_name) == "":
            del os.environ[variable_name]
    try:
        import obstore
        import obstore.exceptions
        import obstore.store
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(_OBSTORE_INSTALL_HINT) from error
    return obstore


def is_bucket_url(value: str) -> bool:
    """Whether ``value`` is an object-store URL (``gs://bucket/...`` etc.)."""
    scheme, separator, _rest = value.partition("://")
    return bool(separator) and scheme.lower() in BUCKET_URL_SCHEMES


# Suffixes the mirror uses for its own bookkeeping files next to real
# entries; a real object with such a name would collide with them (e.g. its
# content silently replaced by etag text), so keys carrying them are refused.
_RESERVED_KEY_SUFFIXES = (".remote-etag", ".mirror-lock")


def _validated_relative_key(relative: str) -> str:
    """Parse a storage-relative key: POSIX-style, inside the root, non-empty.

    ``PurePosixPath.is_absolute()`` only recognizes a leading ``/`` --
    it has no concept of Windows drive letters, so a key like
    ``"C:/Windows/win.ini"`` passes this check as "relative". That matters
    because :class:`LocalStorageRoot` later joins a validated key onto the
    real root with ``self.path / key`` using the *native* platform's
    ``pathlib`` semantics: on native Windows, joining a drive-letter path
    onto an existing path does not append it -- pathlib's documented
    behavior discards the left operand entirely, so ``self.path / key``
    resolves to the drive-letter path outright, escaping the intended root.
    Checking ``PureWindowsPath.is_absolute()`` as well closes that gap
    regardless of which OS actually validates the key.
    """
    key = PurePosixPath(str(relative).replace("\\", "/"))
    if not str(key) or str(key) == ".":
        raise ValueError("storage key must not be empty")
    if key.is_absolute() or PureWindowsPath(str(key)).is_absolute() or ".." in key.parts:
        raise ValueError(
            f"storage key {relative!r} must be relative to the storage root "
            "(no leading '/', no '..', no Windows drive letter)"
        )
    if key.name.endswith(_RESERVED_KEY_SUFFIXES):
        raise ValueError(
            f"storage key {relative!r} ends with a suffix reserved for mirror "
            f"bookkeeping files ({', '.join(_RESERVED_KEY_SUFFIXES)}); rename the file"
        )
    return str(key)


def _default_mirror_directory(url: str) -> Path:
    """The local mirror for one bucket URL: stable across processes.

    Base directory precedence: ``HFLOW_MIRROR_DIR``, then
    ``$XDG_CACHE_HOME/hflow/mirrors``, then ``~/.cache/hflow/mirrors``.
    The per-URL subdirectory is a content hash of the normalized URL, so two
    roots never share a mirror and re-parsing the same URL always finds the
    same cache.
    """
    override = os.environ.get("HFLOW_MIRROR_DIR")
    if override:
        base_directory = Path(override)
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME")
        cache_base = Path(cache_home) if cache_home else Path.home() / ".cache"
        base_directory = cache_base / "hflow" / "mirrors"
    url_digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return base_directory / url_digest


def _apply_default_permissions(path: Path) -> None:
    """Give a temp-created file the conventional umask-honoring mode.

    ``NamedTemporaryFile``/``mkstemp`` create files ``0600`` regardless of
    umask; files renamed from them into a shared data root would then be
    unreadable to teammates and other tools (DuckDB over a team NFS mount).
    Secrets that genuinely need ``0600`` (the bundle ``.env``) set it
    themselves, deliberately.
    """
    current_umask = os.umask(0)
    os.umask(current_umask)
    path.chmod(0o666 & ~current_umask)


@contextmanager
def _mirror_entry_lock(entry_file: Path) -> Iterator[None]:
    """Serialize same-machine mutations of one mirror entry.

    A mirror entry is TWO files (payload + etag sidecar) written by two
    renames; without a lock, interleaved writers (parallel Airflow tasks, a
    dev run overlapping a scheduled one) could pair one version's bytes with
    another version's etag -- pinning stale content behind a current etag
    forever. Advisory ``flock``, POSIX-only, which covers every supported
    runtime (Windows runs via WSL2); the tiny ``.mirror-lock`` file persists
    next to the entry (its suffix is a reserved key suffix).
    """
    lock_file = entry_file.with_name(entry_file.name + ".mirror-lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)


def _write_local_bytes_atomically(destination: Path, data: bytes) -> None:
    """Replace one local file only after all bytes are written to a temp file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_stream:
        temporary_stream.write(data)
        temporary_path = Path(temporary_stream.name)
    try:
        _apply_default_permissions(temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _copy_local_file_atomically(source: Path, destination: Path) -> None:
    """Copy a local file without exposing a partial destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_stream:
        temporary_path = Path(temporary_stream.name)
    try:
        shutil.copyfile(source, temporary_path)
        _apply_default_permissions(temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _download_to_file_atomically(get_result: Any, destination: Path) -> None:
    """Stream one GET into ``destination``; failures leave nothing behind.

    The temp file is unlinked on EVERY exit path -- a mid-stream network
    failure must not leak a partial ``.downloading`` file into the mirror
    (retries over a flaky link would otherwise grow the cache unboundedly).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_stream = tempfile.NamedTemporaryFile(  # noqa: SIM115 -- closed by the with below
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".downloading",
        delete=False,
    )
    temporary_file = Path(temporary_stream.name)
    try:
        with temporary_stream:
            for chunk in get_result:
                temporary_stream.write(chunk)
        _apply_default_permissions(temporary_file)
        temporary_file.replace(destination)
    finally:
        temporary_file.unlink(missing_ok=True)


@dataclass(frozen=True)
class LocalStorageRoot:
    """A data root on the local filesystem -- the pure-stdlib fast path.

    Operations mirror :class:`BucketStorageRoot` exactly so orchestration
    code can hold either; here ``fetch`` and ``publish`` are (near) no-ops
    because the root already IS local.
    """

    path: Path

    def __str__(self) -> str:
        return str(self.path)

    @property
    def workspace(self) -> Path:
        """Where run outputs are staged locally: the root itself."""
        return self.path

    def child(self, name: str) -> "LocalStorageRoot":
        return LocalStorageRoot(self.path / _validated_relative_key(name))

    def exists(self, relative: str) -> bool:
        return (self.path / _validated_relative_key(relative)).is_file()

    def file_size(self, relative: str) -> int:
        return (self.path / _validated_relative_key(relative)).stat().st_size

    def read_bytes(self, relative: str) -> bytes:
        return (self.path / _validated_relative_key(relative)).read_bytes()

    def write_bytes(self, relative: str, data: bytes) -> None:
        destination = self.path / _validated_relative_key(relative)
        _write_local_bytes_atomically(destination, data)

    def write_bytes_if_absent(self, relative: str, data: bytes) -> bool:
        """Create the file with ``data`` unless it exists; True when created."""
        destination = self.path / _validated_relative_key(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_stream:
            temporary_stream.write(data)
            temporary_path = Path(temporary_stream.name)
        try:
            _apply_default_permissions(temporary_path)
            os.link(temporary_path, destination)
        except FileExistsError:
            return False
        finally:
            temporary_path.unlink(missing_ok=True)
        return True

    def store_file_if_absent(self, local_file: Path, relative: str) -> bool:
        """Place ``local_file``'s content at ``relative`` unless it exists.

        Local semantics match the catalog's historical append idiom: copy to
        a sibling temp file, then rename into place (atomic on POSIX), so a
        reader never observes a half-written file.
        """
        destination = self.path / _validated_relative_key(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_stream:
            temporary_destination = Path(temporary_stream.name)
        try:
            shutil.copyfile(local_file, temporary_destination)
            _apply_default_permissions(temporary_destination)
            try:
                os.link(temporary_destination, destination)
            except FileExistsError:
                return False
        finally:
            temporary_destination.unlink(missing_ok=True)
        return True

    def list_names(self, prefix: str = "") -> list[str]:
        """Relative POSIX keys of every file under ``prefix``, sorted."""
        base = self.path / _validated_relative_key(prefix) if prefix else self.path
        if not base.is_dir():
            return []
        return sorted(
            str(entry.relative_to(self.path).as_posix())
            for entry in base.rglob("*")
            if entry.is_file()
        )

    def fetch(self, relative: str) -> Path:
        """The local file at ``relative`` (it already lives here)."""
        local_file = self.path / _validated_relative_key(relative)
        if local_file.is_dir():
            # Deliberately FileNotFoundError and not IsADirectoryError: the accurate
            # class subclasses OSError, not FileNotFoundError, so an `except
            # FileNotFoundError` stops catching it. `_command_doctor` is one such
            # handler, reached through fetch_uri below, and it turns this into a
            # finding rather than a traceback; sixteen more name the class
            # elsewhere under src/hflow/, and callers outside the repo are the real
            # unknown. The errno carries the accurate text without moving the class.
            # Same decision as the four sites #102 covered. #144.
            raise FileNotFoundError(errno.EISDIR, os.strerror(errno.EISDIR), str(local_file))
        if not local_file.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(local_file))
        return local_file

    def uri_for(self, relative: str) -> str:
        """Absolute filesystem URI for a storage-relative file."""
        return str((self.path / _validated_relative_key(relative)).resolve())

    def delete(self, relative: str) -> None:
        """Remove a file if present."""
        (self.path / _validated_relative_key(relative)).unlink(missing_ok=True)

    def sync_into_mirror(self, prefixes: tuple[str, ...]) -> Path:
        """Local roots already are their complete materialized workspace."""
        del prefixes
        return self.path

    def publish(self, local_file: Path, relative: str) -> str:
        """Place ``local_file`` at ``relative``; returns the destination URI.

        A no-op when the file already is the destination (the common case for
        a local root: runs write in place).
        """
        destination = self.path / _validated_relative_key(relative)
        if local_file.resolve() != destination.resolve():
            _copy_local_file_atomically(local_file, destination)
        return self.uri_for(relative)


class BucketStorageRoot:
    """A data root at an object-store prefix, obstore-backed.

    See the module docstring for the spool-through-mirror model. Instances
    are cheap to construct; the store client is built lazily on first I/O
    (which is also where a missing ``hflow[bucket]`` extra is reported).

    :param url: The prefix URL, e.g. ``gs://bucket/robot-data`` (trailing
        slashes are stripped so joins never double them).
    :param mirror: The local mirror directory; defaults to a stable per-URL
        cache directory (see :func:`_default_mirror_directory`).
    :param store_options: Explicit per-root store configuration forwarded to
        ``obstore.store.from_url`` (credentials, region, endpoint overrides).
        ``None`` keeps today's behavior: the provider's ambient environment.
        This is the injection seam for scoped per-workspace credentials -- a
        control plane can hold several roots with distinct credentials in one
        process instead of mutating ``os.environ``. Values may be secrets, so
        they are never written to disk or included in ``repr``.
    """

    def __init__(
        self,
        url: str,
        *,
        mirror: Path | None = None,
        store_options: dict[str, Any] | None = None,
    ) -> None:
        normalized_url = url.rstrip("/")
        _scheme, separator, remainder = normalized_url.partition("://")
        # The bucket is the remainder up to the first slash, so gs:///prefix
        # names none and is refused the same way gs:// is. file:// is exempt
        # because its authority is empty by construction: the tests simulate a
        # bucket with file://<abs path> (tests/conftest.py), and requiring an
        # authority there fails 3 tests and errors 19 more.
        if not separator or (_scheme.lower() != "file" and not remainder.partition("/")[0]):
            raise ValueError(
                f"bucket URL {url!r} names no bucket/container -- expected e.g. gs://bucket/prefix"
            )
        self.url = normalized_url
        self.mirror = mirror if mirror is not None else _default_mirror_directory(normalized_url)
        self.store_options: dict[str, Any] = dict(store_options) if store_options else {}
        self._store: Any | None = None

    def __str__(self) -> str:
        return self.url

    def __repr__(self) -> str:
        return f"BucketStorageRoot({self.url!r}, mirror={str(self.mirror)!r})"

    @property
    def workspace(self) -> Path:
        """The local mirror directory run outputs are staged in (created)."""
        self.mirror.mkdir(parents=True, exist_ok=True)
        breadcrumb = self.mirror / ".mirror-of"
        if not breadcrumb.exists():
            # Debuggability only: says which URL a hash-named mirror caches.
            breadcrumb.write_text(self.url + "\n")
        return self.mirror

    def child(self, name: str) -> "BucketStorageRoot":
        validated_name = _validated_relative_key(name)
        return BucketStorageRoot(
            f"{self.url}/{validated_name}",
            mirror=self.mirror / validated_name,
            store_options=self.store_options or None,
        )

    def uri_for(self, relative: str) -> str:
        return f"{self.url}/{_validated_relative_key(relative)}"

    def _get_store(self) -> Any:
        if self._store is None:
            obstore = _load_obstore()
            # obstore's LocalStore test backend requires its root directory to
            # exist; real bucket prefixes are virtual and need no equivalent.
            store_options: dict[str, Any] = (
                {"mkdir": True} if self.url.startswith("file://") else {}
            )
            # Explicit per-root configuration wins over the defaults above;
            # anything NOT configured here still resolves from the provider's
            # ambient environment inside obstore.
            store_options.update(self.store_options)
            self._store = obstore.store.from_url(self.url, **store_options)
        return self._store

    def exists(self, relative: str) -> bool:
        obstore = _load_obstore()
        try:
            obstore.head(self._get_store(), _validated_relative_key(relative))
        except FileNotFoundError:
            return False
        return True

    def file_size(self, relative: str) -> int:
        obstore = _load_obstore()
        return int(obstore.head(self._get_store(), _validated_relative_key(relative))["size"])

    def read_bytes(self, relative: str) -> bytes:
        obstore = _load_obstore()
        return bytes(obstore.get(self._get_store(), _validated_relative_key(relative)).bytes())

    def write_bytes(self, relative: str, data: bytes) -> None:
        obstore = _load_obstore()
        obstore.put(self._get_store(), _validated_relative_key(relative), data)

    def write_bytes_if_absent(self, relative: str, data: bytes) -> bool:
        """Atomic create-if-absent (``PutMode::Create``); True when created.

        This is the object-store-native form of the catalog's append idiom:
        the store itself refuses the second writer, so two workers appending
        the same content-named file can never corrupt or duplicate it.
        """
        obstore = _load_obstore()
        try:
            obstore.put(self._get_store(), _validated_relative_key(relative), data, mode="create")
        except obstore.exceptions.AlreadyExistsError:
            return False
        return True

    def store_file_if_absent(self, local_file: Path, relative: str) -> bool:
        """Upload ``local_file`` at ``relative`` unless the object exists.

        On success the mirror is warmed with a copy, so a later read of the
        same file never re-downloads it.
        """
        key = _validated_relative_key(relative)
        obstore = _load_obstore()
        try:
            result = obstore.put(self._get_store(), key, local_file, mode="create")
        except obstore.exceptions.AlreadyExistsError:
            return False
        self._warm_mirror(local_file, key, result.get("e_tag"))
        return True

    def list_names(self, prefix: str = "") -> list[str]:
        """Relative POSIX keys of every object under ``prefix``, sorted."""
        obstore = _load_obstore()
        list_prefix = f"{_validated_relative_key(prefix)}/" if prefix else None
        names: list[str] = []
        for batch in obstore.list(self._get_store(), list_prefix):
            names.extend(str(meta["path"]) for meta in batch)
        return sorted(names)

    def fetch(self, relative: str) -> Path:
        """The object at ``relative`` as a local file in the mirror.

        Freshness: one HEAD compares the remote etag against the sidecar
        recorded at download (or publish) time; only a changed or missing
        object downloads. Raises ``FileNotFoundError`` (naming the full
        remote location) when the object does not exist.
        """
        key = _validated_relative_key(relative)
        local_file = self.mirror / key
        sidecar = local_file.with_name(local_file.name + _ETAG_SIDECAR_SUFFIX)
        obstore = _load_obstore()
        store = self._get_store()
        remote_etag = obstore.head(store, key)["e_tag"]
        # The lock covers check-through-sidecar: payload and sidecar are two
        # files, and an unsynchronized concurrent fetch/publish of the same
        # key could pair one version's bytes with another version's etag.
        with _mirror_entry_lock(local_file):
            if (
                local_file.is_file()
                and remote_etag is not None
                and sidecar.is_file()
                and sidecar.read_text() == remote_etag
            ):
                return local_file
            get_result = obstore.get(store, key)
            # The GET's own metadata etag (not the earlier HEAD's) goes into
            # the sidecar, so the recorded etag always matches the downloaded
            # bytes even if the object was replaced between the two requests.
            downloaded_etag = get_result.meta["e_tag"]
            _download_to_file_atomically(get_result, local_file)
            if downloaded_etag is not None:
                _write_local_bytes_atomically(sidecar, downloaded_etag.encode())
            elif sidecar.exists():
                sidecar.unlink()
        return local_file

    def publish(self, local_file: Path, relative: str) -> str:
        """Upload ``local_file`` to ``relative`` (overwrite); returns its URL.

        Overwrite is the correct verb here: canonical episodes and artifacts
        are re-published when their inputs or step versions change, and the
        catalog -- not object identity -- carries the versioning story.
        """
        key = _validated_relative_key(relative)
        obstore = _load_obstore()
        result = obstore.put(self._get_store(), key, local_file)
        self._warm_mirror(local_file, key, result.get("e_tag"))
        return f"{self.url}/{key}"

    def delete(self, relative: str) -> None:
        """Remove an object and its mirror entry if either exists."""
        key = _validated_relative_key(relative)
        obstore = _load_obstore()
        with suppress(FileNotFoundError):
            obstore.delete(self._get_store(), key)
        mirror_file = self.mirror / key
        mirror_file.unlink(missing_ok=True)
        mirror_file.with_name(mirror_file.name + _ETAG_SIDECAR_SUFFIX).unlink(missing_ok=True)

    def sync_into_mirror(self, prefixes: tuple[str, ...]) -> Path:
        """Download everything under ``prefixes`` the mirror lacks; return it.

        Existence is the only check: this is reserved for append-only,
        content-named trees (the catalog tables), where a file that exists
        locally is final by convention. Mutable files go through
        :meth:`fetch` and its etag sidecar instead.
        """
        obstore = _load_obstore()
        store = self._get_store()
        self.workspace  # noqa: B018  -- ensures the mirror directory exists
        for prefix in prefixes:
            for name in self.list_names(prefix):
                local_file = self.mirror / name
                if local_file.is_file():
                    continue
                # No lock or sidecar: immutable-file convention -- concurrent
                # syncers can only write identical bytes, and the replace is
                # atomic either way.
                _download_to_file_atomically(obstore.get(store, name), local_file)
        return self.mirror

    def _warm_mirror(self, local_file: Path, key: str, etag: str | None) -> None:
        """After an upload, make the mirror agree with what was uploaded."""
        mirror_file = self.mirror / key
        # Same lock as fetch(): payload + sidecar must change as one unit.
        with _mirror_entry_lock(mirror_file):
            if local_file.resolve() != mirror_file.resolve():
                _copy_local_file_atomically(local_file, mirror_file)
            sidecar = mirror_file.with_name(mirror_file.name + _ETAG_SIDECAR_SUFFIX)
            if etag is not None:
                _write_local_bytes_atomically(sidecar, etag.encode())
            elif sidecar.exists():
                sidecar.unlink()


StorageRoot = LocalStorageRoot | BucketStorageRoot


def parse_storage_root(value: "str | Path | StorageRoot") -> StorageRoot:
    """Parse a data-root value at the boundary into its storage type.

    Accepted forms: an existing :data:`StorageRoot` (passed through), a
    ``Path`` or plain path string (local), ``file://`` URLs (local), and
    object-store URLs with a scheme in :data:`BUCKET_URL_SCHEMES` (bucket).
    Anything else -- above all a typo'd or unsupported scheme -- is refused
    loudly rather than treated as a strange directory name.
    """
    if isinstance(value, LocalStorageRoot | BucketStorageRoot):
        return value
    if isinstance(value, Path):
        return LocalStorageRoot(value)
    scheme, separator, remainder = value.partition("://")
    if not separator:
        if not value.strip():
            raise ValueError(f"storage root {value!r} must be a directory path or bucket URL")
        return LocalStorageRoot(Path(value))
    normalized_scheme = scheme.lower()
    if normalized_scheme == "file":
        if not remainder.startswith("/"):
            raise ValueError(
                f"file URL {value!r} must be absolute (file:///abs/path); "
                "for a relative directory pass the plain path"
            )
        return LocalStorageRoot(Path(remainder))
    if normalized_scheme in BUCKET_URL_SCHEMES:
        return BucketStorageRoot(value)
    raise ValueError(
        f"unsupported storage scheme {scheme!r} in {value!r} -- supported: a local path, "
        f"file://, or an object-store URL ({', '.join(sorted(BUCKET_URL_SCHEMES))})"
    )


def fetch_uri(uri: "str | Path") -> Path:
    """One-off fetch of a full URI (or local path) to a readable local file.

    Bucket URLs spool into the mirror of their parent prefix (etag-cached, so
    repeated fetches of an unchanged object cost one HEAD). Used by consumers
    that take a single file rather than a data root -- ``hflow doctor``,
    one-off ``app.process("gs://...")`` sources.
    """
    if isinstance(uri, str) and is_bucket_url(uri):
        parent_url, _, name = uri.rpartition("/")
        if not name:
            raise ValueError(f"bucket URI {uri!r} names no object")
        return BucketStorageRoot(parent_url).fetch(name)
    local_file = Path(uri)
    if local_file.is_dir():
        # FileNotFoundError, not IsADirectoryError, for the reason above.
        raise FileNotFoundError(errno.EISDIR, os.strerror(errno.EISDIR), str(local_file))
    if not local_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(local_file))
    return local_file
