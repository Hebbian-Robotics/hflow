"""Storage roots: parsing, local ops, and the bucket spool-through-mirror model.

Bucket tests run obstore's LocalStore backend over a temp directory via a
``file://`` URL passed DIRECTLY to BucketStorageRoot -- every bucket code
path (puts, create-if-absent, etag-sidecar fetches, mirror sync) executes
for real with no network. (``parse_storage_root`` maps ``file://`` to a
LOCAL root on purpose; the direct construction here is the test seam.)

A live object-store integration test runs only when
``HFLOW_TEST_BUCKET_URL`` is set (e.g. ``gs://bucket/tmp-prefix``).
"""

import errno
import os
from pathlib import Path

import pytest
from mcap.exceptions import InvalidMagic

import hflow
from hflow.storage import (
    BucketStorageRoot,
    LocalStorageRoot,
    fetch_uri,
    is_bucket_url,
    parse_storage_root,
)
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

pytest.importorskip("obstore", reason="bucket tests need the hflow[bucket] extra")


class TestParseStorageRoot:
    def test_path_and_plain_string_are_local(self) -> None:
        assert parse_storage_root(Path("/tmp/x")) == LocalStorageRoot(Path("/tmp/x"))
        assert parse_storage_root("./data") == LocalStorageRoot(Path("./data"))
        assert parse_storage_root(".") == LocalStorageRoot(Path())

    def test_existing_roots_pass_through(self) -> None:
        local_root = LocalStorageRoot(Path("/tmp/x"))
        assert parse_storage_root(local_root) is local_root
        bucket_root = BucketStorageRoot("gs://bucket/prefix")
        assert parse_storage_root(bucket_root) is bucket_root

    def test_bucket_url_normalizes_trailing_slash(self) -> None:
        root = parse_storage_root("gs://bucket/prefix/")
        assert isinstance(root, BucketStorageRoot)
        assert root.url == "gs://bucket/prefix"
        assert str(root) == "gs://bucket/prefix"

    def test_file_url_is_local(self) -> None:
        assert parse_storage_root("file:///tmp/x") == LocalStorageRoot(Path("/tmp/x"))

    def test_relative_file_url_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be absolute"):
            parse_storage_root("file://relative/path")

    def test_unknown_scheme_is_refused_loudly(self) -> None:
        with pytest.raises(ValueError, match="unsupported storage scheme 'ftp'"):
            parse_storage_root("ftp://host/x")

    def test_bucketless_url_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names no bucket"):
            parse_storage_root("s3://")
        with pytest.raises(ValueError, match="names no bucket"):
            parse_storage_root("gs:///x")
        with pytest.raises(ValueError, match="names no bucket"):
            parse_storage_root("gs:////x")

    def test_a_file_url_still_builds_a_bucket_root_directly(self) -> None:
        """The exemption the bucket fixtures stand on.

        ``file://<abs path>`` has an empty authority by construction, so the
        bucket check has to skip it. Every bucket test simulates its store this
        way (see this module's docstring and ``tests/conftest.py``); requiring
        an authority here fails 3 tests and errors 19 more, none of which name
        the URL as the cause.
        """
        assert BucketStorageRoot("file:///tmp/remote").url == "file:///tmp/remote"

    def test_empty_local_path_is_refused(self) -> None:
        with pytest.raises(
            ValueError, match=r"storage root .* must be a directory path or bucket URL"
        ):
            parse_storage_root("")
        with pytest.raises(
            ValueError, match="storage root '   ' must be a directory path or bucket URL"
        ):
            parse_storage_root("   ")

    def test_is_bucket_url(self) -> None:
        assert is_bucket_url("gs://b/x.mcap")
        assert is_bucket_url("s3://b/x.mcap")
        assert not is_bucket_url("file:///x.mcap")
        assert not is_bucket_url("/abs/path.mcap")
        assert not is_bucket_url("relative/path.mcap")


class TestLocalStorageRoot:
    def test_round_trip_and_listing(self, tmp_path: Path) -> None:
        root = LocalStorageRoot(tmp_path)
        root.write_bytes("catalog/episodes/a.parquet", b"aa")
        assert root.exists("catalog/episodes/a.parquet")
        assert not root.exists("catalog/episodes/missing.parquet")
        assert root.read_bytes("catalog/episodes/a.parquet") == b"aa"
        assert root.file_size("catalog/episodes/a.parquet") == 2
        assert root.list_names("catalog") == ["catalog/episodes/a.parquet"]
        assert root.list_names("nowhere") == []

    def test_write_bytes_if_absent(self, tmp_path: Path) -> None:
        root = LocalStorageRoot(tmp_path)
        assert root.write_bytes_if_absent("m", b"first") is True
        assert root.write_bytes_if_absent("m", b"second") is False
        assert root.read_bytes("m") == b"first"

    def test_store_file_if_absent(self, tmp_path: Path) -> None:
        root = LocalStorageRoot(tmp_path / "root")
        source_file = tmp_path / "src.parquet"
        source_file.write_bytes(b"rows")
        assert root.store_file_if_absent(source_file, "episodes/x.parquet") is True
        assert root.store_file_if_absent(source_file, "episodes/x.parquet") is False
        assert root.read_bytes("episodes/x.parquet") == b"rows"

    def test_fetch_returns_file_in_place(self, tmp_path: Path) -> None:
        root = LocalStorageRoot(tmp_path)
        root.write_bytes("landing/e.mcap", b"episode")
        assert root.fetch("landing/e.mcap") == tmp_path / "landing" / "e.mcap"
        with pytest.raises(FileNotFoundError):
            root.fetch("landing/missing.mcap")

    def test_fetch_missing_has_oserror_filename(self, tmp_path: Path) -> None:
        root = LocalStorageRoot(tmp_path)
        missing = tmp_path / "landing" / "missing.mcap"
        with pytest.raises(FileNotFoundError) as excinfo:
            root.fetch("landing/missing.mcap")
        assert excinfo.value.errno == errno.ENOENT
        assert excinfo.value.filename == str(missing)
        assert "No such file or directory" in str(excinfo.value)
        assert str(missing) in str(excinfo.value)

    def test_fetch_directory_says_is_a_directory(self, tmp_path: Path) -> None:
        """A directory exists, so ENOENT was the wrong reason (#144)."""
        root = LocalStorageRoot(tmp_path)
        a_directory = tmp_path / "landing"
        a_directory.mkdir()
        with pytest.raises(FileNotFoundError) as excinfo:
            root.fetch("landing")
        assert excinfo.value.errno == errno.EISDIR
        assert excinfo.value.filename == str(a_directory)
        assert "Is a directory" in str(excinfo.value)

    def test_publish_copies_only_when_needed(self, tmp_path: Path) -> None:
        root = LocalStorageRoot(tmp_path / "root")
        outside_file = tmp_path / "made-elsewhere.bin"
        outside_file.write_bytes(b"payload")
        uri = root.publish(outside_file, "episodes/e/canonical.mcap")
        assert uri == str((tmp_path / "root" / "episodes" / "e" / "canonical.mcap").resolve())
        assert root.read_bytes("episodes/e/canonical.mcap") == b"payload"
        in_place = root.fetch("episodes/e/canonical.mcap")
        assert root.publish(in_place, "episodes/e/canonical.mcap") == str(in_place.resolve())

    def test_workspace_and_child(self, tmp_path: Path) -> None:
        root = LocalStorageRoot(tmp_path)
        assert root.workspace == tmp_path
        assert root.child("catalog") == LocalStorageRoot(tmp_path / "catalog")

    def test_escaping_keys_are_refused(self, tmp_path: Path) -> None:
        root = LocalStorageRoot(tmp_path)
        with pytest.raises(ValueError, match="relative to the storage root"):
            root.read_bytes("../outside")
        with pytest.raises(ValueError, match="relative to the storage root"):
            root.read_bytes("/absolute")
        with pytest.raises(ValueError, match="must not be empty"):
            root.read_bytes("")

    def test_windows_drive_letter_keys_are_refused(self, tmp_path: Path) -> None:
        """Storage keys with any Windows drive component are refused.

        ``PurePosixPath`` has no concept of Windows drive letters, so keys
        like ``"C:/Windows/win.ini"`` pass the POSIX absolute check as
        "relative". This guard also checks ``PureWindowsPath(...).drive``
        so that the validator is platform-independent: a key produced on a
        Windows client is refused even when the validating process runs on
        Linux.

        The right predicate is ``.drive``, not ``.is_absolute()``. A
        drive-relative key like ``"C:foo.txt"`` has a drive but no root, so
        ``.is_absolute()`` returns False and the key would slip through.
        Keying on ``.drive`` catches drive-absolute, drive-relative, and
        UNC shapes.
        """
        root = LocalStorageRoot(tmp_path)
        # drive-absolute forms
        with pytest.raises(ValueError, match="relative to the storage root"):
            root.read_bytes("C:/Windows/win.ini")
        with pytest.raises(ValueError, match="relative to the storage root"):
            root.read_bytes("C:\\Windows\\win.ini")
        with pytest.raises(ValueError, match="relative to the storage root"):
            root.read_bytes("D:/secrets.txt")
        # drive-relative forms: has a drive but no leading slash; is_absolute()
        # returns False for these, so they used to slip through
        with pytest.raises(ValueError, match="relative to the storage root"):
            root.read_bytes("C:foo.txt")
        with pytest.raises(ValueError, match="relative to the storage root"):
            root.read_bytes("C:")
        # A colon that is not drive-letter-shaped must still be accepted.
        # (Native Windows forbids colons in filenames at the OS level, so
        # this write is gated to platforms where it can succeed.)
        import sys

        if sys.platform != "win32":
            root.write_bytes("weird/C:notdrive.txt", b"payload")
            assert root.read_bytes("weird/C:notdrive.txt") == b"payload"


class TestBucketStorageRoot:
    def test_round_trip_and_listing(self, bucket_over_tmp: tuple[BucketStorageRoot, Path]) -> None:
        root, remote_dir = bucket_over_tmp
        root.write_bytes("catalog/episodes/a.parquet", b"aa")
        assert (remote_dir / "catalog" / "episodes" / "a.parquet").read_bytes() == b"aa"
        assert root.exists("catalog/episodes/a.parquet")
        assert not root.exists("missing")
        assert root.read_bytes("catalog/episodes/a.parquet") == b"aa"
        assert root.file_size("catalog/episodes/a.parquet") == 2
        assert root.list_names("catalog") == ["catalog/episodes/a.parquet"]
        assert root.list_names() == ["catalog/episodes/a.parquet"]

    def test_write_bytes_if_absent_is_store_native(
        self, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, _ = bucket_over_tmp
        assert root.write_bytes_if_absent("marker", b"v1") is True
        assert root.write_bytes_if_absent("marker", b"v2") is False
        assert root.read_bytes("marker") == b"v1"

    def test_store_file_if_absent_warms_the_mirror(
        self, tmp_path: Path, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, _ = bucket_over_tmp
        source_file = tmp_path / "row.parquet"
        source_file.write_bytes(b"rows")
        assert root.store_file_if_absent(source_file, "catalog/tags/x.parquet") is True
        assert (root.mirror / "catalog" / "tags" / "x.parquet").read_bytes() == b"rows"
        assert root.store_file_if_absent(source_file, "catalog/tags/x.parquet") is False

    def test_fetch_downloads_once_then_reuses_etag(
        self, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, _ = bucket_over_tmp
        root.write_bytes("landing/e.mcap", b"episode-bytes")
        first = root.fetch("landing/e.mcap")
        assert first == root.mirror / "landing" / "e.mcap"
        assert first.read_bytes() == b"episode-bytes"
        sidecar = first.with_name(first.name + ".remote-etag")
        assert sidecar.is_file()
        # Poison the local copy: an etag hit must NOT re-download over it.
        first.write_bytes(b"locally-poisoned")
        assert root.fetch("landing/e.mcap").read_bytes() == b"locally-poisoned"

    def test_fetch_redownloads_when_remote_changed(
        self, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, _ = bucket_over_tmp
        root.write_bytes("landing/e.mcap", b"version-one")
        assert root.fetch("landing/e.mcap").read_bytes() == b"version-one"
        root.write_bytes("landing/e.mcap", b"version-two!")  # changed size => changed etag
        assert root.fetch("landing/e.mcap").read_bytes() == b"version-two!"

    def test_fetch_missing_raises_file_not_found(
        self, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, _ = bucket_over_tmp
        with pytest.raises(FileNotFoundError):
            root.fetch("landing/missing.mcap")

    def test_fetch_local_missing_has_oserror_filename(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.mcap"
        with pytest.raises(FileNotFoundError) as excinfo:
            fetch_uri(missing)
        assert excinfo.value.errno == 2
        assert "No such file or directory" in str(excinfo.value)
        assert str(missing) in str(excinfo.value)

    def test_fetch_local_directory_says_is_a_directory(self, tmp_path: Path) -> None:
        """`hflow doctor <a directory>` reaches this and reported ENOENT (#144)."""
        a_directory = tmp_path / "adir"
        a_directory.mkdir()
        with pytest.raises(FileNotFoundError) as excinfo:
            fetch_uri(a_directory)
        assert excinfo.value.errno == errno.EISDIR
        assert excinfo.value.filename == str(a_directory)
        assert "Is a directory" in str(excinfo.value)

    def test_publish_uploads_and_warms_mirror(
        self, tmp_path: Path, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, remote_dir = bucket_over_tmp
        produced_file = tmp_path / "out.mcap"
        produced_file.write_bytes(b"canonical")
        uri = root.publish(produced_file, "episodes/e/e.canonical.mcap")
        assert uri == f"{root.url}/episodes/e/e.canonical.mcap"
        assert (remote_dir / "episodes" / "e" / "e.canonical.mcap").read_bytes() == b"canonical"
        # A fetch right after a publish is served from the warmed mirror.
        assert root.fetch("episodes/e/e.canonical.mcap").read_bytes() == b"canonical"

    def test_publish_from_inside_mirror(
        self, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, remote_dir = bucket_over_tmp
        staged_file = root.workspace / "episodes" / "e" / "e.canonical.mcap"
        staged_file.parent.mkdir(parents=True)
        staged_file.write_bytes(b"staged")
        root.publish(staged_file, "episodes/e/e.canonical.mcap")
        assert (remote_dir / "episodes" / "e" / "e.canonical.mcap").read_bytes() == b"staged"

    def test_sync_into_mirror_downloads_only_missing(
        self, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, _ = bucket_over_tmp
        root.write_bytes("catalog/episodes/a.parquet", b"aa")
        root.write_bytes("catalog/tags/t.parquet", b"tt")
        mirror = root.sync_into_mirror(("catalog/episodes", "catalog/tags"))
        assert (mirror / "catalog" / "episodes" / "a.parquet").read_bytes() == b"aa"
        # Immutable-file convention: an existing local file is never re-read.
        (mirror / "catalog" / "tags" / "t.parquet").write_bytes(b"local-final")
        root.sync_into_mirror(("catalog/tags",))
        assert (mirror / "catalog" / "tags" / "t.parquet").read_bytes() == b"local-final"

    def test_child_shares_the_mirror_subtree(
        self, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, _ = bucket_over_tmp
        catalog_child = root.child("catalog")
        assert catalog_child.url == f"{root.url}/catalog"
        assert catalog_child.mirror == root.mirror / "catalog"

    def test_workspace_writes_breadcrumb(
        self, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        root, _ = bucket_over_tmp
        workspace = root.workspace
        assert (workspace / ".mirror-of").read_text().strip() == root.url

    def test_uri_for(self, bucket_over_tmp: tuple[BucketStorageRoot, Path]) -> None:
        root, _ = bucket_over_tmp
        assert root.uri_for("episodes/e.mcap") == f"{root.url}/episodes/e.mcap"

    def test_store_options_configure_the_store_and_survive_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit store options reach obstore and propagate to child roots.

        This is the credential-injection seam a control plane uses to scope
        per-workspace credentials without mutating the process environment.
        """
        import hflow.storage as storage_module

        real_obstore = storage_module._load_obstore()
        received_options: list[dict[str, object]] = []
        real_from_url = real_obstore.store.from_url

        def capturing_from_url(url: str, **options: object) -> object:
            received_options.append(dict(options))
            return real_from_url(url, **options)

        monkeypatch.setattr(real_obstore.store, "from_url", capturing_from_url)

        remote_dir = tmp_path / "bucket"
        remote_dir.mkdir()
        root = BucketStorageRoot(
            f"file://{remote_dir}",
            mirror=tmp_path / "mirror",
            store_options={"automatic_cleanup": True},
        )
        root.write_bytes("episodes/e.bin", b"payload")
        assert root.read_bytes("episodes/e.bin") == b"payload"
        assert received_options[-1]["automatic_cleanup"] is True
        # file:// keeps its mkdir default unless the caller overrides it.
        assert received_options[-1]["mkdir"] is True

        catalog_child = root.child("catalog")
        assert catalog_child.store_options == {"automatic_cleanup": True}
        catalog_child.write_bytes("episodes/a.parquet", b"aa")
        assert received_options[-1]["automatic_cleanup"] is True


class TestFetchUri:
    def test_local_path_passes_through(self, tmp_path: Path) -> None:
        local_file = tmp_path / "e.mcap"
        local_file.write_bytes(b"x")
        assert fetch_uri(local_file) == local_file
        assert fetch_uri(str(local_file)) == local_file
        with pytest.raises(FileNotFoundError):
            fetch_uri(tmp_path / "missing.mcap")

    def test_default_mirror_is_stable_per_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HFLOW_MIRROR_DIR", str(tmp_path / "mirrors"))
        first = BucketStorageRoot("gs://bucket/prefix")
        second = BucketStorageRoot("gs://bucket/prefix/")
        other = BucketStorageRoot("gs://bucket/other")
        assert first.mirror == second.mirror
        assert first.mirror != other.mirror
        assert first.mirror.is_relative_to(tmp_path / "mirrors")


class TestBucketPipeline:
    def test_process_publishes_episode_marker_and_catalog(
        self, tmp_path: Path, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        data_root, remote_directory = bucket_over_tmp
        source_path = synthesize_episode(
            tmp_path / "source.mcap",
            SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
        )
        app = hflow.App("bucket-pipeline", data_root=data_root)

        @app.check(version="1", name="always-good")
        def always_good(_episode: hflow.Episode) -> hflow.CheckResult:
            return hflow.CheckResult(verdict=True, measurements={"score": 1.0})

        @app.enrich(version="1", name="write-mask")
        def write_mask(_episode: hflow.Episode) -> hflow.EnrichmentResult:
            artifact_path = tmp_path / "mask.bin"
            artifact_path.write_bytes(b"mask")
            return hflow.EnrichmentResult(artifacts={"left/mask": artifact_path})

        first_report = app.process(source_path)
        episode_directories = list((remote_directory / "episodes").iterdir())
        assert len(episode_directories) == 1
        episode_directory = episode_directories[0]
        assert (episode_directory / "source.canonical.mcap").is_file()
        assert (episode_directory / ".sync-complete.json").is_file()
        published_artifacts = list((episode_directory / "artifacts").rglob("*.bin"))
        assert len(published_artifacts) == 1
        assert published_artifacts[0].read_bytes() == b"mask"
        assert first_report.catalog_entry is not None
        assert first_report.catalog_entry.written is True

        connection = hflow.open_catalog_connection(data_root.child("catalog"))
        try:
            stored_row = connection.execute("SELECT uri, score FROM episodes").fetchone()
            artifact_row = connection.execute(
                "SELECT value_text FROM measurements WHERE key = 'artifact/left/mask'"
            ).fetchone()
        finally:
            connection.close()
        assert stored_row is not None
        stored_uri, score = stored_row
        assert stored_uri.startswith(f"{data_root.url}/episodes/")
        assert score == 1.0
        assert artifact_row is not None
        assert artifact_row[0].startswith(f"{data_root.url}/episodes/")

        repeated_report = app.process(source_path, stages={hflow.Stage.META, hflow.Stage.LABELS})
        assert repeated_report.catalog_entry is not None
        assert repeated_report.catalog_entry.written is False

    def test_failed_bucket_sync_removes_remote_completion_proof(
        self, tmp_path: Path, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        data_root, remote_directory = bucket_over_tmp
        source_path = synthesize_episode(
            tmp_path / "source.mcap",
            SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
        )
        app = hflow.App("bucket-sync-proof", data_root=data_root)
        app.process(source_path, record=False, stages={hflow.Stage.SYNC})
        episode_directory = next((remote_directory / "episodes").iterdir())
        published_canonical = episode_directory / "source.canonical.mcap"
        previous_bytes = published_canonical.read_bytes()

        source_path.write_bytes(b"not an mcap")
        with pytest.raises(InvalidMagic):
            app.process(source_path, record=False, stages={hflow.Stage.SYNC})

        assert published_canonical.read_bytes() == previous_bytes
        assert not (episode_directory / ".sync-complete.json").exists()
        with pytest.raises(FileNotFoundError, match="sync completion marker"):
            app.process(source_path, record=False, stages={hflow.Stage.META})


@pytest.mark.skipif(
    not os.environ.get("HFLOW_TEST_BUCKET_URL"),
    reason="set HFLOW_TEST_BUCKET_URL=gs://bucket/tmp-prefix to run live",
)
class TestLiveBucket:
    def test_live_round_trip(self, tmp_path: Path) -> None:
        base_url = os.environ["HFLOW_TEST_BUCKET_URL"].rstrip("/")
        root = BucketStorageRoot(f"{base_url}/storage-live-test", mirror=tmp_path / "mirror")
        key = "round-trip/probe.bin"
        try:
            root.write_bytes(key, b"live-payload")
            assert root.exists(key)
            assert root.file_size(key) == len(b"live-payload")
            assert root.fetch(key).read_bytes() == b"live-payload"
            assert root.write_bytes_if_absent(key, b"other") is False
            assert key in root.list_names("round-trip")
        finally:
            import obstore

            obstore.delete(root._get_store(), key)


class TestHardening:
    """Regressions from the adversarial review of the bucket feature."""

    def test_reserved_suffix_keys_are_refused(
        self, tmp_path: Path, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        # A real object named like a sidecar would collide with mirror
        # bookkeeping (fetch would overwrite it with etag text).
        root, _ = bucket_over_tmp
        with pytest.raises(ValueError, match="reserved"):
            root.write_bytes("landing/e.mcap.remote-etag", b"x")
        with pytest.raises(ValueError, match="reserved"):
            LocalStorageRoot(tmp_path).read_bytes("a/b.mirror-lock")

    def test_failed_download_leaves_no_temp_files(self, tmp_path: Path) -> None:
        from hflow.storage import _download_to_file_atomically

        class ExplodingStream:
            def __iter__(self):  # noqa: ANN204
                yield b"first-chunk"
                raise ConnectionError("link dropped mid-stream")

        destination = tmp_path / "mirror" / "episodes" / "e.mcap"
        with pytest.raises(ConnectionError):
            _download_to_file_atomically(ExplodingStream(), destination)
        assert not destination.exists()
        leftovers = list((tmp_path / "mirror").rglob("*"))
        assert [entry for entry in leftovers if entry.is_file()] == []

    def test_local_writes_honor_umask(self, tmp_path: Path) -> None:
        previous_umask = os.umask(0o022)
        try:
            root = LocalStorageRoot(tmp_path)
            root.write_bytes("catalog/format_version", b"v")
            source_file = tmp_path / "src.parquet"
            source_file.write_bytes(b"rows")
            root.store_file_if_absent(source_file, "catalog/episodes/x.parquet")
            for relative in ("catalog/format_version", "catalog/episodes/x.parquet"):
                mode = (tmp_path / relative).stat().st_mode & 0o777
                assert mode == 0o644, f"{relative} has mode {oct(mode)}"
        finally:
            os.umask(previous_umask)

    def test_empty_credential_env_vars_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Compose ${VAR} passthrough resolves unset variables to "" -- which
        # object_store would treat as a present credential, breaking the
        # metadata-server fallback. The obstore boundary drops empty values.
        from hflow.storage import _load_obstore

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "real-looking-value")
        _load_obstore()
        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "real-looking-value"

    def test_fetch_and_publish_share_a_mirror_entry_lock(
        self, tmp_path: Path, bucket_over_tmp: tuple[BucketStorageRoot, Path]
    ) -> None:
        # Smoke-level: the lock file appears next to the entry and repeated
        # fetch/publish cycles stay correct (the interleaving itself needs
        # multiple processes; the lock's existence is the observable seam).
        root, _ = bucket_over_tmp
        produced = tmp_path / "canonical.mcap"
        produced.write_bytes(b"v1")
        root.publish(produced, "episodes/e/e.canonical.mcap")
        fetched = root.fetch("episodes/e/e.canonical.mcap")
        assert fetched.read_bytes() == b"v1"
        assert fetched.with_name(fetched.name + ".mirror-lock").exists()
