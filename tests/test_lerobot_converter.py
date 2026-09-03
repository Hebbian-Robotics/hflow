"""Outcome-focused coverage for first-class LeRobot Dataset v3 import.

These tests exercise the metadata-driven discovery and fail-loud behavior
with a synthetic v3-style corpus, without asserting third-party
implementation details or touching the network.
"""

import io
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import cast

import pytest

import hflow.importers.lerobot as prep
from hflow.cli import main as cli_main

_DERIVE = prep._derive_numeric_schema
_ENCODE = prep._encode_cdr_float32_array

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

# Only the remux test below shells out. Scoping this to the module would skip
# the five metadata tests too, and none of those touch ffmpeg at all.
_requires_system_ffmpeg = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="system ffmpeg/ffprobe required to construct and inspect the test video",
)


def _build_fake_corpus(tmp_path: Path) -> dict:
    """Synthetic v3 metadata: 4 episodes, 2 cameras, 6-dim state/action."""
    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [6]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "observation.images.up": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.side": {"dtype": "video", "shape": [480, 640, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
        "robot_type": "so101",
    }
    (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))

    import duckdb

    conn = duckdb.connect()
    rows = []
    for i in range(4):
        rows.append(
            [
                i,
                60 + i * 5,
                "chunk-000",
                "file-000",
                i * 200,
                i * 200 + 60 + i * 5,
                "chunk-000",
                "file-000",
                0.0,
                2.0 + i * 0.2,
                "chunk-000",
                "file-000",
                0.0,
                2.0 + i * 0.2,
                [f"task-{i}"],
            ]
        )
    ep_cols = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "videos/observation.images.up/chunk_index",
        "videos/observation.images.up/file_index",
        "videos/observation.images.up/from_timestamp",
        "videos/observation.images.up/to_timestamp",
        "videos/observation.images.side/chunk_index",
        "videos/observation.images.side/file_index",
        "videos/observation.images.side/from_timestamp",
        "videos/observation.images.side/to_timestamp",
        "tasks",
    ]
    ep_path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    ep_quoted = str(ep_path).replace("'", "''")
    vals_sql = ",".join(
        "("
        + ",".join(
            "[" + ",".join(f"'{x}'" for x in v) + "]"
            if isinstance(v, list)
            else f"'{v!s}'"
            if isinstance(v, str)
            else str(v)
            for v in row
        )
        + ")"
        for row in rows
    )
    ep_cols_q = ",".join(f'"{c}"' for c in ep_cols)
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {vals_sql}) AS t({ep_cols_q})) "
        f"TO '{ep_quoted}' (FORMAT parquet)"
    )

    # data parquet with contiguous `index` column matching episode windows
    data_rows = []
    idx = 0
    for i in range(4):
        length = 60 + i * 5
        for f in range(length):
            state = "[" + ",".join(str(float(f)) for _ in range(6)) + "]"
            action = "[" + ",".join(str(float(f + 0.5)) for _ in range(6)) + "]"
            data_rows.append([idx, i, f, round(f / 30.0, 6), state, action])
            idx += 1
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_quoted = str(data_path).replace("'", "''")
    data_vals = ",".join("(" + ",".join(str(v) for v in row) + ")" for row in data_rows)
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {data_vals}) AS "
        't(index, episode_index, frame_index, timestamp, "observation.state", action)) '
        f"TO '{data_quoted}' (FORMAT parquet)"
    )
    conn.close()

    return {
        "info": info,
        "fps": 30,
        "data_path": info["data_path"],
        "video_path": info["video_path"],
        "cache_dir": tmp_path,
        "data_chunk": "chunk-000",
        "data_file": "file-000",
    }


def test_derive_numeric_schema_float32_vector() -> None:
    schema = _DERIVE("observation.state", {"dtype": "float32", "shape": [6]})
    assert schema.name == "observation.state"
    assert schema.dim == 6


def test_derive_numeric_schema_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="unsupported feature"):
        _DERIVE("action", {"dtype": "float64", "shape": [6]})
    with pytest.raises(ValueError, match="unsupported feature"):
        _DERIVE("observation.state", {"dtype": "float32", "shape": [2, 3]})
    with pytest.raises(ValueError, match="unsupported feature"):
        _DERIVE("observation.state", {"dtype": "float32", "shape": []})
    for shape in ([True], [False]):
        with pytest.raises(
            ValueError,
            match=rf"unsupported feature action: dtype=float32, shape=\[{shape[0]}\]",
        ):
            _DERIVE("action", {"dtype": "float32", "shape": shape})


def test_import_rejects_required_boolean_dimension_without_dataset_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    monkeypatch.setattr(
        prep,
        "_hf_repo_info",
        lambda repo, revision: {"sha": "abc", "license": "apache-2.0"},
    )
    monkeypatch.setattr(
        prep,
        "_ensure_source_archive",
        lambda source, cache_dir: {
            "numeric_features": {
                "action": {"dtype": "float32", "shape": [True]},
                "observation.state": {"dtype": "float32", "shape": [6]},
            },
            "video_keys": [prep.DEFAULT_CAMERA_KEY],
            "episodes": [],
            "dataset": dataset_source,
        },
    )

    with pytest.raises(
        ValueError,
        match=r"unsupported feature action: dtype=float32, shape=\[True\]",
    ):
        prep.import_lerobot_dataset(dataset_repo="fake/repo", output_dir=output_dir)

    assert not (output_dir / "landing").exists()
    assert not (output_dir / "prepared-manifest.json").exists()


def test_cdr_float32_array_n_byte_compatible() -> None:
    out = _ENCODE([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # CDR XCDR1 little-endian encapsulation header (same bytes hflow decodes)
    assert out[:4] == b"\x00\x01\x00\x00"
    assert len(out) == 4 + 4 * 6


def test_index_discovery_multi_camera_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _build_fake_corpus(tmp_path)
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, rev: {"sha": rev, "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: corpus["info"])
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda repo, rev, path: (
            [{"path": "meta/episodes/chunk-000/file-000.parquet", "type": "file"}]
            if "episodes" in path
            else [{"path": "meta/info.json", "type": "file"}]
        ),
    )
    monkeypatch.setattr(prep, "_download_file", lambda url, dest, **kw: None)

    def fake_dl(url: str, dest: Path, **kw: object) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "meta/episodes" in url:
            import shutil

            shutil.copy(
                str(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"), dest
            )
        elif url.endswith("info.json"):
            import shutil

            shutil.copy(str(tmp_path / "meta" / "info.json"), dest)
        else:
            import shutil

            shutil.copy(str(tmp_path / "data" / "chunk-000" / "file-000.parquet"), dest)

    monkeypatch.setattr(prep, "_download_file", fake_dl)

    ds = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    found = prep._ensure_source_archive(ds, tmp_path)
    assert len(found["episodes"]) == 4
    assert found["episodes"][0]["length"] == 60
    assert found["episodes"][1]["length"] == 65
    assert found["episodes"][0]["data_from"] == 0
    assert found["episodes"][0]["data_to"] == 60
    assert set(found["video_keys"]) == {"observation.images.up", "observation.images.side"}
    assert found["episodes"][0]["video_windows"]["observation.images.up"][
        "to_timestamp"
    ] == pytest.approx(2.0)


def test_video_cache_distinguishes_file_indices_and_reuses_same_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _build_fake_corpus(tmp_path)
    camera_key = "observation.images.up"
    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")

    def episode_row(episode_index: int, video_file_index: int) -> dict:
        return {
            "episode_index": episode_index,
            "task": f"task-{episode_index}",
            "length": 1,
            "data_chunk": "000",
            "data_file": "000",
            "data_from": 0,
            "data_to": 1,
            "video_windows": {
                camera_key: {
                    "chunk_index": "000",
                    "file_index": f"{video_file_index:03d}",
                    "from_timestamp": 0.0,
                    "to_timestamp": 0.0,
                }
            },
        }

    source_archive = cast(
        prep._SourceArchive,
        {
            **corpus,
            "episodes": [episode_row(0, 0), episode_row(1, 1)],
            "video_keys": [camera_key],
        },
    )
    numeric_schemas = {
        "observation.state": prep._NumericSchema(name="observation.state", dim=6),
        "action": prep._NumericSchema(name="action", dim=6),
    }
    video_downloaded_urls: set[str] = set()
    cache_path_by_url: dict[str, Path] = {}
    converted_sources: list[bytes] = []

    def fake_download(url: str, destination_path: Path, **_kwargs: object) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if "/videos/" in url:
            if url in video_downloaded_urls:
                raise AssertionError(f"source was downloaded twice: {url}")
            video_downloaded_urls.add(url)
            cache_path_by_url[url] = destination_path
            destination_path.write_bytes(url.encode())
            return
        shutil.copy(tmp_path / "data" / "chunk-000" / "file-000.parquet", destination_path)

    def fake_transcode(mp4_path: Path, *_args: object, **_kwargs: object) -> list[bytes]:
        converted_sources.append(mp4_path.read_bytes())
        return [b"access-unit"]

    monkeypatch.setattr(prep, "_download_file", fake_download)
    monkeypatch.setattr(prep, "_transcode_mp4_to_h264", fake_transcode)
    monkeypatch.setattr(prep, "_get_video_pts_times", lambda path: [0])
    monkeypatch.setattr(prep, "ffmpeg_version", lambda: "test-ffmpeg")
    monkeypatch.setattr(
        prep,
        "write_canonical_episode",
        lambda source_path, output_path, *_args, **_kwargs: shutil.copy(source_path, output_path),
    )

    for episode_index in (0, 1, 0):
        prep._convert_single_episode(
            source_archive=source_archive,
            dataset_source=dataset_source,
            output_dir=tmp_path / "output",
            episode_index=episode_index,
            camera_keys=(camera_key,),
            numeric_schemas=numeric_schemas,
            frames_per_second=30,
        )

    video_urls = [
        "https://huggingface.co/datasets/fake/repo/resolve/abc/videos/"
        "observation.images.up/chunk-000/file-000.mp4",
        "https://huggingface.co/datasets/fake/repo/resolve/abc/videos/"
        "observation.images.up/chunk-000/file-001.mp4",
    ]
    assert converted_sources == [
        video_urls[0].encode(),
        video_urls[1].encode(),
        video_urls[0].encode(),
    ]
    assert cache_path_by_url[video_urls[0]] != cache_path_by_url[video_urls[1]]


def test_camera_selection_validates_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _build_fake_corpus(tmp_path)
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, rev: {"sha": rev, "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: corpus["info"])
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda repo, rev, path: (
            [{"path": "meta/episodes/chunk-000/file-000.parquet", "type": "file"}]
            if "episodes" in path
            else [{"path": "meta/info.json", "type": "file"}]
        ),
    )

    def fake_dl(url: str, dest: Path, **kw: object) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        if "meta/episodes" in url:
            shutil.copy(
                str(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"), dest
            )
        elif url.endswith("info.json"):
            shutil.copy(str(tmp_path / "meta" / "info.json"), dest)
        else:
            shutil.copy(str(tmp_path / "data" / "chunk-000" / "file-000.parquet"), dest)

    monkeypatch.setattr(prep, "_download_file", fake_dl)

    with pytest.raises(ValueError, match="not found"):
        prep.import_lerobot_dataset(
            dataset_repo="fake/repo",
            revision="abc",
            output_dir=tmp_path / "out",
            camera_keys="observation.nope",
        )


def test_import_namespaces_source_cache_by_resolved_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Realistic valid shas the production validator at
    # src/hflow/importers/lerobot.py would accept: 40-character hexadecimal,
    # visibly distinct at the start so a reader can tell them apart at a
    # glance. ``branch-a`` and ``tag-a`` resolve to the same sha on purpose:
    # they are the two-revisions-one-cache leg of the contract.
    sha_a = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
    sha_b = "f0e1d2c3b4a5968778695a4b3c2d1e0f00112233"
    resolved_shas = {
        "branch-a": sha_a,
        "branch-b": sha_b,
        "tag-a": sha_a,
    }
    cache_observations: list[tuple[str, Path, str]] = []

    monkeypatch.setattr(
        prep,
        "_hf_repo_info",
        lambda repo, revision: {"sha": resolved_shas[revision], "license": "apache-2.0"},
    )

    def fake_ensure_source_archive(dataset_source: prep.DatasetSource, cache_dir: Path) -> dict:
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_marker = cache_dir / "source-marker.txt"
        if not source_marker.exists():
            source_marker.write_text(dataset_source.revision)
        cache_observations.append((dataset_source.revision, cache_dir, source_marker.read_text()))
        return {
            "info": {},
            "fps": 30,
            "data_path": "data/{chunk_index}/{file_index}.parquet",
            "video_path": "videos/{camera_key}/{chunk_index}/{file_index}.mp4",
            "episodes": [],
            "video_keys": [prep.DEFAULT_CAMERA_KEY],
            "numeric_features": {
                "action": {"dtype": "float32", "shape": [1]},
                "observation.state": {"dtype": "float32", "shape": [1]},
            },
            "cache_dir": cache_dir,
            "dataset": dataset_source,
        }

    monkeypatch.setattr(prep, "_ensure_source_archive", fake_ensure_source_archive)

    for revision in ("branch-a", "branch-b", "tag-a"):
        prep.import_lerobot_dataset(
            dataset_repo="fake/repo", revision=revision, output_dir=tmp_path
        )

    assert cache_observations == [
        (sha_a, tmp_path / "_lerobot_cache" / sha_a, sha_a),
        (sha_b, tmp_path / "_lerobot_cache" / sha_b, sha_b),
        (sha_a, tmp_path / "_lerobot_cache" / sha_a, sha_a),
    ]
    assert sorted(path.name for path in (tmp_path / "_lerobot_cache").iterdir()) == [
        sha_a,
        sha_b,
    ]


@pytest.mark.parametrize("resolved_sha", ["../../evil", "/tmp/probe-328-absolute"])
def test_hf_repo_info_rejects_malformed_commit_sha(
    resolved_sha: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    response_body = json.dumps({"sha": resolved_sha, "cardData": {"license": "apache-2.0"}})
    monkeypatch.setattr(
        prep.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(response_body.encode()),
    )

    with pytest.raises(ValueError, match="malformed commit sha"):
        prep._hf_repo_info("fake/repo", "main")


def test_import_refuses_invalid_arguments_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset_repo must not be empty"):
        prep.import_lerobot_dataset(dataset_repo="", output_dir=tmp_path)
    with pytest.raises(ValueError, match="revision must not be empty"):
        prep.import_lerobot_dataset(dataset_repo="lerobot/pusht", revision=" ", output_dir=tmp_path)
    with pytest.raises(ValueError, match="episode_index must be zero or greater"):
        prep.import_lerobot_dataset(
            dataset_repo="lerobot/pusht", episode_index=-1, output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="camera_keys must name at least one"):
        prep.import_lerobot_dataset(
            dataset_repo="lerobot/pusht", camera_keys=(), output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="camera_keys must not contain duplicates"):
        prep.import_lerobot_dataset(
            dataset_repo="lerobot/pusht",
            camera_keys=("observation.image", "observation.image"),
            output_dir=tmp_path,
        )


def test_cli_routes_lerobot_import_refusals_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli_main(["import", "lerobot", "--repo", "", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "import lerobot: dataset_repo must not be empty\n"


@_requires_system_ffmpeg
def test_converter_output_remuxes_without_tail_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The converter's stream must decode in full through the hflow remux.

    B-frame H.264 loses its reorder-buffer tail when remuxed from raw Annex
    B to MP4, so decoded_frame_count undercounts healthy episodes (#250).
    The converter encodes with bframes=0; prove the full chain decodes
    every source frame.
    """

    assert _FFMPEG is not None and _FFPROBE is not None
    system_ffmpeg_path = Path(_FFMPEG)
    system_ffprobe_path = Path(_FFPROBE)
    monkeypatch.setattr(prep, "ffmpeg_path", lambda: system_ffmpeg_path)
    monkeypatch.setattr(prep, "ffprobe_path", lambda: system_ffprobe_path)

    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            _FFMPEG,
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=30:duration=3,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            str(source),
        ],
        capture_output=True,
        check=True,
    )

    from hflow.video import write_access_units_to_mp4

    units = prep._transcode_mp4_to_h264(source, gop_seconds=1.0, frames_per_second=30.0)
    muxed = write_access_units_to_mp4(units, fps=30.0, output=tmp_path / "remux.mp4")

    probe = subprocess.run(
        [
            _FFPROBE,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1",
            str(muxed),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "nb_read_frames=90" in probe.stdout, probe.stdout


# --- Hugging Face JSON boundary contextual errors (#301) ---------------------
#
# The LeRobot importer decodes response bytes at three HF boundaries (repo
# info, tree listings, meta/info.json). Invalid UTF-8 or malformed JSON must
# surface as a contextual ValueError naming the source, with the original
# decoder exception chained as the cause. These tests stub urlopen with local
# bytes; no network request is made.


class _StubResponse:
    """Minimal urlopen response double: a context manager with read()."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self.headers = headers or {}

    def __enter__(self) -> "_StubResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class _StubUrlopen:
    """Routes urlopen calls to fixed bytes by a marker in the request URL."""

    def __init__(
        self,
        routes: dict[str, bytes],
        response_headers: dict[str, dict[str, str]] | None = None,
        max_requests: int | None = None,
    ) -> None:
        self._routes = routes
        self._response_headers = response_headers or {}
        self._max_requests = max_requests
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: int = 60) -> _StubResponse:
        url = request.full_url
        self.requests.append(request)
        if self._max_requests is not None and len(self.requests) > self._max_requests:
            raise AssertionError(f"urlopen exceeded test request limit: {self._max_requests}")
        for marker, body in self._routes.items():
            if marker in url:
                return _StubResponse(body, self._response_headers.get(marker))
        raise AssertionError(f"unexpected urlopen URL: {url}")


def _stub_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[str, bytes],
    response_headers: dict[str, dict[str, str]] | None = None,
    max_requests: int | None = None,
) -> _StubUrlopen:
    stub = _StubUrlopen(routes, response_headers, max_requests)
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    return stub


_INVALID_UTF8 = b"\xff\xfe\x00"
_MALFORMED_JSON = b"{not json"


def test_hf_repo_info_invalid_utf8_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/api/datasets/": _INVALID_UTF8})
    with pytest.raises(ValueError) as excinfo:
        prep._hf_repo_info("lerobot/pusht", "main")
    assert "lerobot/pusht@main" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_hf_repo_info_malformed_json_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/api/datasets/": _MALFORMED_JSON})
    with pytest.raises(ValueError) as excinfo:
        prep._hf_repo_info("lerobot/pusht", "main")
    assert "lerobot/pusht@main" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_hf_tree_invalid_utf8_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/tree/": _INVALID_UTF8})
    with pytest.raises(ValueError) as excinfo:
        prep._hf_tree("lerobot/pusht", "main", "meta")
    message = str(excinfo.value)
    assert "lerobot/pusht@main" in message
    assert "'meta'" in message
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_hf_tree_malformed_json_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/tree/": b"[1, 2"})
    with pytest.raises(ValueError) as excinfo:
        prep._hf_tree("lerobot/pusht", "main", "meta")
    message = str(excinfo.value)
    assert "lerobot/pusht@main" in message
    assert "'meta'" in message
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_fetch_info_json_invalid_utf8_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree_body = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    _stub_urlopen(
        monkeypatch,
        {"recursive=true": tree_body, "meta/info.json": _INVALID_UTF8},
    )
    with pytest.raises(ValueError) as excinfo:
        prep._fetch_info_json("lerobot/pusht", "main", tmp_path)
    message = str(excinfo.value)
    assert "meta/info.json" in message
    assert "lerobot/pusht@main" in message
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_fetch_info_json_malformed_json_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree_body = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    _stub_urlopen(
        monkeypatch,
        {"recursive=true": tree_body, "meta/info.json": b"{oops"},
    )
    with pytest.raises(ValueError) as excinfo:
        prep._fetch_info_json("lerobot/pusht", "main", tmp_path)
    message = str(excinfo.value)
    assert "meta/info.json" in message
    assert "lerobot/pusht@main" in message
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_hf_repo_info_valid_json_still_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real resolved commit sha: at least the 7 hex characters the sha
    # validation requires, since a cache directory is named after it.
    body = json.dumps({"sha": "abc1234", "cardData": {"license": "apache-2.0"}}).encode()
    _stub_urlopen(monkeypatch, {"/api/datasets/": body})
    assert prep._hf_repo_info("lerobot/pusht", "main") == {
        "sha": "abc1234",
        "license": "apache-2.0",
    }


def test_hf_tree_valid_json_still_returns_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    stub = _stub_urlopen(monkeypatch, {"/tree/": body})
    assert prep._hf_tree("lerobot/pusht", "main", "meta") == [
        {"path": "meta/info.json", "type": "file"}
    ]
    assert len(stub.requests) == 1


def test_hf_tree_follows_next_link_preserves_headers_and_deduplicates_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_page = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    second_page = json.dumps(
        [
            {"path": "meta/info.json", "type": "file"},
            {"path": "meta/episodes/file-000.parquet", "type": "file"},
        ]
    ).encode()
    next_url = (
        "https://huggingface.co/api/datasets/lerobot/pusht/tree/main/meta"
        "?recursive=true&cursor=page-2"
    )
    stub = _stub_urlopen(
        monkeypatch,
        {"cursor=page-2": second_page, "/tree/": first_page},
        {"/tree/": {"Link": f'<{next_url}>; rel="next"'}},
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")

    assert prep._hf_tree("lerobot/pusht", "main", "meta") == [
        {"path": "meta/info.json", "type": "file"},
        {"path": "meta/episodes/file-000.parquet", "type": "file"},
    ]
    assert [request.full_url for request in stub.requests] == [
        "https://huggingface.co/api/datasets/lerobot/pusht/tree/main/meta?recursive=true",
        next_url,
    ]
    for request in stub.requests:
        assert request.get_header("User-agent") == "hflow-lerobot"
        assert request.get_header("Authorization") == "Bearer test-token"


def test_hf_tree_rejects_next_link_with_different_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_page = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    unsafe_next_url = "https://attacker.example/tree/secret?cursor=page-2"
    stub = _stub_urlopen(
        monkeypatch,
        {"/tree/": first_page},
        {"/tree/": {"Link": f'<{unsafe_next_url}>; rel="next"'}},
        max_requests=1,
    )
    monkeypatch.setenv("HF_TOKEN", "secret-user-token")

    with pytest.raises(ValueError, match="different scheme and host"):
        prep._hf_tree("lerobot/pusht", "main", "meta")

    assert len(stub.requests) == 1


def test_hf_tree_rejects_a_repeated_next_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_page = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    page_url = "https://huggingface.co/api/datasets/lerobot/pusht/tree/main/meta?recursive=true"
    stub = _stub_urlopen(
        monkeypatch,
        {"/tree/": first_page},
        {"/tree/": {"Link": f'<{page_url}>; rel="next"'}},
        max_requests=1,
    )

    with pytest.raises(ValueError, match="already fetched"):
        prep._hf_tree("lerobot/pusht", "main", "meta")

    assert len(stub.requests) == 1


def test_hf_tree_malformed_later_page_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_page = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    next_url = (
        "https://huggingface.co/api/datasets/lerobot/pusht/tree/main/meta"
        "?recursive=true&cursor=page-2"
    )
    _stub_urlopen(
        monkeypatch,
        {"cursor=page-2": _MALFORMED_JSON, "/tree/": first_page},
        {"/tree/": {"Link": f'<{next_url}>; rel="next"'}},
    )

    with pytest.raises(ValueError) as excinfo:
        prep._hf_tree("lerobot/pusht", "main", "meta")

    message = str(excinfo.value)
    assert "lerobot/pusht@main" in message
    assert "'meta'" in message
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_hf_repo_info_wrong_shape_refusal_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/api/datasets/": b"[1, 2, 3]"})
    with pytest.raises(ValueError, match="not a JSON object"):
        prep._hf_repo_info("lerobot/pusht", "main")


@pytest.mark.parametrize(
    "bad_fps",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        None,
        "30",
        0,
        -30,
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
        "boolean",
        "missing",
        "nonnumeric",
        "zero",
        "negative",
    ],
)
def test_info_json_refuses_non_finite_or_non_positive_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_fps: object
) -> None:
    """Invalid fps metadata must be refused before any episode discovery runs."""
    corpus = _build_fake_corpus(tmp_path)
    info = dict(corpus["info"])
    if bad_fps is None:
        info.pop("fps")
    else:
        info["fps"] = bad_fps
    expected_value = info.get("fps")
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, rev: {"sha": rev, "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: info)

    def fail_tree(repo: str, rev: str, path: str) -> list[dict]:
        raise AssertionError("episode metadata discovery must not run after invalid fps")

    monkeypatch.setattr(prep, "_hf_tree", fail_tree)

    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    cache_dir = tmp_path / "cache"
    with pytest.raises(ValueError, match="fps") as excinfo:
        prep._ensure_source_archive(dataset_source, cache_dir)

    message = str(excinfo.value)
    assert "FPS must be finite and positive" in message
    assert repr(expected_value) in message
    # Refusal happens before episode metadata discovery: no downloads, no output.
    assert not cache_dir.exists() or not (cache_dir / "meta" / "episodes").exists()


def test_info_json_accepts_normal_positive_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _build_fake_corpus(tmp_path)
    corpus["info"]["fps"] = 30
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, rev: {"sha": rev, "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: corpus["info"])
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda repo, rev, path: (
            [{"path": "meta/episodes/chunk-000/file-000.parquet", "type": "file"}]
            if "episodes" in path
            else [{"path": "meta/info.json", "type": "file"}]
        ),
    )

    def fake_dl(url: str, dest: Path, **kw: object) -> None:
        import shutil

        dest.parent.mkdir(parents=True, exist_ok=True)
        if "meta/episodes" in url:
            shutil.copy(
                str(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"), dest
            )

    monkeypatch.setattr(prep, "_download_file", fake_dl)

    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    source_archive = prep._ensure_source_archive(dataset_source, tmp_path / "cache")
    assert source_archive["fps"] == 30
