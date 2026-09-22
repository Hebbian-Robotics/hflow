"""Tests for the reproducible real-camera workflow (examples/lerobot/workflow.py).

The pinned network import is replaced by a fake importer that publishes the
same artifacts the public adapter writes: doctor-conforming canonical landing
MCAPs (foxglove CompressedVideo channels plus the episode/v1 and
source-provenance/v1 metadata records), a prepared-manifest.json carrying the
pinned revision, and a revision-namespaced source cache consumed by the v3
exporter. The pinned network corpus stays a documented validation run, not a
test download.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import duckdb
import pytest
from lerobot_test_helpers import CorpusEpisodeRow, two_camera_v3_info, write_v3_corpus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hflow
from examples.lerobot import workflow
from hflow.curation import curate

REVISION = "f641879e22172be7e8161d5e6c1503c2d2feb657"
CAMS = ("observation.images.up", "observation.images.side")
FIXTURE_MANIFEST = {
    "repository": "lerobot/fixture_pickplace",
    "revision": REVISION,
    "license": "Apache-2.0",
    "cameras": list(CAMS),
    "episodes": [0, 1],
}
START_NS = 1_755_000_000_000_000_000
FPS = 15
FRAMES = 30


def _frame_units(stream: bytes) -> list[bytes]:
    """Split an Annex-B H.264 stream into frame access units.

    Parameter sets and other non-VCL NALs ride with the next VCL frame, so
    every unit published as one CompressedVideo message contains a coded
    picture -- the same shape the real converter's per-frame access units
    have.
    """
    nals: list[bytes] = []
    i = 0
    while i + 4 <= len(stream):
        if stream[i : i + 4] == b"\x00\x00\x00\x01":
            start = i + 4
            i += 4
        elif stream[i : i + 3] == b"\x00\x00\x01":
            start = i + 3
            i += 3
        else:
            i += 1
            continue
        j = i
        while j + 3 <= len(stream) and not (
            stream[j : j + 4] == b"\x00\x00\x00\x01" or stream[j : j + 3] == b"\x00\x00\x01"
        ):
            j += 1
        body = stream[start:j]
        if body:
            nals.append(b"\x00\x00\x00\x01" + body)
        i = j
    frames: list[bytes] = []
    pending: bytes = b""
    for nal in nals:
        kind = nal[4] & 0x1F
        if kind in (1, 2, 3, 4, 5):
            frames.append(pending + nal)
            pending = b""
        else:
            pending += nal
    return frames


@pytest.fixture(scope="module")
def clip_units(tmp_path_factory: pytest.TempPathFactory) -> list[bytes]:
    """Animated H.264 access units (testsrc2 never freezes a frame)."""
    root = tmp_path_factory.mktemp("clip")
    ffmpeg = hflow.ffmpeg.ffmpeg_path()
    encoded = root / "clip.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=15",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "15",
            "-bf",
            "0",
            str(encoded),
        ],
        check=True,
        capture_output=True,
    )
    annexb = root / "clip.h264"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(encoded),
            "-c",
            "copy",
            "-bsf:v",
            "h264_mp4toannexb",
            "-f",
            "h264",
            str(annexb),
        ],
        check=True,
        capture_output=True,
    )
    units = _frame_units(annexb.read_bytes())
    assert len(units) >= FRAMES
    return units


@pytest.fixture(scope="module")
def fixture_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic v3 source cache: info, episode parquet, data, video chunks."""
    root = tmp_path_factory.mktemp("archive")
    info = {
        "code": "LeRobotDataset/v3",
        **two_camera_v3_info(frames_per_second=FPS, camera_frame_shape=(240, 320, 3)),
        "total_episodes": 3,
        "total_frames": FRAMES * 3,
    }
    write_v3_corpus(
        root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=index,
                length=FRAMES,
                dataset_from_index=index * FRAMES,
                video_to_timestamp=2.0,
                tasks=(),
            )
            for index in range(3)
        ],
        camera_keys=CAMS,
    )
    for camera_key in CAMS:
        video_directory = root / "videos" / camera_key / "chunk-000"
        video_directory.mkdir(parents=True)
        (video_directory / "file-000.mp4").write_bytes(b"fake-mp4-content")
    return root


def _write_landing(landing_path: Path, episode_index: int, units: list[bytes]) -> Path:
    """Mirror the public importer: foxglove video channels + metadata records."""
    from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
    from mcap.writer import Writer as McapWriter
    from mcap_protobuf.schema import build_file_descriptor_set

    landing_path.parent.mkdir(parents=True, exist_ok=True)
    source = landing_path.with_suffix(".source.mcap")
    with source.open("wb") as stream:
        writer = McapWriter(stream)
        writer.start(profile="", library="hflow test fixture")
        schema_id = writer.register_schema(
            "foxglove.CompressedVideo",
            "protobuf",
            build_file_descriptor_set(CompressedVideo).SerializeToString(),
        )
        channels = {
            cam: writer.register_channel(
                topic=f"/{cam}", message_encoding="protobuf", schema_id=schema_id
            )
            for cam in CAMS
        }
        for frame, unit in enumerate(units):
            log_ns = START_NS + round(frame * 1_000_000_000 / FPS)
            for cam in CAMS:
                message = CompressedVideo()
                message.timestamp.seconds = log_ns // 1_000_000_000
                message.timestamp.nanos = log_ns % 1_000_000_000
                message.frame_id = cam
                message.data = unit
                message.format = "h264"
                writer.add_message(
                    channel_id=channels[cam],
                    log_time=log_ns,
                    data=message.SerializeToString(),
                    publish_time=log_ns,
                    sequence=frame,
                )
        writer.add_metadata(
            name="episode/v1",
            data={
                "task": "fixture_task",
                "operator": "lerobot_converter",
                "embodiment": "so101",
                "source_dataset": cast(str, FIXTURE_MANIFEST["repository"]),
                "source_revision": REVISION,
                "source_episode_index": str(episode_index),
                "camera_keys": json.dumps(list(CAMS), separators=(",", ":")),
                "gop_seconds": "1",
            },
        )
        writer.add_metadata(
            name="source-provenance/v1",
            data={
                "converter_version": "test",
                "ffmpeg_version": "test",
                "source_uri": f"hf://datasets/{FIXTURE_MANIFEST['repository']}@{REVISION}",
            },
        )
        writer.finish()

    from hflow.transform import write_canonical_episode

    write_canonical_episode(source, landing_path)
    return landing_path


@pytest.fixture()
def installed_fake_import(
    monkeypatch: pytest.MonkeyPatch, fixture_archive: Path, clip_units: list[bytes]
) -> Path:
    """Fake importer that publishes landing files, manifest, and cache."""

    def _fake_import(
        dataset_repo: str,
        revision: str,
        output_dir: Path,
        episode_index: object = None,
        camera_keys: tuple[str, ...] = (),
    ) -> list[str]:
        out = Path(output_dir)
        cache = out / "_lerobot_cache" / str(revision)
        if not cache.exists():
            shutil.copytree(fixture_archive, cache)
        selected = [0, 1, 2] if episode_index is None else [cast(int, episode_index)]
        landing = out / "landing"
        uris: list[str] = []
        for index in selected:
            canonical = _write_landing(
                landing / f"lerobot_episode_{index + 1:04d}.mcap", index, clip_units
            )
            uris.append(str(canonical))
        (out / "prepared-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "dataset": {
                        "repo_id": FIXTURE_MANIFEST["repository"],
                        "revision": REVISION,
                        "license": "Apache-2.0",
                    },
                    "camera_keys": list(CAMS),
                    "episodes_converted": len(uris),
                    "episodes": [
                        {"uri": uri, "content_id": "fixture", "size_bytes": 1} for uri in uris
                    ],
                    "converter_version": "test",
                },
                indent=2,
            )
        )
        return uris

    monkeypatch.setattr(workflow.hflow, "import_lerobot_dataset", _fake_import)
    return fixture_archive


def test_committed_curation_sql_matches_the_policy() -> None:
    committed = (
        (Path(__file__).resolve().parents[1] / "examples" / "lerobot" / "curation.sql")
        .read_text()
        .strip()
    )
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1] / "examples" / "lerobot" / "source-manifest.json"
        ).read_text()
    )
    assert workflow.curation_policy_sql(tuple(manifest["cameras"])) == committed


def test_workflow_end_to_end_on_small_fixture(tmp_path: Path, installed_fake_import: Path) -> None:
    data_dir = tmp_path / "run"
    assert workflow.main(["--data-dir", str(data_dir)]) == 0

    # Selection manifest exists and is deterministic on a second cut.
    manifest_path = data_dir / "manifest.parquet"
    assert manifest_path.exists()
    second = data_dir / "manifest-again.parquet"
    report = curate(data_dir / "catalog", workflow.curation_policy_sql(CAMS), output=second)
    assert report.row_count == 3
    conn = duckdb.connect()
    try:
        first_rows = conn.execute(
            "SELECT * FROM read_parquet('"
            + str(manifest_path).replace("'", "''")
            + "') ORDER BY episode_id"
        ).fetchall()
        second_rows = conn.execute(
            "SELECT * FROM read_parquet('"
            + str(second).replace("'", "''")
            + "') ORDER BY episode_id"
        ).fetchall()
        assert first_rows == second_rows
        assert len(first_rows) == 3  # all fixture episodes pass the policy

        # Both cameras appear in the recorded quality evidence.
        keys = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT key FROM read_parquet('"
                + str(data_dir / "catalog" / "measurements" / "*.parquet").replace("'", "''")
                + "') ORDER BY key"
            ).fetchall()
        ]
        assert any(key.startswith("/observation.images.up/") for key in keys)
        assert any(key.startswith("/observation.images.side/") for key in keys)
        # The catalog records the fixture source provenance.
        meta_rows = conn.execute(
            "SELECT DISTINCT metadata_json FROM read_parquet('"
            + str(data_dir / "catalog" / "episodes" / "*.parquet").replace("'", "''")
            + "') WHERE metadata_json IS NOT NULL"
        ).fetchall()
        assert meta_rows
        meta = json.loads(meta_rows[0][0])
        assert meta["source_dataset"] == FIXTURE_MANIFEST["repository"]
        assert meta["source_revision"] == REVISION
    finally:
        conn.close()

    # Snapshot: standard parquet tables plus copied media.
    snapshot = data_dir / "snapshot"
    assert list(snapshot.glob("*.parquet"))
    assets = snapshot / "assets"
    assert assets.is_dir() and any(assets.rglob("*"))

    # v3 export contains exactly the selected episodes.
    info = json.loads((data_dir / "v3" / "meta" / "info.json").read_text())
    assert info["code"] == "LeRobotDataset/v3"
    assert info["total_episodes"] == 3
    conn_read = duckdb.connect()
    try:
        indexes = sorted(
            row[0]
            for row in conn_read.execute(
                "SELECT DISTINCT episode_index FROM read_parquet('"
                + str(data_dir / "v3" / "meta" / "episodes" / "**" / "*.parquet").replace("'", "''")
                + "')"
            ).fetchall()
        )
        assert indexes == [0, 1, 2]
    finally:
        conn_read.close()

    # Clean-process verifier accepts the export.
    verify_script = Path(__file__).resolve().parents[1] / "examples" / "lerobot" / "verify.py"
    proc = subprocess.run(
        [sys.executable, str(verify_script), str(data_dir / "v3"), "--expect", "0,1,2"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_verify_rejects_a_missing_reference(tmp_path: Path) -> None:
    """The clean-process verifier fails when an exported file goes missing."""
    v3 = tmp_path / "v3"
    (v3 / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (v3 / "data" / "chunk-000").mkdir(parents=True)
    (v3 / "videos" / "camera.up" / "chunk-000").mkdir(parents=True)
    info = {
        "code": "LeRobotDataset/v3",
        "total_episodes": 1,
        "total_frames": 2,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {"camera.up": {"dtype": "video", "shape": [240, 320, 3]}},
    }
    (v3 / "meta" / "info.json").write_text(json.dumps(info))
    example_row = [
        [
            0,
            2,
            0,
            0,
            0,
            2,
            0,
            0,
            0.0,
            2.0,
        ]
    ]
    conn = duckdb.connect()
    ep_parquet = v3 / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    vals = "(" + ",".join(str(v) for v in example_row[0]) + ")"
    quoted = str(ep_parquet).replace("'", "''")
    conn.execute(
        "COPY (SELECT * FROM (VALUES "
        f'{vals}) AS t(episode_index, length, "data/chunk_index", "data/file_index", '
        'dataset_from_index, dataset_to_index, "videos/camera.up/chunk_index", '
        '"videos/camera.up/file_index", "videos/camera.up/from_timestamp", '
        f"\"videos/camera.up/to_timestamp\")) TO '{quoted}' (FORMAT parquet)"
    )
    data_parquet = v3 / "data" / "chunk-000" / "file-000.parquet"
    dquoted = str(data_parquet).replace("'", "''")
    conn.execute(
        "COPY (SELECT * FROM (VALUES (0, 0, 0, 0.0), (1, 0, 1, 0.066667)) AS "
        "t(index, episode_index, frame_index, timestamp)) TO '" + dquoted + "' (FORMAT parquet)"
    )
    (v3 / "videos" / "camera.up" / "chunk-000" / "file-000.mp4").write_bytes(b"mp4")
    conn.close()

    verify_script = Path(__file__).resolve().parents[1] / "examples" / "lerobot" / "verify.py"
    ok = subprocess.run(
        [sys.executable, str(verify_script), str(v3), "--expect", "0"],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert "OK" in ok.stdout

    (v3 / "videos" / "camera.up" / "chunk-000" / "file-000.mp4").unlink()
    broken = subprocess.run(
        [sys.executable, str(verify_script), str(v3), "--expect", "0"],
        capture_output=True,
        text=True,
    )
    assert broken.returncode != 0
    assert "FAIL" in broken.stderr

    mismatch = subprocess.run(
        [sys.executable, str(verify_script), str(v3), "--expect", "0,1"],
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode != 0
