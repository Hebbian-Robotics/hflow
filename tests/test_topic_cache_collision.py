"""#535: one camera's pixels must never be served to another through a
sanitized-name collision.

``_sanitize_topic`` keys the MP4 cache, both frames caches, the contact-sheet
files, and the built-in artifact keys. Two distinct topics that sanitize alike
(``/cam/front`` vs ``/cam_front``) previously shared one cache file, and the
existence check at ``Episode.video()`` made the second camera measure the
first one's pixels with no error anywhere. These tests pin the fix: a readable
stem plus a digest of the full topic, so the name is injective.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from mcap.writer import Writer
from mcap_protobuf.schema import build_file_descriptor_set

import hflow
from hflow import Episode
from hflow.episode import _sanitize_topic
from hflow.ffmpeg import ffmpeg_path
from hflow.format import METADATA_RECORD_EPISODE

RED = "/cam/front"  # sanitizes (stem-only) to cam_front
BLUE = "/cam_front"  # sanitizes (stem-only) to cam_front too

_DIGEST_RE = re.compile(r"^[A-Za-z0-9_.-]+-[0-9a-f]{12}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _solid_color_jpegs(color: str, workdir: Path) -> list[bytes]:
    """Two identical solid-color frames, one second apart, as JPEG bytes."""
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=64x64:d=2:r=1",
            "-q:v",
            "2",
            str(workdir / "f_%03d.jpg"),
        ],
        check=True,
    )
    paths = sorted(workdir.glob("f_*.jpg"))[:2]
    return [path.read_bytes() for path in paths]


def _write_source_episode(path: Path, camera_frames: dict[str, list[bytes]]) -> Path:
    channel_ids: dict[str, int] = {}
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start(library="test-535")
        schema_id = writer.register_schema(
            name="foxglove.CompressedImage",
            encoding="protobuf",
            data=build_file_descriptor_set(CompressedImage).SerializeToString(),
        )
        for topic in camera_frames:
            channel_ids[topic] = writer.register_channel(
                topic=topic, message_encoding="protobuf", schema_id=schema_id
            )
        writer.add_metadata(METADATA_RECORD_EPISODE, {"task": "collision", "embodiment": "probe"})
        for frame_index in range(2):
            for topic, frames in camera_frames.items():
                message = CompressedImage()
                message.timestamp.seconds = frame_index
                message.format = "jpeg"
                message.data = frames[frame_index]
                writer.add_message(
                    channel_id=channel_ids[topic],
                    log_time=1_000_000_000 * frame_index,
                    publish_time=1_000_000_000 * frame_index,
                    data=message.SerializeToString(),
                )
        writer.finish()
    canonical = path.with_name(f"{path.stem}.canonical.mcap")
    hflow.write_canonical_episode(path, canonical)
    return canonical


def test_sanitize_topic_is_injective_across_the_colliding_classes() -> None:
    """Distinct topics that collapse to one readable stem get distinct names."""
    front = _sanitize_topic(RED)
    side = _sanitize_topic(BLUE)
    assert front != side
    assert front.startswith("cam_front-")
    assert side.startswith("cam_front-")
    # The double-slash class Kingston named in #535:
    assert _sanitize_topic("/a//b") != _sanitize_topic("/a/b")
    for name in (front, side, _sanitize_topic("/"), _sanitize_topic("/x/y")):
        assert _DIGEST_RE.fullmatch(name), name
    # Names are deterministic: one topic, one cache key, every time.
    assert _sanitize_topic(RED) == front


def _video_bytes(canonical: Path, topic: str, workdir: Path) -> bytes:
    with Episode(canonical, workdir=workdir) as episode:
        return episode.video(topic).read_bytes()


def test_colliding_caches_serve_each_camera_its_own_pixels(tmp_path: Path) -> None:
    """The red-vs-blue proof: each topic's cache file and extracted frames are
    byte-exactly ITS OWN recording, matching the single-camera reference."""
    red_only = _write_source_episode(
        tmp_path / "red.mcap", {RED: _solid_color_jpegs("red", tmp_path / "red_jpegs")}
    )
    blue_only = _write_source_episode(
        tmp_path / "blue.mcap", {BLUE: _solid_color_jpegs("blue", tmp_path / "blue_jpegs")}
    )
    pair = _write_source_episode(
        tmp_path / "pair.mcap",
        {
            RED: _solid_color_jpegs("red", tmp_path / "red_pair"),
            BLUE: _solid_color_jpegs("blue", tmp_path / "blue_pair"),
        },
    )

    with Episode(pair, workdir=tmp_path / "pair-wd") as episode:
        red_mp4 = episode.video(RED)
        blue_mp4 = episode.video(BLUE)
        assert red_mp4 != blue_mp4, "both cameras shared one cache path (#535)"
        cache_files = sorted(path.name for path in (tmp_path / "pair-wd").glob("*.mp4"))
        assert len(cache_files) == 2, cache_files
        red_sha = _sha256(red_mp4.read_bytes())
        blue_sha = _sha256(blue_mp4.read_bytes())
        assert red_sha != blue_sha
        # Attribution: the shared workdir's per-camera mux equals that camera's
        # own single-camera reference byte for byte.
        assert red_sha == _sha256(_video_bytes(red_only, RED, tmp_path / "red-wd"))
        assert blue_sha == _sha256(_video_bytes(blue_only, BLUE, tmp_path / "blue-wd"))

        red_frames = episode.frames(RED, fps=1.0)
        blue_frames = episode.frames(BLUE, fps=1.0)
        red_frame_sha = _sha256(b"".join(frame.path.read_bytes() for frame in red_frames))
        blue_frame_sha = _sha256(b"".join(frame.path.read_bytes() for frame in blue_frames))
        assert red_frame_sha != blue_frame_sha


def test_three_cameras_produce_three_distinct_contact_sheets(tmp_path: Path) -> None:
    """The artifact half of #535: N cameras must mean N distinct published
    contact-sheet paths with N distinct payloads, not one overwritten sheet."""
    green_frames = _solid_color_jpegs("green", tmp_path / "green_jpegs")
    canonical = tmp_path / "three.mcap"
    _write_source_episode(
        canonical,
        {
            "/a/b": _solid_color_jpegs("red", tmp_path / "a_b_jpegs"),
            "/a_b": _solid_color_jpegs("blue", tmp_path / "a_und_b_jpegs"),
            "/a//b": green_frames,
        },
    )
    app = hflow.App(
        "535-three-cameras",
        data_root=tmp_path / "data",
        default_checks=(),
    )
    report = app.process(canonical, stages={hflow.Stage.SYNC, hflow.Stage.MEDIA})
    (media_run,) = [
        run for run in report.enrichments if run.enrichment.name == "media/contact_sheet"
    ]
    assert media_run.result is not None
    artifacts = media_run.result.artifacts
    assert set(artifacts) == {"/a/b", "/a_b", "/a//b"}
    paths = [artifacts[topic] for topic in ("/a/b", "/a_b", "/a//b")]
    assert len(set(paths)) == 3, f"contact-sheet paths collided: {paths}"
    payload_shas = {_sha256(path.read_bytes()) for path in paths}
    assert len(payload_shas) == 3, "two cameras' contact sheets were byte-identical"
