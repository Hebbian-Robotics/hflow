"""The opt-in ``camera_video`` enrichment: a playable MP4 per camera, aligned
to the episode's time axis through its labels."""

import subprocess
from pathlib import Path

import pytest

import hflow
from hflow.camera_video import CAMERA_VIDEO_VERSION, camera_video, video_artifact_name
from hflow.curation import open_catalog_connection
from hflow.ffmpeg import ffprobe_path
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

TWO_CAMERAS_AT_15_HZ = SyntheticEpisodeSpec(
    duration_s=2.0, cameras=("wrist_cam", "top_cam"), image_hz=15.0
)


def _decoded_frame_count(mp4_path: Path) -> int:
    completed = subprocess.run(
        [
            str(ffprobe_path()),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(completed.stdout.strip())


def test_camera_video_publishes_a_playable_mp4_per_camera_with_its_clock(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    app = hflow.App("camera-video", data_root=data_root)
    app.enrich(version=CAMERA_VIDEO_VERSION)(camera_video)
    source = synthesize_episode(tmp_path / "episode.mcap", TWO_CAMERAS_AT_15_HZ)

    report = app.test(source, verbose=False, record=True)

    (video_run,) = [run for run in report.enrichments if run.enrichment.name == "camera_video"]
    assert video_run.status is hflow.CheckStatus.MEASURED, video_run.error
    assert video_run.result is not None
    camera_topics = ["/top_cam/compressed", "/wrist_cam/compressed"]
    assert sorted(video_run.artifact_uris) == [video_artifact_name(t) for t in camera_topics]

    with hflow.Episode(report.canonical_path) as canonical_episode:
        time_bounds = canonical_episode.time_bounds
        assert time_bounds is not None
        first_frame_stamps = {
            topic: int(canonical_episode.channel(topic).timestamps[0]) for topic in camera_topics
        }
    for topic in camera_topics:
        published_mp4 = Path(video_run.artifact_uris[video_artifact_name(topic)])
        # Published under the data root, not left in the scratch workdir.
        assert published_mp4.is_relative_to(data_root)
        assert published_mp4.suffix == ".mp4"
        labels = video_run.result.labels
        assert labels[f"{topic}/video_fps"] == pytest.approx(15.0, rel=0.05)
        assert labels[f"{topic}/video_frame_count"] == _decoded_frame_count(published_mp4)
        expected_start_s = (first_frame_stamps[topic] - time_bounds.start_ns) / 1e9
        assert labels[f"{topic}/video_start_s"] == pytest.approx(expected_start_s)
        assert 0.0 <= expected_start_s < 1.0

    connection = open_catalog_connection(data_root / "catalog")
    try:
        cataloged_artifacts = connection.execute(
            "SELECT key FROM measurements_latest "
            "WHERE check_name = 'camera_video' AND key LIKE 'artifact/%' ORDER BY key"
        ).fetchall()
    finally:
        connection.close()
    assert cataloged_artifacts == [
        (f"artifact/{video_artifact_name(topic)}",) for topic in camera_topics
    ]


def test_camera_video_on_a_camera_less_episode_records_nothing(tmp_path: Path) -> None:
    app = hflow.App("camera-video-empty", data_root=tmp_path / "data")
    app.enrich(version=CAMERA_VIDEO_VERSION)(camera_video)
    source = synthesize_episode(
        tmp_path / "episode.mcap", SyntheticEpisodeSpec(duration_s=1.0, cameras=())
    )

    report = app.test(source, verbose=False)

    (video_run,) = report.enrichments
    assert video_run.status is hflow.CheckStatus.MEASURED, video_run.error
    assert video_run.result == hflow.EnrichmentResult()
