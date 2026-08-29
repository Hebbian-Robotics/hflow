"""Quality-control pipeline for the egocentric factory corpus.

One file serves both entry points: ``python pipeline.py <episodes>`` runs the
checks in-process for the dev loop, and ``hflow up --pipeline
examples/egocentric/pipeline.py:app`` runs the identical App under Airflow.
"""

import sys
from pathlib import Path

import hflow

# Aliased because the wrappers below take the built-ins' names. Their explicit
# versions belong to this pipeline and must be bumped when behavior changes.
from hflow.checks import camera_frame_stats as measure_camera_frame_stats
from hflow.checks import timestamp_regularity as measure_timestamp_regularity

# Inside the runtime's containers the data root is always mounted at
# /opt/airflow/data; on the host the corpus lives where prepare.py wrote it.
# Both vantage points name the same directory, so local runs and Airflow runs
# share one catalog.
CONTAINER_DATA_ROOT = Path("/opt/airflow/data")
DATA_ROOT = CONTAINER_DATA_ROOT if CONTAINER_DATA_ROOT.is_dir() else Path("data/egocentric")

app = hflow.App("egocentric", data_root=DATA_ROOT)


@app.check(version="1")
def timestamp_regularity(episode: hflow.Episode) -> hflow.CheckResult:
    return measure_timestamp_regularity(episode, expected_hz={episode.cameras[0]: 10.0})


@app.check(version="1", critical=True)
def camera_health(episode: hflow.Episode) -> hflow.CheckResult:
    camera_topic = episode.cameras[0]
    evidence = measure_camera_frame_stats(episode, cameras=[camera_topic])
    black_frame_percent = evidence.measurements[f"{camera_topic}/black_frame_pct"]
    freeze_total_seconds = evidence.measurements[f"{camera_topic}/freeze_total_s"]
    average_luma_mean = evidence.measurements[f"{camera_topic}/luma_avg_mean"]
    decoded_frame_count = evidence.measurements[f"{camera_topic}/decoded_frame_count"]
    assert isinstance(black_frame_percent, float)
    assert isinstance(freeze_total_seconds, float)
    return hflow.CheckResult(
        measurements={
            "black_frame_pct": black_frame_percent,
            "freeze_total_s": freeze_total_seconds,
            "luma_avg_mean": average_luma_mean,
            "frame_count": decoded_frame_count,
        },
        intervals=[
            hflow.Interval(
                start_ns=interval.start_ns,
                end_ns=interval.end_ns,
                label="camera_freeze",
            )
            for interval in evidence.intervals
        ],
        verdict=black_frame_percent < 5.0 and freeze_total_seconds < 2.0,
    )


@app.enrich(version="1")
def contact_sheet(episode: hflow.Episode) -> hflow.EnrichmentResult:
    # episode.cameras is sorted, so this is the alphabetically first camera and
    # not necessarily the one the recorder considered primary. Recorded in the
    # labels because a sheet that cannot say which camera it shows is a trap on
    # any multi-camera corpus.
    camera_topic = episode.cameras[0]
    artifact_path = DATA_ROOT / "artifacts" / f"{episode.path.stem}-contact-sheet.jpg"
    sheet = hflow.ffmpeg.contact_sheet(
        episode.frames(camera=camera_topic, fps=1.0),
        artifact_path,
        columns=5,
        max_tiles=20,
    )
    return hflow.EnrichmentResult(
        labels={
            "contact_sheet_frame_count": sheet.frames_sampled_from,
            "contact_sheet_camera": camera_topic,
        },
        artifacts={"contact_sheet": sheet.path},
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <episode.mcap> [episode.mcap ...]")
    for episode_path in map(Path, sys.argv[1:]):
        app.test(episode_path, record=True)


if __name__ == "__main__":
    main()
