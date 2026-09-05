"""A built-in enrichment that publishes each camera as a browser-playable MP4.

Canonical episodes carry H.264 in-band as ``foxglove.CompressedVideo``
messages. Foxglove and Rerun play that directly; a browser ``<video>`` element
does not. This enrichment remuxes each camera stream losslessly (no
re-encode) into an MP4 artifact so a workspace UI, a notebook, or a plain
HTML page can scrub the footage next to the catalog's evidence.

It is opt-in because it roughly doubles per-camera storage: the MP4 holds the
same access units the canonical file already stores. Register it like any
other enrichment::

    import hflow
    from hflow.camera_video import CAMERA_VIDEO_VERSION, camera_video

    app.enrich(version=CAMERA_VIDEO_VERSION)(camera_video)

Per camera topic it records:

- artifact ``video:<topic>``: the MP4's published URI (served by
  ``hflow-server`` under ``/api/v1/episodes/{id}/media/video:<topic>``);
- label ``<topic>/video_start_s``: seconds from the episode's first log
  timestamp (``Episode.time_bounds.start_ns``) to the first video frame, so a
  player maps video time ``t`` to log time ``start_ns + (video_start_s + t) * 1e9``.
  Absent when the file records no time bounds;
- label ``<topic>/video_fps``: the constant frame rate the MP4 was muxed at
  (the stream's median frame interval), which is what video time counts in;
- label ``<topic>/video_frame_count``: how many access units were muxed.

The MP4 plays at a constant rate, so a stream with irregular stamps drifts
from log time by its jitter; the ``timestamp_regularity`` check measures that
jitter, and callers needing exact per-frame times read the message stamps.
"""

from pathlib import Path

from hflow.episode import Episode
from hflow.steps import EnrichmentResult, MeasurementValue
from hflow.video import estimate_fps_from_log_times

NANOSECONDS_PER_SECOND = 1_000_000_000

# The author-owned compatibility promise for this step's outputs. Bump when the
# artifact names, label keys, or the muxed file's timing model change.
CAMERA_VIDEO_VERSION = "1"

VIDEO_ARTIFACT_PREFIX = "video:"


def video_artifact_name(camera_topic: str) -> str:
    """The artifact name this enrichment records for one camera topic."""
    return f"{VIDEO_ARTIFACT_PREFIX}{camera_topic}"


def camera_video(episode: Episode) -> EnrichmentResult:
    """Remux every camera stream to an MP4 artifact with its timing labels.

    Requires a canonical episode (``hflow.write_canonical_episode``): the
    remux reads the canonical ``foxglove.CompressedVideo`` channel. A stream
    the frame-rate estimate refuses (duplicate or non-increasing stamps)
    raises, which the runner records as this step's error, not the run's.
    """
    labels: dict[str, MeasurementValue] = {}
    artifacts: dict[str, Path] = {}
    time_bounds = episode.time_bounds
    for camera_topic in episode.cameras:
        mp4_path = episode.video(camera_topic)
        channel = episode.channel(camera_topic)
        artifacts[video_artifact_name(camera_topic)] = mp4_path
        # The same estimate Episode.video() muxed at, so the label is the
        # file's actual clock.
        labels[f"{camera_topic}/video_fps"] = estimate_fps_from_log_times(
            channel.timestamps.tolist(), topic=camera_topic
        )
        labels[f"{camera_topic}/video_frame_count"] = len(channel)
        if time_bounds is not None and len(channel) > 0:
            first_frame_offset_ns = int(channel.timestamps[0]) - time_bounds.start_ns
            labels[f"{camera_topic}/video_start_s"] = first_frame_offset_ns / NANOSECONDS_PER_SECOND
    return EnrichmentResult(labels=labels, artifacts=artifacts)
