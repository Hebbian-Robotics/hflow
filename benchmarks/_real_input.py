"""Shared real-footage input handling for the benchmark scripts.

The synthetic fixtures answer "do the mechanisms behave as designed"; a
real recording answers "what do the numbers look like on real
content". Both benchmark scripts accept ``--input <mcap>`` and use the
helpers here to discover the camera and state topics of an arbitrary
recording instead of assuming the synthetic fixture's layout.

The reference real inputs are not redistributed. Fetch the nuScenes sample:

    curl -fLo nuscenes-mini-sample.mcap \
        https://mcap-proxy.lichtblick.workers.dev/NuScenes-v1.0-mini-scene-sample.mcap

nuScenes v1.0-mini scene sample as converted by nuscenes2mcap and hosted by
the Lichtblick project. License: CC BY-NC-SA 4.0 (non-commercial research).
Attribution: "Adapted from nuScenes dataset. Copyright 2020 nuScenes."

Issue #170 also uses one manipulation episode from RobotisAI's public MCAP
corpus, pinned to an immutable repository revision. The URL and checksum are
constants below so the exact camera + proprio recording can be reproduced.
"""

from dataclasses import dataclass
from pathlib import Path

from mcap.reader import make_reader
from mcap.summary import Summary

from hflow.format import CAMERA_SCHEMA_NAMES

NUSCENES_SAMPLE_URL = (
    "https://mcap-proxy.lichtblick.workers.dev/NuScenes-v1.0-mini-scene-sample.mcap"
)
NUSCENES_SAMPLE_SHA256 = "089b5be708d2536a8c7e341f4d2b62dc618a581525e4db8d15c93ec5ce446547"
NUSCENES_ATTRIBUTION = "Adapted from nuScenes dataset. Copyright 2020 nuScenes."

ROBOTIS_BUTTON_PUSH_SAMPLE_REVISION = "93a0ba73c6b82689696d9c4909b3b48af0294cf8"
ROBOTIS_BUTTON_PUSH_SAMPLE_URL = (
    "https://huggingface.co/datasets/RobotisAI/evButtonPush-260615-0-MCAP/"
    f"resolve/{ROBOTIS_BUTTON_PUSH_SAMPLE_REVISION}/107/107_0.mcap?download=true"
)
ROBOTIS_BUTTON_PUSH_SAMPLE_SHA256 = (
    "51bec431162844b2bb239f65ea158ea1030bb3fce5312ad81768978dc68f82c3"
)

# The training-sample "state" analog, in preference order: proprioception-like
# telemetry first, densest non-camera topic as the fallback.
_STATE_TOPIC_PREFERENCE = ("/joint_states", "/imu", "/odom")


@dataclass(frozen=True)
class InputTopics:
    """Camera and state topics discovered from an arbitrary recording."""

    camera_topics: list[str]
    state_topic: str
    duration_s: float
    message_count: int


def read_summary(path: Path) -> Summary:
    with path.open("rb") as stream:
        summary = make_reader(stream).get_summary()
    if summary is None:
        raise ValueError(f"{path} has no summary section")
    return summary


# Read-pattern grouping proxy for arbitrary recordings: non-camera topics
# whose MEAN message size is at or below this are proprio-like telemetry that
# training samples co-read ("state"); larger ones are bulk modalities (point
# clouds, grids) that are not, and would pollute the state group's chunks.
SMALL_STATE_MESSAGE_BYTES = 16_384


def plan_topic_groups(path: Path, camera_topics: list[str]) -> dict[str, str]:
    """A read-pattern ``topic_groups`` assignment for a real recording.

    The canonical default (cameras vs everything-else) assumes a manipulation
    robot's layout: cameras plus small proprio channels. A sensor-heavy
    recording (lidar/radar point clouds, grids) breaks that assumption -- the
    rule is to group topics that SHARE A READ PATTERN, so bulk
    modalities get their own group here, approximated by mean message size.
    One decode-free pass over the file.
    """
    payload_bytes: dict[str, int] = {}
    message_counts: dict[str, int] = {}
    with path.open("rb") as stream:
        for _schema, channel, message in make_reader(stream).iter_messages(log_time_order=False):
            payload_bytes[channel.topic] = payload_bytes.get(channel.topic, 0) + len(message.data)
            message_counts[channel.topic] = message_counts.get(channel.topic, 0) + 1

    camera_topic_set = set(camera_topics)
    groups: dict[str, str] = {}
    for topic, total_bytes in payload_bytes.items():
        if topic in camera_topic_set:
            continue  # schema-based default already sends cameras to "cameras"
        mean_bytes = total_bytes / max(1, message_counts[topic])
        groups[topic] = "state" if mean_bytes <= SMALL_STATE_MESSAGE_BYTES else "bulk"
    return groups


def discover_topics(path: Path) -> InputTopics:
    """Camera topics (by schema name) and the state topic of a recording.

    Works on both input-shaped files (CompressedImage cameras) and canonical
    ones (CompressedVideo cameras): ``CAMERA_SCHEMA_NAMES`` covers both.
    """
    summary = read_summary(path)
    statistics = summary.statistics
    if statistics is None:
        raise ValueError(f"{path} has no statistics record")
    message_counts = dict(statistics.channel_message_counts)

    camera_topics: list[str] = []
    state_candidates: dict[str, int] = {}
    for channel in summary.channels.values():
        schema = summary.schemas.get(channel.schema_id)
        schema_name = schema.name if schema else ""
        if schema_name in CAMERA_SCHEMA_NAMES:
            camera_topics.append(channel.topic)
        else:
            state_candidates[channel.topic] = message_counts.get(channel.id, 0)
    if not camera_topics:
        raise ValueError(f"{path} has no camera channels (schemas: {CAMERA_SCHEMA_NAMES})")
    if not state_candidates:
        raise ValueError(f"{path} has no non-camera channels to use as the state topic")

    state_topic = next(
        (topic for topic in _STATE_TOPIC_PREFERENCE if topic in state_candidates),
        max(state_candidates, key=lambda topic: state_candidates[topic]),
    )
    return InputTopics(
        camera_topics=sorted(camera_topics),
        state_topic=state_topic,
        duration_s=(statistics.message_end_time - statistics.message_start_time) / 1e9,
        message_count=statistics.message_count,
    )
