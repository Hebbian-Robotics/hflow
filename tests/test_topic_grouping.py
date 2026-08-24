"""Canonical layout: which topics share chunks, and saying so in provenance.

Grouping is a READ-PATTERN decision, not a schema one. A training sample
co-reads the cameras and the small state channels and never touches the point
clouds, so a lidar topic sharing chunks with `/imu` makes every `/imu` read
drag bulk bytes along -- 230 MB against 4 MB on the real footage in
docs/BENCHMARKS.md.
"""

import json
from pathlib import Path

import pytest
from mcap.writer import Writer as StockWriter

import hflow
from hflow.format import (
    BULK_MESSAGE_BYTES,
    DEFAULT_BULK_GROUP,
    DEFAULT_CAMERA_GROUP,
    DEFAULT_STATE_GROUP,
    METADATA_RECORD_EPISODE,
    PROVENANCE_KEY_TOPIC_GROUP_PREFIX,
)
from hflow.transform import TransformConfig, write_canonical_episode

STATE_TOPIC = "/imu"
BULK_TOPIC = "/lidar/points"


def _source_with_a_bulk_channel(path: Path) -> Path:
    """Small telemetry beside a channel of point-cloud-sized messages."""
    small_payload = json.dumps({"angular_velocity": 0.1}).encode()
    bulk_payload = json.dumps({"points": "x" * (BULK_MESSAGE_BYTES * 2)}).encode()
    with path.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        state_channel = writer.register_channel(
            topic=STATE_TOPIC, message_encoding="json", schema_id=0
        )
        bulk_channel = writer.register_channel(
            topic=BULK_TOPIC, message_encoding="json", schema_id=0
        )
        for index in range(8):
            log_time = index * 10**8
            writer.add_message(
                state_channel, log_time=log_time, data=small_payload, publish_time=log_time
            )
            writer.add_message(
                bulk_channel, log_time=log_time, data=bulk_payload, publish_time=log_time
            )
        writer.add_metadata(METADATA_RECORD_EPISODE, {"task": "grouping-demo"})
        writer.finish()
    return path


@pytest.fixture
def canonical_with_bulk(tmp_path: Path) -> Path:
    source = _source_with_a_bulk_channel(tmp_path / "source.mcap")
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    return canonical


def _topic_groups(canonical: Path) -> dict[str, str]:
    with hflow.Episode(canonical) as episode:
        return {
            key.removeprefix(PROVENANCE_KEY_TOPIC_GROUP_PREFIX): value
            for key, value in episode.metadata.items()
            if key.startswith(PROVENANCE_KEY_TOPIC_GROUP_PREFIX)
        }


def test_a_bulk_channel_gets_its_own_group(canonical_with_bulk: Path) -> None:
    groups = _topic_groups(canonical_with_bulk)

    assert groups[STATE_TOPIC] == DEFAULT_STATE_GROUP
    assert groups[BULK_TOPIC] == DEFAULT_BULK_GROUP


def test_the_resolved_layout_is_readable_from_the_published_episode(
    canonical_with_bulk: Path,
) -> None:
    """Group names appear nowhere in the MCAP itself, and the assignment is
    partly data-derived, so without provenance the layout of a published
    episode could only be guessed at from chunk membership."""
    assert _topic_groups(canonical_with_bulk) == {
        STATE_TOPIC: DEFAULT_STATE_GROUP,
        BULK_TOPIC: DEFAULT_BULK_GROUP,
    }


def test_an_explicit_override_still_wins(tmp_path: Path) -> None:
    """The escape hatch: a user who knows their read pattern beats the proxy."""
    source = _source_with_a_bulk_channel(tmp_path / "source.mcap")
    canonical = tmp_path / "override.canonical.mcap"
    write_canonical_episode(
        source, canonical, TransformConfig(topic_groups={BULK_TOPIC: DEFAULT_STATE_GROUP})
    )

    assert _topic_groups(canonical)[BULK_TOPIC] == DEFAULT_STATE_GROUP


def test_a_manipulation_recording_is_laid_out_exactly_as_before(tmp_path: Path) -> None:
    """Cameras plus proprio: the shape the old default already got right, so
    nothing about it changes except that it now says so in provenance."""
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

    source = synthesize_episode(
        tmp_path / "manipulation.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam",), black_segment=None),
    )
    canonical = tmp_path / "manipulation.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())

    groups = _topic_groups(canonical)
    assert DEFAULT_BULK_GROUP not in groups.values()
    assert groups["/joint_states"] == DEFAULT_STATE_GROUP
    assert any(group == DEFAULT_CAMERA_GROUP for group in groups.values())
