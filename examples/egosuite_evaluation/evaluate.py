"""Compare VLM hand counts with projected Lightwheel EgoSuite hand joints.

EgoSuite stores synchronized world-space hand joints, camera poses, camera
intrinsics, and head-camera video in each annotated MCAP episode. This example
projects every selected hand joint into a selected head image and treats a hand
as in frame when at least one of its 21 labeled joints lands inside the image.
Inspect AI then asks any OpenAI-compatible vision model to count visible hands
from the image alone and scores that answer against the projected label.

Run from the repository root. See
``docs/how-to/run-egosuite-hand-evaluation.md`` for the data contract,
prerequisites, and exact commands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import hflow
from inspect_ai import SampleSource, Task, eval_set
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from inspect_ai.model import (
    ChatMessageUser,
    ContentImage,
    ContentText,
    GenerateConfig,
    ResponseSchema,
)
from inspect_ai.scorer import Score, Scorer, Target, categorical, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import JSONSchema

REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from examples.egosuite_evaluation.geometry import (  # noqa: E402
    CameraPoseInWorld,
    PinholeCameraCalibration,
    Point3D,
    Quaternion,
    project_world_joints,
)
from examples.egosuite_evaluation.judgment import (  # noqa: E402
    DEFAULT_PROMPT_PATH,
    HAND_COUNT_RESPONSE_SCHEMA,
    ResponseFormat,
    image_file_data_url,
    parse_hand_count_response,
)

SCHEMA_VERSION = 1
EXPECTED_HAND_JOINT_COUNT = 21
DEFAULT_RUNS_DIRECTORY = Path("data/egosuite-evaluation/runs")
DEFAULT_LABELS_DIRECTORY = Path("data/egosuite-evaluation/labels")
INSPECT_LOGS_DIRECTORY_NAME = "logs"
RUN_METADATA_FILE_NAME = "run.json"
SUMMARY_FILE_NAME = "summary.json"
INSPECT_OPENAI_COMPATIBLE_SERVICE_NAME = "hflow-egosuite-evaluation"


class CameraView(StrEnum):
    HEAD_LEFT = "head-left"
    HEAD_RIGHT = "head-right"


@dataclass(frozen=True)
class CameraTopics:
    video: str
    intrinsic: str
    extrinsic: str


@dataclass
class _PartialFrameGeometry:
    left_hand_joints: tuple[Point3D, ...] | None = None
    right_hand_joints: tuple[Point3D, ...] | None = None
    camera_pose_in_world: CameraPoseInWorld | None = None
    calibration: PinholeCameraCalibration | None = None
    left_hand_issue_reasons: tuple[str, ...] = field(default_factory=tuple)
    right_hand_issue_reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProjectedHandFrameLabel:
    source_path: Path
    source_episode: str
    camera_view: CameraView
    frame_index: int
    left_in_frame_joint_count: int
    right_in_frame_joint_count: int
    expected_hand_count: int
    left_hand_issue_reasons: tuple[str, ...]
    right_hand_issue_reasons: tuple[str, ...]


def _required_label_record_string(
    frame_record: Mapping[str, object], field_name: str, record_context: str
) -> str:
    field_value = frame_record.get(field_name)
    if not isinstance(field_value, str) or not field_value:
        raise ValueError(f"{record_context} field {field_name!r} must be a string")
    return field_value


def _required_label_record_integer(
    frame_record: Mapping[str, object], field_name: str, record_context: str
) -> int:
    field_value = frame_record.get(field_name)
    if not isinstance(field_value, int) or isinstance(field_value, bool):
        raise ValueError(f"{record_context} field {field_name!r} must be an integer")
    return field_value


def _label_record_issue_reasons(
    frame_record: Mapping[str, object], field_name: str, record_context: str
) -> tuple[str, ...]:
    field_value = frame_record.get(field_name)
    if not isinstance(field_value, list) or not all(
        isinstance(reason, str) for reason in field_value
    ):
        raise ValueError(f"{record_context} field {field_name!r} must be an array of strings")
    return tuple(field_value)


def load_projected_hand_label_report(
    report_path: Path,
) -> dict[str, list[ProjectedHandFrameLabel]]:
    """Parse a saved label report, keyed by its stable source episode name."""

    try:
        report_payload = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read projected-hand label report {report_path}: {error}"
        ) from error
    if not isinstance(report_payload, dict):
        raise ValueError(f"projected-hand label report {report_path} must contain a JSON object")
    raw_frame_records = report_payload.get("frames")
    if not isinstance(raw_frame_records, list):
        raise ValueError(f"projected-hand label report {report_path} must contain a 'frames' array")

    labels_by_source_episode: dict[str, list[ProjectedHandFrameLabel]] = defaultdict(list)
    seen_frame_keys: set[tuple[str, int]] = set()
    for record_index, raw_frame_record in enumerate(raw_frame_records):
        record_context = f"{report_path} frame record {record_index}"
        if not isinstance(raw_frame_record, dict):
            raise ValueError(f"{record_context} must be a JSON object")
        source_path = Path(
            _required_label_record_string(raw_frame_record, "source_path", record_context)
        ).resolve()
        source_episode = _required_label_record_string(
            raw_frame_record, "source_episode", record_context
        )
        if source_episode != source_path.stem:
            raise ValueError(
                f"{record_context} source_episode {source_episode!r} does not match "
                f"source_path stem {source_path.stem!r}"
            )
        try:
            camera_view = CameraView(
                _required_label_record_string(raw_frame_record, "camera_view", record_context)
            )
        except ValueError as error:
            raise ValueError(f"{record_context} has an unsupported camera_view") from error
        frame_index = _required_label_record_integer(
            raw_frame_record, "frame_index", record_context
        )
        left_in_frame_joint_count = _required_label_record_integer(
            raw_frame_record, "left_in_frame_joint_count", record_context
        )
        right_in_frame_joint_count = _required_label_record_integer(
            raw_frame_record, "right_in_frame_joint_count", record_context
        )
        expected_hand_count = _required_label_record_integer(
            raw_frame_record, "expected_hand_count", record_context
        )
        if frame_index < 0:
            raise ValueError(f"{record_context} frame_index must be nonnegative")
        if not 0 <= left_in_frame_joint_count <= EXPECTED_HAND_JOINT_COUNT:
            raise ValueError(f"{record_context} left joint count must be between 0 and 21")
        if not 0 <= right_in_frame_joint_count <= EXPECTED_HAND_JOINT_COUNT:
            raise ValueError(f"{record_context} right joint count must be between 0 and 21")
        if expected_hand_count not in {0, 1, 2}:
            raise ValueError(f"{record_context} expected_hand_count must be 0, 1, or 2")
        frame_key = (source_episode, frame_index)
        if frame_key in seen_frame_keys:
            raise ValueError(
                f"{record_context} duplicates source episode {source_episode!r} frame {frame_index}"
            )
        seen_frame_keys.add(frame_key)
        labels_by_source_episode[source_episode].append(
            ProjectedHandFrameLabel(
                source_path=source_path,
                source_episode=source_episode,
                camera_view=camera_view,
                frame_index=frame_index,
                left_in_frame_joint_count=left_in_frame_joint_count,
                right_in_frame_joint_count=right_in_frame_joint_count,
                expected_hand_count=expected_hand_count,
                left_hand_issue_reasons=_label_record_issue_reasons(
                    raw_frame_record, "left_hand_issue_reasons", record_context
                ),
                right_hand_issue_reasons=_label_record_issue_reasons(
                    raw_frame_record, "right_hand_issue_reasons", record_context
                ),
            )
        )

    for source_labels in labels_by_source_episode.values():
        source_labels.sort(key=lambda label: label.frame_index)
    return dict(labels_by_source_episode)


@dataclass(frozen=True)
class PreparedEvaluationFrame:
    label: ProjectedHandFrameLabel
    image_path: Path


@dataclass(frozen=True)
class EvaluationConfiguration:
    source_paths: tuple[Path, ...]
    camera_view: CameraView
    frame_stride: int
    limit_per_episode: int | None
    episode_count: int | None
    samples_per_episode: int | None
    samples_per_hand_count: int | None
    sample_seed: int
    output_directory: Path
    model: str
    base_url: str
    api_key_environment_variable: str
    allow_missing_api_key: bool
    response_format: ResponseFormat
    temperature: float | None
    max_tokens: int
    max_retries: int
    worker_count: int
    prompt: str
    prompt_path: Path
    label: str


def camera_topics(camera_view: CameraView) -> CameraTopics:
    camera_name = camera_view.value.replace("-", "_")
    topic_prefix = f"/sensor/camera/{camera_name}"
    return CameraTopics(
        video=f"{topic_prefix}/video",
        intrinsic=f"{topic_prefix}/intrinsic",
        extrinsic=f"{topic_prefix}/extrinsic",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sanitize_base_url(base_url: str) -> str:
    parsed_url = urlsplit(base_url)
    if parsed_url.hostname is None:
        return base_url
    host = parsed_url.hostname
    if parsed_url.port is not None:
        host = f"{host}:{parsed_url.port}"
    return urlunsplit((parsed_url.scheme, host, parsed_url.path, "", ""))


def _resolved_mcap_paths(raw_paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved_paths: list[Path] = []
    for raw_path in raw_paths:
        if raw_path.is_dir():
            resolved_paths.extend(sorted(path.resolve() for path in raw_path.rglob("*.mcap")))
        elif raw_path.is_file() and raw_path.suffix.lower() == ".mcap":
            resolved_paths.append(raw_path.resolve())
        elif raw_path.exists():
            raise ValueError(f"input is not an MCAP file or directory: {raw_path}")
        else:
            raise FileNotFoundError(f"input does not exist: {raw_path}")
    deduplicated_paths = tuple(dict.fromkeys(resolved_paths))
    if not deduplicated_paths:
        raise ValueError("no MCAP files were found in the selected inputs")
    return deduplicated_paths


def _source_episode_name(source_path: Path) -> str:
    return source_path.stem


def select_episode_paths(
    source_paths: Sequence[Path],
    *,
    episode_count: int | None,
    sample_seed: int,
) -> tuple[Path, ...]:
    """Select a reproducible uniform sample of episodes."""

    if episode_count is None:
        return tuple(source_paths)
    if episode_count <= 0:
        raise ValueError("episode count must be positive")
    if sample_seed < 0:
        raise ValueError("sample seed must be nonnegative")
    ranked_source_paths = sorted(
        source_paths,
        key=lambda source_path: _sha256_text(f"{sample_seed}\0{_source_episode_name(source_path)}"),
    )
    selected_source_paths = set(ranked_source_paths[:episode_count])
    return tuple(
        source_path for source_path in source_paths if source_path in selected_source_paths
    )


def _selected_frame_indices(
    source_path: Path,
    source_frame_count: int,
    *,
    frame_stride: int,
    limit_per_episode: int | None,
    samples_per_episode: int | None,
    sample_seed: int,
) -> list[int]:
    if limit_per_episode is not None and samples_per_episode is not None:
        raise ValueError("limit per episode and samples per episode are mutually exclusive")
    eligible_frame_indices = list(range(0, source_frame_count, frame_stride))
    if limit_per_episode is not None:
        return eligible_frame_indices[:limit_per_episode]
    if samples_per_episode is None:
        return eligible_frame_indices
    if samples_per_episode <= 0:
        raise ValueError("samples per episode must be positive")
    if sample_seed < 0:
        raise ValueError("sample seed must be nonnegative")
    ranked_frame_indices = sorted(
        eligible_frame_indices,
        key=lambda frame_index: _sha256_text(
            f"{sample_seed}\0{_source_episode_name(source_path)}\0{frame_index}"
        ),
    )
    return sorted(ranked_frame_indices[:samples_per_episode])


def _message_header_count(decoded_message: Any) -> int | None:
    header = getattr(decoded_message, "header", None)
    header_count = getattr(header, "count", None)
    return int(header_count) if isinstance(header_count, int) else None


def _validate_header_count(topic: str, source_frame_index: int, decoded_message: Any) -> None:
    header_count = _message_header_count(decoded_message)
    if header_count is not None and header_count != source_frame_index:
        raise ValueError(
            f"{topic} message {source_frame_index} declares header.count={header_count}"
        )


def _hand_joints_from_message(topic: str, decoded_message: Any) -> tuple[Point3D, ...]:
    transforms = list(decoded_message.transforms)
    if len(transforms) != EXPECTED_HAND_JOINT_COUNT:
        raise ValueError(
            f"{topic} must contain {EXPECTED_HAND_JOINT_COUNT} joints, got {len(transforms)}"
        )
    return tuple(
        Point3D(x=float(transform.pos.x), y=float(transform.pos.y), z=float(transform.pos.z))
        for transform in transforms
    )


def _calibration_from_message(topic: str, decoded_message: Any) -> PinholeCameraCalibration:
    intrinsic_matrix = list(decoded_message.K)
    if len(intrinsic_matrix) != 9:
        raise ValueError(f"{topic} must contain a 3x3 intrinsic matrix")
    distortion_coefficients = [float(value) for value in decoded_message.D]
    if any(abs(value) > 1e-12 for value in distortion_coefficients):
        raise ValueError(
            f"{topic} has nonzero distortion coefficients; this pinhole evaluator "
            "requires rectified images"
        )
    return PinholeCameraCalibration(
        width=int(decoded_message.width),
        height=int(decoded_message.height),
        focal_length_x=float(intrinsic_matrix[0]),
        focal_length_y=float(intrinsic_matrix[4]),
        principal_point_x=float(intrinsic_matrix[2]),
        principal_point_y=float(intrinsic_matrix[5]),
    )


def _camera_pose_from_message(
    topic: str, expected_camera_frame: str, decoded_message: Any
) -> CameraPoseInWorld:
    transforms = list(decoded_message.transforms)
    if len(transforms) != 1:
        raise ValueError(f"{topic} must contain exactly one camera transform")
    transform = transforms[0]
    if transform.parent_frame_id != "world" or transform.child_frame_id != expected_camera_frame:
        raise ValueError(
            f"{topic} must describe world -> {expected_camera_frame}, got "
            f"{transform.parent_frame_id} -> {transform.child_frame_id}"
        )
    return CameraPoseInWorld(
        translation=Point3D(
            x=float(transform.translation.x),
            y=float(transform.translation.y),
            z=float(transform.translation.z),
        ),
        rotation=Quaternion(
            x=float(transform.rotation.x),
            y=float(transform.rotation.y),
            z=float(transform.rotation.z),
            w=float(transform.rotation.w),
        ),
    )


def _hand_issue_reasons(decoded_message: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left_reasons: list[str] = []
    right_reasons: list[str] = []
    for raw_problem_type in decoded_message.problem_type:
        side, separator, reason = str(raw_problem_type).partition(":")
        if not separator or not reason:
            raise ValueError(f"invalid EgoSuite hand problem_type: {raw_problem_type!r}")
        match side:
            case "left":
                left_reasons.append(reason)
            case "right":
                right_reasons.append(reason)
            case unknown_side:
                raise ValueError(f"unknown EgoSuite hand problem side: {unknown_side!r}")
    return tuple(left_reasons), tuple(right_reasons)


def _topic_message_counts(episode: hflow.Episode) -> dict[str, int]:
    counts_by_topic: dict[str, int] = {}
    for channel_info in episode.channels.values():
        if channel_info.topic in counts_by_topic:
            raise ValueError(f"duplicate MCAP topic {channel_info.topic!r} in {episode.path}")
        counts_by_topic[channel_info.topic] = channel_info.message_count
    return counts_by_topic


def _required_frame_count(
    source_path: Path,
    camera_view: CameraView,
    counts_by_topic: Mapping[str, int],
) -> int:
    topics = camera_topics(camera_view)
    required_topics = (
        topics.video,
        topics.intrinsic,
        topics.extrinsic,
        "/pose/left_hand",
        "/pose/right_hand",
        "/annotation/bad_frame/pose/hand",
    )
    missing_topics = [topic for topic in required_topics if topic not in counts_by_topic]
    if missing_topics:
        raise ValueError(f"{source_path} is missing required topics: {missing_topics}")
    count_by_topic = {topic: counts_by_topic[topic] for topic in required_topics}
    distinct_counts = set(count_by_topic.values())
    if len(distinct_counts) != 1:
        raise ValueError(f"required EgoSuite topics are not frame-aligned: {count_by_topic}")
    frame_count = distinct_counts.pop()
    if frame_count <= 0:
        raise ValueError(f"required EgoSuite topics contain no frames: {source_path}")
    return frame_count


def _complete_projected_label(
    source_path: Path,
    camera_view: CameraView,
    frame_index: int,
    partial_geometry: _PartialFrameGeometry,
) -> ProjectedHandFrameLabel:
    if partial_geometry.left_hand_joints is None:
        raise ValueError(f"frame {frame_index} is missing left-hand joints")
    if partial_geometry.right_hand_joints is None:
        raise ValueError(f"frame {frame_index} is missing right-hand joints")
    if partial_geometry.camera_pose_in_world is None:
        raise ValueError(f"frame {frame_index} is missing the camera extrinsic")
    if partial_geometry.calibration is None:
        raise ValueError(f"frame {frame_index} is missing camera intrinsics")
    projected_left_hand = project_world_joints(
        partial_geometry.left_hand_joints,
        partial_geometry.camera_pose_in_world,
        partial_geometry.calibration,
    )
    projected_right_hand = project_world_joints(
        partial_geometry.right_hand_joints,
        partial_geometry.camera_pose_in_world,
        partial_geometry.calibration,
    )
    return ProjectedHandFrameLabel(
        source_path=source_path,
        source_episode=_source_episode_name(source_path),
        camera_view=camera_view,
        frame_index=frame_index,
        left_in_frame_joint_count=projected_left_hand.in_frame_joint_count,
        right_in_frame_joint_count=projected_right_hand.in_frame_joint_count,
        expected_hand_count=sum(
            (projected_left_hand.is_in_frame, projected_right_hand.is_in_frame)
        ),
        left_hand_issue_reasons=partial_geometry.left_hand_issue_reasons,
        right_hand_issue_reasons=partial_geometry.right_hand_issue_reasons,
    )


def load_projected_hand_labels(
    source_path: Path,
    *,
    camera_view: CameraView,
    frame_stride: int,
    limit_per_episode: int | None,
    samples_per_episode: int | None = None,
    sample_seed: int = 42,
) -> list[ProjectedHandFrameLabel]:
    """Read synchronized EgoSuite labels without decoding the video payloads."""

    if frame_stride <= 0:
        raise ValueError("frame stride must be positive")
    if limit_per_episode is not None and limit_per_episode <= 0:
        raise ValueError("limit per episode must be positive")
    selected_camera_topics = camera_topics(camera_view)
    expected_camera_frame = camera_view.value.replace("-", "_") + "_camera"
    decoded_topics = (
        "/pose/left_hand",
        "/pose/right_hand",
        "/annotation/bad_frame/pose/hand",
        selected_camera_topics.intrinsic,
        selected_camera_topics.extrinsic,
    )
    with hflow.Episode(source_path) as episode:
        source_frame_count = _required_frame_count(
            source_path,
            camera_view,
            _topic_message_counts(episode),
        )
        selected_frame_indices = _selected_frame_indices(
            source_path,
            source_frame_count,
            frame_stride=frame_stride,
            limit_per_episode=limit_per_episode,
            samples_per_episode=samples_per_episode,
            sample_seed=sample_seed,
        )
        partial_geometry_by_frame = {
            frame_index: _PartialFrameGeometry() for frame_index in selected_frame_indices
        }
        message_index_by_topic: defaultdict[str, int] = defaultdict(int)
        for decoded_batch in episode.iter_decoded_batches(topics=decoded_topics):
            topic = decoded_batch.topic
            for decoded_message in decoded_batch.messages:
                source_frame_index = message_index_by_topic[topic]
                message_index_by_topic[topic] += 1
                _validate_header_count(topic, source_frame_index, decoded_message)
                partial_geometry = partial_geometry_by_frame.get(source_frame_index)
                if partial_geometry is None:
                    continue
                match topic:
                    case "/pose/left_hand":
                        partial_geometry.left_hand_joints = _hand_joints_from_message(
                            topic, decoded_message
                        )
                    case "/pose/right_hand":
                        partial_geometry.right_hand_joints = _hand_joints_from_message(
                            topic, decoded_message
                        )
                    case "/annotation/bad_frame/pose/hand":
                        (
                            partial_geometry.left_hand_issue_reasons,
                            partial_geometry.right_hand_issue_reasons,
                        ) = _hand_issue_reasons(decoded_message)
                    case intrinsic_topic if intrinsic_topic == selected_camera_topics.intrinsic:
                        partial_geometry.calibration = _calibration_from_message(
                            topic, decoded_message
                        )
                    case extrinsic_topic if extrinsic_topic == selected_camera_topics.extrinsic:
                        partial_geometry.camera_pose_in_world = _camera_pose_from_message(
                            topic, expected_camera_frame, decoded_message
                        )
                    case unexpected_topic:
                        raise AssertionError(f"unexpected decoded topic: {unexpected_topic}")
        decoded_count_by_topic = {topic: message_index_by_topic[topic] for topic in decoded_topics}
        if any(count != source_frame_count for count in decoded_count_by_topic.values()):
            raise ValueError(
                f"decoded EgoSuite topics are not frame-aligned in {source_path}: "
                f"{decoded_count_by_topic}"
            )
        return [
            _complete_projected_label(
                source_path,
                camera_view,
                frame_index,
                partial_geometry_by_frame[frame_index],
            )
            for frame_index in selected_frame_indices
        ]


def select_stratified_labels(
    labels_by_source: Mapping[Path, Sequence[ProjectedHandFrameLabel]],
    *,
    samples_per_hand_count: int | None,
    sample_seed: int,
) -> dict[Path, list[ProjectedHandFrameLabel]]:
    """Select up to a fixed number of reproducible samples for each hand count."""

    copied_labels_by_source = {
        source_path: list(labels) for source_path, labels in labels_by_source.items()
    }
    if samples_per_hand_count is None:
        return copied_labels_by_source
    if samples_per_hand_count <= 0:
        raise ValueError("samples per hand count must be positive")
    if sample_seed < 0:
        raise ValueError("sample seed must be nonnegative")

    all_labels = [label for labels in labels_by_source.values() for label in labels]
    selected_frame_keys: set[tuple[Path, int]] = set()
    for expected_hand_count in range(3):
        matching_labels = [
            label for label in all_labels if label.expected_hand_count == expected_hand_count
        ]
        matching_labels.sort(
            key=lambda label: _sha256_text(
                f"{sample_seed}\0{label.source_path}\0{label.frame_index}"
            )
        )
        selected_frame_keys.update(
            (label.source_path, label.frame_index)
            for label in matching_labels[:samples_per_hand_count]
        )

    return {
        source_path: [
            label
            for label in labels
            if (label.source_path, label.frame_index) in selected_frame_keys
        ]
        for source_path, labels in copied_labels_by_source.items()
        if any((label.source_path, label.frame_index) in selected_frame_keys for label in labels)
    }


def _frame_cache_key(source_path: Path) -> str:
    path_hash = hashlib.sha256(str(source_path).encode()).hexdigest()[:12]
    sanitized_episode_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", source_path.stem).strip("-")
    return f"{sanitized_episode_name or 'episode'}-{path_hash}"


def _extract_sampled_frames(
    source_path: Path,
    labels: Sequence[ProjectedHandFrameLabel],
    *,
    camera_view: CameraView,
    frame_cache_directory: Path,
) -> list[Path]:
    if not labels:
        return []
    selected_frame_indices = [label.frame_index for label in labels]
    if selected_frame_indices != sorted(set(selected_frame_indices)):
        raise ValueError(f"selected frame indices must be unique and ordered: {source_path}")
    if any(label.source_path != source_path for label in labels):
        raise ValueError(f"selected labels do not all belong to {source_path}")
    episode_cache_directory = frame_cache_directory / _frame_cache_key(source_path)
    camera_topic = camera_topics(camera_view).video
    with hflow.Episode(source_path, workdir=episode_cache_directory) as episode:
        extracted_frames = episode.frames_at_indices(
            camera_topic,
            frame_indices=selected_frame_indices,
        )
    return [extracted_frame.path for extracted_frame in extracted_frames]


def prepare_evaluation_frames(
    labels_by_source: Mapping[Path, Sequence[ProjectedHandFrameLabel]],
    *,
    camera_view: CameraView,
    frame_cache_directory: Path,
) -> list[PreparedEvaluationFrame]:
    prepared_frames: list[PreparedEvaluationFrame] = []
    for source_path, labels in labels_by_source.items():
        image_paths = _extract_sampled_frames(
            source_path,
            labels,
            camera_view=camera_view,
            frame_cache_directory=frame_cache_directory,
        )
        prepared_frames.extend(
            PreparedEvaluationFrame(label=label, image_path=image_path)
            for label, image_path in zip(labels, image_paths, strict=True)
        )
    return prepared_frames


def _inspect_sample(frame: PreparedEvaluationFrame, prompt: str) -> Sample:
    label = frame.label
    return Sample(
        id=f"{label.source_episode}:{label.camera_view.value}:{label.frame_index:06d}",
        input=[
            ChatMessageUser(
                content=[
                    ContentText(text=prompt),
                    ContentImage(image=image_file_data_url(frame.image_path)),
                ]
            )
        ],
        target=str(label.expected_hand_count),
        metadata={
            "source_path": str(label.source_path),
            "source_episode": label.source_episode,
            "camera_view": label.camera_view.value,
            "frame_index": label.frame_index,
            "left_in_frame_joint_count": label.left_in_frame_joint_count,
            "right_in_frame_joint_count": label.right_in_frame_joint_count,
            "left_hand_issue_reasons": list(label.left_hand_issue_reasons),
            "right_hand_issue_reasons": list(label.right_hand_issue_reasons),
        },
    )


class _PreparedFrameSampleSource(SampleSource):
    def __init__(
        self,
        *,
        frames: Sequence[PreparedEvaluationFrame],
        prompt: str,
        batch_size: int,
    ) -> None:
        self._frame_iterator: Iterator[PreparedEvaluationFrame] = iter(frames)
        self._prompt = prompt
        self._batch_size = batch_size

    def _next_batch(self) -> list[Sample]:
        frames = list(islice(self._frame_iterator, self._batch_size))
        return [_inspect_sample(frame, self._prompt) for frame in frames]

    def initial_samples(self) -> list[Sample]:
        return self._next_batch()

    async def next_samples(self) -> list[Sample] | None:
        samples = self._next_batch()
        return samples or None


@scorer(
    metrics={"prediction": categorical(["0", "1", "2"]), "agreement": [mean()]},
    name="egosuite_projected_hand_count",
)
def projected_hand_count_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score | None:
        try:
            predicted_value = parse_hand_count_response(state.output.completion)
        except ValueError:
            return None
        return Score(
            value={
                "prediction": str(predicted_value),
                "agreement": str(predicted_value) == target.text,
            },
            answer=str(predicted_value),
        )

    return score


def _inspect_generate_config(configuration: EvaluationConfiguration) -> GenerateConfig:
    response_schema: ResponseSchema | None = None
    extra_body: dict[str, object] | None = None
    match configuration.response_format:
        case ResponseFormat.JSON_SCHEMA:
            response_schema = ResponseSchema(
                name="egosuite_projected_hand_count",
                json_schema=JSONSchema.model_validate(HAND_COUNT_RESPONSE_SCHEMA),
            )
        case ResponseFormat.JSON_OBJECT:
            extra_body = {"response_format": {"type": "json_object"}}
        case ResponseFormat.TEXT:
            pass
    return GenerateConfig(
        max_retries=configuration.max_retries,
        max_tokens=configuration.max_tokens,
        temperature=configuration.temperature,
        response_schema=response_schema,
        extra_body=extra_body,
    )


def _label_record(label: ProjectedHandFrameLabel) -> dict[str, object]:
    return {
        **asdict(label),
        "source_path": str(label.source_path),
        "camera_view": label.camera_view.value,
        "left_hand_issue_reasons": list(label.left_hand_issue_reasons),
        "right_hand_issue_reasons": list(label.right_hand_issue_reasons),
    }


def summarize_reference_labels(
    labels: Sequence[ProjectedHandFrameLabel],
) -> dict[str, object]:
    expected_counts = Counter(label.expected_hand_count for label in labels)
    left_joint_counts = Counter(label.left_in_frame_joint_count for label in labels)
    right_joint_counts = Counter(label.right_in_frame_joint_count for label in labels)
    left_issue_reasons = Counter(
        reason for label in labels for reason in label.left_hand_issue_reasons
    )
    right_issue_reasons = Counter(
        reason for label in labels for reason in label.right_hand_issue_reasons
    )
    return {
        "frame_count": len(labels),
        "expected_hand_count_counts": dict(sorted(expected_counts.items())),
        "expected_hand_count_fractions": {
            str(hand_count): count / len(labels)
            for hand_count, count in sorted(expected_counts.items())
        }
        if labels
        else {},
        "left_in_frame_joint_count_counts": dict(sorted(left_joint_counts.items())),
        "right_in_frame_joint_count_counts": dict(sorted(right_joint_counts.items())),
        "frames_with_pose_quality_issue": sum(
            bool(label.left_hand_issue_reasons or label.right_hand_issue_reasons)
            for label in labels
        ),
        "left_hand_issue_reason_counts": dict(sorted(left_issue_reasons.items())),
        "right_hand_issue_reason_counts": dict(sorted(right_issue_reasons.items())),
    }


def _write_json_atomically(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _print_reference_summary(
    labels_by_source: Mapping[Path, Sequence[ProjectedHandFrameLabel]],
) -> None:
    print("| source episode | frames | 0 hands | 1 hand | 2 hands | pose issue frames |")
    print("|---|---:|---:|---:|---:|---:|")
    for source_path, labels in labels_by_source.items():
        summary = summarize_reference_labels(labels)
        counts = cast(dict[int, int], summary["expected_hand_count_counts"])
        print(
            f"| {source_path.stem} | {summary['frame_count']} | {counts.get(0, 0)} "
            f"| {counts.get(1, 0)} | {counts.get(2, 0)} "
            f"| {summary['frames_with_pose_quality_issue']} |"
        )
    all_labels = [label for labels in labels_by_source.values() for label in labels]
    if len(labels_by_source) > 1:
        summary = summarize_reference_labels(all_labels)
        counts = cast(dict[int, int], summary["expected_hand_count_counts"])
        print(
            f"| **all** | {summary['frame_count']} | {counts.get(0, 0)} | {counts.get(1, 0)} "
            f"| {counts.get(2, 0)} | {summary['frames_with_pose_quality_issue']} |"
        )


def write_label_report(
    labels_by_source: Mapping[Path, Sequence[ProjectedHandFrameLabel]],
    *,
    camera_view: CameraView,
    frame_stride: int,
    limit_per_episode: int | None,
    episode_count: int | None,
    samples_per_episode: int | None,
    samples_per_hand_count: int | None,
    sample_seed: int,
    output_path: Path,
) -> None:
    all_labels = [label for labels in labels_by_source.values() for label in labels]
    report = {
        "schema_version": SCHEMA_VERSION,
        "label_type": "projected-hand-joints",
        "camera_view": camera_view.value,
        "frame_stride": frame_stride,
        "limit_per_episode": limit_per_episode,
        "episode_count": episode_count,
        "samples_per_episode": samples_per_episode,
        "samples_per_hand_count": samples_per_hand_count,
        "sample_seed": sample_seed,
        "projection_rule": "hand is in frame when at least one labeled joint has positive depth and projects inside the image bounds",
        "summary": summarize_reference_labels(all_labels),
        "sources": [str(path) for path in labels_by_source],
        "frames": [_label_record(label) for label in all_labels],
    }
    _write_json_atomically(output_path, report)


def _required_run_metadata_string(
    document: Mapping[str, object], field_name: str, record_context: str
) -> str:
    field_value = document.get(field_name)
    if not isinstance(field_value, str) or not field_value:
        raise ValueError(f"{record_context} field {field_name!r} must be a string")
    return field_value


def _required_run_metadata_integer(
    document: Mapping[str, object], field_name: str, record_context: str
) -> int:
    field_value = document.get(field_name)
    if not isinstance(field_value, int) or isinstance(field_value, bool):
        raise ValueError(f"{record_context} field {field_name!r} must be an integer")
    return field_value


def _optional_run_metadata_integer(
    document: Mapping[str, object], field_name: str, record_context: str
) -> int | None:
    field_value = document.get(field_name)
    if field_value is None:
        return None
    if not isinstance(field_value, int) or isinstance(field_value, bool):
        raise ValueError(f"{record_context} field {field_name!r} must be an integer or null")
    return field_value


@dataclass(frozen=True)
class RunMetadata:
    """Saved run.json metadata, parsed once at the load boundary.

    ``document`` is the full persisted schema; the other fields are the
    consumed subset, typed so task creation and summary generation never
    index back into the JSON shape.
    """

    label: str
    fingerprint: str
    model: str
    camera_view: str
    frame_stride: int
    episode_count: int | None
    samples_per_episode: int | None
    samples_per_hand_count: int | None
    sample_seed: int
    document: Mapping[str, object]

    def to_json_dict(self) -> dict[str, object]:
        return dict(self.document)

    @classmethod
    def from_json_file(cls, metadata_path: Path) -> RunMetadata:
        try:
            document = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read run metadata {metadata_path}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError(f"{metadata_path} must contain a JSON object")
        context = str(metadata_path)
        return cls(
            label=_required_run_metadata_string(document, "label", context),
            fingerprint=_required_run_metadata_string(document, "fingerprint", context),
            model=_required_run_metadata_string(document, "model", context),
            camera_view=_required_run_metadata_string(document, "camera_view", context),
            frame_stride=_required_run_metadata_integer(document, "frame_stride", context),
            episode_count=_optional_run_metadata_integer(document, "episode_count", context),
            samples_per_episode=_optional_run_metadata_integer(
                document, "samples_per_episode", context
            ),
            samples_per_hand_count=_optional_run_metadata_integer(
                document, "samples_per_hand_count", context
            ),
            sample_seed=_required_run_metadata_integer(document, "sample_seed", context),
            document=document,
        )


def _run_metadata_document(configuration: EvaluationConfiguration) -> dict[str, object]:
    source_descriptors = [
        {"path": str(path), "size_bytes": path.stat().st_size}
        for path in configuration.source_paths
    ]
    result_contract = {
        "adapter_schema_version": SCHEMA_VERSION,
        "inspect_ai_version": importlib.metadata.version("inspect-ai"),
        "sources": source_descriptors,
        "camera_view": configuration.camera_view.value,
        "frame_stride": configuration.frame_stride,
        "limit_per_episode": configuration.limit_per_episode,
        "episode_count": configuration.episode_count,
        "samples_per_episode": configuration.samples_per_episode,
        "samples_per_hand_count": configuration.samples_per_hand_count,
        "sample_seed": configuration.sample_seed,
        "projection_contract": {
            "world_to_camera": "inverse of the world-parent to camera-child pose",
            "projection": "pinhole K matrix over positive camera depth",
            "hand_presence": "at least one of 21 joints projects inside image bounds",
            "distortion": "rectified images only; nonzero distortion coefficients rejected",
            "pose_quality_flags": "recorded as metadata and do not change hand presence",
        },
        "model": configuration.model,
        "base_url": _sanitize_base_url(configuration.base_url),
        "response_format": configuration.response_format.value,
        "temperature": configuration.temperature,
        "max_tokens": configuration.max_tokens,
        "prompt": {
            "path": str(configuration.prompt_path),
            "sha256": _sha256_text(configuration.prompt),
            "text": configuration.prompt,
            "response_schema": HAND_COUNT_RESPONSE_SCHEMA,
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "label": configuration.label,
        "fingerprint": hflow.fingerprint_contract(result_contract),
        **result_contract,
        "api_key_environment_variable": configuration.api_key_environment_variable,
        "worker_count": configuration.worker_count,
        "max_retries": configuration.max_retries,
    }


def _run_metadata(configuration: EvaluationConfiguration) -> RunMetadata:
    document = _run_metadata_document(configuration)
    return RunMetadata(
        label=configuration.label,
        fingerprint=str(document["fingerprint"]),
        model=configuration.model,
        camera_view=configuration.camera_view.value,
        frame_stride=configuration.frame_stride,
        episode_count=configuration.episode_count,
        samples_per_episode=configuration.samples_per_episode,
        samples_per_hand_count=configuration.samples_per_hand_count,
        sample_seed=configuration.sample_seed,
        document=document,
    )


def _prepare_output_directory(configuration: EvaluationConfiguration) -> RunMetadata:
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = configuration.output_directory / RUN_METADATA_FILE_NAME
    current_metadata = _run_metadata(configuration)
    if metadata_path.is_file():
        existing_metadata = RunMetadata.from_json_file(metadata_path)
        if existing_metadata.fingerprint != current_metadata.fingerprint:
            raise ValueError(
                f"{metadata_path} describes a different experiment; choose another --output"
            )
        return existing_metadata
    _write_json_atomically(metadata_path, current_metadata.to_json_dict())
    return current_metadata


def _inspect_model_name(requested_model: str) -> str:
    return f"openai-api/{INSPECT_OPENAI_COMPATIBLE_SERVICE_NAME}/{requested_model.lstrip('/')}"


def _sample_expected_value(sample: EvalSample) -> int:
    raw_target = sample.target[0] if isinstance(sample.target, list) else sample.target
    return parse_hand_count_response(raw_target)


@dataclass(frozen=True)
class SampleResponseMetadata:
    """Provider response fields retained for every sample outcome variant."""

    response_model: str | None
    latency_seconds: float | None
    usage: Mapping[str, object] | None


@dataclass(frozen=True)
class SuccessfulSampleOutcome:
    """A completed response parsed into the hand-count result vocabulary."""

    raw_response: str
    response_metadata: SampleResponseMetadata
    predicted_value: int


@dataclass(frozen=True)
class InvalidResponseSampleOutcome:
    """A completed response outside the hand-count result vocabulary."""

    raw_response: str
    response_metadata: SampleResponseMetadata
    parse_error: str


@dataclass(frozen=True)
class ExecutionErrorSampleOutcome:
    """Inspect reported an execution failure instead of a completed response."""

    raw_response: str
    response_metadata: SampleResponseMetadata
    error: str


SampleOutcome = SuccessfulSampleOutcome | InvalidResponseSampleOutcome | ExecutionErrorSampleOutcome


@dataclass(frozen=True)
class EvaluatedSample:
    """One evaluated Inspect sample: identification plus exactly one outcome variant."""

    frame_id: str
    source_episode: str
    expected_value: int
    outcome: SampleOutcome


def _sample_result(log: EvalLog, sample: EvalSample) -> EvaluatedSample:
    sample_metadata = sample.metadata or {}
    response_metadata = SampleResponseMetadata(
        response_model=sample.output.model or None,
        latency_seconds=sample.output.time,
        usage=(
            sample.output.usage.model_dump(exclude_none=True)
            if sample.output.usage is not None
            else None
        ),
    )
    if sample.error is not None or sample.output.error is not None:
        error_message = sample.error.message if sample.error is not None else sample.output.error
        outcome: SampleOutcome = ExecutionErrorSampleOutcome(
            raw_response=sample.output.completion,
            response_metadata=response_metadata,
            error=str(error_message)[:1000],
        )
    else:
        try:
            predicted_value = parse_hand_count_response(sample.output.completion)
        except ValueError as error:
            outcome = InvalidResponseSampleOutcome(
                raw_response=sample.output.completion,
                response_metadata=response_metadata,
                parse_error=str(error),
            )
        else:
            outcome = SuccessfulSampleOutcome(
                raw_response=sample.output.completion,
                response_metadata=response_metadata,
                predicted_value=predicted_value,
            )
    return EvaluatedSample(
        frame_id=str(sample.id),
        source_episode=str(sample_metadata["source_episode"]),
        expected_value=_sample_expected_value(sample),
        outcome=outcome,
    )


def _results_from_inspect_logs(
    log_headers: Sequence[EvalLog],
) -> tuple[list[EvaluatedSample], list[str]]:
    results: list[EvaluatedSample] = []
    log_locations: list[str] = []
    for log_header in log_headers:
        log_locations.append(log_header.location)
        completed_log = read_eval_log(log_header.location)
        results.extend(
            _sample_result(completed_log, sample) for sample in completed_log.samples or []
        )
    return results, log_locations


def _evaluation_result_summary(results: Sequence[EvaluatedSample]) -> dict[str, object]:
    expected_counts = Counter(result.expected_value for result in results)
    valid_pairs = [
        (result.expected_value, result.outcome.predicted_value)
        for result in results
        if isinstance(result.outcome, SuccessfulSampleOutcome)
    ]
    predicted_counts = Counter(predicted_value for _, predicted_value in valid_pairs)
    confusion_counts = Counter(valid_pairs)
    agreement_count = sum(
        expected_value == predicted_value for expected_value, predicted_value in valid_pairs
    )
    per_class_agreement: dict[str, dict[str, int | float | None]] = {}
    valid_class_agreement_fractions: list[float] = []
    attempted_class_agreement_fractions: list[float] = []
    for expected_value, attempted_count in sorted(expected_counts.items()):
        valid_count = sum(
            confusion_counts.get((expected_value, predicted_value), 0)
            for predicted_value in (0, 1, 2)
        )
        class_agreement_count = confusion_counts.get((expected_value, expected_value), 0)
        valid_class_agreement_fraction = (
            class_agreement_count / valid_count if valid_count else None
        )
        attempted_class_agreement_fraction = class_agreement_count / attempted_count
        per_class_agreement[str(expected_value)] = {
            "attempted_count": attempted_count,
            "valid_count": valid_count,
            "agreement_count": class_agreement_count,
            "agreement_fraction": valid_class_agreement_fraction,
            "attempted_agreement_fraction": attempted_class_agreement_fraction,
        }
        if valid_class_agreement_fraction is not None:
            valid_class_agreement_fractions.append(valid_class_agreement_fraction)
        attempted_class_agreement_fractions.append(attempted_class_agreement_fraction)
    latency_values = [
        float(result.outcome.response_metadata.latency_seconds)
        for result in results
        if result.outcome.response_metadata.latency_seconds is not None
    ]
    usage_totals: Counter[str] = Counter()
    for result in results:
        usage = result.outcome.response_metadata.usage
        if usage is not None:
            usage_totals.update(
                {
                    str(name): float(value)
                    for name, value in usage.items()
                    if isinstance(value, int | float) and not isinstance(value, bool)
                }
            )
    return {
        "attempted_count": len(results),
        "valid_count": len(valid_pairs),
        "invalid_count": sum(
            isinstance(result.outcome, InvalidResponseSampleOutcome) for result in results
        ),
        "error_count": sum(
            isinstance(result.outcome, ExecutionErrorSampleOutcome) for result in results
        ),
        "expected_value_counts": dict(sorted(expected_counts.items())),
        "predicted_value_counts": dict(sorted(predicted_counts.items())),
        "agreement_count": agreement_count,
        "agreement_fraction": agreement_count / len(valid_pairs) if valid_pairs else None,
        "attempted_agreement_fraction": agreement_count / len(results) if results else None,
        "macro_agreement_fraction": (
            sum(valid_class_agreement_fractions) / len(valid_class_agreement_fractions)
            if valid_class_agreement_fractions
            else None
        ),
        "macro_attempted_agreement_fraction": (
            sum(attempted_class_agreement_fractions) / len(attempted_class_agreement_fractions)
            if attempted_class_agreement_fractions
            else None
        ),
        "per_class_agreement": per_class_agreement,
        "confusion_matrix": {
            str(expected): {
                str(predicted): confusion_counts.get((expected, predicted), 0)
                for predicted in (0, 1, 2)
            }
            for expected in (0, 1, 2)
        },
        "average_latency_seconds": (
            sum(latency_values) / len(latency_values) if latency_values else None
        ),
        "usage_totals": dict(sorted(usage_totals.items())),
    }


def summarize_evaluation_results(
    run_metadata: RunMetadata, results: Sequence[EvaluatedSample]
) -> dict[str, object]:
    latest_results = {result.frame_id: result for result in results}
    deduplicated_results = list(latest_results.values())
    source_episodes = sorted({result.source_episode for result in deduplicated_results})
    return {
        "schema_version": SCHEMA_VERSION,
        "label": run_metadata.label,
        "fingerprint": run_metadata.fingerprint,
        "model": run_metadata.model,
        "camera_view": run_metadata.camera_view,
        "frame_stride": run_metadata.frame_stride,
        "episode_count": run_metadata.episode_count,
        "samples_per_episode": run_metadata.samples_per_episode,
        "samples_per_hand_count": run_metadata.samples_per_hand_count,
        "sample_seed": run_metadata.sample_seed,
        "overall": _evaluation_result_summary(deduplicated_results),
        "episodes": {
            source_episode: _evaluation_result_summary(
                [
                    result
                    for result in deduplicated_results
                    if result.source_episode == source_episode
                ]
            )
            for source_episode in source_episodes
        },
    }


def _format_percentage(value: object) -> str:
    return f"{100.0 * float(value):.2f}%" if isinstance(value, int | float) else "-"


def _print_evaluation_summary(summary: Mapping[str, object]) -> None:
    overall = cast(dict[str, object], summary["overall"])
    expected_counts = cast(dict[int, int], overall["expected_value_counts"])
    predicted_counts = cast(dict[int, int], overall["predicted_value_counts"])
    print(f"\nEgoSuite projected-hand evaluation: {summary['label']}")
    print(
        "| valid / attempted | target 0 | target 1 | target 2 | predicted 0 | predicted 1 | "
        "predicted 2 | valid accuracy | end-to-end accuracy | macro end-to-end accuracy |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {overall['valid_count']} / {overall['attempted_count']} "
        f"| {expected_counts.get(0, 0)} | {expected_counts.get(1, 0)} "
        f"| {expected_counts.get(2, 0)} | {predicted_counts.get(0, 0)} "
        f"| {predicted_counts.get(1, 0)} | {predicted_counts.get(2, 0)} "
        f"| {_format_percentage(overall['agreement_fraction'])} "
        f"| {_format_percentage(overall['attempted_agreement_fraction'])} "
        f"| {_format_percentage(overall['macro_attempted_agreement_fraction'])} |"
    )


def run_evaluation(configuration: EvaluationConfiguration) -> dict[str, object]:
    api_key = os.environ.get(configuration.api_key_environment_variable)
    if not api_key and not configuration.allow_missing_api_key:
        raise ValueError(
            f"{configuration.api_key_environment_variable} is not set; use --allow-missing-api-key "
            "only for an endpoint that does not authenticate"
        )
    run_metadata = _prepare_output_directory(configuration)
    candidate_labels_by_source = {
        source_path: load_projected_hand_labels(
            source_path,
            camera_view=configuration.camera_view,
            frame_stride=configuration.frame_stride,
            limit_per_episode=configuration.limit_per_episode,
            samples_per_episode=configuration.samples_per_episode,
            sample_seed=configuration.sample_seed,
        )
        for source_path in configuration.source_paths
    }
    labels_by_source = select_stratified_labels(
        candidate_labels_by_source,
        samples_per_hand_count=configuration.samples_per_hand_count,
        sample_seed=configuration.sample_seed,
    )
    _print_reference_summary(labels_by_source)
    prepared_frames = prepare_evaluation_frames(
        labels_by_source,
        camera_view=configuration.camera_view,
        frame_cache_directory=configuration.output_directory / "frames",
    )
    inspect_logs_directory = configuration.output_directory / INSPECT_LOGS_DIRECTORY_NAME
    inspect_logs_directory.mkdir(parents=True, exist_ok=True)
    task = Task(
        name="egosuite_projected_hand_count",
        version=run_metadata.fingerprint[:16],
        dataset=_PreparedFrameSampleSource(
            frames=prepared_frames,
            prompt=configuration.prompt,
            batch_size=max(configuration.worker_count * 2, 1),
        ),
        scorer=projected_hand_count_scorer(),
        config=_inspect_generate_config(configuration),
        metadata={
            "requested_model": configuration.model,
            "camera_view": configuration.camera_view.value,
            "frame_stride": configuration.frame_stride,
            "episode_count": configuration.episode_count,
            "samples_per_episode": configuration.samples_per_episode,
            "samples_per_hand_count": configuration.samples_per_hand_count,
            "sample_seed": configuration.sample_seed,
            "prompt_sha256": _sha256_text(configuration.prompt),
            "run_fingerprint": run_metadata.fingerprint,
        },
    )
    inserted_placeholder_api_key = api_key is None
    if inserted_placeholder_api_key:
        os.environ[configuration.api_key_environment_variable] = "not-needed"
    try:
        all_tasks_succeeded, log_headers = eval_set(
            [task],
            log_dir=str(inspect_logs_directory),
            model=_inspect_model_name(configuration.model),
            model_base_url=configuration.base_url,
            model_args={"api_key_var": configuration.api_key_environment_variable},
            metadata={
                "run_fingerprint": run_metadata.fingerprint,
                "label": configuration.label,
            },
            max_connections=configuration.worker_count,
            max_samples=configuration.worker_count,
            max_tasks=1,
            fail_on_error=False,
            retry_on_error=0,
            retry_attempts=1,
            log_format="eval",
            log_images=False,
            log_model_api=False,
            log_dir_allow_dirty=True,
        )
    finally:
        if inserted_placeholder_api_key:
            os.environ.pop(configuration.api_key_environment_variable, None)
    results, log_locations = _results_from_inspect_logs(log_headers)
    summary = summarize_evaluation_results(run_metadata, results)
    summary["inspect_logs"] = log_locations
    summary["reference"] = summarize_reference_labels(
        [label for labels in labels_by_source.values() for label in labels]
    )
    summary_path = configuration.output_directory / SUMMARY_FILE_NAME
    _write_json_atomically(summary_path, summary)
    _print_evaluation_summary(summary)
    print(f"\nInspect logs: {inspect_logs_directory}")
    print(f"summary: {summary_path}")
    if not all_tasks_succeeded:
        raise RuntimeError("the Inspect task failed; inspect the logs for details")
    return summary


def compare_summaries(summary_paths: Sequence[Path]) -> None:
    print(
        "| run | model | camera | valid / attempted | valid accuracy | end-to-end accuracy "
        "| macro end-to-end accuracy | predicted 0 | predicted 1 | predicted 2 |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text())
        overall = summary["overall"]
        predicted_counts = overall["predicted_value_counts"]
        attempted_agreement_fraction = overall.get("attempted_agreement_fraction")
        if attempted_agreement_fraction is None and overall["attempted_count"]:
            attempted_agreement_fraction = overall["agreement_count"] / overall["attempted_count"]
        print(
            f"| {summary['label']} | {summary['model']} | {summary['camera_view']} "
            f"| {overall['valid_count']} / {overall['attempted_count']} "
            f"| {_format_percentage(overall['agreement_fraction'])} "
            f"| {_format_percentage(attempted_agreement_fraction)} "
            f"| {_format_percentage(overall.get('macro_attempted_agreement_fraction'))} "
            f"| {predicted_counts.get('0', 0)} | {predicted_counts.get('1', 0)} "
            f"| {predicted_counts.get('2', 0)} |"
        )


def _positive_integer(raw_value: str) -> int:
    parsed_value = int(raw_value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value


def _nonnegative_integer(raw_value: str) -> int:
    parsed_value = int(raw_value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed_value


def _default_output_directory(model: str, camera_view: CameraView) -> Path:
    sanitized_model_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-") or "model"
    return DEFAULT_RUNS_DIRECTORY / f"{camera_view.value}-{sanitized_model_name}"


def _labels_by_source_from_arguments(
    arguments: argparse.Namespace,
) -> dict[Path, list[ProjectedHandFrameLabel]]:
    source_paths = select_episode_paths(
        _resolved_mcap_paths(arguments.inputs),
        episode_count=arguments.episode_count,
        sample_seed=arguments.sample_seed,
    )
    camera_view = CameraView(arguments.camera)
    candidate_labels_by_source = {
        source_path: load_projected_hand_labels(
            source_path,
            camera_view=camera_view,
            frame_stride=arguments.frame_stride,
            limit_per_episode=arguments.limit_per_episode,
            samples_per_episode=arguments.samples_per_episode,
            sample_seed=arguments.sample_seed,
        )
        for source_path in source_paths
    }
    return select_stratified_labels(
        candidate_labels_by_source,
        samples_per_hand_count=arguments.samples_per_hand_count,
        sample_seed=arguments.sample_seed,
    )


def _run_configuration_from_arguments(arguments: argparse.Namespace) -> EvaluationConfiguration:
    if not arguments.model:
        raise ValueError("--model or OPENAI_MODEL is required")
    if not arguments.base_url:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    camera_view = CameraView(arguments.camera)
    output_directory = arguments.output or _default_output_directory(arguments.model, camera_view)
    source_paths = select_episode_paths(
        _resolved_mcap_paths(arguments.inputs),
        episode_count=arguments.episode_count,
        sample_seed=arguments.sample_seed,
    )
    return EvaluationConfiguration(
        source_paths=source_paths,
        camera_view=camera_view,
        frame_stride=arguments.frame_stride,
        limit_per_episode=arguments.limit_per_episode,
        episode_count=arguments.episode_count,
        samples_per_episode=arguments.samples_per_episode,
        samples_per_hand_count=arguments.samples_per_hand_count,
        sample_seed=arguments.sample_seed,
        output_directory=output_directory,
        model=arguments.model,
        base_url=arguments.base_url,
        api_key_environment_variable=arguments.api_key_env,
        allow_missing_api_key=arguments.allow_missing_api_key,
        response_format=ResponseFormat(arguments.response_format),
        temperature=arguments.temperature,
        max_tokens=arguments.max_tokens,
        max_retries=arguments.max_retries,
        worker_count=arguments.workers,
        prompt=arguments.prompt.read_text(),
        prompt_path=arguments.prompt,
        label=arguments.label or arguments.model,
    )


def _add_input_and_projection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", type=Path, help="MCAP files or directories")
    parser.add_argument(
        "--camera",
        type=CameraView,
        choices=CameraView,
        default=CameraView.HEAD_LEFT,
    )
    parser.add_argument(
        "--frame-stride",
        type=_positive_integer,
        default=30,
        help="evaluate every Nth source frame; default 30 (about 1 fps)",
    )
    per_episode_selection = parser.add_mutually_exclusive_group()
    per_episode_selection.add_argument(
        "--limit-per-episode",
        type=_positive_integer,
        default=None,
        help="stop after this many selected frames in each episode",
    )
    parser.add_argument(
        "--episode-count",
        type=_positive_integer,
        default=None,
        help="select up to N input episodes deterministically",
    )
    per_episode_selection.add_argument(
        "--samples-per-episode",
        type=_positive_integer,
        default=None,
        help="randomly select up to N eligible frames in each episode deterministically",
    )
    parser.add_argument(
        "--samples-per-hand-count",
        type=_positive_integer,
        default=None,
        help="select up to N deterministic samples for each projected count (0, 1, and 2)",
    )
    parser.add_argument(
        "--sample-seed",
        type=_nonnegative_integer,
        default=42,
        help="seed for deterministic episode, frame, and class selection; default 42",
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    labels_parser = subparsers.add_parser(
        "labels", help="calculate projected hand-count labels without calling a model"
    )
    _add_input_and_projection_arguments(labels_parser)
    labels_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LABELS_DIRECTORY / "projected-hand-labels.json",
    )

    run_parser = subparsers.add_parser("run", help="run the image-only VLM evaluation")
    _add_input_and_projection_arguments(run_parser)
    run_parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    run_parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    run_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_parser.add_argument("--allow-missing-api-key", action="store_true")
    run_parser.add_argument(
        "--response-format",
        type=ResponseFormat,
        choices=ResponseFormat,
        default=ResponseFormat.JSON_SCHEMA,
    )
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument("--max-tokens", type=_positive_integer, default=512)
    run_parser.add_argument("--max-retries", type=_positive_integer, default=5)
    run_parser.add_argument("--workers", type=_positive_integer, default=8)
    run_parser.add_argument("--output", type=Path, default=None)
    run_parser.add_argument("--label", default=None, help="display label used by compare")
    run_parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)

    compare_parser = subparsers.add_parser("compare", help="compare completed run summaries")
    compare_parser.add_argument("summaries", nargs="+", type=Path)
    return parser


def main() -> None:
    parser = _argument_parser()
    arguments = parser.parse_args()
    try:
        match arguments.command:
            case "labels":
                labels_by_source = _labels_by_source_from_arguments(arguments)
                _print_reference_summary(labels_by_source)
                write_label_report(
                    labels_by_source,
                    camera_view=CameraView(arguments.camera),
                    frame_stride=arguments.frame_stride,
                    limit_per_episode=arguments.limit_per_episode,
                    episode_count=arguments.episode_count,
                    samples_per_episode=arguments.samples_per_episode,
                    samples_per_hand_count=arguments.samples_per_hand_count,
                    sample_seed=arguments.sample_seed,
                    output_path=arguments.output,
                )
                print(f"\nlabels: {arguments.output}")
            case "run":
                run_evaluation(_run_configuration_from_arguments(arguments))
            case "compare":
                compare_summaries(arguments.summaries)
            case unknown_command:
                raise AssertionError(f"unhandled command: {unknown_command}")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
