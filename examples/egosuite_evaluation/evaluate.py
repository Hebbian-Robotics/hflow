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
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
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
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory

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

SCHEMA_VERSION = 1
EXPECTED_HAND_JOINT_COUNT = 21
DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts") / "hand_count.txt"
DEFAULT_RUNS_DIRECTORY = Path("data/egosuite-evaluation/runs")
DEFAULT_LABELS_DIRECTORY = Path("data/egosuite-evaluation/labels")
INSPECT_LOGS_DIRECTORY_NAME = "logs"
RUN_METADATA_FILE_NAME = "run.json"
SUMMARY_FILE_NAME = "summary.json"
INSPECT_OPENAI_COMPATIBLE_SERVICE_NAME = "hflow-egosuite-evaluation"
HAND_COUNT_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"hand_count": {"type": "integer", "enum": [0, 1, 2]}},
    "required": ["hand_count"],
}


class CameraView(StrEnum):
    HEAD_LEFT = "head-left"
    HEAD_RIGHT = "head-right"


class ResponseFormat(StrEnum):
    JSON_SCHEMA = "json-schema"
    JSON_OBJECT = "json-object"
    TEXT = "text"


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


def _camera_topics(camera_view: CameraView) -> CameraTopics:
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


def _selected_frame_index(
    source_frame_index: int,
    frame_stride: int,
    limit_per_episode: int | None,
) -> bool:
    if source_frame_index % frame_stride != 0:
        return False
    selected_ordinal = source_frame_index // frame_stride
    return limit_per_episode is None or selected_ordinal < limit_per_episode


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


def _topic_message_counts(source_path: Path) -> dict[str, int]:
    with source_path.open("rb") as source_stream:
        summary = make_reader(source_stream).get_summary()
    if summary is None or summary.statistics is None:
        raise ValueError(f"MCAP summary statistics are required: {source_path}")
    counts_by_topic: dict[str, int] = {}
    for channel_id, channel in summary.channels.items():
        if channel.topic in counts_by_topic:
            raise ValueError(f"duplicate MCAP topic {channel.topic!r} in {source_path}")
        counts_by_topic[channel.topic] = int(
            summary.statistics.channel_message_counts.get(channel_id, 0)
        )
    return counts_by_topic


def _required_frame_count(
    source_path: Path,
    camera_view: CameraView,
    counts_by_topic: Mapping[str, int],
) -> int:
    topics = _camera_topics(camera_view)
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
) -> list[ProjectedHandFrameLabel]:
    """Read synchronized EgoSuite labels without decoding the video payloads."""

    if frame_stride <= 0:
        raise ValueError("frame stride must be positive")
    if limit_per_episode is not None and limit_per_episode <= 0:
        raise ValueError("limit per episode must be positive")
    camera_topics = _camera_topics(camera_view)
    expected_camera_frame = camera_view.value.replace("-", "_") + "_camera"
    decoded_topics = (
        "/pose/left_hand",
        "/pose/right_hand",
        "/annotation/bad_frame/pose/hand",
        camera_topics.intrinsic,
        camera_topics.extrinsic,
    )
    source_frame_count = _required_frame_count(
        source_path,
        camera_view,
        _topic_message_counts(source_path),
    )
    selected_frame_indices = [
        frame_index
        for frame_index in range(source_frame_count)
        if _selected_frame_index(frame_index, frame_stride, limit_per_episode)
    ]
    partial_geometry_by_frame = {
        frame_index: _PartialFrameGeometry() for frame_index in selected_frame_indices
    }
    message_index_by_topic: defaultdict[str, int] = defaultdict(int)
    with source_path.open("rb") as source_stream:
        reader = make_reader(
            source_stream,
            decoder_factories=[ProtobufDecoderFactory()],
        )
        for _schema, channel, _message, decoded_message in reader.iter_decoded_messages(
            topics=list(decoded_topics), log_time_order=True
        ):
            topic = channel.topic
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
                case intrinsic_topic if intrinsic_topic == camera_topics.intrinsic:
                    partial_geometry.calibration = _calibration_from_message(topic, decoded_message)
                case extrinsic_topic if extrinsic_topic == camera_topics.extrinsic:
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
    frame_stride: int,
    frame_cache_directory: Path,
) -> list[Path]:
    if not labels:
        return []
    selected_frame_indices = [label.frame_index for label in labels]
    if selected_frame_indices != sorted(set(selected_frame_indices)):
        raise ValueError(f"selected frame indices must be unique and ordered: {source_path}")
    if any(label.source_path != source_path for label in labels):
        raise ValueError(f"selected labels do not all belong to {source_path}")
    selection_hash = _sha256_text(
        ",".join(str(frame_index) for frame_index in selected_frame_indices)
    )[:12]
    episode_cache_directory = frame_cache_directory / _frame_cache_key(source_path)
    extraction_directory = episode_cache_directory / (
        f"{camera_view.value}-selection-{selection_hash}-count-{len(labels)}"
    )
    expected_frame_paths = [
        extraction_directory / f"frame_{output_index:06d}.jpg"
        for output_index in range(len(labels))
    ]
    if all(path.is_file() for path in expected_frame_paths):
        return expected_frame_paths
    if extraction_directory.exists():
        shutil.rmtree(extraction_directory)
    staging_directory = extraction_directory.with_name(f"{extraction_directory.name}.tmp")
    if staging_directory.exists():
        shutil.rmtree(staging_directory)
    staging_directory.mkdir(parents=True)
    camera_topic = _camera_topics(camera_view).video
    with hflow.Episode(source_path, workdir=episode_cache_directory) as episode:
        video_path = episode.video(camera_topic)
    ffmpeg_executable = shutil.which("ffmpeg")
    if ffmpeg_executable is None:
        raise RuntimeError("ffmpeg is required to extract EgoSuite evaluation images")
    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
    ]
    regular_frame_indices = [
        selected_ordinal * frame_stride for selected_ordinal in range(len(labels))
    ]
    if selected_frame_indices == regular_frame_indices:
        if frame_stride > 1:
            command += ["-vf", f"select=not(mod(n\\,{frame_stride}))"]
    else:
        selection_expression = "+".join(
            f"eq(n\\,{frame_index})" for frame_index in selected_frame_indices
        )
        command += ["-vf", f"select={selection_expression}"]
    command += [
        "-fps_mode",
        "vfr",
        "-frames:v",
        str(len(labels)),
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(staging_directory / "frame_%06d.jpg"),
    ]
    completed_process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    staged_frame_paths = sorted(staging_directory.glob("frame_*.jpg"))
    if completed_process.returncode != 0 or len(staged_frame_paths) != len(labels):
        stderr_tail = completed_process.stderr.strip().splitlines()[-8:]
        shutil.rmtree(staging_directory)
        raise RuntimeError(
            f"ffmpeg extracted {len(staged_frame_paths)} of {len(labels)} frames from "
            f"{source_path} (exit {completed_process.returncode}): {stderr_tail}"
        )
    staging_directory.replace(extraction_directory)
    return expected_frame_paths


def prepare_evaluation_frames(
    labels_by_source: Mapping[Path, Sequence[ProjectedHandFrameLabel]],
    *,
    camera_view: CameraView,
    frame_stride: int,
    frame_cache_directory: Path,
) -> list[PreparedEvaluationFrame]:
    prepared_frames: list[PreparedEvaluationFrame] = []
    for source_path, labels in labels_by_source.items():
        image_paths = _extract_sampled_frames(
            source_path,
            labels,
            camera_view=camera_view,
            frame_stride=frame_stride,
            frame_cache_directory=frame_cache_directory,
        )
        prepared_frames.extend(
            PreparedEvaluationFrame(label=label, image_path=image_path)
            for label, image_path in zip(labels, image_paths, strict=True)
        )
    return prepared_frames


def _image_data_url(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"evaluation frame is not JPEG: {image_path}")
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_image}"


def _strip_markdown_code_fence(response_text: str) -> str:
    stripped_response = response_text.strip()
    code_fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return code_fence_match.group(1).strip() if code_fence_match else stripped_response


def parse_hand_count_response(response_text: str) -> int:
    stripped_response = _strip_markdown_code_fence(response_text)
    try:
        parsed_response: object = json.loads(stripped_response)
    except json.JSONDecodeError:
        parsed_response = stripped_response
    if isinstance(parsed_response, dict):
        parsed_response = parsed_response.get("hand_count")
    if isinstance(parsed_response, bool):
        raise ValueError("hand count must be 0, 1, or 2")
    if isinstance(parsed_response, int):
        hand_count = parsed_response
    elif isinstance(parsed_response, str) and re.fullmatch(r"[012]", parsed_response.strip()):
        hand_count = int(parsed_response)
    else:
        raise ValueError("hand count must be 0, 1, or 2")
    if hand_count not in {0, 1, 2}:
        raise ValueError("hand count must be 0, 1, or 2")
    return hand_count


def _inspect_sample(frame: PreparedEvaluationFrame, prompt: str) -> Sample:
    label = frame.label
    return Sample(
        id=f"{label.source_episode}:{label.camera_view.value}:{label.frame_index:06d}",
        input=[
            ChatMessageUser(
                content=[
                    ContentText(text=prompt),
                    ContentImage(image=_image_data_url(frame.image_path)),
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
        "samples_per_hand_count": samples_per_hand_count,
        "sample_seed": sample_seed,
        "projection_rule": "hand is in frame when at least one labeled joint has positive depth and projects inside the image bounds",
        "summary": summarize_reference_labels(all_labels),
        "sources": [str(path) for path in labels_by_source],
        "frames": [_label_record(label) for label in all_labels],
    }
    _write_json_atomically(output_path, report)


def _run_metadata(configuration: EvaluationConfiguration) -> dict[str, object]:
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
    serialized_contract = json.dumps(result_contract, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": SCHEMA_VERSION,
        "label": configuration.label,
        "fingerprint": _sha256_text(serialized_contract),
        **result_contract,
        "api_key_environment_variable": configuration.api_key_environment_variable,
        "worker_count": configuration.worker_count,
        "max_retries": configuration.max_retries,
    }


def _prepare_output_directory(configuration: EvaluationConfiguration) -> dict[str, object]:
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = configuration.output_directory / RUN_METADATA_FILE_NAME
    current_metadata = _run_metadata(configuration)
    if metadata_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text())
        if existing_metadata.get("fingerprint") != current_metadata["fingerprint"]:
            raise ValueError(
                f"{metadata_path} describes a different experiment; choose another --output"
            )
        return existing_metadata
    _write_json_atomically(metadata_path, current_metadata)
    return current_metadata


def _inspect_model_name(requested_model: str) -> str:
    return f"openai-api/{INSPECT_OPENAI_COMPATIBLE_SERVICE_NAME}/{requested_model.lstrip('/')}"


def _sample_expected_value(sample: EvalSample) -> int:
    raw_target = sample.target[0] if isinstance(sample.target, list) else sample.target
    return parse_hand_count_response(raw_target)


def _sample_result(
    log: EvalLog,
    sample: EvalSample,
    configuration: EvaluationConfiguration,
) -> dict[str, object]:
    sample_metadata = sample.metadata or {}
    base_result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(sample_metadata["source_path"]),
        "source_episode": str(sample_metadata["source_episode"]),
        "camera_view": str(sample_metadata["camera_view"]),
        "frame_id": str(sample.id),
        "frame_index": int(sample_metadata["frame_index"]),
        "left_in_frame_joint_count": int(sample_metadata["left_in_frame_joint_count"]),
        "right_in_frame_joint_count": int(sample_metadata["right_in_frame_joint_count"]),
        "left_hand_issue_reasons": sample_metadata["left_hand_issue_reasons"],
        "right_hand_issue_reasons": sample_metadata["right_hand_issue_reasons"],
        "expected_value": _sample_expected_value(sample),
        "model": configuration.model,
        "raw_response": sample.output.completion,
    }
    if sample.output.model:
        base_result["response_model"] = sample.output.model
    if sample.output.time is not None:
        base_result["latency_seconds"] = sample.output.time
    if sample.output.usage is not None:
        base_result["usage"] = sample.output.usage.model_dump(exclude_none=True)
    if sample.error is not None or sample.output.error is not None:
        error_message = sample.error.message if sample.error is not None else sample.output.error
        return {**base_result, "status": "error", "error": str(error_message)[:1000]}
    try:
        predicted_value = parse_hand_count_response(sample.output.completion)
    except ValueError as error:
        return {**base_result, "status": "invalid", "error": str(error)}
    return {**base_result, "status": "ok", "predicted_value": predicted_value}


def _results_from_inspect_logs(
    log_headers: Sequence[EvalLog], configuration: EvaluationConfiguration
) -> tuple[list[dict[str, object]], list[str]]:
    results: list[dict[str, object]] = []
    log_locations: list[str] = []
    for log_header in log_headers:
        log_locations.append(log_header.location)
        completed_log = read_eval_log(log_header.location)
        results.extend(
            _sample_result(completed_log, sample, configuration)
            for sample in completed_log.samples or []
        )
    return results, log_locations


def _result_integer(result: Mapping[str, object], name: str) -> int:
    value = result[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"result field {name!r} must be an integer, got {value!r}")
    return value


def _evaluation_result_summary(results: Sequence[dict[str, object]]) -> dict[str, object]:
    valid_results = [result for result in results if result.get("status") == "ok"]
    expected_counts = Counter(_result_integer(result, "expected_value") for result in results)
    predicted_counts = Counter(
        _result_integer(result, "predicted_value") for result in valid_results
    )
    confusion_counts = Counter(
        (
            _result_integer(result, "expected_value"),
            _result_integer(result, "predicted_value"),
        )
        for result in valid_results
    )
    agreement_count = sum(
        result["expected_value"] == result["predicted_value"] for result in valid_results
    )
    latency_values = [
        float(latency)
        for result in results
        if isinstance((latency := result.get("latency_seconds")), int | float)
    ]
    usage_totals: Counter[str] = Counter()
    for result in results:
        usage = result.get("usage")
        if isinstance(usage, dict):
            usage_totals.update(
                {
                    str(name): float(value)
                    for name, value in usage.items()
                    if isinstance(value, int | float) and not isinstance(value, bool)
                }
            )
    return {
        "attempted_count": len(results),
        "valid_count": len(valid_results),
        "invalid_count": sum(result.get("status") == "invalid" for result in results),
        "error_count": sum(result.get("status") == "error" for result in results),
        "expected_value_counts": dict(sorted(expected_counts.items())),
        "predicted_value_counts": dict(sorted(predicted_counts.items())),
        "agreement_count": agreement_count,
        "agreement_fraction": agreement_count / len(valid_results) if valid_results else None,
        "attempted_agreement_fraction": agreement_count / len(results) if results else None,
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
    run_metadata: Mapping[str, object], results: Sequence[dict[str, object]]
) -> dict[str, object]:
    latest_results = {str(result["frame_id"]): result for result in results}
    deduplicated_results = list(latest_results.values())
    source_episodes = sorted({str(result["source_episode"]) for result in deduplicated_results})
    return {
        "schema_version": SCHEMA_VERSION,
        "label": run_metadata["label"],
        "fingerprint": run_metadata["fingerprint"],
        "model": run_metadata["model"],
        "camera_view": run_metadata["camera_view"],
        "frame_stride": run_metadata["frame_stride"],
        "samples_per_hand_count": run_metadata["samples_per_hand_count"],
        "sample_seed": run_metadata["sample_seed"],
        "overall": _evaluation_result_summary(deduplicated_results),
        "episodes": {
            source_episode: _evaluation_result_summary(
                [
                    result
                    for result in deduplicated_results
                    if result["source_episode"] == source_episode
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
        "predicted 2 | valid accuracy | end-to-end accuracy |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {overall['valid_count']} / {overall['attempted_count']} "
        f"| {expected_counts.get(0, 0)} | {expected_counts.get(1, 0)} "
        f"| {expected_counts.get(2, 0)} | {predicted_counts.get(0, 0)} "
        f"| {predicted_counts.get(1, 0)} | {predicted_counts.get(2, 0)} "
        f"| {_format_percentage(overall['agreement_fraction'])} "
        f"| {_format_percentage(overall['attempted_agreement_fraction'])} |"
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
        frame_stride=configuration.frame_stride,
        frame_cache_directory=configuration.output_directory / "frames",
    )
    inspect_logs_directory = configuration.output_directory / INSPECT_LOGS_DIRECTORY_NAME
    inspect_logs_directory.mkdir(parents=True, exist_ok=True)
    task = Task(
        name="egosuite_projected_hand_count",
        version=str(run_metadata["fingerprint"])[:16],
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
            "samples_per_hand_count": configuration.samples_per_hand_count,
            "sample_seed": configuration.sample_seed,
            "prompt_sha256": _sha256_text(configuration.prompt),
            "run_fingerprint": run_metadata["fingerprint"],
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
                "run_fingerprint": run_metadata["fingerprint"],
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
    results, log_locations = _results_from_inspect_logs(log_headers, configuration)
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
        "| predicted 0 | predicted 1 | predicted 2 |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
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
    source_paths = _resolved_mcap_paths(arguments.inputs)
    camera_view = CameraView(arguments.camera)
    candidate_labels_by_source = {
        source_path: load_projected_hand_labels(
            source_path,
            camera_view=camera_view,
            frame_stride=arguments.frame_stride,
            limit_per_episode=arguments.limit_per_episode,
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
    return EvaluationConfiguration(
        source_paths=_resolved_mcap_paths(arguments.inputs),
        camera_view=camera_view,
        frame_stride=arguments.frame_stride,
        limit_per_episode=arguments.limit_per_episode,
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
    parser.add_argument(
        "--limit-per-episode",
        type=_positive_integer,
        default=None,
        help="stop after this many selected frames in each episode",
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
        help="seed for deterministic class-stratified selection; default 42",
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
    match arguments.command:
        case "labels":
            try:
                labels_by_source = _labels_by_source_from_arguments(arguments)
                _print_reference_summary(labels_by_source)
                write_label_report(
                    labels_by_source,
                    camera_view=CameraView(arguments.camera),
                    frame_stride=arguments.frame_stride,
                    limit_per_episode=arguments.limit_per_episode,
                    samples_per_hand_count=arguments.samples_per_hand_count,
                    sample_seed=arguments.sample_seed,
                    output_path=arguments.output,
                )
                print(f"\nlabels: {arguments.output}")
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                parser.error(str(error))
        case "run":
            try:
                run_evaluation(_run_configuration_from_arguments(arguments))
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                parser.error(str(error))
        case "compare":
            compare_summaries(arguments.summaries)
        case unknown_command:
            raise AssertionError(f"unhandled command: {unknown_command}")


if __name__ == "__main__":
    main()
