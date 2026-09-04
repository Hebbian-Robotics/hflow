"""Run EgoSuite projected hand visibility as an HFlow episode check.

The companion ``evaluate.py`` adapter uses Inspect AI to compare models over a
dataset slice. This pipeline applies the same prompt, response schema, parser,
geometric reference, and exact-frame extraction to one HFlow episode and
records aggregate and per-frame evidence in HFlow's catalog boundary.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import os
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, assert_never

import hflow
from examples.egosuite_evaluation.evaluate import (
    DEFAULT_HFLOW_DATA_ROOT,
    CameraView,
    ProjectedHandFrameLabel,
    ProjectedHandLabelReport,
    camera_topics,
    load_projected_hand_label_report,
    load_projected_hand_labels,
)
from examples.egosuite_evaluation.judgment import (
    DEFAULT_PROMPT_PATH,
    HandCountOutcome,
    ParsedHandCountOutcome,
    ResponseFormat,
    UnparsedHandCountOutcome,
    evaluate_image_with_model,
    image_file_data_url,
)

DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "model-not-configured"
MEASUREMENT_PREFIX = "egosuite/hand_count"


class PipelineConfiguration(NamedTuple):
    endpoint: str
    model: str
    api_key_environment_variable: str
    allow_missing_api_key: bool
    response_format: ResponseFormat
    temperature: float | None
    max_tokens: int
    max_retries: int
    worker_count: int
    camera_view: CameraView
    frame_stride: int
    limit_per_episode: int | None
    sample_seed: int
    prompt: str
    label_manifest_path: Path | None


def _optional_float_environment_variable(name: str) -> float | None:
    raw_value = os.environ.get(name)
    return float(raw_value) if raw_value is not None else None


def _boolean_environment_variable(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _optional_positive_int_environment_variable(name: str, default: str) -> int | None:
    raw_value = os.environ.get(name, default).strip().lower()
    if raw_value in {"", "all", "none"}:
        return None
    parsed_value = int(raw_value)
    if parsed_value <= 0:
        raise ValueError(f"{name} must be positive, 'all', or 'none'")
    return parsed_value


def _pipeline_configuration() -> PipelineConfiguration:
    prompt_path = Path(os.environ.get("EGOSUITE_HAND_COUNT_PROMPT", str(DEFAULT_PROMPT_PATH)))
    raw_label_manifest_path = os.environ.get("EGOSUITE_LABEL_MANIFEST")
    return PipelineConfiguration(
        endpoint=os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_COMPATIBLE_BASE_URL),
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL_NAME),
        api_key_environment_variable=os.environ.get("EGOSUITE_API_KEY_ENV", "OPENAI_API_KEY"),
        allow_missing_api_key=_boolean_environment_variable("EGOSUITE_ALLOW_MISSING_API_KEY"),
        response_format=ResponseFormat(
            os.environ.get("EGOSUITE_RESPONSE_FORMAT", ResponseFormat.JSON_SCHEMA.value)
        ),
        temperature=_optional_float_environment_variable("EGOSUITE_TEMPERATURE"),
        max_tokens=int(os.environ.get("EGOSUITE_MAX_TOKENS", "512")),
        max_retries=int(os.environ.get("EGOSUITE_MAX_RETRIES", "5")),
        worker_count=int(os.environ.get("EGOSUITE_WORKERS", "4")),
        camera_view=CameraView(os.environ.get("EGOSUITE_CAMERA", CameraView.HEAD_LEFT.value)),
        frame_stride=int(os.environ.get("EGOSUITE_FRAME_STRIDE", "30")),
        limit_per_episode=_optional_positive_int_environment_variable(
            "EGOSUITE_LIMIT_PER_EPISODE", "10"
        ),
        sample_seed=int(os.environ.get("EGOSUITE_SAMPLE_SEED", "42")),
        prompt=prompt_path.read_text(),
        label_manifest_path=(
            Path(raw_label_manifest_path).resolve() if raw_label_manifest_path else None
        ),
    )


pipeline_configuration = _pipeline_configuration()
manifest_label_report = (
    load_projected_hand_label_report(pipeline_configuration.label_manifest_path)
    if pipeline_configuration.label_manifest_path is not None
    else None
)
app = hflow.App(
    "egosuite-projected-hand-visibility-example",
    data_root=os.environ.get("HFLOW_DATA_ROOT", str(DEFAULT_HFLOW_DATA_ROOT)),
    default_checks=(),
)


def _check_version() -> hflow.StepVersion:
    version_contract = {
        "prompt": pipeline_configuration.prompt,
        "endpoint": pipeline_configuration.endpoint,
        "model": pipeline_configuration.model,
        "response_format": pipeline_configuration.response_format.value,
        "temperature": pipeline_configuration.temperature,
        "max_tokens": pipeline_configuration.max_tokens,
        "camera_view": pipeline_configuration.camera_view.value,
        "frame_stride": pipeline_configuration.frame_stride,
        "limit_per_episode": pipeline_configuration.limit_per_episode,
        "sample_seed": pipeline_configuration.sample_seed,
        "label_manifest_sha256": (
            hashlib.sha256(pipeline_configuration.label_manifest_path.read_bytes()).hexdigest()
            if pipeline_configuration.label_manifest_path is not None
            else None
        ),
    }
    return hflow.step_version_from_contract(
        "egosuite-projected-hand-visibility-v2",
        version_contract,
    )


def labels_for_pipeline_episode(
    episode_path: Path,
    episode_metadata: Mapping[str, object],
    label_report: ProjectedHandLabelReport,
) -> list[ProjectedHandFrameLabel]:
    """Select one canonical episode's declared frames from a label manifest."""

    source_uri = episode_metadata.get("source_uri")
    if not isinstance(source_uri, str) or not source_uri:
        raise ValueError(f"canonical episode {episode_path} has no source_uri provenance")
    canonical_suffix = ".canonical.mcap"
    source_episode = (
        episode_path.name[: -len(canonical_suffix)]
        if episode_path.name.endswith(canonical_suffix)
        else episode_path.stem
    )
    source_identity = source_episode if label_report.uses_legacy_episode_names else source_uri
    try:
        return label_report.labels_by_source_identity[source_identity]
    except KeyError:
        raise ValueError(
            f"label manifest has no frames for source identity {source_identity!r}"
        ) from None


_client_by_thread = threading.local()


def _client_for_current_thread() -> Any:
    if pipeline_configuration.model == DEFAULT_MODEL_NAME:
        raise ValueError("OPENAI_MODEL is required")
    api_key = os.environ.get(pipeline_configuration.api_key_environment_variable)
    if not api_key and not pipeline_configuration.allow_missing_api_key:
        raise ValueError(
            f"{pipeline_configuration.api_key_environment_variable} is not set; set "
            "EGOSUITE_ALLOW_MISSING_API_KEY=1 only for an unauthenticated endpoint"
        )
    client_cache_key = (pipeline_configuration.endpoint, api_key)
    if getattr(_client_by_thread, "cache_key", None) != client_cache_key:
        openai_module = importlib.import_module("openai")
        _client_by_thread.client = openai_module.OpenAI(
            api_key=api_key or "not-needed",
            base_url=pipeline_configuration.endpoint,
            max_retries=pipeline_configuration.max_retries,
        )
        _client_by_thread.cache_key = client_cache_key
    return _client_by_thread.client


def _evaluate_frame(extracted_frame: hflow.ExtractedFrame) -> HandCountOutcome:
    return evaluate_image_with_model(
        client=_client_for_current_thread(),
        model=pipeline_configuration.model,
        prompt=pipeline_configuration.prompt,
        image_data_url=image_file_data_url(extracted_frame.path),
        response_format=pipeline_configuration.response_format,
        temperature=pipeline_configuration.temperature,
        max_tokens=pipeline_configuration.max_tokens,
    )


def hand_visibility_check_result(
    labels: Sequence[ProjectedHandFrameLabel],
    extracted_frames: Sequence[hflow.ExtractedFrame],
    judgments: Sequence[HandCountOutcome],
    *,
    requested_model: str,
) -> hflow.CheckResult:
    """Record aggregate and timestamped per-frame HFlow evidence."""

    if not (len(labels) == len(extracted_frames) == len(judgments)):
        raise ValueError("labels, extracted frames, and judgments must have equal lengths")
    reference_counts = Counter(label.expected_hand_count for label in labels)
    predicted_counts: Counter[int] = Counter()
    valid_count = 0
    agreement_count = 0
    for label, judgment in zip(labels, judgments, strict=True):
        match judgment:
            case ParsedHandCountOutcome(predicted_hand_count=predicted_hand_count):
                predicted_counts[predicted_hand_count] += 1
                valid_count += 1
                agreement_count += predicted_hand_count == label.expected_hand_count
            case UnparsedHandCountOutcome():
                pass
            case unexpected_judgment:
                assert_never(unexpected_judgment)
    attempted_count = len(labels)
    measurements: dict[str, hflow.MeasurementValue] = {
        f"{MEASUREMENT_PREFIX}/attempted_count": attempted_count,
        f"{MEASUREMENT_PREFIX}/valid_count": valid_count,
        f"{MEASUREMENT_PREFIX}/invalid_count": attempted_count - valid_count,
        f"{MEASUREMENT_PREFIX}/agreement_count": agreement_count,
        f"{MEASUREMENT_PREFIX}/attempted_agreement_fraction": (
            agreement_count / attempted_count if attempted_count else 0.0
        ),
        f"{MEASUREMENT_PREFIX}/requested_model": requested_model,
        f"{MEASUREMENT_PREFIX}/pose_issue_frame_count": sum(
            bool(label.left_hand_issue_reasons or label.right_hand_issue_reasons)
            for label in labels
        ),
    }
    if valid_count:
        measurements[f"{MEASUREMENT_PREFIX}/valid_agreement_fraction"] = (
            agreement_count / valid_count
        )
    for hand_count in range(3):
        measurements[f"{MEASUREMENT_PREFIX}/reference/{hand_count}"] = reference_counts[hand_count]
        measurements[f"{MEASUREMENT_PREFIX}/predicted/{hand_count}"] = predicted_counts[hand_count]

    response_models = sorted(
        {
            judgment.response_metadata.response_model
            for judgment in judgments
            if judgment.response_metadata.response_model is not None
        }
    )
    if response_models:
        measurements[f"{MEASUREMENT_PREFIX}/response_models"] = ",".join(response_models)
    usage_totals: dict[str, float] = {}
    for judgment in judgments:
        for usage_name, usage_value in judgment.response_metadata.usage.items():
            if isinstance(usage_value, int | float) and not isinstance(usage_value, bool):
                usage_totals[usage_name] = usage_totals.get(usage_name, 0.0) + usage_value
    for usage_name, usage_total in usage_totals.items():
        measurements[f"{MEASUREMENT_PREFIX}/usage/{usage_name}"] = usage_total

    observations: list[hflow.Observation] = []
    intervals: list[hflow.Interval] = []
    for label, extracted_frame, judgment in zip(labels, extracted_frames, judgments, strict=True):
        observation_values: dict[str, hflow.MeasurementValue] = {
            "frame_index": label.frame_index,
            "reference_hand_count": label.expected_hand_count,
            "raw_response": judgment.raw_response,
            "requested_model": requested_model,
            "left_in_frame_joint_count": label.left_in_frame_joint_count,
            "right_in_frame_joint_count": label.right_in_frame_joint_count,
            "pose_issue": bool(label.left_hand_issue_reasons or label.right_hand_issue_reasons),
        }
        if judgment.response_metadata.response_model is not None:
            observation_values["response_model"] = judgment.response_metadata.response_model
        for usage_name, usage_value in judgment.response_metadata.usage.items():
            if isinstance(usage_value, int | float | str | bool):
                observation_values[f"usage/{usage_name}"] = usage_value
        match judgment:
            case ParsedHandCountOutcome(predicted_hand_count=predicted_hand_count):
                observation_values["predicted_hand_count"] = predicted_hand_count
                observation_values["valid"] = True
                observation_values["agreement"] = predicted_hand_count == label.expected_hand_count
                if predicted_hand_count != label.expected_hand_count:
                    intervals.append(
                        hflow.Interval(
                            start_ns=extracted_frame.log_time_ns,
                            end_ns=extracted_frame.log_time_ns,
                            label=(
                                f"{MEASUREMENT_PREFIX}/reference_{label.expected_hand_count}_"
                                f"predicted_{predicted_hand_count}"
                            ),
                        )
                    )
            case UnparsedHandCountOutcome(parse_error=parse_error):
                observation_values["valid"] = False
                observation_values["agreement"] = False
                observation_values["parse_error"] = parse_error
                intervals.append(
                    hflow.Interval(
                        start_ns=extracted_frame.log_time_ns,
                        end_ns=extracted_frame.log_time_ns,
                        label=f"{MEASUREMENT_PREFIX}/unparsed",
                    )
                )
            case unexpected_judgment:
                assert_never(unexpected_judgment)
        observations.append(
            hflow.Observation(
                observation_id=f"frame:{label.frame_index}",
                timestamp_ns=extracted_frame.log_time_ns,
                values=observation_values,
            )
        )

    tags = [f"{MEASUREMENT_PREFIX}/has_unparsed_output"] if valid_count < attempted_count else []
    return hflow.CheckResult(
        measurements=measurements,
        observations=observations,
        intervals=intervals,
        tags=tags,
    )


@app.check(
    name="egosuite_projected_hand_visibility",
    requires=("vision-model",),
    version=_check_version(),
)
def egosuite_projected_hand_visibility(episode: hflow.Episode) -> hflow.CheckResult:
    """Compare image-only VLM hand counts with projected EgoSuite joints."""

    source_uri = episode.metadata.get("source_uri")
    if not isinstance(source_uri, str) or not source_uri:
        raise ValueError(f"canonical episode {episode.path} has no source_uri provenance")
    if manifest_label_report is not None:
        labels = labels_for_pipeline_episode(episode.path, episode.metadata, manifest_label_report)
        mismatched_camera_views = sorted(
            {
                label.camera_view.value
                for label in labels
                if label.camera_view is not pipeline_configuration.camera_view
            }
        )
        if mismatched_camera_views:
            raise ValueError(
                f"label manifest camera views {mismatched_camera_views} do not match configured "
                f"camera {pipeline_configuration.camera_view.value!r}"
            )
    else:
        labels = load_projected_hand_labels(
            episode.path,
            source_uri=source_uri,
            camera_view=pipeline_configuration.camera_view,
            frame_stride=pipeline_configuration.frame_stride,
            limit_per_episode=pipeline_configuration.limit_per_episode,
            sample_seed=pipeline_configuration.sample_seed,
        )
    extracted_frames = episode.frames_at_indices(
        camera_topics(pipeline_configuration.camera_view).video,
        frame_indices=[label.frame_index for label in labels],
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=pipeline_configuration.worker_count
    ) as thread_pool:
        judgments = list(thread_pool.map(_evaluate_frame, extracted_frames))
    return hand_visibility_check_result(
        labels,
        extracted_frames,
        judgments,
        requested_model=pipeline_configuration.model,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="annotated EgoSuite MCAP episode")
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    report = app.test(arguments.episode)
    if report.has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
