"""Hand detection with Google's MediaPipe Hand Landmarker.

This is **not** one of hflow's deterministic built-in checks. It runs a model,
and the model is named at every call site on purpose: what it records is what
MediaPipe *detected*, which is not the same claim as what was in the frame.
A gloved hand is present, is visible, and is not detected. That gap is the
whole reason this module is separate from :mod:`hflow.checks`, and the
limitations a user is accepting are documented in
``docs/how-to/measure-hand-presence-with-mediapipe.md``. Read them before
gating anything on these numbers -- which is also why nothing here ships a
recommended gate.

Mechanism: frames are sampled from a camera at a declared rate, decoded
straight to RGB by the pinned ffmpeg, and passed to the Hand Landmarker in
VIDEO mode. Per frame, how many hands it found and which; per camera, the
share of sampled frames those add up to, plus the spans where it found none.
Landmark coordinates are deliberately not recorded: 21 image and 21 world
landmarks per hand is a nested shape the flat catalog cannot hold, and the
counts are what the shares need.

Counts are over FRAMES, never an average of per-episode shares, so a
measurement here is comparable to a frame-labelled reference set.

Requires the ``mediapipe`` extra (``pip install 'hflow[mediapipe]'``) and a
model asset, which is downloaded once and digest-verified like the pinned
ffmpeg build.
"""

import importlib
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np

from hflow._pinned_asset import (
    PinnedAssetError,
    download_verified_asset,
    sha256_hex_of_file,
    user_cache_dir,
)
from hflow.episode import Episode
from hflow.ffmpeg import rgb_frames
from hflow.steps import CheckResult, Interval, MeasurementValue
from hflow.video import source_log_times_for_sampled_frames

logger = logging.getLogger(__name__)

# COORDINATE: this digest is the model's identity, and it is also part of the
# check's identity -- hand_landmarker_model_path reads it, so the version walk
# folds it into every measurement row. Bumping the pin is therefore a
# deliberate re-versioning of every hand measurement, never a silent one.
PINNED_HAND_MODEL_FILENAME = "hand_landmarker_float16_v1.task"
PINNED_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
PINNED_HAND_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

HAND_LANDMARKER_MODEL_ENV_VAR = "HFLOW_HAND_LANDMARKER_MODEL"

# The model's own cap. Two is right for one operator and wrong for a frame with
# a bystander in it: the extra hands do not raise the count past two, they
# compete for the two slots. Not a parameter, because raising it would change
# what every recorded share means without changing its name.
MAXIMUM_HANDS = 2
LANDMARKS_PER_HAND = 21

# One frame a second: enough to characterize an episode, cheap at ~11 ms of
# inference per frame, and the rate a frames-only VLM reads the same footage
# at -- which is what makes the two answers comparable.
DEFAULT_SAMPLE_FPS = 1.0
DEFAULT_MINIMUM_CONFIDENCE = 0.5

# MediaPipe's video clock is integer milliseconds, so two frames inside one
# millisecond cannot carry distinct timestamps.
_MAXIMUM_SAMPLE_FPS = 1000.0
_MILLISECONDS_PER_SECOND = 1000


class MediapipeExtraNotInstalledError(RuntimeError):
    pass


class HandModelNotAvailableError(RuntimeError):
    pass


class HandLandmarkerError(RuntimeError):
    """MediaPipe returned a structurally unexpected result."""


class Handedness(StrEnum):
    """MediaPipe's handedness labels, plus a preserved unknown case.

    ``UNKNOWN`` is kept rather than guessed: a detection whose label did not
    parse is still a detected hand, and silently calling it left or right
    would put a fabricated fact in the catalog.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FrameHandCounts:
    """What the model found in one frame.

    ``left``/``right`` are recorded beside the total because "two hands" and
    "one hand seen twice" are different facts that a count alone cannot tell
    apart, and because the handedness label is the only thing that
    distinguishes them.
    """

    hand_count: int
    left_hand_count: int
    right_hand_count: int


@dataclass(frozen=True)
class HandDetectionSummary:
    """One camera's frames, reduced to the counts a share needs."""

    frame_count: int
    frames_with_any_hand: int
    frames_with_two_hands: int
    frames_with_a_left_hand: int
    frames_with_a_right_hand: int

    @property
    def any_hand_share(self) -> float | None:
        """Share of sampled frames with at least one detection.

        ``None`` when nothing was measured, which is not a share of zero: an
        episode whose camera could not be sampled did not have hands absent
        from it, and recording 0.0 would put that claim in the catalog.
        """
        return self._share(self.frames_with_any_hand)

    @property
    def two_hand_share(self) -> float | None:
        return self._share(self.frames_with_two_hands)

    def _share(self, part: int) -> float | None:
        return part / self.frame_count if self.frame_count > 0 else None


def summarize_hand_detections(detections: Sequence[FrameHandCounts]) -> HandDetectionSummary:
    """Reduce per-frame counts to per-camera counts.

    Pure: it takes rows that have already been measured and returns data, so
    the arithmetic is testable without the model or the extra installed.
    """
    return HandDetectionSummary(
        frame_count=len(detections),
        frames_with_any_hand=sum(1 for frame in detections if frame.hand_count >= 1),
        frames_with_two_hands=sum(1 for frame in detections if frame.hand_count >= MAXIMUM_HANDS),
        frames_with_a_left_hand=sum(1 for frame in detections if frame.left_hand_count >= 1),
        frames_with_a_right_hand=sum(1 for frame in detections if frame.right_hand_count >= 1),
    )


def _no_detection_intervals(
    log_times_ns: Sequence[int],
    detections: Sequence[FrameHandCounts],
    label: str,
) -> list[Interval]:
    """Spans of consecutive frames in which the model found no hand.

    A run ends at the log time of the next frame that DID show a hand, so the
    interval covers the footage the absence actually spans. A run that reaches
    the end of the episode can only be closed at the last sampled frame, so a
    trailing absence is under-reported by up to one sample period -- stated
    rather than papered over, because the alternative is inventing an end
    time from a rate.
    """
    intervals: list[Interval] = []
    run_start_index: int | None = None
    for frame_index, frame in enumerate(detections):
        if frame.hand_count == 0 and run_start_index is None:
            run_start_index = frame_index
        elif frame.hand_count > 0 and run_start_index is not None:
            intervals.append(
                Interval(
                    start_ns=log_times_ns[run_start_index],
                    end_ns=log_times_ns[frame_index],
                    label=label,
                )
            )
            run_start_index = None
    if run_start_index is not None and run_start_index < len(log_times_ns) - 1:
        intervals.append(
            Interval(
                start_ns=log_times_ns[run_start_index],
                end_ns=log_times_ns[-1],
                label=label,
            )
        )
    return intervals


def _import_mediapipe() -> ModuleType:
    """MediaPipe, imported on use so the core install never needs it.

    Typed as the module rather than the package so a missing extra is one
    clear error at the call site instead of an ImportError at ``import
    hflow``. Nothing in this module may reach the import at REGISTRATION time
    either: a check whose registration touches its instrument cannot be
    registered without the instrument installed.
    """
    # Imported by name rather than with an import statement, which
    # hflow.motion does not need to do for OpenCV: that extra is mirrored into
    # the dev group, so the typechecker can resolve it, while this one is
    # deliberately absent (~430 MB and a second OpenCV distribution). A plain
    # import here would be unresolvable in a default environment and resolvable
    # in an opted-in one, so any suppression for it would itself be flagged as
    # unused half the time.
    try:
        return importlib.import_module("mediapipe")
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by hand
        raise MediapipeExtraNotInstalledError(
            "hand detection needs MediaPipe, which ships in the optional "
            "'mediapipe' extra: pip install 'hflow[mediapipe]' (or uv add "
            "'hflow[mediapipe]'). It is a model, not a signal statistic, so it "
            "is deliberately not part of the core install."
        ) from error


@lru_cache(maxsize=1)
def hand_landmarker_model_path() -> Path:
    """The Hand Landmarker weights: an override, else the pinned download.

    Resolution mirrors the pinned ffmpeg build (:mod:`hflow.ffmpeg._binary`),
    for the same reason -- the asset is a measuring instrument, so hflow never
    uses whatever happens to be on the machine:

    1. ``HFLOW_HAND_LANDMARKER_MODEL``, for a user managing their own asset.
       Verified against the pin only if it matches by name; a deliberately
       different model is the user's to own, and recorded as its own digest.
    2. The pinned float16 asset, downloaded once into
       ``<user cache>/hflow/models`` and sha256-verified.

    MediaPipe 1.0.0 bundles no weights, and the legacy API that used to is
    gone, so there is no third option.
    """
    override = os.environ.get(HAND_LANDMARKER_MODEL_ENV_VAR)
    if override:
        override_path = Path(override)
        if not override_path.is_file():
            raise HandModelNotAvailableError(
                f"{HAND_LANDMARKER_MODEL_ENV_VAR}={override} does not exist"
            )
        return override_path

    model_path = user_cache_dir("models") / PINNED_HAND_MODEL_FILENAME
    if model_path.is_file():
        return model_path
    logger.info("downloading the pinned MediaPipe hand landmarker to %s", model_path)
    try:
        return download_verified_asset(PINNED_HAND_MODEL_URL, PINNED_HAND_MODEL_SHA256, model_path)
    except PinnedAssetError as error:
        raise HandModelNotAvailableError(
            f"{error} Retry with network access, or set "
            f"{HAND_LANDMARKER_MODEL_ENV_VAR}=/path/to/hand_landmarker.task."
        ) from error


@lru_cache(maxsize=1)
def _resolved_hand_model_digest() -> str:
    """The digest of the model actually in use, hashed once per process.

    Not assumed from the pin: an override may point at a different asset, and
    the recorded digest has to describe what ran rather than what was
    intended.
    """
    return sha256_hex_of_file(hand_landmarker_model_path())


def _handedness_of(handedness_categories: Sequence[object]) -> Handedness:
    """The highest-scoring label MediaPipe assigned to one detected hand.

    The score is a handedness confidence, NOT a detection confidence, however
    much it looks like one: it says how sure the model is that the hand it
    already found is a left hand. It is used only to pick between the
    categories here, and never recorded as though it graded the detection.
    """
    best_category = max(
        handedness_categories,
        key=lambda category: getattr(category, "score", None) or float("-inf"),
        default=None,
    )
    raw_label = getattr(best_category, "category_name", None)
    if not isinstance(raw_label, str):
        return Handedness.UNKNOWN
    match raw_label.casefold():
        case "left":
            return Handedness.LEFT
        case "right":
            return Handedness.RIGHT
        case _:
            return Handedness.UNKNOWN


def detect_hands_in_frames(
    frames: Iterable[np.ndarray],
    *,
    sample_fps: float,
    minimum_hand_detection_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
    minimum_hand_presence_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
    minimum_tracking_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
) -> list[FrameHandCounts]:
    """Run the Hand Landmarker over RGB frames, in order.

    One landmarker per call, which makes the call boundary a real tracker
    reset boundary: VIDEO mode carries tracking state between frames, and
    state leaking across episodes would make an episode's numbers depend on
    which episode preceded it.
    """
    mediapipe = _import_mediapipe()
    options = mediapipe.tasks.vision.HandLandmarkerOptions(
        base_options=mediapipe.tasks.BaseOptions(
            model_asset_path=str(hand_landmarker_model_path())
        ),
        running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
        num_hands=MAXIMUM_HANDS,
        min_hand_detection_confidence=minimum_hand_detection_confidence,
        min_hand_presence_confidence=minimum_hand_presence_confidence,
        min_tracking_confidence=minimum_tracking_confidence,
    )
    detections: list[FrameHandCounts] = []
    with mediapipe.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for frame_index, frame in enumerate(frames):
            # A copy because MediaPipe takes ownership of the buffer, and the
            # decoder's frames are read-only views over the pipe's bytes.
            image = mediapipe.Image(
                image_format=mediapipe.ImageFormat.SRGB, data=np.array(frame, dtype=np.uint8)
            )
            timestamp_ms = int(frame_index * _MILLISECONDS_PER_SECOND / sample_fps)
            detections.append(_frame_hand_counts(landmarker.detect_for_video(image, timestamp_ms)))
    return detections


def _frame_hand_counts(result: object) -> FrameHandCounts:
    """One MediaPipe result, reduced to counts.

    A structural mismatch raises rather than being coerced: it means this
    build of MediaPipe returns a shape this code was not written against, and
    a quietly wrong count is worse than a failed run.
    """
    hand_landmarks = getattr(result, "hand_landmarks", None) or []
    handedness_per_hand = getattr(result, "handedness", None) or []
    if len(handedness_per_hand) != len(hand_landmarks):
        raise HandLandmarkerError(
            f"MediaPipe returned {len(hand_landmarks)} landmark sets and "
            f"{len(handedness_per_hand)} handedness sets for one frame"
        )
    for landmarks in hand_landmarks:
        if len(landmarks) != LANDMARKS_PER_HAND:
            raise HandLandmarkerError(
                f"MediaPipe returned {len(landmarks)} landmarks for one hand, "
                f"expected {LANDMARKS_PER_HAND}"
            )
    labels = [_handedness_of(categories) for categories in handedness_per_hand]
    return FrameHandCounts(
        hand_count=len(hand_landmarks),
        left_hand_count=labels.count(Handedness.LEFT),
        right_hand_count=labels.count(Handedness.RIGHT),
    )


def mediapipe_hand_detection(
    episode: Episode,
    *,
    cameras: Sequence[str] | None = None,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    inference_long_edge_pixels: int | None = None,
    minimum_hand_detection_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
    minimum_hand_presence_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
    minimum_tracking_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
) -> CheckResult:
    """How often MediaPipe detected hands in each camera's footage.

    Requires the ``mediapipe`` extra and downloads a pinned 7.8 MB model on
    first use. Evidence only, and it ships no recommended gate: a low share
    means the model saw nothing, which on gloved or dim footage says nothing
    about whether hands were there. Read
    ``docs/how-to/measure-hand-presence-with-mediapipe.md`` before treating
    these numbers as a quality signal.

    ``{topic}/hand_detected_frame_share`` is over
    ``{topic}/hand_detection_frame_count`` -- the frames actually sampled, not
    the frames the camera recorded. A camera that could not be sampled records
    the count alone and no share.

    ``inference_long_edge_pixels`` resizes each frame before inference. It
    changes which hands are found at all on small footage, so it is part of
    the question rather than a performance knob, and it is folded into this
    check's version like any other captured setting.
    """
    if not 0.0 < sample_fps <= _MAXIMUM_SAMPLE_FPS:
        raise ValueError(
            f"sample_fps must be in (0, {_MAXIMUM_SAMPLE_FPS:g}], got {sample_fps}: "
            "MediaPipe timestamps video frames in whole milliseconds"
        )

    selected_cameras = list(cameras) if cameras is not None else episode.cameras
    measurements: dict[str, MeasurementValue] = {}
    intervals: list[Interval] = []
    for topic in selected_cameras:
        stamps_ns = episode.channel(topic).timestamps
        if len(stamps_ns) < 2:
            measurements[f"{topic}/hand_detection_frame_count"] = 0
            continue
        # The stream's real rate, not a nominal one: it decides which recorded
        # message each sampled frame is attributed to.
        median_interval_s = float(np.median(np.diff(stamps_ns)) / 1e9)
        if median_interval_s <= 0:
            measurements[f"{topic}/hand_detection_frame_count"] = 0
            continue

        with rgb_frames(
            episode.video(topic),
            fps=sample_fps,
            long_edge_pixels=inference_long_edge_pixels,
        ) as frames:
            detections = detect_hands_in_frames(
                frames,
                sample_fps=sample_fps,
                minimum_hand_detection_confidence=minimum_hand_detection_confidence,
                minimum_hand_presence_confidence=minimum_hand_presence_confidence,
                minimum_tracking_confidence=minimum_tracking_confidence,
            )

        summary = summarize_hand_detections(detections)
        measurements[f"{topic}/hand_detection_frame_count"] = summary.frame_count
        measurements[f"{topic}/hand_sample_fps"] = sample_fps
        if inference_long_edge_pixels is not None:
            measurements[f"{topic}/hand_inference_long_edge_pixels"] = inference_long_edge_pixels
        # No share when nothing was sampled, rather than a share of zero. The
        # optional return type is what makes that a type error to get wrong
        # here instead of a wrong number in the catalog.
        any_hand_share = summary.any_hand_share
        two_hand_share = summary.two_hand_share
        if any_hand_share is None or two_hand_share is None:
            continue
        measurements.update(
            {
                f"{topic}/hand_detected_frame_share": any_hand_share,
                f"{topic}/two_hand_detected_frame_share": two_hand_share,
                f"{topic}/left_hand_detected_frame_count": summary.frames_with_a_left_hand,
                f"{topic}/right_hand_detected_frame_count": summary.frames_with_a_right_hand,
                # The instrument, recorded as evidence: the model digest is in
                # this check's version, but the MediaPipe build is not (a
                # library version is a poor proxy for "does this compute
                # differently"), so a run's own row is where it is auditable.
                f"{topic}/mediapipe_version": _import_mediapipe().__version__,
                f"{topic}/hand_model_digest": _resolved_hand_model_digest(),
            }
        )
        log_times_ns = source_log_times_for_sampled_frames(
            stamps_ns.tolist(),
            source_fps=1.0 / median_interval_s,
            sample_fps=sample_fps,
            start_s=0.0,
            frame_count=summary.frame_count,
        )
        intervals.extend(
            _no_detection_intervals(log_times_ns, detections, f"no_hand_detected:{topic}")
        )
    return CheckResult(measurements=measurements, intervals=intervals)
