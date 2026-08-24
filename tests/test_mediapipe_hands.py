"""Tests for the MediaPipe hand-detection check.

The extra is deliberately absent from the development environment (it is a
model; see pyproject.toml), so almost everything here tests the layer that does
not need it: the frame arithmetic, the interval construction, the model-asset
resolution, and the promise that registering the check does not require the
model to be installed. The end-to-end case is opt-in via
``HFLOW_MEDIAPIPE_TESTS=1``.

That split is the same one the original implementation made -- its own tests
covered selection and resize arithmetic, and the detector was exercised by a
manual probe -- and it leaves one honest gap, recorded at the bottom of this
file.
"""

import functools
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

import hflow
from hflow.mediapipe_hands import (
    HAND_LANDMARKER_MODEL_ENV_VAR,
    PINNED_HAND_MODEL_SHA256,
    FrameHandCounts,
    HandModelNotAvailableError,
    _no_detection_intervals,
    _resolved_hand_model_digest,
    hand_landmarker_model_path,
    mediapipe_hand_detection,
    summarize_hand_detections,
)
from hflow.steps import (
    UNDESCRIBED_CONFIGURATION_KEY,
    compute_check_version,
    step_identity_payload,
)
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import TransformConfig, write_canonical_episode

MEDIAPIPE_TESTS_ENABLED = os.environ.get("HFLOW_MEDIAPIPE_TESTS") == "1"


def _frame(hand_count: int, left: int = 0, right: int = 0) -> FrameHandCounts:
    return FrameHandCounts(hand_count=hand_count, left_hand_count=left, right_hand_count=right)


def _check_version() -> str:
    return compute_check_version("hands", mediapipe_hand_detection, False, frozenset(), None)


@pytest.fixture
def cleared_model_caches() -> Iterator[None]:
    hand_landmarker_model_path.cache_clear()
    _resolved_hand_model_digest.cache_clear()
    yield None
    hand_landmarker_model_path.cache_clear()
    _resolved_hand_model_digest.cache_clear()


class TestFrameArithmetic:
    def test_shares_count_frames(self) -> None:
        summary = summarize_hand_detections(
            [_frame(0), _frame(1, left=1), _frame(2, left=1, right=1), _frame(2, right=2)]
        )
        assert summary.frame_count == 4
        assert summary.frames_with_any_hand == 3
        assert summary.frames_with_two_hands == 2
        assert summary.any_hand_share == 0.75
        assert summary.two_hand_share == 0.5

    def test_nothing_measured_is_not_a_share_of_zero(self) -> None:
        """A camera that could not be sampled did not have hands absent from
        it. Recording 0.0 would put that claim in the catalog, so the share is
        absent and only the denominator is recorded.
        """
        summary = summarize_hand_detections([])
        assert summary.frame_count == 0
        assert summary.any_hand_share is None
        assert summary.two_hand_share is None

    def test_handedness_counts_separate_two_hands_from_one_hand_twice(self) -> None:
        """Two detections labelled the same way is a different fact from one of
        each, and the total alone cannot tell them apart.
        """
        one_of_each = summarize_hand_detections([_frame(2, left=1, right=1)])
        assert (one_of_each.frames_with_a_left_hand, one_of_each.frames_with_a_right_hand) == (1, 1)

        same_hand_twice = summarize_hand_detections([_frame(2, right=2)])
        assert (
            same_hand_twice.frames_with_a_left_hand,
            same_hand_twice.frames_with_a_right_hand,
        ) == (0, 1)
        # Both are still two-hand frames: the count is what the share is over.
        assert same_hand_twice.frames_with_two_hands == 1

    def test_an_unlabelled_detection_is_still_a_detection(self) -> None:
        summary = summarize_hand_detections([_frame(1)])
        assert summary.frames_with_any_hand == 1
        assert summary.frames_with_a_left_hand == 0
        assert summary.frames_with_a_right_hand == 0


class TestAbsenceIntervals:
    def test_a_run_ends_where_detection_resumes(self) -> None:
        log_times_ns = [0, 1_000_000_000, 2_000_000_000, 3_000_000_000]
        intervals = _no_detection_intervals(
            log_times_ns, [_frame(1), _frame(0), _frame(0), _frame(2)], "no_hand_detected:/cam"
        )
        (interval,) = intervals
        assert (interval.start_ns, interval.end_ns) == (1_000_000_000, 3_000_000_000)
        assert interval.label == "no_hand_detected:/cam"

    def test_a_trailing_run_closes_at_the_last_sampled_frame(self) -> None:
        log_times_ns = [0, 1_000_000_000, 2_000_000_000]
        (interval,) = _no_detection_intervals(
            log_times_ns, [_frame(1), _frame(0), _frame(0)], "absent"
        )
        assert (interval.start_ns, interval.end_ns) == (1_000_000_000, 2_000_000_000)

    def test_footage_the_model_never_saw_a_hand_in_is_one_interval(self) -> None:
        log_times_ns = [0, 1_000_000_000, 2_000_000_000]
        (interval,) = _no_detection_intervals(
            log_times_ns, [_frame(0), _frame(0), _frame(0)], "absent"
        )
        assert (interval.start_ns, interval.end_ns) == (0, 2_000_000_000)

    def test_no_intervals_when_every_frame_shows_a_hand(self) -> None:
        assert _no_detection_intervals([0, 1_000_000_000], [_frame(1), _frame(2)], "absent") == []

    def test_a_lone_trailing_absent_frame_spans_no_footage(self) -> None:
        """One absent frame at the very end has nothing after it to bound the
        span, and an interval from a point to itself would claim a duration the
        sampling cannot support.
        """
        assert _no_detection_intervals([0, 1_000_000_000], [_frame(1), _frame(0)], "absent") == []


class TestModelAcquisition:
    def test_the_environment_override_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleared_model_caches: None
    ) -> None:
        own_model = tmp_path / "my-hand-landmarker.task"
        own_model.write_bytes(b"not really a model")
        monkeypatch.setenv(HAND_LANDMARKER_MODEL_ENV_VAR, str(own_model))
        assert hand_landmarker_model_path() == own_model
        # The recorded digest describes what ran, not what was pinned.
        assert _resolved_hand_model_digest() == hashlib.sha256(b"not really a model").hexdigest()

    def test_an_override_to_a_missing_file_names_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, cleared_model_caches: None
    ) -> None:
        monkeypatch.setenv(HAND_LANDMARKER_MODEL_ENV_VAR, "/nonexistent/model.task")
        with pytest.raises(HandModelNotAvailableError, match=HAND_LANDMARKER_MODEL_ENV_VAR):
            hand_landmarker_model_path()

    def test_the_pinned_asset_is_downloaded_once_and_verified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleared_model_caches: None
    ) -> None:
        """The whole acquisition path, offline: a ``file://`` URL stands in for
        the pinned one so the download, the digest check, and the atomic
        publish into the user cache are all exercised without network access.
        """
        published_asset = tmp_path / "source" / "hand_landmarker.task"
        published_asset.parent.mkdir()
        published_asset.write_bytes(b"pretend weights")
        monkeypatch.delenv(HAND_LANDMARKER_MODEL_ENV_VAR, raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.setattr("hflow.mediapipe_hands.PINNED_HAND_MODEL_URL", published_asset.as_uri())
        monkeypatch.setattr(
            "hflow.mediapipe_hands.PINNED_HAND_MODEL_SHA256",
            hashlib.sha256(b"pretend weights").hexdigest(),
        )

        resolved = hand_landmarker_model_path()
        assert resolved.is_relative_to(tmp_path / "cache")
        assert resolved.read_bytes() == b"pretend weights"

        # A second resolution uses the cache: deleting the source proves it.
        published_asset.unlink()
        hand_landmarker_model_path.cache_clear()
        assert hand_landmarker_model_path() == resolved

    def test_a_digest_mismatch_refuses_and_leaves_nothing_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleared_model_caches: None
    ) -> None:
        """Unverified weights must never occupy the cache path, even briefly:
        the next run would read them as the pinned asset.
        """
        wrong_asset = tmp_path / "source" / "hand_landmarker.task"
        wrong_asset.parent.mkdir()
        wrong_asset.write_bytes(b"tampered weights")
        monkeypatch.delenv(HAND_LANDMARKER_MODEL_ENV_VAR, raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.setattr("hflow.mediapipe_hands.PINNED_HAND_MODEL_URL", wrong_asset.as_uri())

        with pytest.raises(HandModelNotAvailableError) as error_info:
            hand_landmarker_model_path()
        message = str(error_info.value)
        assert PINNED_HAND_MODEL_SHA256 in message
        assert HAND_LANDMARKER_MODEL_ENV_VAR in message
        assert not list((tmp_path / "cache").rglob("*.task"))


class TestRegistrationWithoutTheModel:
    def test_registering_the_check_needs_neither_the_extra_nor_the_model(
        self, tmp_path: Path
    ) -> None:
        """The documented shortest path, on a machine that has neither.

        Registration computes a content hash, and anything it touched that
        reached for the instrument would make the check impossible to register
        until the instrument was installed -- exactly the regression that
        ``camera_frame_stats`` shipped with when it read the ffmpeg version at
        registration.
        """
        app = hflow.App("hands", data_root=tmp_path)
        app.check()(mediapipe_hand_detection)
        (registered,) = app.checks
        assert registered.name == "mediapipe_hand_detection"
        assert registered.version

    def test_changing_the_model_pin_moves_the_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different weights cannot share one identity: they are what produced
        the numbers, so a re-pin has to append new-version rows rather than
        mixing two models' measurements under one version.
        """
        version_on_the_pinned_model = _check_version()
        monkeypatch.setattr("hflow.mediapipe_hands.PINNED_HAND_MODEL_SHA256", "0" * 64)
        assert _check_version() != version_on_the_pinned_model

    def test_the_version_leaves_nothing_undescribed(self) -> None:
        """Same bar as the built-ins: a marker here means someone can edit the
        code below it and no version will move.
        """
        payload = step_identity_payload(
            "mediapipe_hand_detection", mediapipe_hand_detection, False, frozenset(), None
        )
        assert UNDESCRIBED_CONFIGURATION_KEY not in json.dumps(payload)

    def test_retuning_a_confidence_moves_the_version(self) -> None:
        default_version = _check_version()
        retuned_version = compute_check_version(
            "hands",
            functools.partial(mediapipe_hand_detection, minimum_hand_detection_confidence=0.8),
            False,
            frozenset(),
            None,
        )
        assert retuned_version != default_version

    def test_a_sampling_rate_finer_than_the_model_clock_is_refused(self, tmp_path: Path) -> None:
        """MediaPipe timestamps video frames in whole milliseconds, so two
        frames inside one millisecond cannot be distinguished. Refused up
        front rather than producing silently misordered inference.
        """
        source = synthesize_episode(
            tmp_path / "episode.mcap", SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam",))
        )
        canonical = tmp_path / "episode.canonical.mcap"
        write_canonical_episode(source, canonical, TransformConfig())
        with hflow.Episode(canonical) as episode, pytest.raises(ValueError, match="sample_fps"):
            mediapipe_hand_detection(episode, sample_fps=5000.0)


@pytest.fixture(scope="module")
def hands_episode_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A canonical episode with one camera, built once for the whole module."""
    directory = tmp_path_factory.mktemp("hands")
    source = synthesize_episode(
        directory / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=3.0, cameras=("wrist_cam",)),
    )
    canonical = directory / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    return canonical


@pytest.mark.skipif(
    not MEDIAPIPE_TESTS_ENABLED,
    reason="needs the mediapipe extra and the model asset; set HFLOW_MEDIAPIPE_TESTS=1",
)
class TestEndToEnd:
    """The plumbing, against the real model.

    What this proves: the extra imports, the model resolves and verifies, the
    RGB frames are shaped the way MediaPipe accepts, inference runs, and the
    measurements land under the documented keys.

    What it cannot prove: that a POSITIVE detection ever happens. Synthetic
    footage has no hands in it and there is no hand fixture in this repo, so a
    zero share here is the correct answer and also indistinguishable from a
    detector that never fires. Confirm that by hand on real footage; the
    how-to shows the snippet.
    """

    def test_footage_without_hands_records_a_zero_share_over_a_real_denominator(
        self, hands_episode_path: Path
    ) -> None:
        with hflow.Episode(hands_episode_path) as episode:
            result = mediapipe_hand_detection(episode)

        topic = "/wrist_cam/compressed"
        assert result.measurements[f"{topic}/hand_detection_frame_count"] == 3
        assert result.measurements[f"{topic}/hand_detected_frame_share"] == 0.0
        assert result.measurements[f"{topic}/two_hand_detected_frame_share"] == 0.0
        assert result.measurements[f"{topic}/hand_sample_fps"] == 1.0
        # The instrument, recorded as evidence rather than hashed into the version.
        assert result.measurements[f"{topic}/mediapipe_version"]
        assert result.measurements[f"{topic}/hand_model_digest"] == PINNED_HAND_MODEL_SHA256
        # No detection anywhere is one absence interval over the sampled span.
        (interval,) = result.intervals
        assert interval.label == f"no_hand_detected:{topic}"

    def test_the_inference_size_is_recorded_when_it_is_set(self, hands_episode_path: Path) -> None:
        """It changes which hands are found at all, so a reader has to be able
        to see which size produced the row.
        """
        with hflow.Episode(hands_episode_path) as episode:
            result = mediapipe_hand_detection(episode, inference_long_edge_pixels=256)
        assert result.measurements["/wrist_cam/compressed/hand_inference_long_edge_pixels"] == 256

    def test_no_key_collides_with_a_built_in_check(self, hands_episode_path: Path) -> None:
        """A pipeline may register this beside the built-ins, and the catalog
        ranks measurements per (episode_id, key): a shared key is a tie one
        producer silently loses.
        """
        with hflow.Episode(hands_episode_path) as episode:
            hand_keys = set(mediapipe_hand_detection(episode).measurements)
            builtin_keys = {
                key
                for check in (
                    hflow.checks.camera_frame_stats,
                    hflow.checks.camera_signal_quality,
                    hflow.checks.camera_fps_conformance,
                    hflow.checks.camera_stability,
                    hflow.checks.keyframe_interval,
                    hflow.checks.media_digest,
                    hflow.checks.content_digest,
                    hflow.checks.episode_duration,
                    hflow.checks.timestamp_regularity,
                )
                for key in check(episode).measurements
            }
        assert hand_keys.isdisjoint(builtin_keys)
