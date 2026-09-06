"""Sampled Build AI checks: per-frame observations folded into absence intervals."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import httpx2
import pytest

import hflow
from hflow.build_ai_vlm_checks import (
    FrameSampling,
    HFlowHostedExecution,
    register_active_manipulation,
    register_hand_visibility,
)
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

NANOSECONDS_PER_SECOND = 1_000_000_000
CAMERA_TOPIC = "/head_camera/compressed"


class _StubHostedResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _StubHostedResponse:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self) -> Iterator[bytes]:
        yield self._body


def _scripted_hosted_answers(
    monkeypatch: pytest.MonkeyPatch, answers: list[object]
) -> list[object]:
    """Answer hosted requests in order; unparsed answers are strings the parser refuses."""
    remaining = list(answers)
    served: list[object] = []

    def hosted_response(method: str, url: str, **_request: object) -> _StubHostedResponse:
        answer = remaining.pop(0)
        served.append(answer)
        if isinstance(answer, str) and answer.startswith("unparsed:"):
            return _StubHostedResponse(
                {"outcome": "unparsed", "raw_response": answer, "parse_error": "not a count"}
            )
        return _StubHostedResponse(
            {"outcome": "parsed", "prediction": answer, "raw_response": str(answer)}
        )

    monkeypatch.setattr(httpx2, "stream", hosted_response)
    return served


def _episode(tmp_path: Path, duration_s: float) -> Path:
    return synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(
            duration_s=duration_s,
            cameras=("head_camera",),
            image_hz=10.0,
            black_segment=None,
            joint_jump_at_s=None,
            timestamp_offset_segment=None,
        ),
    )


def test_sampled_hand_visibility_folds_no_hand_frames_into_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = _scripted_hosted_answers(monkeypatch, [2, 0, 0, 1])
    application = hflow.App("sampled", data_root=tmp_path / "data", default_checks=())
    register_hand_visibility(
        application, execution=HFlowHostedExecution(), sampling=FrameSampling(fps=1.0)
    )

    report = application.test(_episode(tmp_path, duration_s=4.0), verbose=False)

    run = report.check("build_ai_hand_visibility")
    assert run.status is hflow.CheckStatus.MEASURED, run.error
    assert run.result is not None
    assert len(served) == 4
    stamps = [observation.timestamp_ns for observation in run.result.observations]
    assert len(stamps) == 4 and stamps == sorted(stamps)
    assert [observation.values["prediction"] for observation in run.result.observations] == [
        2,
        0,
        0,
        1,
    ]
    # The run opens at the first 0-hands frame and closes at the frame that
    # saw a hand again, on the log clock.
    assert run.result.intervals == [
        hflow.Interval(start_ns=stamps[1], end_ns=stamps[3], label=f"hands_absent:{CAMERA_TOPIC}")
    ]
    measurements = run.result.measurements
    assert measurements["build_ai/hand_count/sampled_frame_count"] == 4
    assert measurements["build_ai/hand_count/hands_absent_frame_count"] == 2
    assert measurements["build_ai/hand_count/hands_absent_frame_pct"] == pytest.approx(50.0)
    assert measurements["build_ai/hand_count/hands_absent_total_s"] == pytest.approx(
        (stamps[3] - stamps[1]) / NANOSECONDS_PER_SECOND
    )
    assert measurements["build_ai/hand_count/unparsed_frame_count"] == 0
    assert run.result.tags == []


def test_sampled_run_reaching_the_end_closes_one_period_after_the_last_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted_hosted_answers(monkeypatch, ["yes", "no", "no"])
    application = hflow.App("sampled-end", data_root=tmp_path / "data", default_checks=())
    register_active_manipulation(
        application, execution=HFlowHostedExecution(), sampling=FrameSampling(fps=1.0)
    )

    report = application.test(_episode(tmp_path, duration_s=3.0), verbose=False)

    run = report.check("build_ai_active_manipulation")
    assert run.result is not None, run.error
    stamps = [observation.timestamp_ns for observation in run.result.observations]
    assert run.result.intervals == [
        hflow.Interval(
            start_ns=stamps[1],
            end_ns=stamps[2] + NANOSECONDS_PER_SECOND,
            label=f"no_manipulation:{CAMERA_TOPIC}",
        )
    ]


def test_an_unparsed_answer_ends_a_run_without_counting_as_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted_hosted_answers(monkeypatch, [0, "unparsed: three", 0, 2])
    application = hflow.App("sampled-unparsed", data_root=tmp_path / "data", default_checks=())
    register_hand_visibility(
        application, execution=HFlowHostedExecution(), sampling=FrameSampling(fps=1.0)
    )

    report = application.test(_episode(tmp_path, duration_s=4.0), verbose=False)

    run = report.check("build_ai_hand_visibility")
    assert run.result is not None, run.error
    stamps = [observation.timestamp_ns for observation in run.result.observations]
    assert [interval.start_ns for interval in run.result.intervals] == [stamps[0], stamps[2]]
    assert [interval.end_ns for interval in run.result.intervals] == [stamps[1], stamps[3]]
    assert run.result.measurements["build_ai/hand_count/hands_absent_frame_count"] == 2
    assert run.result.measurements["build_ai/hand_count/unparsed_frame_count"] == 1
    assert run.result.tags == ["build_ai/hand_count/unparsed"]
    assert run.result.observations[1].values["valid"] is False


def test_a_transient_hosted_failure_is_retried_within_the_sampled_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One gateway timeout mid-run costs a delay, not the whole check."""
    import httpx2 as httpx_module

    sleeps: list[float] = []
    monkeypatch.setattr("hflow.build_ai_vlm_checks.time.sleep", sleeps.append)
    answers: list[object] = [2, "http-504", 0, 2]

    class _FailingResponse(_StubHostedResponse):
        def __init__(self, status_code: int) -> None:
            super().__init__({})
            self._status_code = status_code

        def raise_for_status(self) -> None:
            request = httpx_module.Request("POST", "https://api.hflow.dev/v1/checks")
            response = httpx_module.Response(
                self._status_code, request=request, headers={"Retry-After": "3"}
            )
            raise httpx_module.HTTPStatusError("gateway", request=request, response=response)

    def hosted_response(method: str, url: str, **_request: object) -> _StubHostedResponse:
        answer = answers.pop(0)
        if answer == "http-504":
            return _FailingResponse(504)
        return _StubHostedResponse(
            {"outcome": "parsed", "prediction": answer, "raw_response": str(answer)}
        )

    monkeypatch.setattr(httpx2, "stream", hosted_response)
    application = hflow.App("sampled-retry", data_root=tmp_path / "data", default_checks=())
    register_hand_visibility(
        application, execution=HFlowHostedExecution(), sampling=FrameSampling(fps=1.0)
    )

    report = application.test(_episode(tmp_path, duration_s=3.0), verbose=False)

    run = report.check("build_ai_hand_visibility")
    assert run.status is hflow.CheckStatus.MEASURED, run.error
    assert run.result is not None
    assert [observation.values["prediction"] for observation in run.result.observations] == [
        2,
        0,
        2,
    ]
    # The server's Retry-After was honoured and the failed request was repeated.
    assert sleeps == [3.0]
    assert answers == []


def test_hosted_retries_are_bounded_and_the_last_status_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx2 as httpx_module

    sleeps: list[float] = []
    monkeypatch.setattr("hflow.build_ai_vlm_checks.time.sleep", sleeps.append)
    attempts = 0

    class _AlwaysBusy(_StubHostedResponse):
        def raise_for_status(self) -> None:
            request = httpx_module.Request("POST", "https://api.hflow.dev/v1/checks")
            response = httpx_module.Response(429, request=request)
            raise httpx_module.HTTPStatusError("busy", request=request, response=response)

    def hosted_response(method: str, url: str, **_request: object) -> _StubHostedResponse:
        nonlocal attempts
        attempts += 1
        return _AlwaysBusy({})

    monkeypatch.setattr(httpx2, "stream", hosted_response)
    application = hflow.App("sampled-busy", data_root=tmp_path / "data", default_checks=())
    register_hand_visibility(
        application,
        execution=HFlowHostedExecution(max_retries=2),
        sampling=FrameSampling(fps=1.0),
    )

    report = application.test(_episode(tmp_path, duration_s=1.0), verbose=False)

    run = report.check("build_ai_hand_visibility")
    assert run.status is hflow.CheckStatus.ERROR
    assert run.error is not None and "HTTP 429" in run.error
    assert attempts == 3
    # No Retry-After: exponential from one second.
    assert sleeps == [1.0, 2.0]


def test_black_frames_are_skipped_and_never_read_as_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blackout says nothing about hands: the model is not asked about a
    black frame, the frame is recorded as skipped, and it ends a run of
    absence instead of extending one."""
    served = _scripted_hosted_answers(monkeypatch, [0, 0, 2])
    application = hflow.App("sampled-black", data_root=tmp_path / "data", default_checks=())
    register_hand_visibility(
        application, execution=HFlowHostedExecution(), sampling=FrameSampling(fps=1.0)
    )
    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(
            duration_s=4.0,
            cameras=("head_camera",),
            image_hz=10.0,
            # Frames sampled at 0, 1, 2, 3 s: the one at 2 s is black.
            black_segment=(1.8, 2.6),
            joint_jump_at_s=None,
            timestamp_offset_segment=None,
        ),
    )

    report = application.test(source, verbose=False)

    run = report.check("build_ai_hand_visibility")
    assert run.status is hflow.CheckStatus.MEASURED, run.error
    assert run.result is not None
    # Three model calls for four frames: the black one was never sent.
    assert len(served) == 3
    observations = run.result.observations
    assert len(observations) == 4
    assert observations[2].values == {
        "task": "hand-count",
        "valid": False,
        "skipped": "black_frame",
    }
    stamps = [observation.timestamp_ns for observation in observations]
    # 0 hands at 0 s and 1 s, then the black frame closes the run at 2 s.
    assert run.result.intervals == [
        hflow.Interval(start_ns=stamps[0], end_ns=stamps[2], label=f"hands_absent:{CAMERA_TOPIC}")
    ]
    measurements = run.result.measurements
    assert measurements["build_ai/hand_count/sampled_frame_count"] == 4
    assert measurements["build_ai/hand_count/skipped_black_frame_count"] == 1
    assert measurements["build_ai/hand_count/hands_absent_frame_count"] == 2
    # Percentages are over the frames the model actually saw.
    assert measurements["build_ai/hand_count/hands_absent_frame_pct"] == pytest.approx(200 / 3)


def test_sampling_is_part_of_the_check_version(tmp_path: Path) -> None:
    versions: list[str] = []
    for sampling in (None, FrameSampling(fps=1.0), FrameSampling(fps=2.0)):
        application = hflow.App("versions", data_root=tmp_path / "data", default_checks=())
        register_hand_visibility(application, execution=HFlowHostedExecution(), sampling=sampling)
        versions.append(application.checks[0].version)
    assert len(set(versions)) == 3


@pytest.mark.parametrize(
    "bad_sampling",
    [
        {"fps": 0.0},
        {"fps": float("inf")},
        {"start_s": -1.0},
        {"start_s": 5.0, "end_s": 5.0},
    ],
)
def test_frame_sampling_refuses_an_empty_or_unbounded_window(bad_sampling: dict) -> None:
    with pytest.raises(ValueError):
        FrameSampling(**bad_sampling)
