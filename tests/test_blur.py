"""Raw-score and missing-evidence outcomes for the CPU blur adapter."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from hflow.blur import measure_video_blur, summarize_blur_scores


def test_summary_preserves_raw_scores_and_excludes_unavailable_frames() -> None:
    summary = summarize_blur_scores(iter((120.0, math.nan, 240.0, math.inf, -math.inf)))

    assert summary.frame_count == 5
    assert summary.scored_frame_count == 2
    assert summary.mean_blur_score == pytest.approx(180.0)


@pytest.mark.parametrize("scores", [(), (math.nan, math.inf, -math.inf)])
def test_missing_blur_evidence_has_no_score(scores: tuple[float, ...]) -> None:
    summary = summarize_blur_scores(iter(scores))

    assert summary.frame_count == len(scores)
    assert summary.scored_frame_count == 0
    assert summary.mean_blur_score is None


def write_static_video(video_path: Path, frame: NDArray[np.uint8]) -> None:
    frame_height, frame_width = frame.shape
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter.fourcc(*"FFV1"),
        5.0,
        (frame_width, frame_height),
        isColor=False,
    )
    if not writer.isOpened():
        raise RuntimeError("Could not generate blur test video")
    try:
        for _ in range(3):
            writer.write(frame)
    finally:
        writer.release()


def test_video_adapter_distinguishes_defocus_from_missing_detail(tmp_path: Path) -> None:
    row_indices, column_indices = np.indices((128, 192))
    sharp_frame = (((row_indices // 16 + column_indices // 16) % 2) * 255).astype(np.uint8)
    defocused_frame = cv2.GaussianBlur(sharp_frame, (0, 0), sigmaX=3.0).astype(np.uint8)
    featureless_frame = np.full_like(sharp_frame, 128)
    sharp_path = tmp_path / "sharp.mkv"
    defocused_path = tmp_path / "defocused.mkv"
    featureless_path = tmp_path / "featureless.mkv"
    write_static_video(sharp_path, sharp_frame)
    write_static_video(defocused_path, defocused_frame)
    write_static_video(featureless_path, featureless_frame)

    sharp_summary = measure_video_blur(sharp_path)
    defocused_summary = measure_video_blur(defocused_path)
    featureless_summary = measure_video_blur(featureless_path)

    assert sharp_summary.frame_count == sharp_summary.scored_frame_count == 3
    assert defocused_summary.frame_count == defocused_summary.scored_frame_count == 3
    assert sharp_summary.mean_blur_score is not None
    assert defocused_summary.mean_blur_score is not None
    assert defocused_summary.mean_blur_score > sharp_summary.mean_blur_score
    assert featureless_summary.frame_count == 3
    assert featureless_summary.scored_frame_count == 0
    assert featureless_summary.mean_blur_score is None
