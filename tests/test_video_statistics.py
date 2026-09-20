"""Public full-range measurements match explicit lossless normalization."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from hflow.ffmpeg import ffmpeg_path
from hflow.video_statistics import (
    FrameStatisticsSettings,
    LumaRangePolicy,
    measure_video_frame_statistics,
)


@pytest.mark.parametrize("color", ["black", "gray", "white"])
def test_full_range_statistics_match_explicit_normalization_without_persistent_outputs(
    tmp_path: Path, color: str
) -> None:
    source = tmp_path / "source.mp4"
    normalized = tmp_path / "normalized.mp4"
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color={color}:s=96x64:r=8:d=3",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    source_digest = hashlib.sha256(source.read_bytes()).digest()
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            "scale=in_range=auto:out_range=full",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "pc",
            str(normalized),
        ],
        check=True,
        capture_output=True,
    )
    reference = measure_video_frame_statistics(
        normalized, settings=FrameStatisticsSettings(luma_range=LumaRangePolicy.FULL)
    )
    observed = measure_video_frame_statistics(
        source, settings=FrameStatisticsSettings(luma_range=LumaRangePolicy.FULL)
    )
    for field_name in (
        "black_frame_percent",
        "clipped_highlight_frame_percent",
        "crushed_shadow_frame_percent",
        "freeze_total_seconds",
        "average_luma_mean",
    ):
        assert getattr(observed, field_name) == pytest.approx(getattr(reference, field_name))
    assert observed.average_luma_mean == pytest.approx(
        {"black": 0, "gray": 128, "white": 255}[color]
    )
    assert observed.provenance.settings.luma_range is LumaRangePolicy.FULL
    assert hashlib.sha256(source.read_bytes()).digest() == source_digest
    assert set(tmp_path.iterdir()) == {source, normalized}
