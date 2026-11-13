"""Regression tests for malformed H.264 inputs rejected by hflow.video."""

import numpy as np
import pytest

from hflow.video import (
    source_log_times_for_sampled_frames,
)


@pytest.mark.parametrize("timestamps", [[], np.empty(0, dtype=np.int64)])
def test_sampled_frame_mapping_rejects_empty_source_stream(
    timestamps: list[int] | np.ndarray,
) -> None:
    message = "cannot map sampled frames onto an empty source stream"
    with pytest.raises(ValueError, match=message):
        source_log_times_for_sampled_frames(
            timestamps, source_fps=30.0, sample_fps=10.0, start_s=0.0, frame_count=1
        )
