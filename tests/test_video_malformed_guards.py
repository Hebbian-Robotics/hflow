"""Regression tests for malformed H.264 inputs rejected by hflow.video."""

import pytest

from hflow.video import (
    _nal_header_offset,
    ensure_access_unit_delimiter,
    source_log_times_for_sampled_frames,
)

_ANNEX_B_4_BYTE_START_CODE = b"\x00\x00\x00\x01"
_IDR_NAL_HEADER = b"\x65"  # NAL type 5: IDR coded slice.


def _single_slice_idr(rbsp: bytes) -> bytes:
    """Build one Annex B IDR NAL with an explicitly supplied slice-header RBSP."""
    return _ANNEX_B_4_BYTE_START_CODE + _IDR_NAL_HEADER + rbsp


def test_sampled_frame_mapping_rejects_empty_source_stream() -> None:
    message = "cannot map sampled frames onto an empty source stream"
    with pytest.raises(ValueError, match=message):
        source_log_times_for_sampled_frames(
            [], source_fps=30.0, sample_fps=10.0, start_s=0.0, frame_count=1
        )


def test_nal_header_rejects_invalid_annex_b_start_code() -> None:
    with pytest.raises(ValueError, match="invalid Annex B start code at byte 3"):
        _nal_header_offset(b"\x00\x00\x00\xff", 3)


def test_ensure_aud_rejects_slice_header_without_complete_first_mb_value() -> None:
    # first_mb_in_slice is unsigned Exp-Golomb. An all-zero RBSP has no
    # terminating one bit, so the decoder must reject it as incomplete.
    malformed_slice = _single_slice_idr(b"\x00")
    message = "slice header has no complete first_mb_in_slice value"

    with pytest.raises(ValueError, match=message):
        ensure_access_unit_delimiter(malformed_slice)


def test_ensure_aud_rejects_truncated_first_mb_value() -> None:
    # 00000100 starts an Exp-Golomb value with five leading zero bits and a
    # one bit, but only two suffix bits remain in the byte. This deliberately
    # reaches the truncation guard rather than the no-complete-value guard.
    malformed_slice = _single_slice_idr(b"\x04")
    message = "slice header truncates its first_mb_in_slice value"

    with pytest.raises(ValueError, match=message):
        ensure_access_unit_delimiter(malformed_slice)
