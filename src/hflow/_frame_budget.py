"""The default decoded-frame size budget shared by video probing and raw-frame readers.

This module has no dependencies so the extractable ``hflow._video_measurements``
package can share the default with ``hflow.media.VideoLimits``.
"""

DEFAULT_MAXIMUM_FRAME_PIXELS = 33554432
_FFMPEG_MAXIMUM_PIXELS_OPTION_LIMIT = 2**31 - 1


def ffmpeg_decoder_maximum_pixels(maximum_frame_pixels: int) -> int:
    """Return FFmpeg's ``-max_pixels`` decoder option for a frame budget.

    FFmpeg checks the stride-aligned width (up to 64 padding columns) against
    ``-max_pixels``, so the decoder needs headroom to accept every frame that fits
    the budget. Twice the budget still bounds what a stream that changes after it
    was probed can make the decoder allocate.
    """
    return min(2 * maximum_frame_pixels, _FFMPEG_MAXIMUM_PIXELS_OPTION_LIMIT)
