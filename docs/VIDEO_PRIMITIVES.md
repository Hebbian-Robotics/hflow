# Local video preparation and measurements

These file-level APIs do not require a catalog, scheduler, model service, or
persistent workspace. Callers retain source identity and own output lifetimes.
FFmpeg-based APIs use HFlow's binary policy and may download managed binaries
unless the caller configures an installed toolchain.

## MCAP camera export

`hflow.mcap_video.export_mcap_camera(source, output, camera_topic=None, limits=...)`
requires the optional `hflow[video]` dependency. It streams one camera from an MCAP
into MP4, returning `PreparedMcapVideo` with the output path, selected topic,
original first log timestamp, and duration in milliseconds. Multiple cameras
require an exact topic. Existing destinations are never overwritten.

Frame times are MCAP log times relative to the first selected frame, rounded to
microseconds. Each frame lasts until the next; the final frame repeats the prior
positive interval. At least two frames are required. Duplicate rounded timestamps,
unsupported formats, changing dimensions, B-frames, missing H.264 codec headers,
and gaps beyond the MP4 duration representation are rejected. H.264 access units
are remuxed with lossless AUD repair; supported JPEG/PNG and 8-bit raw images are
encoded as lossless RGB H.264. Raw recordings remain unchanged.

This is synchronous native decoding. A caller needing a hard execution deadline
must isolate it in a process and reap that process before deleting temporary media.
The existing canonical `camera_video` enrichment keeps its constant-rate contract.

## Direct model-video preparation

`hflow.importers.video.prepare_model_video(source, output, config, limits=...,
transform_config=...)` uses `VideoImportConfig` and `TransformConfig` to produce
canonical model-input pixels without first writing an MCAP. It shares the video
importer's direct source-to-H.264 encoding: fixed-rate sampling, aspect-preserving
resize and letterboxing, then libx264, with no JPEG intermediate. Output is an
atomically published caller-owned MP4. Removing the former lossy JPEG step
intentionally changes pixels compared with older imports.

`import_video_episode` and `prepare_video_episode` also accept `transform_config`.
Use the same `TransformConfig` for import, canonical SYNC, and direct preparation
to obtain identical decoded pixels. Custom `crf`, `gop_preset`, and explicit
`gop_seconds` are applied at import/direct preparation; explicit seconds override
the preset. Canonical SYNC validates and copies the encoded access units without
a second lossy encode. If requested CRF or effective GOP seconds differ from the
import metadata (or those settings are missing), SYNC raises `SourceNotConforming`
with instructions to re-import the original video with the requested settings.
Changing only grouping/compression settings does not require re-import. Ordinary
recorded H.264 without the first-party import record retains its existing
pass-through behavior; legacy JPEG landing episodes still transcode at SYNC.

For example, pass `TransformConfig(crf=18, gop_seconds=2.0)` as `transform_config`
to either import function and to `prepare_model_video`, and as `config` to
`write_canonical_episode`. Encoding choices live in `video_import/v1`, along with
source identity, sampling settings, and the actual FFmpeg version. Landing remains
a source episode: only caller metadata enters `episode/v1`; the canonical
transform still owns grouping, QC boundaries, and `provenance/v1`.

Samples retain the half-open excerpt grid and MCAP timestamps. When there is only
one sample, its H.264 encoder timing and exported MP4 packet/container duration
are one second, even for sampling rates below or above 1 Hz. Multi-frame output
uses the requested sampling rate.

`VideoImportConfig.maximum_encoded_bytes` defaults to 64 MiB (exclusive) for all
three entrypoints. FFmpeg is asked to stop at that size (it may overshoot by one
encoded packet); size is checked before Python reads/splits/validates the stream.
Reaching the limit rejects the excerpt without publishing output. Split larger
excerpts or explicitly raise the budget. Buffering is bounded by the selected
encoded-byte budget but includes copied access units, parser objects and validation
buffers: this is not a 64 MiB total-memory/RSS guarantee. Existing source/output
dimension limits, timeouts, bounded diagnostics, and temporary-file cleanup remain.

Expected media failures return `UnreadableVideo` or `UnsupportedVideo`; operational
failures raise. The exception-style `import_video_episode` raises `ValueError` for
unsupported excerpts, including the encoded-byte limit. Existing destinations raise
`FileExistsError`. New import identities reflect the new encoded bytes and importer
metadata; canonical encoding defaults and generic H.264 pass-through are unchanged.

## Frame statistics

`hflow.video_statistics.measure_video_frame_statistics(video, settings=...,
toolchain=None, instrument_output_cache_path=None)` exposes file-level statistics,
settings, provenance, and errors as a public API. No persistent cache is created
unless explicitly requested.

`FrameStatisticsSettings.luma_range` accepts `LumaRangePolicy.PRESERVE` (the existing
instrument behavior) or `FULL` (convert the declared input range to full-range luma
inside the measurement filter graph). The graph and settings are recorded in
provenance. Measure unpadded source framing; black model-input borders would affect
pixel statistics. Full-range measurement needs no intermediate video encode.

## Raw blur and shake summaries

`hflow.blur.measure_video_blur(video, timeout_seconds=120, executable=None)` uses
FFmpeg's default `blurdetect` settings at the supplied frame cadence and resolution.
`BlurSummary` contains observed/scored frame counts and the mean finite raw score.
Nonfinite scores are unassessed, zero assessed frames produce a null mean, and
negative scores fail. `summarize_blur_scores` applies the same accounting to a
caller-supplied score stream.

`hflow.camera_motion.summarize_camera_shake(observations)` consumes filtered motion
observations in constant summary memory. Mean and RMS weight adjacent-pair duration;
maximum, assessment duration, and missing-context counts remain explicit. The final
frame has no following pair and contributes no extra duration. Missing observations
never become zero shake. Field of view is caller configuration, not calibration.

These measurements do not classify footage as blurry or unstable and are not
accuracy estimates. See [streaming camera motion](how-to/stream-camera-motion.md)
for extraction and filtering contracts, and [source sampling](how-to/sample-source-video.md)
for original-frame evidence selection.

## Shared window measurements

`measure_video_window_independently` uses the same single decode and selection,
but returns an `IndependentVideoWindowMeasurements` with a typed
`MeasurementFailure` for a branch whose Python motion calculation or output
parsing fails. `None` still means the branch was not selected. Source probe,
FFmpeg graph, timeout, and filesystem failures are shared and cannot yield
independent measurements. The original `measure_video_window` keeps its strict
all-or-nothing behavior.

`hflow.window_measurements.measure_video_window(source, window, selection,
limits=VideoLimits(), toolchain=None)` measures one `VideoWindow` of the original
source with a single decode. FFmpeg seeks the source, applies the window's `fps`
filter and its default display-rotation handling, then splits the frames between
the measurements chosen by `WindowMeasurementSelection`: `frame_statistics`
settings, `blur`, and `camera_shake` settings. No intermediate video is written.

Each branch applies exactly the filters of its file-level measurement, so results
equal measuring an uncompressed copy of the same window with
`measure_video_frame_statistics`, `measure_video_blur`, and `stream_camera_motion`.
They differ from measuring `prepare_video_window` output, whose lossy H.264 encode
changes pixel values. Unselected results are `None`. The window is clamped to the
source end. A start at or beyond the source end, a damaged source, or a window that
decodes to less than half its duration (at most 0.5 seconds) returns
`UnreadableVideo`; unsupported sources return `UnsupportedVideo`, as
`prepare_video_window` does. Timeouts and other process failures raise.
