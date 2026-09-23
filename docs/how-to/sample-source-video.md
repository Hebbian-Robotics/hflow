# Sample original video frames with complete window coverage

Use this when a worker needs bounded previews of an original local video before
importing it into a canonical episode. The
[runnable example](../../examples/sample_source_video.py) probes the original
duration, plans complete windows, writes JPEGs, and prints one JSON record per
window with source timestamps and any keyframe fallback reason. To send each
window's frames to a model while later windows are sampled, see
[Score source windows with a model](./score-source-windows-with-a-model.md).

## Run the example

From the repository root, with the normal uv environment and a local video:

```bash
uv run python examples/sample_source_video.py recording.mp4 \
  --output data/source-frames --maximum-window-millis 120000 \
  --maximum-frames 16 --mode keyframes_first
```

FFmpeg and ffprobe follow HFlow's normal binary policy, which may download a
managed build. The example reads the source and writes JPEGs under the new output
directory. It makes no model calls and uses no cloud credentials or scheduler.
Use a fresh output directory for each run; existing output is never overwritten.

## Plan every part of the source

`hflow.plan_source_windows(duration_millis, maximum_window_millis=...)` produces
the fewest windows that cover the entire duration. Windows are half-open
`SourceWindow(start_millis, end_millis)` values in playback time. Integer
millisecond arithmetic prevents gaps and overlaps. Window lengths differ by at
most one millisecond, with longer windows first.

For example, a 120,001 ms source with a 120,000 ms maximum becomes `[0, 60001)`
and `[60001, 120001)`. It never creates or discards a 1 ms tail.

`minimum_window_millis` defaults to 1 and `maximum_windows` to 10,000. An
impossible minimum or excessive count raises `ValueError` before allocating the
plan. Obtain the duration from `hflow.media.probe_video` and its
`VideoProperties.duration_millis`, which rounds the measured duration up to the
next millisecond. Object identity, revisions, hashes, source grouping, and worker
admission limits remain the caller's responsibility.

The planning module uses only the standard library and numeric guards. It can
run in an owner-side process without loading media, model, or pipeline modules.

## Choose a frame policy

Pass a `SourceFrameSampling` configuration to `hflow.sample_source_frames`:

| Mode | Selection |
| --- | --- |
| `UNIFORM` | First original frame in each temporal bin. Bin length is the greater of `minimum_interval_millis` (default 1,000) and window duration divided by `maximum_frames` (default 16). |
| `KEYFRAMES` | First encoded keyframe in each of `maximum_frames` equal temporal bins. |
| `KEYFRAMES_FIRST` | Try keyframes; fall back to uniform if fewer than two were selected or their timestamp span covers less than half the window. |

Bins start at the window boundary. The sampler probes the source time base and
selects bins using integer presentation timestamps, avoiding floating-point
rounding at frame boundaries. Timelines exceeding FFmpeg's exact arithmetic
range are rejected. A frame belongs to at most one bin. Empty
bins stay empty: the sampler neither duplicates frames nor invents presentation
timestamps to fill gaps. An empty keyframe window is a valid empty result;
callers must not interpret it as a negative model observation. Fallback depends
only on temporal coverage and is reported through `KeyframeFallbackReason`.

The result pairs every JPEG path with its actual `timestamp_seconds` as a Python
`Fraction`. Times are relative to the source container's playback origin, even
when its stored timestamps begin at a nonzero offset. They are not relative to
the requested window and are not MCAP log times. The example converts them to
floating-point seconds for JSON display; library callers can retain the exact
fractions. A preceding keyframe or a frame at the exclusive end is never included.

## Set limits and handle failures

Defaults allow 120,000 ms per window, at most 16 JPEGs on a 640 by 360 canvas,
2 MiB per accepted JPEG, 2 MiB of process diagnostics, and 120 seconds for the
entire extraction including fallback. Aspect ratio is retained using black
padding. Width and height must be positive even integers. Increase
`maximum_window_millis` explicitly to sample a longer excerpt with the same frame
cap. Sparse samples over a long excerpt do not represent continuous 1 fps coverage.

The timeout starts after FFmpeg/ffprobe binary resolution and includes the time-base
probe. The existing media runner
kills and reaps the process on timeout or excessive diagnostics. Frame size is
checked after encoding. Source download limits and decoder memory limits belong
to the surrounding worker.

`SourceSamplingError` reports extraction failures without raw diagnostics. A
failed extraction removes its new output directory; completed earlier windows
in the example remain available. The caller owns successful local output and its
cleanup. These files do not establish resumable or durable run state.

## Include sampling in check identity

Checks that adopt this sampler should include `SOURCE_FRAME_SAMPLING_VERSION`,
their effective `SourceFrameSampling` settings, window bounds, source identity,
and selected FFmpeg build in their version or input contract. Changing selection
can change measured results. This is a new original-source API; existing
`Episode.frames()` and Build AI `FrameSampling` keep their declared frame-rate
behavior. No canonical transform version changes are needed to use these helpers.

## Nearest keyframes and unpadded evidence

`SourceSamplingMode.NEAREST_KEYFRAMES` selects unique codec keyframes nearest
caller-supplied `keyframe_positions`, expressed as fractions of the requested
window. Ties choose the earlier frame; results are chronological. Empty windows
remain empty and this mode has no uniform fallback. Position count must fit
`maximum_frames`.

Import `SourceFrameResize` from `hflow.source_sampling`. Set `resize=SourceFrameResize.FIT`
to fit within `width` and `height` without adding padding; the default `PAD` retains
the existing padded canvas. `scaling_algorithm` accepts `lanczos` (default) or
`bicubic`, and `jpeg_quality` accepts FFmpeg quality values 1 through 31 (default 5).
For example, a caller can choose positions `(0.15, 0.5, 0.85)`, a 960×960 fit box,
and JPEG quality 2. These are evidence settings, not a classification policy.

Nearest-keyframe probing shares the extraction deadline and is bounded by
`maximum_probe_bytes`. Returned timestamps retain the source's playback clock and
rational time base; subtract the window start explicitly for relative timestamps.
