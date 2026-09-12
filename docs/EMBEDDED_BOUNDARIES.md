# Embedded integration boundaries

These APIs support local services, external schedulers, and workers that own
their temporary storage. They do not schedule jobs, retain run state, or infer
customer quality policy. See [embedded workers](./how-to/run-embedded-workers.md)
for processing with `App.process_many(record=False)`.

## Video preparation

`hflow.media.probe_video(path, limits=VideoLimits(...))` returns one of:

- `VideoProperties`: positive dimensions, finite positive frame rate and duration.
  Duration uses decimal arithmetic; `duration_millis` rounds up for coverage plans.
- `UnreadableVideo`: recognized input decode failure or malformed media properties.
- `UnsupportedVideo`: properties exceed the caller's supported limits.

Duration comes from the first video stream, its `DURATION` tag, or the container
when it contains exactly one stream. Longer audio never extends video coverage.
Limits include input frame pixels, frame rate, duration, command timeout, and
combined probe/diagnostic output bytes. Limits are per command, not a deadline
for an entire multi-command pipeline. Filesystem failures, missing tools,
timeouts, output limits, and ambiguous tool failures propagate as operational
errors. Bounded diagnostic text is used internally to recognize input decode
failures; it is never included in public error messages.

`prepare_video_window(source, output, VideoWindow(...), limits=...)` produces
`PreparedVideoWindow` or a rejection outcome. It selects the inspected first
video stream, seeks with millisecond precision, resamples using FFmpeg's `fps`
filter, and encodes H.264 with the `veryfast` preset. The caller owns its output.
Existing output paths are refused and unsuccessful work leaves no partial output.

`hflow.importers.prepare_video_episode(source, output, config, limits=...)`
returns `ImportedVideoEpisode` or a rejection outcome. It shares the importer
implementation and preserves `VideoImportConfig`'s covering-frame sampling grid.
The existing `import_video_episode` API returns a path and raises for rejected
media. Both import APIs use the shared bounded probe and command execution.
Window encoding and episode import have distinct sampling contracts; bypassing
window encoding in an existing integration requires checking metric equivalence.

## Immutable source transfers

`hflow.sources.download_source(store, expectation, destination, limits=...)`
uses a caller-configured obstore store, available through `hflow[bucket]`.
A custom `object_getter` may implement `SourceObjectGetter` instead; that path
does not require obstore. Core never discovers credentials or writes to the store.

`SourceExpectation` contains a `SourceRevision(object_name, version)` and optional
expected size and lowercase SHA-256. A planner may not know the hash yet; a worker
can require the planner's verified hash. The transfer verifies returned revision,
object key when supplied by the provider, declared and streamed size, and any
expected hash before atomically replacing the destination. Failures preserve an
existing destination and remove temporary files.

`DownloadedSource` retains the verified revision, byte count, and hash. It describes
the completed transfer; callers remain responsible for subsequent file mutation
and lifetime. `SourceReadError` has a generic public message; its exception cause
is private provider context and must not be serialized into external results.
The elapsed budget is checked between chunks. Configure provider request/read
timeouts as well, since a synchronous provider can block inside a read.

## Evidence and result projection

`CameraQualityEvidence.from_measurements(measurements, camera_topic)` parses the
built-in black/highlight/shadow percentages and freeze duration into typed fields.
`measurement_names(camera_topic)` defines the corresponding measurement selection.
These measurements describe the processed stream; the caller defines coverage,
assessment duration, and acceptance thresholds. `finite_measurement` rejects
booleans, nonfinite values, and values outside an explicit range.

`hflow.results.project_results(report, selections)` produces an immutable
`ResultProjection`. Each `CheckSelection` names an internal check, a public alias,
and a mapping from public measurement names to internal keys. There is no implicit
export-all mode. Missing requested checks or measurements raise instead of silently
omitting evidence. Callers remain responsible for the contents of selected values.

Successful execution produces `MeasuredCheck` with scalar measurements and an
independent optional quality verdict. Other outcomes are `UnavailableCheck` with
`error`, `skipped`, or `superseded`. Paths, registration objects, diagnostic text,
tags, and unselected measurements are absent. `to_payload()` returns schema version
1; `to_json(maximum_bytes=...)` rejects an oversized UTF-8 result. The limit applies
to serialized output, not to the memory already occupied by the input report.
Transport adapters own parsing their incoming wire contracts and their disclosure
and payload-size policies.

## Offline tool verification

`hflow.ffmpeg.verify_media_binary(path, MediaBinarySpec(...))` verifies SHA-256
before running a bounded version command and returns `VerifiedMediaBinary`.
It resolves the path before both operations and never downloads binaries.
`PINNED_LINUX_X86_64_FFMPEG` and `PINNED_LINUX_X86_64_FFPROBE` describe the retained
executables from the pinned Linux x86-64 distribution. Other builds supply their
reviewed specifications. Callers own directory permissions and protection against
concurrent modification; verification is not a filesystem sandbox.
