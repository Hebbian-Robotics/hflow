# Write an input MCAP converter

A converter only needs to produce a source MCAP that the built-in transform can read. It does not need to produce a [canonical episode](../FORMAT.md): HFlow establishes that output contract during ingest.

## Required input contract

The source must be a valid MCAP file with a summary section. HFlow opens it with the stock Python `mcap` reader and reads channels from the summary; an unindexed or truncated file is rejected. Finishing the file with a conforming MCAP writer normally supplies the required summary. See [`PythonMcapEpisodeReader`](../../src/hflow/reader.py).

Non-camera channels have no HFlow-specific payload contract. Their topic, schema when present, message encoding, payload bytes, log time, and publish time pass through unchanged. A schema-less channel is allowed, as are multiple channels sharing one topic. The transform operates on channel IDs rather than assuming that a topic identifies one channel. See [`write_canonical_episode`](../../src/hflow/transform.py).

Camera handling is selected by exact schema name. A custom schema name that happens to contain images is treated as an ordinary pass-through channel, so it will not appear in `Episode.cameras` and camera checks will not inspect it. The accepted camera inputs are:

- `sensor_msgs/msg/CompressedImage` or `foxglove.CompressedImage`: the message encoding and schema must be decodable by the installed ROS 2 or protobuf MCAP decoder. Every message on one channel must use the same image format, and that format string must contain `jpeg`, `jpg`, or `png`. The payloads must contain valid images of that format. For a channel with at least two messages, the median difference between consecutive log times must be positive so HFlow can infer a frame rate. HFlow accepts a one-message channel using a 1 fps fallback and drops an empty camera channel with a warning. See [`_decode_compressed_images`](../../src/hflow/transform.py), [`_input_codec_for_image_format`](../../src/hflow/transform.py), and [`estimate_fps_from_log_times`](../../src/hflow/video.py).
- `foxglove.CompressedVideo` or `foxglove_msgs/msg/CompressedVideo`: the message encoding and schema must be decodable. Each message must declare `format="h264"` and contain one Annex B access unit. Every keyframe must include SPS and PPS, the first message on a channel must be a keyframe, and the stream must contain no B-frames. Pass-through video must also already match the configured fixed-GOP cadence: HFlow measures the channel frame rate from log times and requires keyframes exactly on the `gop_frames = max(1, round(gop_seconds × fps))` message grid. HFlow inserts a missing access-unit delimiter without re-encoding when the message can also be serialized by its protobuf or ROS 2 CDR encoder; it rejects other stream repairs because v1 does not transcode `CompressedVideo`. See [`_resolve_decoder`, `_resolve_encoder`, and `_validate_passthrough_video_payload`](../../src/hflow/transform.py).

Raw `sensor_msgs/msg/Image` and `foxglove.RawImage` channels are not supported. Record compressed JPEG or PNG images instead. Any derived topic configured by the pipeline must also be absent from the source because HFlow refuses to create a second channel with that topic; this is normally a pipeline-author concern rather than converter logic. These refusals are in [`write_canonical_episode`](../../src/hflow/transform.py).

Those are the HFlow-specific acceptance rules. Other malformed records or payloads can still be rejected by the MCAP decoder, the selected message decoder, or ffmpeg. HFlow does not require source provenance metadata, canonical chunk layout, globally sorted messages, or canonical identifiers before ingest.

When ingest cannot create an episode, `failure_kind` in the [`ingest_failures` table](../CATALOG.md) now tells you which side of the boundary it was: `source-unsupported` means the transform read the file fine but refused on its contents (the cases named above -- an unsupported image format, mixed image formats on one channel, a passthrough video violation, a raw image channel, a derived-topic collision), `source-unreadable` means the file was not a readable MCAP at all, and `infrastructure` is everything else the engine did not recognize. Still inspect `error_type` and `message` alongside `failure_kind` for the specific reason -- the kind tells you where to look, not what went wrong. The classification logic is in [`classify_ingest_failure`](../../src/hflow/ingest_ledger.py).

## What HFlow adds

The transform turns supported compressed-image channels into `foxglove.CompressedVideo` H.264, or validates and minimally repairs already encoded H.264. It globally orders output messages, preserves other channels, source metadata, and attachments, and replaces any source `provenance/v1` record with its own.

HFlow also assigns topics to `cameras`, `bulk`, and `state` chunk groups, derives per-group chunk targets from the episode's byte rates unless configured otherwise, and records both decisions in `provenance/v1`. It adds the canonical schema and pipeline versions, ffmpeg and GOP facts, transform provenance, and the identifiers derived from the canonical bytes. A converter should not pre-group or pre-chunk data to imitate this layout.

The [canonical format conformance rules](../FORMAT.md#conformance) describe this output. `hflow doctor` checks those rules, so a source file can be a valid converter output even when `doctor` reports that it is not canonical yet.

## Conventions that improve the result

Use stable, descriptive topic names such as `/joint_states`, `/action`, `/imu`, and camera-specific names, and keep units consistent with the schema. HFlow does not require those spellings for ingest, but pipeline-authored checks and downstream queries need predictable names.

Use nanosecond timestamps from one clock, keep log times increasing within each channel, and align streams that describe the same event. The six checks registered by default are `episode_duration`, `timestamp_regularity`, `camera_frame_stats`, `keyframe_interval`, `content_digest`, and `media_digest`. The duration and timestamp checks can inspect any populated topic; the three camera-specific checks only produce camera evidence for the recognized schema names above. Clean timestamps and correct camera schemas therefore determine how much baseline evidence an otherwise accepted file produces. See [`DEFAULT_CHECKS`](../../src/hflow/checks.py).

Put stable episode semantics such as task, operator, success, embodiment, and robot software version in an `episode/v1` metadata record when they are available. Record converter and source-dataset details under a separate name such as `source-provenance/v1`; do not write `provenance/v1`, because the transform owns and replaces that record.

## Follow the LeRobot example

HFlow's first-party
[`hflow.importers.lerobot`](../../src/hflow/importers/lerobot.py) implementation
demonstrates the boundary: it opens a standard MCAP writer, registers schemas
and channels, writes timestamped messages and source metadata, finishes the
source file, and then calls `hflow.write_canonical_episode`. That sequence is
the reusable converter pattern. The runnable
[`examples/lerobot/prepare.py`](../../examples/lerobot/prepare.py) wrapper calls
the same `hflow import lerobot` command users install with HFlow.

Its Hugging Face download, LeRobot v3 Parquet mapping, topic names, metadata
values, and video timestamp checks belong to that source format. It also
arrives with encoded video, so the importer prepares a pass-through H.264
stream that already meets the constraints above. A converter whose source
contains JPEG or PNG frames can write `CompressedImage` messages instead and
leave H.264 encoding, GOP structure, grouping, chunk sizing, and canonical
provenance to HFlow.
