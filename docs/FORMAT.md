# Canonical MCAP episode format

**Version 1**: a convention for training-ready robot episodes in standard [MCAP](https://mcap.dev/spec).

> **Project maturity:** until HFlow 1.0, canonical episodes are derived artifacts, not the source-of-record. HFlow may
> change their exact bytes or require regeneration from retained source recordings between
> releases. Version stamps make those rewrites visible; they are not a promise of byte stability.

A *canonical episode* is one MCAP file, one episode: cameras stored as in-band H.264 video, state streams preserved verbatim, semantics and version stamps carried in-file. It is what the HFlow transform writes and what the rest of the pipeline (checks, catalog, curation) consumes. This page is normative: a third party should be able to implement a conforming writer from it without reading HFlow's code.

Two load-bearing ideas shape the format: match GOP length to the video read
pattern, and group topics that are read together. The exact video schema,
metadata layout, codec, and numeric defaults are HFlow conventions with
reproducible measurements in the [benchmark report](./BENCHMARKS.md).

The single overriding rule: **a canonical episode is spec-conforming MCAP.** Every convention here constrains *how* the file is written, never *what format* it is. Any conforming MCAP reader reads these files unmodified.

## File container

- MCAP magic, `Header`, data section, `DataEnd`, summary section, `Footer`, closing magic, per the [MCAP spec](https://mcap.dev/spec).
- `Header.profile` is the empty string `""` (the file mixes protobuf video channels with pass-through channels of arbitrary encoding, so no single profile applies).
- `Header.library` is informational only (currently `hflow._grouped_mcap_writer`). Its exact value is not part of the format contract, and no reader may key behavior off it (see [Identifier rules](#identifier-rules)).
- Chunks are compressed with **zstd** by default (`"none"` is permitted). Each `Chunk` record carries `uncompressed_crc`; the `Footer` carries a summary CRC.
- The summary section repeats all `Schema` and `Channel` records and contains `Statistics`, all `ChunkIndex` records, `AttachmentIndex`/`MetadataIndex` records, and `SummaryOffset` records. A canonical episode always has a complete summary; unindexed files are not canonical.

## Topic-group chunking

Default MCAP writing gives each topic its own chunks, so one training sample costs a read per
topic. A canonical episode instead groups topics that share a read pattern into shared,
time-major chunk streams, so a sample costs one read per *group*.

**The rule**: every channel is assigned to exactly one named *group*. Messages from channels in different groups MUST NOT share a chunk. Within a group, messages are written time-major (ascending `log_time` across all of the group's channels interleaved), and the group's chunks form their own sequence with non-decreasing time ranges.

| Convention | Value |
|---|---|
| Default groups | `cameras` (topics whose schema is a camera schema; see below), `bulk` (non-camera topics averaging over 16 384 bytes per message: point clouds, occupancy grids), `state` (everything else) |
| Group override | Per-topic, at transform configuration time (`TransformConfig.topic_groups`); beats both defaults |
| Resolved layout | Recorded per topic in `provenance/v1` as `group/<topic>` |
| Target uncompressed chunk size | Derived **per group** as `rate x read window`, clamped to [800 KB, 8 MB]. Measured: 3.79 chunk fetches per training sample fetching 11.21 MB, against 9.18 fetching 6.65 MB at the flat 800 KB it replaced -- half the round trips for 1.7x the bytes, and not the optimum on that recording (docs/BENCHMARKS.md has the full table and says why the default stands anyway). `TransformConfig.chunk_size_bytes=<int>` pins one target for every group instead |
| Derived chunk targets | Recorded per group in `provenance/v1` as `chunk-target/<group>`, and only when derived (a pinned target is already in the config) |
| Camera schemas | `foxglove.CompressedVideo`, `foxglove.CompressedImage`, `foxglove.RawImage`, `foxglove_msgs/msg/CompressedVideo`, `sensor_msgs/msg/CompressedImage`, `sensor_msgs/msg/Image` |

Writer mechanics (currently implemented by HFlow's private `hflow._grouped_mcap_writer`
incubation package; equivalent layouts are conforming as long as the rule above holds):

- One chunk buffer per group. A message routes to its channel's group buffer; when a group's uncompressed buffer exceeds the chunk-size target, that group's chunk is finalized. At `finish()`, every group flushes its remaining buffer.
- Each `Chunk` record is followed immediately by its `MessageIndex` records (one per channel present in that chunk), and a `ChunkIndex` in the summary points at both. Consequence: the set of channel IDs in any chunk's `message_index_offsets` maps to exactly one group; this is the property conformance checks assert.
- `Schema` and `Channel` records are written *unchunked* in the data section at registration time and repeated in the summary (spec-legal; readers that only consult the summary see them either way).

Chunking changes write order, not the format.

## In-band video

The transform re-encodes per-frame JPEG cameras to in-band H.264 with GOP length matched to
the read pattern, without giving up native visualization. HFlow defines the
following in-file convention:

Camera streams are messages of [`foxglove.CompressedVideo`](https://docs.foxglove.dev/docs/sdk/schemas/compressed-video), protobuf-encoded:

- `Schema`: name `foxglove.CompressedVideo`, encoding `protobuf`, data = the serialized `FileDescriptorSet` for the message.
- `Channel`: `message_encoding` `protobuf`, topic = the source camera topic, unchanged.

| Field | Contents |
|---|---|
| `timestamp` | The frame's capture time; equals the MCAP message `log_time` |
| `frame_id` | The source image's `header.frame_id`; if absent, the topic name without its leading `/` |
| `data` | Exactly one H.264 access unit (one decodable frame), Annex B format |
| `format` | `"h264"` |

The H.264 bitstream constraints (all MUST):

1. **Annex B** byte stream (start-code delimited), one complete access unit per message.
2. Every access unit **begins with an access-unit delimiter** (AUD, NAL type 9); this is what makes message boundaries recoverable from the raw stream.
3. **SPS and PPS attached to every keyframe**: each IDR access unit contains its parameter-set NALs (x264 `repeat-headers=1`), so decoding can start at any keyframe message without out-of-band state.
4. **No B-frames** (`bframes=0`): decode order equals presentation order equals message order, and frame *i* of the source maps to access unit *i*.
5. **Fixed GOP** (`keyint = min-keyint`, `scenecut=0`): keyframes land exactly every `gop_frames` messages, so seek cost is uniform and predictable.
6. 4:2:0 chroma (`yuv420p`), the universally decodable baseline.

### GOP presets

GOP length is a storage-versus-seek tradeoff determined by how training reads
the video. The writer provides presets for common access patterns:

| Preset | Keyframe interval | Read pattern it serves |
|---|---|---|
| `vla` | 1.0 s | Short sparse windows (a keyframe seek per sample, so keep it cheap) |
| `world_model` | 6.0 s | Long contiguous sequences (keyframe cost amortizes, so favor compression) |

`gop_frames = max(1, round(gop_seconds × fps))`, with `fps` measured from the source stream (`1e9 / median(Δ log_time)`). The seconds values are configurable defaults, now with measured results in [the benchmark report](./BENCHMARKS.md); the provenance record (below) stamps what was used.

## State channels

Every non-camera channel passes through **byte-for-byte**: original topic name, original schema (name, encoding, data), original message encoding, original payload bytes, original `log_time`/`publish_time`. A canonical episode is therefore a *superset* format: whatever the robot recorded (ROS 2 CDR, protobuf, JSON) is still exactly there, in `state`-group chunks. Messages across all channels are written in global `log_time` order (the writer distributes them to per-group chunk streams).

Raw uncompressed image channels (`sensor_msgs/msg/Image`, `foxglove.RawImage`) are not accepted by the v1 transform; record compressed images upstream.

## Metadata records

Version stamps and episode semantics live in MCAP `Metadata` records, keeping the episode
self-contained, viewer-inspectable, and traceable to the schema, pipeline, and
robot software that produced it. All values are strings.

### `episode/v1`: what this episode is

| Key | Meaning |
|---|---|
| `task` | What the robot was doing (e.g. `fold_napkin`) |
| `operator` | Who/what collected it |
| `success` | Collector-labeled outcome, `"true"`/`"false"` |
| `embodiment` | Robot/platform identifier |
| `robot_software_version` | Software running on the robot at record time |
| *(any user key)* | Additional semantics pass through untouched |

All keys are optional; the record is copied/merged from the source recording. Recorders are encouraged to write it at collection time.

### `provenance/v1`: what produced this file

| Key | Meaning |
|---|---|
| `schema_version` | This format's version: `"1"` |
| `pipeline_version` | Content hash (12 hex chars, SHA-256 prefix) of the producing transform configuration; two files with equal `pipeline_version` were produced by identical configuration |
| `ffmpeg_version` | Version line of the encoding instrument |
| `gop_preset` | `vla` or `world_model` |
| `gop_seconds` | The keyframe interval actually used |
| `source_uri` | Where the source recording came from (optional) |

During a rewrite, the derived episode replaces `provenance/v1`; all other source `Metadata`
records are copied through unchanged. A corpus may be mixed-version while regeneration is in
progress, so curation pins or excludes exact versions rather than assuming an atomic migration.

## Attachments

MCAP `Attachment` records carry episode-scoped files that are not message streams: calibration files, URDFs. The transform copies all source attachments through (name, media type, timestamps, bytes).

## Identifier rules

Stored-data identifiers (metadata record names, their keys, group names) are **neutral and format-versioned**: they never embed a project name, and a breaking change to a record's meaning bumps its `/v1` suffix rather than mutating existing files. Topic names are the source recording's own. The only place a project name may appear is the informational `Header.library` field, which no reader may depend on.

## Conformance

These rules describe the canonical output written by HFlow. A converter should follow the separate [input MCAP contract](./how-to/write-a-converter.md) instead of implementing these guarantees before ingest.

A file claiming this convention must satisfy, in increasing strictness:

1. **It is valid MCAP**: readable by the stock [`mcap` package](https://mcap.dev/docs/python/) with CRC validation on, with a complete summary section.
2. **It opens in the standard viewers**: [Foxglove](https://foxglove.dev/) and [Rerun](https://rerun.io/). Opening in both is the acceptance test for every file the pipeline writes.
3. **Chunk purity**: for every `ChunkIndex`, the channel IDs in `message_index_offsets` belong to exactly one group; each group's chunk sequence is time-ordered.
4. **Video constraints**: every `foxglove.CompressedVideo` message is one Annex B access unit beginning with an AUD; every IDR access unit contains SPS and PPS; no B-frames; `format="h264"`.
5. **Stamps**: `provenance/v1` present with `schema_version` and `pipeline_version`.

The no-B-frame constraint is enforced by refusal at two boundaries, because a `-c:v copy` remux
to MP4 cannot represent the reorder tail: the trailing B frames are silently undecodable from
the muxed file (measured in [#250](https://github.com/Hebbian-Robotics/hflow/issues/250): 303
samples in, 301 decoded). The transform refuses pass-through video that carries B-frames, and
the MP4 remux behind `Episode.video` refuses any B-frame payload outright, with an error naming
the observed reorder depth and the frames at risk. `hflow doctor` reports any B picture it
classifies in a video message as the `video-b-picture` error finding (below), so a clean
report now covers this constraint alongside the refusals.

`hflow doctor <file.mcap> [more.mcap ...]` checks every file given and prints
one report each in argument order; its aggregate result follows the
[exit code rules](#exit-codes). It validates the container, summary, indexes,
stamps, chunk purity (against the file's own group map, or a video-versus-state
approximation when it has none), per-topic time order, per-group chunk time
order, and the H.264 access-unit properties listed below. It classifies H.264
picture coding types to detect B-frames as the `video-b-picture` error finding.
A payload whose slice headers cannot be parsed still cannot be classified, and
is reported as `video-invalid-slice-header` instead. The doctor does not reject
non-VCL NAL units before the first AUD, so a clean report does not prove the
canonical AUD-first constraint. An unreadable or unparseable path is reported
in place, in the same per-file shape (`[error] unreadable: ...`), and the run
continues with the remaining files.

### Doctor finding codes

Finding codes are stable, kebab-case identifiers suitable for matching in
automation. An `error` breaks the canonical convention (or the MCAP spec); a
`warning` is legal but differs from the layout HFlow writes by default.

| Code | Level | Reported when |
|---|---|---|
| `unreadable` | error | The path cannot be opened or parsed as an MCAP file. |
| `no-summary` | error | The file has no summary section. |
| `no-statistics` | error | The summary has no `Statistics` record. |
| `no-chunk-indexes` | error | The summary has no `ChunkIndex` records. |
| `chunk-missing-message-indexes` | error | A chunk index has no per-channel `MessageIndex` offsets. |
| `chunk-mixes-groups` | warning | One chunk contains channels assigned to different groups; custom grouping can make this intentional. |
| `group-chunks-out-of-order` | warning | A group's chunks are not ascending by `message_start_time`. |
| `chunk-mixes-video-and-state` | warning | One chunk contains both video and state channels, reported only when the file has no group map; custom grouping can make this intentional. |
| `missing-provenance` | error | The `provenance/v1` metadata record is absent. |
| `provenance-missing-key` | error | `provenance/v1` lacks `schema_version` or `pipeline_version`. |
| `missing-episode-record` | warning | The optional `episode/v1` semantics record is absent. |
| `topic-time-order` | error | A channel's `log_time` decreases between messages. |
| `video-format` | error | A supported video message does not declare `format="h264"`. |
| `video-invalid-slice-header` | error | The H.264 payload's picture count cannot be determined from its slice headers. |
| `video-b-picture` | error | A video message's slice headers classify at least one picture as a B picture; canonical video requires no B-frames. |
| `video-multiple-access-units` | error | A video message contains more than one picture or access unit. |
| `video-not-aud-delimited` | error | No AUD is present, or VCL data precedes the first AUD. |
| `video-keyframe-missing-parameter-sets` | error | A keyframe does not carry both SPS and PPS. |
| `video-stream-starts-mid-gop` | error | A video channel's first message is not a keyframe. |
| `read-failed` | error | Full message reading, CRC validation, or video decoding fails. |

At most three findings with the same code are printed. The report then gives
the number of further occurrences suppressed for that code.

### Exit codes

Every `hflow` command follows the same three-value convention:

- `0` - success: the command did what it was asked.
- `1` - ran, and found something to report: `doctor`, a non-conforming file;
  `stale --exit-code`, stale episodes found; `up`, started then failed, so
  containers may still be running; `ingest`, episodes failed.
- `2` - bad input, nothing useful happened: an invalid flag combination, an
  unreachable endpoint, a missing catalog, a `serve` launch that never started
  (a port outside 1-65535, a data root that exists and is not a directory, a
  host with no free port), or a `doctor` run in which no file could be
  diagnosed at all. A data root that is not there yet is not bad input: it is a
  workspace nothing has ingested into, and `serve` and `up` both accept one.

`doctor` treats an unreadable file as at least as bad as a non-conforming one
(exit 1), and reserves 2 for runs where nothing was diagnosed, so a batch run
never both loses its reports and erases the fact that it read anything.

## References

- [MCAP specification](https://mcap.dev/spec) and [Python libraries](https://mcap.dev/docs/python/)
- [foxglove.CompressedVideo schema](https://docs.foxglove.dev/docs/sdk/schemas/compressed-video)
- [Architecture](./ARCHITECTURE.md): the full design this format serves
- [How HFlow fits the robotics data stack](./INTEGRATIONS.md)
