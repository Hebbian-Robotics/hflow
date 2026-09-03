# HFlow storage, read, and camera-check benchmarks

These reproducible benchmarks measure HFlow's in-band video compression,
topic-group chunking, and cold camera-evidence throughput. They report what the
current implementation achieves at honest small scale alongside the
million-hour results Dyna published in
[Training Dyna-2 at million-hour scale, repeatably](https://www.dyna.co/research/dyna-2-infrastructure)
(Figure 3). Every number below comes from a real run of the scripts in
[`benchmarks/`](../benchmarks); nothing is extrapolated.

## Results at a glance

| Workload | Measured result | Why it matters |
| --- | --- | --- |
| Six-camera real footage | **48-50.5% less video payload** than the source JPEG payloads | Quantifies the storage effect without relying on synthetic test patterns |
| One-camera 1080p30 evidence pass | **4.90 s median** to inspect every frame of a 30 s episode | Establishes the cold `camera_frame_stats` cost without mixing in transform or cache time |
| Four-camera synthetic training windows | **2.42x fewer chunk fetches** than per-topic chunking | Shows how grouping topics by read pattern reduces sample assembly work |
| Six-camera real footage with 8 MB chunks | **2.81x fewer fetches and 3.21x fewer bytes fetched** than per-topic chunking | Demonstrates that chunk size and grouping policy must be tuned together |
| Selective state scans | Naive schema grouping fetched **230 MB** for a 0.2 MB `/imu` stream | Shows why HFlow exposes per-topic group overrides instead of treating grouping as a fixed schema rule |
| Three-camera manipulation footage | A 400 KB candidate improves training fetches x MB by **15.3%**, but raises state-scan fetches from **7 to 13** and regresses the nuScenes cross-check | Measures the 800 KB derived-target floor on the camera + compact state/proprio workload it is meant to serve |

These are measurements of specific workloads, not universal performance
claims. The methodology, caveats, input recordings, and reproduction commands
are included below.

**How to rerun** (each script prints these tables; `--quick` for a fast pass;
`--input <recording.mcap>` measures a real recording; see the real-footage
sections for the reference datasets):

```bash
uv run python benchmarks/storage_benchmark.py
uv run python benchmarks/read_benchmark.py
uv run python benchmarks/camera_frame_stats_benchmark.py
uv run python benchmarks/camera_frame_stats_benchmark.py --profile-filters
uv run python benchmarks/storage_benchmark.py --input nuscenes-mini-sample.mcap
uv run python benchmarks/read_benchmark.py --input nuscenes-mini-sample.mcap \
    --grouping read-pattern --chunk-size-bytes 8000000
uv run python benchmarks/read_benchmark.py --input robotis-button-push-107.mcap \
    --grouping schema-default --chunk-size-bytes derived
```

**Method caveats, up front.**

- Three content sources: the synthetic fixture
  (`hflow.testing.synthesize_episode`, ffmpeg test patterns at 320x240,
  defect injections disabled), a real six-camera driving recording, and a
  real three-camera manipulation recording. Synthetic patterns are
  unrealistically compressible; the two storage results quantify exactly how
  much.
- Reads run against local disk, where a "fetch" costs a chunk decompression,
  not an object-storage round trip. Wall-clock ratios here therefore
  *understate* the benefit that matters on S3/GCS; the fetch and
  bytes-fetched counts are layout facts and transfer directly.
- Scale: 11.6-60 s episodes, not the forty-three million of Dyna's corpus. The
  point is that the mechanisms behave as Dyna's article describes, not that
  the ratios match.
- Camera-check wall time is machine- and content-dependent. Its benchmark uses
  a fresh `Episode` workdir for every repetition, so neither the remuxed MP4 nor
  the persistent FFmpeg instrument cache can turn a cold run into a cache hit.
- Storage reductions compare **video payload bytes** (the codec effect, the
  number comparable to Dyna's ~68%). File sizes are shown alongside but carry
  every non-camera channel passed through byte-for-byte; on a lidar-heavy
  recording those dwarf the cameras and would swamp a file-level comparison.

## Camera evidence: cold per-frame throughput (issue #365)

`camera_frame_stats` measures blackout, freeze, luma, frame-difference,
temporal-outlier, and broadcast-range evidence for every decoded frame. The
implementation already shares one FFmpeg filter graph between those
measurements and caches its instrument output, so this benchmark measures the
remaining first-run cost rather than reintroducing the repeated decode removed
by #175.

The script first transforms one deterministic synthetic episode, reports that
time separately, then runs the check three times with a new workdir each time:

```bash
uv run python benchmarks/camera_frame_stats_benchmark.py
# --quick uses a 3 s, 320x180 development fixture
```

Measured on a Ryzen 9 8945H (8 cores/16 threads), 32 GB RAM, Linux x86_64, and
FFmpeg `n8.1.2-50-g1a748fe2cd-20260901` at commit `050d145`:

| phase | wall-clock | decoded frames | instrument cache |
|---|---:|---:|---:|
| transform to canonical | 3.510 s | - | - |
| cold `camera_frame_stats` #1 | 4.897 s | 900 | 840.3 KB |
| cold `camera_frame_stats` #2 | 4.940 s | 900 | 840.3 KB |
| cold `camera_frame_stats` #3 | 4.797 s | 900 | 840.3 KB |

The cold median is **4.897 s per camera** for 30 seconds of 1920x1080 video at
30 FPS. All three runs decoded all 900 frames. This synthetic result matches
the shape of the separately reported real Egocentric-10K control in #365:
seconds per camera, stable across cold repetitions, and independent of whether
the canonical H.264 arrived directly or through a JPEG transform.

### Where the time goes

`--profile-filters` runs one-variable FFmpeg controls against the same
canonical MP4 to isolate the cost:

| filter path | wall-clock |
|---|---:|
| decode only | 0.422 s |
| decode + `format=yuv420p` | 0.518 s |
| decode + `blackframe` | 0.537 s |
| decode + `freezedetect` | 0.508 s |
| decode + `signalstats=stat=tout+brng` | 4.536 s |
| complete shipped graph | 4.711 s |

`signalstats` is the bottleneck, not H.264 decoding. The additional controls
recorded in #365 found no repeatable gain from overriding FFmpeg's automatic
filter threading, splitting the graph into parallel branches, or using NVDEC
before the software-only evidence filters. A luma-only input is not equivalent
either: FFmpeg's `BRNG` statistic deliberately evaluates the Y, U, and V
planes, so removing chroma would silently weaken the recorded evidence.

**Conclusion: keep the measurement path unchanged.** The tested changes do not
produce a repeatable win without changing evidence or adding complexity that
outweighs noise-level movement. The benchmark is the durable outcome: future
FFmpeg releases or alternative instruments now have a reproducible baseline
and must beat it while preserving every per-frame result.

## Storage: per-frame JPEG vs canonical MCAP by GOP preset (issue #26)

HFlow's transform re-encodes per-frame JPEG into in-band H.264 with GOP
length matched to the read pattern. Dyna's article reports that the same
move (from H5 holding per-frame JPEG) cut their storage ~68%.

**Measured** (30 s synthetic episode, 2 cameras @ 15 Hz, 320x240,
zstd chunks; baseline = the sum of the JPEG payload bytes, i.e. the storage
floor of any per-frame-JPEG layout; reduction = canonical video payload vs
that baseline):

| layout | file size | video payload | video reduction vs JPEG | transform time |
|---|---|---|---|---|
| per-frame JPEG payloads (baseline) | 6.11 MB | 6.11 MB | - | - |
| source MCAP (JPEG in-band, as recorded) | 4.28 MB | 6.11 MB | - | - |
| canonical MCAP, `gop_preset=vla` (1 s GOP) | 1.41 MB | 1.16 MB | **80.9%** | 0.7 s |
| canonical MCAP, `gop_preset=world_model` (6 s GOP) | 1.29 MB | 1.01 MB | **83.5%** | 0.8 s |

Observations:

- The reduction direction and magnitude match the claim; the ~13-15 points
  over Dyna's 68% are synthetic-content flattery, quantified against real
  footage below (48-51% on a real recording).
- Longer GOPs buy a few points more on top of short GOPs (fewer keyframes):
  the storage side of the storage-vs-seek trade the presets encode.
- The source MCAP row is itself smaller than its own JPEG payloads (4.28 vs
  6.11 MB): zstd found redundancy *across* the synthetic JPEGs inside each
  chunk. Real footage does not do that (see below: 512 MB file for 200 MB of
  JPEG payloads); it is a fixture artifact, reported because hiding it would
  overstate the H.264 gain.

## Reads: chunk fetches and throughput by chunk layout (issue #27)

Default MCAP writing gives each topic its own chunks, so one training sample
costs a read per topic. HFlow's writer instead lays out topic *groups*
time-major (cameras in one chunk stream, proprioception+actions in another),
so a sample costs one read per group; Dyna's article reports ~3.4x fewer
chunk fetches and ~2.9x faster reads from the same layout change at their
scale.

**Measured**: three layouts holding identical messages (60 s episode,
4 cameras @ 15 Hz + `/joint_states` @ 100 Hz, 800 KB chunks, same stock
reader): **per-topic** (Dyna's baseline), **interleaved** (the stock
Python writer's single chunk builder, what most tooling writes today), and
**topic-group** (ours).

Training-style samples (200 seeded 1 s windows, each reading all 4 cameras +
state):

| layout | fetches/sample | compressed MB fetched/sample | wall-clock (local) |
|---|---|---|---|
| per-topic (Dyna's baseline) | 5.04 | 0.963 | 2251 ms |
| interleaved (stock python writer) | 1.11 | 0.537 | 821 ms |
| topic-group (ours) | **2.08** | 0.830 | 1508 ms |

State-only scan (the full `/joint_states` stream, a curation/QC access
pattern, x10):

| layout | fetches/scan | compressed MB fetched/scan | wall-clock (local) |
|---|---|---|---|
| per-topic (Dyna's baseline) | 3.00 | 0.437 | 310 ms |
| interleaved (stock python writer) | 6.00 | 2.889 | 363 ms |
| topic-group (ours) | **3.00** | **0.437** | 302 ms |

Observations:

- Against Dyna's per-topic baseline, topic-group chunking gives **2.42x
  fewer fetches** and 1.49x faster local reads on training samples. The
  arithmetic ceiling at 4 cameras + 1 state topic is (4+1)/2 = 2.5x, and the
  measurement sits on it; Dyna's 3.4x implies more topics per sample at
  their scale, exactly their article's "adding a camera no longer adds a
  round trip".
- The interleaved layout is *best* on full multi-view samples (every byte in
  its chunks is needed) but pays 6.6x the bytes and 2x the fetches the
  moment a read is selective: the state-only scan drags the entire video
  stream through the reader. Topic-group is the only layout that is near-best
  under **both** access patterns, which is the actual claim under test:
  read layouts are tuned per consumer, and a corpus serves more than one
  consumer.
- Local wall-clock tracks bytes-decompressed, not round trips; on object
  storage the fetch counts dominate and the gap widens toward Dyna's
  numbers.

## Real footage: nuScenes mini scene

**Dataset**: one nuScenes v1.0-mini scene converted to MCAP by
`nuscenes2mcap`, hosted by the
[Lichtblick](https://github.com/lichtblick-suite/lichtblick) project.
19.2 s, six 1600x900 cameras at ~11.5 Hz as `foxglove.CompressedImage` JPEG
(199.94 MB of image payloads), plus lidar, five radars, `/imu` (~99 Hz),
odometry, TF, grids, and 22k ROS 1 diagnostics messages, 512 MB in all.
Not redistributed here; fetch it yourself:

```bash
curl -fLo nuscenes-mini-sample.mcap \
    https://mcap-proxy.lichtblick.workers.dev/NuScenes-v1.0-mini-scene-sample.mcap
# sha256 089b5be708d2536a8c7e341f4d2b62dc618a581525e4db8d15c93ec5ce446547
```

License CC BY-NC-SA 4.0 (non-commercial research). *Adapted from nuScenes
dataset. Copyright 2020 nuScenes.*

### Storage, real footage

| layout | file size | video payload | video reduction vs JPEG | transform time |
|---|---|---|---|---|
| per-frame JPEG payloads (baseline) | 199.94 MB | 199.94 MB | - | - |
| source MCAP (JPEG in-band, as recorded) | 512.26 MB | 199.94 MB | - | - |
| canonical MCAP, `gop_preset=vla` (1 s GOP) | 349.84 MB | 103.90 MB | **48.0%** | 18.9 s |
| canonical MCAP, `gop_preset=world_model` (6 s GOP) | 344.89 MB | 98.96 MB | **50.5%** | 22.6 s |

Real footage lands at **48-51% video reduction** against Dyna's ~68%. The gap
has honest explanations rather than excuses: these cameras run at ~11.5 Hz
(far less temporal redundancy than a 30 Hz wrist camera), the source JPEGs
are already well compressed (~150 KB per 1600x900 frame), and the encode uses
the default `crf=23` with no per-deployment tuning. Manipulation-robot
footage (static scenes, high frame rates) sits between this and the
synthetic upper bound. Transform throughput on this box: the full 512 MB
recording (decode 1,318 JPEGs, encode six H.264 streams, pass 312 MB of
sensor data through) takes ~19-23 s per preset.

### Reads, real footage, and a finding about grouping

The first run applied the **schema-default grouping** (cameras vs
everything-else) blindly, and it made reads *worse* than per-topic: this
recording's "everything else" is ~300 MB of point clouds and diagnostics, so
`/imu` shared chunks with lidar and a full `/imu` scan dragged **230 MB**
through the reader (vs 0.2 MB per-topic), while training samples fetched
37.5 chunks/sample vs per-topic's 12.9. The lesson is the instruction in
Dyna's article read carefully: group topics that **share a read pattern**.
Grouping is not a schema decision, and `TransformConfig.topic_groups` exists
precisely to say so. The benchmark's `--grouping read-pattern` mode assigns
bulk modalities (mean message size > 16 KB) to their own group; the default
stays right for camera+proprio manipulation recordings.

With read-pattern grouping, training samples (200 seeded 1 s windows, all six
cameras + `/imu`):

| chunk size | layout | fetches/sample | compressed MB fetched/sample |
|---|---|---|---|
| 800 KB flat | per-topic (Dyna's baseline) | 12.93 | 10.14 |
| 800 KB flat | topic-group (read-pattern) | **9.18** | **6.65** |
| 8 MB flat | per-topic (Dyna's baseline) | 7.56 | 49.26 |
| 8 MB flat | topic-group (read-pattern) | **2.69** | **15.34** |
| 5.41 MB flat | topic-group (read-pattern) | **3.08** | **12.32** |
| **derived per group** (the default) | topic-group (read-pattern) | **3.79** | **11.21** |

### Per-group derived targets: what they buy, and what they do not

Chunk size stopped being a flat tuned constant. Each group's target is derived
from the rate *that group* is written at and the read window its workload
implies (`hflow.format.derived_chunk_size_bytes`): a read covering `W` seconds
of a group written at `R` costs about `1 + R*W/C` fetches and `C + R*W` bytes,
so with `x = C/(R*W)` their product is `R*W*(2 + x + 1/x)` -- minimized exactly
at `C = R*W`. For this recording's camera group that is 5.4 MB/s x 1 s; every
other group here falls below the 800 KB floor and takes it.

Measured against the flat targets on the product they trade off:

| chunk size | fetches x MB |
|---|---|
| 800 KB flat | 61.0 |
| 8 MB flat | 41.3 |
| **derived per group** (the default) | **42.5** |
| 5.41 MB flat | 38.0 |

Read that honestly: **the derived policy is not the minimum here.** It roughly
halves the fetches of the 800 KB default it replaced (3.79 against 9.18) at
1.7x the bytes, which is the trade worth making when round trips dominate --
and it fetches the fewest bytes of any layout that gets under four fetches per
sample. But a flat 5.41 MB measured better than it on both the product and the
fetch count, because a flat target also lifts this recording's *state* group
off the floor, and the training sample reads a state topic too. The same shows
up in the full `/imu` scan: 16 fetches derived against 3 at flat 5.41 MB, for
the same 4 MB.

It is still the default, for a reason that is about defaults rather than about
this table: 5.41 MB is this recording's camera rate, not a principle. Shipping
it as a constant would fit nuScenes and mis-fit the next corpus, which is the
mistake the flat 800 KB default already made. The formula transfers; the
number does not. If your workload is one recording you can measure, pin
`TransformConfig(chunk_size_bytes=...)` and beat the default -- that is what
the knob is for.

The floor now has a second real measurement on a manipulation workload with a
proprio-rate state stream. That result points lower than 800 KB on the
fetches-times-bytes balance, while the nuScenes state group points higher; the
same-run cross-check below shows that moving the global floor to 400 KB would
trade one corpus's result for the other's rather than establish a better
default.

Reproduce the default with `--chunk-size-bytes derived` against the same
input; the flat rows reproduce at `800000`, `5411000` and `8000000`. A flat
`5411000` is deliberately NOT the same layout as `derived`, which is what made
an earlier version of this table wrong.

Observations:

- At a flat 800 KB, six 1600x900 cameras produce ~5.4 MB/s, so a 1 s window
  spans ~7 camera chunks by sheer byte volume: fetch counts are byte-bound,
  not layout-bound, and grouping buys only 1.41x. Chunk size is a tuned
  parameter (docs/ARCHITECTURE.md); at 8 MB the layout effect dominates:
  **2.81x fewer fetches and 3.21x fewer bytes** than the per-topic baseline,
  against Dyna's ~3.4x with more topics per sample. The shipped derived
  default lands at **3.41x fewer fetches** than per-topic -- the best fetch
  ratio in this report, and the closest to Dyna's -- while fetching 0.90x its
  bytes.
- The `/imu` scan under read-pattern grouping fetches 4 MB (vs 230 MB
  naive-grouped, 322-344 MB interleaved); per-topic remains the optimum for
  single-topic scans, as always.
- Local wall-clock *inverts* here (grouped reads slower on NVMe despite
  fewer fetches and bytes): the pure-Python reader decompresses whole chunks,
  and big mixed chunks cost more per touch locally. On object storage the
  round trips these tables count are the dominant term; the wall-clock
  column is the least transferable number in this report, on real data as on
  synthetic.

## Manipulation footage for the chunk floor

**Dataset**: episode 107 (`107/107_0.mcap`) from
`RobotisAI/evButtonPush-260615-0-MCAP`, a public button-push manipulation
corpus hosted on Hugging Face. The pinned MCAP is 75,946,038 bytes (75.9 MB)
and records 11.6167 s with three compressed camera streams. Its schema-default
state group contains `/arm/tactile_broadcaster/gpio_states`, `/joint_states`,
`/leader/joint_states`, and `/tf`: 4,769,296 payload bytes, or **410.6 kB/s**
over the episode. That is the low-rate camera + state/proprio workload missing
from the original floor measurement, and directly satisfies #170's requirement
that the state group be in the kilobytes-per-second range rather than
megabytes-per-second.

The recording is not redistributed here. Fetch the exact bytes used below:

```bash
curl -fLo robotis-button-push-107.mcap \
    'https://huggingface.co/datasets/RobotisAI/evButtonPush-260615-0-MCAP/resolve/93a0ba73c6b82689696d9c4909b3b48af0294cf8/107/107_0.mcap?download=true'
# sha256 51bec431162844b2bb239f65ea158ea1030bb3fce5312ad81768978dc68f82c3
```

Repository revision:
`93a0ba73c6b82689696d9c4909b3b48af0294cf8`.

To reproduce the floor sweep, rerun the manipulation benchmark above with
`--chunk-size-bytes` set in turn to `200000`, `400000`, `800000`, `1600000`,
`3200000`, and `derived`.

### Floor sweep

The sweep varies only the flat chunk target. The input episode, VLA GOP preset
(1 s), schema-default grouping (cameras vs state), 200 seeded 1 s training
windows, and state-only scan are held fixed. The derived row uses the shipped
800 KB floor; on this recording it produces the same grouped layout as the
flat 800 KB row.

| chunk target | topic-group fetches/sample | compressed MB/sample | fetches x MB |
|---|---:|---:|---:|
| 200 KB flat | 5.58 | 0.550 | 3.07 |
| 400 KB flat | 3.80 | 0.729 | **2.77** |
| 800 KB flat | 2.88 | 1.128 | 3.25 |
| 1.6 MB flat | 2.48 | 1.937 | 4.80 |
| 3.2 MB flat | 2.23 | 3.360 | 7.49 |
| **derived per group** (800 KB floor) | **2.88** | **1.128** | **3.25** |

The 400 KB point is the best measured flat target on the balanced
fetches-times-bytes product, but it is not a free improvement: training fetches
rise from 2.88 to 3.80 and the full `/joint_states` scan rises from 7 to 13
chunk fetches. Because round trips are the expensive term on object storage,
that trade needs a cross-workload check before it can become a default.

### Same-run floor comparison

To isolate the floor itself, both recordings were rerun in one GitHub Actions
Ubuntu 24.04 job with Python 3.14, the same source commit, VLA GOP preset, 200
seeded 1 s windows, and unchanged grouping for each corpus. The candidate run
changed only `MINIMUM_DERIVED_CHUNK_SIZE_BYTES`: shipped 800 KB versus 400 KB.
nuScenes keeps its documented read-pattern grouping; the manipulation episode
keeps schema-default camera + state grouping.

| corpus | derived floor | fetches/sample | compressed MB/sample | fetches x MB | state-scan fetches |
|---|---:|---:|---:|---:|---:|
| manipulation | **800 KB (shipped)** | **2.88** | 1.128 | 3.25 | **7** |
| manipulation | 400 KB candidate | 3.77 | **0.730** | **2.75** | 13 |
| nuScenes | **800 KB (shipped)** | **3.77** | 11.169 | **42.11** | **16** |
| nuScenes | 400 KB candidate | 4.01 | **11.101** | 44.52 | 21 |

**Conclusion: keep the 800 KB floor.** Lowering it to 400 KB improves the
balanced training product by about 15.3% on this manipulation episode, but
increases its training round trips by about 31% and nearly doubles selective
state-scan fetches. The same change also regresses both the balanced product
and fetch counts on the nuScenes cross-check. Neither fixed value dominates
across the two measured corpus shapes and access patterns.

Replacing the floor with a new adaptive rule would need another measured
signal beyond the group byte rate and read window already used by the derived
formula. Two recordings do not justify inventing that signal. Keeping 800 KB
preserves the current behavior and leaves `TransformConfig(chunk_size_bytes=...)`
as the explicit tuning path for a corpus whose workload has been measured.
`TRANSFORM_BEHAVIOR_VERSION` therefore does not change.

## Provenance

The synthetic tables and the original nuScenes tables were recorded on the
development box (local NVMe, Linux) and were included in the initial public
repository snapshot (`6ab6ef9`). The issue #170 manipulation sweep and the
same-run 800 KB versus 400 KB comparison were measured on GitHub Actions
Ubuntu 24.04 with Python 3.14. Those new rows compare fetch counts and
compressed bytes; wall-clock values are deliberately not compared across
machines.

ffmpeg is the build recorded in each transformed episode's `provenance/v1`;
seeds and fixture parameters are constants at the top of each script.
Deterministic fixture and encoder settings make the storage tables
reproducible byte-for-byte, while read wall-clock values remain subject to
machine noise. The real recordings are fetched, never vendored; their sha256
values are pinned above.
