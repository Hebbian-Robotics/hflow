# HFlow MCAP storage and read benchmarks

These reproducible benchmarks measure the two performance decisions in HFlow's
canonical MCAP writer: in-band video compression and topic-group chunking.
They report what the current implementation achieves at honest small scale
alongside the million-hour results Dyna published in
[Training Dyna-2 at million-hour scale, repeatably](https://www.dyna.co/research/dyna-2-infrastructure)
(Figure 3). Every number below comes from a real run of the scripts in
[`benchmarks/`](../benchmarks); nothing is extrapolated.

## Results at a glance

| Workload | Measured result | Why it matters |
| --- | --- | --- |
| Six-camera real footage | **48-50.5% less video payload** than the source JPEG payloads | Quantifies the storage effect without relying on synthetic test patterns |
| Four-camera synthetic training windows | **2.42x fewer chunk fetches** than per-topic chunking | Shows how grouping topics by read pattern reduces sample assembly work |
| Six-camera real footage with 8 MB chunks | **2.81x fewer fetches and 3.21x fewer bytes fetched** than per-topic chunking | Demonstrates that chunk size and grouping policy must be tuned together |
| Selective state scans | Naive schema grouping fetched **230 MB** for a 0.2 MB `/imu` stream | Shows why HFlow exposes per-topic group overrides instead of treating grouping as a fixed schema rule |

These are measurements of specific workloads, not universal performance
claims. The methodology, caveats, input recording, and reproduction commands
are included below.

**How to rerun** (each script prints these tables; `--quick` for a fast pass;
`--input <recording.mcap>` measures a real recording; see the real-footage
section for the reference dataset):

```bash
uv run python benchmarks/storage_benchmark.py
uv run python benchmarks/read_benchmark.py
uv run python benchmarks/storage_benchmark.py --input nuscenes-mini-sample.mcap
uv run python benchmarks/read_benchmark.py --input nuscenes-mini-sample.mcap \
    --grouping read-pattern --chunk-size-bytes 8000000
```

**Method caveats, up front.**

- Two content sources: the synthetic fixture
  (`hflow.testing.synthesize_episode`, ffmpeg test patterns at 320x240,
  defect injections disabled) and a real six-camera recording (the
  real-footage section below). Synthetic patterns are unrealistically
  compressible; the two storage results quantify exactly how much.
- Reads run against local disk, where a "fetch" costs a chunk decompression,
  not an object-storage round trip. Wall-clock ratios here therefore
  *understate* the benefit that matters on S3/GCS; the fetch and
  bytes-fetched counts are layout facts and transfer directly.
- Scale: 19-60 s episodes, not the forty-three million of Dyna's corpus. The
  point is that the mechanisms behave as Dyna's article describes, not that
  the ratios match.
- Storage reductions compare **video payload bytes** (the codec effect, the
  number comparable to Dyna's ~68%). File sizes are shown alongside but carry
  every non-camera channel passed through byte-for-byte; on a lidar-heavy
  recording those dwarf the cameras and would swamp a file-level comparison.

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

The floor is the part with the least evidence behind it. 800 KB is where the
measurements start, not a measured optimum, and this table is the first
evidence that it is too low for a busy state group. Worth revisiting with a
recording whose state group is proprio-rate rather than 20 topics of vehicle
telemetry.

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

## Provenance

Machine: the development box (local NVMe, Linux); ffmpeg: the pinned build
recorded in each episode's `provenance/v1`; seeds and fixture parameters are
constants at the top of each script. These results were included in the
initial public repository snapshot (`6ab6ef9`). Rerunning reproduces the
storage tables byte-for-byte
(deterministic fixtures and encoder settings) and the read tables up to
wall-clock noise. The real recording is fetched, never vendored; its sha256
is pinned above.
