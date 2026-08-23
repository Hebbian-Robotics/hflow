# HFlow architecture for Physical AI data pipelines

HFlow applies a production data-pipeline architecture to multimodal physical-AI data from human egocentric collection, robot teleoperation, autonomous policy rollouts, and other sensor-rich sources.

The architecture was inspired by Dyna Robotics' ["Training Dyna-2 at million-hour scale,
repeatably"](https://www.dyna.co/research/dyna-2-infrastructure) (Aug 2026). HFlow is an
independent project: not affiliated with Dyna, containing none of their code, and claiming no
compatibility with their internal systems.

Load-bearing decisions in this document cite their sources: Dyna's article where a mechanism or
measurement is theirs (the article describes goals and mechanisms without publishing a complete
file-format or service specification), and cited research everywhere else.

The governing trade: **democratize the architecture, defer the optimizations.**

## Implementation status

This table is the short version of where the implementation stands relative to the full
architecture at production scale. **Implemented**
means the capability exists in this repository at small scale; **simplified**
means HFlow preserves the interface or intent with a single-tenant design;
**deferred** means possible future work, not a release commitment; and **out of
scope** means it is deliberately not part of this repository.

| Area | At million-hour scale (per Dyna's article) | HFlow today | Status |
|---|---|---|---|
| Lifecycle boundary | Collection through training-batch loading | Starts from landed MCAP files and ends at curated MCAP plus a Parquet manifest; data-collection systems and training/loading are not included | **Partial by design** |
| Input formats and triggering | Data from multiple sources and vendors lands in a bucket | MCAP input only, triggered explicitly through the SDK/CLI or Airflow API; no bucket/filesystem watcher | **Simplified; more readers deferred** |
| Canonical episode format | MCAP, H.264 GOPs matched to read patterns, topic-group chunking, and version stamps | An independently specified, standard-MCAP convention covering the same ground (GOP presets, topic-group chunking, version stamps); interoperability with standard readers is tested, but compatibility with Dyna's undisclosed internal layout is not claimed | **Implemented independently** |
| Ingestion graph and gates | Airflow transformation, quality-check, and enrichment stages with run profiles and critical/non-critical steps | The stage graph, profiles, quarantine gates, local Compose runtime, and bring-your-own Airflow bundle are implemented | **Implemented at small scale** |
| Checkpoint and replay | Durable pipeline state, checkpointable multi-day runs, replay from any step, cross-DAG artifact sharing, and selective reprocessing | Durable outputs, a sync completion marker, versioned catalog facts, stage-profile reruns that reuse the previous run's published canonical episode (cross-run artifact sharing through the data root), and `hflow stale` for selective reprocessing; no general arbitrary-step checkpoint/replay engine | **Partial; arbitrary-step replay deferred** |
| Per-step compute | Resources and worker allocation tailored to each step | `requires=`/`uses=` record intent, validate configured endpoint aliases, and put cheaper steps first; they do not yet route tasks to heterogeneous worker pools or probe GPU/endpoint health | **Deferred** |
| Batch scheduling | Byte-balanced batches and staggered starts, plus joint optimization against network, I/O, database, and worker limits | First-fit-decreasing byte balancing and deterministic stagger are implemented; the joint optimizer is not | **Simplified; optimizer deferred** |
| Catalog and curation | Transactional database, CDC, analytical warehouse, and a memory-mapped training manifest | Parquet catalog plus DuckDB SQL and Parquet manifests; no database/CDC/warehouse stack and no training dataloader | **Simplified; training loader out of scope** |
| Corpus caching and fleet orchestration | Alluxio/NVMe cache warming and training-fleet orchestration at million-hour scale | No distributed corpus cache, cache warmer, Slurm/Ansible fleet layer, or topology-aware training orchestration | **Out of scope** |
| Model-based enrichment | Production labels, captions, and segmentations | Hooks, frame/contact-sheet helpers, and examples; users supply models, clients, prompts, aggregation, and model-serving infrastructure | **Extension surface implemented; models out of scope** |
| Product/tenancy | An internal production system operating at that scale | One open-source, single-tenant workspace with no accounts, RBAC, or hosted control plane | **Simplified; hosted control plane is only a future direction** |

The detailed sections below explain why each simplification was made. The
remaining scale mechanisms are collected in [The scale path](#the-scale-path);
the issue tracker, rather than this document, is the source of truth for work
that is actually scheduled.

## Design tenets

1. **Evidence, not verdicts.** Quality checks record measurements, intervals, and tags. Pass/fail policy belongs to the consumer at curation time, never hardcoded into the corpus.
2. **Standard formats at every boundary; no captive UIs.** Episodes are standard MCAP (Foxglove/Rerun open them), runs are standard Airflow DAGs (Airflow's UI shows them), the catalog and manifests are Parquet (DuckDB/pandas/anything reads them). The optional workspace UI (`hflow serve`, shipped separately as `hflow-server`) is a strict client of these same open surfaces through a documented JSON API -- it hides nothing, gates nothing, and everything it shows stays reachable without it. The system is extensible without touching our code.
3. **Your code stays your code.** Transformations, checks, and enrichments are plain Python functions in the user's own environment. Existing processing code plugs in through small adapters rather than being rebuilt inside a framework.
4. **Ship code only where it earns its place.** Either the canonical format forces bridging (video lives in-band; nothing can read it without our accessors) or the code encodes a painfully-rediscoverable pitfall. We ship no client wrappers around things users already know (`openai`, `subprocess.run(["ffmpeg", ...])`); the examples are the documentation.
5. **Coarse-grained steps.** One task processes one episode or one batch and runs for seconds to minutes. Hot loops live inside a task, never across tasks.

## The data lifecycle

The pipeline follows a four-stage lifecycle -- in the words of Dyna's article, "collection into a
landing bucket, ingestion into training-ready episodes, curation into the dataset for a given
experiment, and loading into training batches." HFlow applies that lifecycle to physical-AI
episode data regardless of who or what generated it: human-worn sensors, a teleoperated robot,
an autonomous policy, or another multimodal collection system.

```
multimodal              landing              ingestion DAG                    catalog + curation
recording ───────────►  bucket/dir  ──API──► transform ─► QC gate ─► enrich ─► episodes.parquet
(human or robot)        (raw MCAP)  trigger  (canonical    (user      (user     │
                                              MCAP)         checks)    steps)   ▼ DuckDB SQL
                                                                               manifest.parquet
                                                                                │
                                                                                ▼
                                                                 delivery / conversion (separate package)
```

HFlow implements the first three stages. **Loading/training is out of scope**: the pipeline ends at curated, quality-tagged, version-stamped episodes plus a manifest, whether the next hop is a trainer or a data buyer.

### Landing and triggering

Collection lands raw recordings in a bucket or directory; ingestion is triggered by an API call.
The SDK/CLI (`hflow ingest <uri>`) wraps Airflow's REST API v2 `dagRuns` endpoint with the
episode/batch URI in the run conf. A collection rig or upload script calls it directly. A
filesystem/bucket watcher is an optional convenience, not the core mechanism (Airflow cannot
natively watch buckets; level-based file-exists triggers are explicitly prohibited upstream).

**v1 accepts MCAP only** (any chunking, any message encoding; this includes every rosbag2
recording, since MCAP is the ROS 2 default bag format). This is a format boundary, not a
restriction to robot-originated data. Field data arrives from many vendors in many formats, each
with its own quality quirks; starting MCAP-in is our simplification, and other readers
(video + CSV, HDF5) are later plugins.

## The episode container: canonical MCAP

Canonical episodes are standard [MCAP](https://mcap.dev/): video-optimizable, randomly
accessible, flexibly chunked, and natively visualizable. Dyna's article describes moving from
H5-holding-per-frame-JPEG to MCAP for the same reasons and names the two tunings that mattered
most at scale; HFlow implements both, plus the version stamps the article calls for:

1. **H.264 with GOP length matched to the read pattern.** Inter-frame compression cut Dyna's
   storage ~68% versus per-frame JPEG (their measurement). GOP length is a storage-vs-seek trade
   that depends on the consumer: VLA-style training samples short sparse windows (pays a keyframe
   seek per sample → short GOPs), world-model training reads long contiguous sequences (amortizes
   keyframes → long GOPs). HFlow exposes GOP as a writer preset keyed to model class, because --
   as Dyna's article is explicit -- this is effectively a training hyperparameter.
2. **Topic-group chunking.** MCAP's default writing gives each topic its own chunks, so
   assembling one training sample costs a read per topic. HFlow's canonical writer instead groups
   topics that share a read pattern and writes each group time-major: cameras interleaved in one
   chunk stream, proprioception + actions in another, never sharing a chunk. A sample then costs
   one read per *group*: an added camera costs no extra fetch. The mechanism comes from
   Dyna's article, which reports ~3.4× fewer chunk fetches and ~2.9× faster reads; no other
   open-source MCAP writer implements it. Chunking changes write order, not the format, so **any
   conforming MCAP reader (Foxglove, Rerun, the stock `mcap` package) reads our files
   unmodified.**
3. **Version stamps.** HFlow stamps every canonical episode with `schema_version`,
   `pipeline_version` (a content hash of the step configuration that produced it), and
   `robot_software_version` (from the source recording's metadata, when present). Dyna's article
   states the requirement plainly: "Every processed episode also carries a stamp of what produced
   it: the schema version, the ingestion pipeline version, and the software version running on
   the robot when it was recorded." The corpus is assumed permanently mixed-version (reprocessing
   takes too long to ever be atomic), so curation filters by version range and stale episodes are
   found and reprocessed selectively.

The article leaves several specifics open; these are HFlow's design decisions:

| Decision | Choice | Rationale |
|---|---|---|
| How H.264 sits in MCAP | [`foxglove.CompressedVideo`](https://docs.foxglove.dev/docs/sdk/schemas/compressed-video) messages (Annex B, SPS/PPS on every keyframe, no B-frames) | The ecosystem-standard encoding; it is what makes Foxglove/Rerun viewability real, and lack of native visualization was one of the two reasons Dyna's article gives for abandoning H5 |
| Where stamps and episode semantics live | MCAP Metadata records (task, operator, success label, embodiment, version stamps); Attachments for calibration/URDF | Keeps the episode self-contained and viewer-inspectable; Dyna's article says only that metadata is indexed for manifest queries (a database concern), not where it lives in-file |
| Compression codec | zstd chunks | Dyna's article mentions tuned compression without naming a codec; zstd is the MCAP ecosystem default |
| Numeric parameters (GOP seconds, chunk sizes) | Configurable, with measured defaults | Not disclosed in Dyna's article; our benchmark report will publish what we measure |

On-disk identifiers (metadata record names, channel naming) are neutral and format-versioned: they never embed the project name, so stored data is independent of branding.

**Acceptance test for every file the pipeline writes: it opens cleanly in Foxglove and Rerun.**

The full normative convention lives in [FORMAT.md](./FORMAT.md).

## Ingestion: the Airflow DAG

Ingestion is an Airflow DAG with three stage families: **data transformation** (resample all
streams onto a common timestamp grid, compute derived signals, encode canonical MCAP, index
metadata), **quality check** (a runtime-toggleable pre/post gate around transformation), and
**feature enrichment** (performance labels, captions, segmentations). The structure follows
Dyna's article, which describes rewriting an original single-Kubernetes-job pipeline into this
DAG shape to gain per-step resource allocation, critical/non-critical step tagging (a
non-critical failure doesn't cancel the run), and per-run step toggles ("run profiles": full
processing, metadata-only backfill, re-label pass), with pipeline state backed by durable
storage rather than kept in a live process -- enabling checkpointable runs, replay from any
step, cross-DAG artifact sharing, and selective reprocessing.

HFlow adopts the stage model, run profiles, and the critical/non-critical gate
semantics: the three-stage skeleton is fixed, and user steps hang off it. Its
replay boundary is currently a stage, not every individual user step; see
[Implementation status](#implementation-status).

### Transform

The built-in transform resamples streams onto a common grid, computes derived signals, and writes the canonical episode. Users override at three levels: **configure** it (grid rate, GOP preset, topic-group assignment, per-signal interpolate-vs-nearest policy); **extend** it (register derived-signal functions); or **replace** it (`@app.transform`); a custom transform still ends by calling the canonical-writer library function, so downstream contracts hold. The multi-rate alignment policy is explicit, versioned, and configurable because it is where format converters silently diverge.

### Steps and resources

Each DAG step should get resources matched to its own requirements (Dyna's
article counts per-step tailoring among the DAG rewrite's wins). In HFlow, a
step can declare `requires={"gpu"}` and `uses="judge"` (a named endpoint
alias), but the current implementation uses those declarations only to validate
that named endpoint aliases are configured and to order plain steps before
resource-declaring steps. It does **not** yet probe endpoint health or GPU
visibility, or route individual steps to heterogeneous Airflow worker pools. A
bring-your-own Airflow deployment must currently arrange those resources
outside HFlow.

### Batching

HFlow ships simple versions of two scheduler-protecting mechanisms in v1: a planning task runs
first-fit-decreasing over file sizes into near-equal-byte groups for mapped tasks (file sizes
vary widely and unbalance workers), and batch starts are spaced by a configurable stagger
interval (deterministic even spacing, which desynchronizes identical steps at least as well as
random jitter and reproduces exactly). Both mechanisms come from Dyna's article: at their scale
the scheduler database became the choke point, and the fixes were staggered batch start times
and bin-packing input batches into near-equal *bytes*. What stays on
[the scale path](#the-scale-path) is the joint optimizer the article describes, tuning worker
counts against network/IO/DB constraints.

### Deployment modes

Airflow cannot be a normal pip dependency: its own maintainers state unconstrained installs "will not work from time to time," and its ~128-package footprint would collide with users' ML stacks. The SDK is therefore dependency-light and *provisions or targets* a runtime rather than importing one:

1. **Docker mode (default).** `hflow up` renders a Compose bundle: Airflow 3.x services + Postgres + a worker container; the data root is a host directory mounted into the containers, or an object-store prefix (`s3://`, `gs://`, Azure) the tasks talk to natively (see [the runtime guide](./RUNTIME.md)). The SDK generates a private DAG-bundle directory (Airflow 3's `dag_bundle_config_list`) and pre-sets the configuration that traps newcomers (`dags_are_paused_at_creation=False`, examples off, object-storage-backed XCom so large accidental returns never hit the metadata DB).
2. **Bring-your-own Airflow.** `hflow deploy` emits the same DAG bundle as plain files plus a light runtime package for an existing deployment (Astronomer, MWAA, Cloud Composer, self-managed); the data root is either an absolute filesystem path visible on every worker or an object-store prefix (`s3://`, `gs://`, Azure). Bucket roots use an obstore-backed local mirror, so media processing still receives ordinary local files while durable publication stays store-native.

**Dependency isolation:** the worker image carries two environments, Airflow's own and a dedicated pip-built venv from the user's requirements. User steps execute in the user's venv (external-python pattern), so user dependencies and Airflow's pins never meet. The worker image is user-extensible (own base image, extra system packages) so a check needing a system library never requires touching HFlow.

**Dev loop:** `app.test(episode)` runs the entire pipeline in-process on one episode: no Docker, no scheduler, no Airflow import at all (a plain Python runner with the same gate semantics, wrapping the `app.process()` operation the DAG maps over). Iterate on a check in seconds; `app.run()` when it works.

**Observability is Airflow's own UI**, exposed on localhost in Compose mode. Clear per-step
status in the DAG view is part of what's being democratized (Dyna's article counts it among the
rewrite's benefits). The SDK adds plain-language diagnostics for embedder-specific traps
(`hflow status`), it does not replace the UI.

### Data passing

Steps pass references (URIs, episode ids) and small measurements, never media bytes. Multi-gigabyte artifacts move through the data root; Airflow XCom is configured with an object-storage backend as a safety net.

## Quality checks and curation

Quality checks run as their own runtime-toggleable pre/post gate around transformation, and
their outcomes stay queryable at curation time (filter by whether the run succeeded, by whether
a camera dropped out). Dyna's article pins down that placement and the issue classes production
QC catches -- camera blackout, choppy joint states, missing/occluded hand positions, bad
frames -- but not what a check outputs, any thresholds, or whether a filtered-out episode is
deleted or tagged.

HFlow's answer is a three-layer model, grounded in published curation research:

**Layer 1: checks output measurements, not verdicts.** A check returns numbers, time intervals, and optional tags; everything lands in the episode's catalog row regardless of pass/fail. Rationale: the same stored measurements support different policies without rerunning media processing, and quality heuristics are known to *invert* on real defects ([Voxel51's audit](https://voxel51.com/blog/robot-data-quality-scoring) found smoothness metrics scoring an early-gripper-release defect *better* than clean demos). A hardcoded verdict bakes the wrong call into the corpus; a measurement lets you re-decide with a query.

**Layer 2: gates are optional policy on critical checks.** A check may declare a user-owned `verdict` predicate, or a pipeline may attach a declarative `Gate` at registration (`@app.check(gate=...)`) that the runner evaluates over the measurements the check returned. HFlow ships *recommended* gates as values (`hflow.checks.RECOMMENDED_CAMERA_INTEGRITY`) so a starting threshold is documented and tunable, but nothing gates until a pipeline passes one in: a corpus is never quarantined by a number its owner did not choose. Gates are evaluated outside the check on purpose -- a threshold applied inside raises on a key that episode never produced, and the runner records that as an infrastructure error, discarding every measurement already computed. A gate composes with a check's own verdict by conjunction, so it can only ever tighten policy; and because rejecting a conjunction needs only one false conjunct while accepting needs every threshold read, a gate that cannot read a threshold *abstains* (recording a `gate-unevaluated:<pattern>` tag) rather than approving what it never inspected. A failed **critical** verdict tags the episode `quarantined:<check>` and skips its downstream steps (no enrichment spend on an episode with a dead camera); a failed non-critical verdict records a tag and the run proceeds (the critical/non-critical gate semantics described above). Quarantine is a tag, never a deletion: the field's strongest teams keep failures ([DROID](https://droid-dataset.github.io/) releases them, [1X trains on them](https://www.1x.tech/discover/redwood-ai), RoboMIND annotates their causes). A check *crashing* is infrastructure, not data: it retries and can fail the run, but is never recorded as a bad episode.

**Layer 3: curation is SQL over everything layer 1 recorded.** Measurements are catalog columns; a curation query emits `manifest.parquet`. Every measurement row carries the check's version (a content hash of its configuration), so re-running a changed check appends new-version rows and curation can pin exact versions (versions are content hashes, so pins are equality or set membership, never ordered ranges): the mixed-version-corpus reality, applied to checks. The other half of that reality is selective reprocessing: `hflow stale` (`hflow.stale_episodes`) lists exactly which sources' latest runs predate the current pipeline/format versions, ready to pipe back into `hflow ingest` -- the selective reprocessing Dyna's article calls for: find exactly the stale episodes and reprocess only those.

Dataset-level reporting includes **coverage denominators** (which checks ran on what fraction of episodes) because a statistic over half a delivery must not look like a statistic over all of it.

### The check library and the porting story

Porting existing QC code is the primary onboarding story:

```python
from your_existing_qc import check_joint_smoothness  # untouched


@app.check()
def joint_smoothness(ep: hflow.Episode) -> hflow.CheckResult:
    joints = ep.channel("/joint_states").to_numpy()  # extract
    result = check_joint_smoothness(joints, rate_hz=100)  # unchanged
    return hflow.CheckResult(measurements=result)  # record
```

The accessor surface is chosen by asking what existing robotics QC scripts take as input: `ep.video(camera)` (lossless remux of in-band H.264 to MP4, no re-encode), `ep.frames(camera, fps=...)` (JPEG frames), `ep.channel(topic).to_numpy() / .to_arrow()`, `ep.metadata`. Users who want none of it can open `ep.path` with the raw `mcap` package; the file is standard MCAP.

Built-in checks ship in the same shape users write, doubling as documentation. The starting set covers the issue classes named in Dyna's article plus the field's established integrity checks: timestamp regularity (1/fps ± tolerance, [LeRobot's check](https://github.com/huggingface/lerobot)) with cross-stream sync, camera blackout/freeze/exposure and frame count vs expected rate (`camera_frame_stats`), joint discontinuity vs velocity limits, and idle fraction. The camera checks run as a **single-decode ffmpeg instrument**: blackframe + freezedetect + signalstats in one filter graph, so all three share one frame denominator and one decode. Beyond those: `camera_signal_quality` measures coding range, range-gated exposure, impulse noise, and stillness off the same decode pass; `action_integrity` reads the recorded values themselves (non-finite samples, publisher stalls that repeat a sample bit-for-bit, dimensions that never move); `keyframe_interval` reports how seekable and stream-copy-cuttable the footage is; `camera_fps_conformance` classifies a stream's timestamp-derived rate against the rate the corpus declares; `trajectory_metrics` and `trajectory_segments` report episode-scope motion facts and where inside the episode the motion did something worth looking at; and `action_rate` reports per-topic message rates. Two of the classic cuts are corpus-relative judgments, so their checks record evidence and the decision is a curation query: `episode_duration` for length outliers, and `content_digest` for exact-duplicate detection (a `GROUP BY ... HAVING count(*) > 1`) -- with `media_digest` answering the narrower question of duplicate *footage*, which survives re-stamping and differing telemetry. Motion-smoothness metrics ship as flags only, never default reject rules (the Voxel51 inversion result), and no shipped gate thresholds one -- `tests/test_gates.py` walks the keys those checks emit and fails if any does. Built-in checks themselves always return evidence only. HFlow does ship *recommended* thresholds as gate values a pipeline can pass to `@app.check(gate=...)` and tune, so a starting bar is documented rather than folklore; choosing to gate on one is always the user's explicit act. No two built-ins claim the same measurement key: every step of one run shares its fingerprint, so a shared key is a tie the catalog resolves arbitrarily, and the runner refuses it.

### Model-based checks

Enrichment generates model-derived features -- performance labels, video captions, segmentations
(the classes Dyna's article names; the mechanisms are unspecified there).

HFlow supports frames-only VLM usage in v1: most models and the OpenAI-compatible protocol don't
natively support video, so the honest unit is the frame. The user extracts frames explicitly
(`ep.frames(fps=...)`), calls their own client (any OpenAI-compatible endpoint, hosted or
self-run vLLM/Ollama; each step names its endpoint), and owns the aggregation of per-frame
answers. There is no bundled VLM client; examples show plain `openai` calls. Two helpers survive
because they encode non-obvious value: the **contact sheet** (N timestamped frames composited
into one image; works even on single-image models, cheap on vision tokens) and frame extraction
itself. Missing/occluded hand positions -- one of the issue classes Dyna's article names -- is
the canonical example of this surface: hand visibility is a model judgment, not a signal
statistic, and the [OpenAI vision example](../examples/openai_vision/pipeline.py) shows it as a
contact-sheet VLM check. Native-video protocols are a contributor-shaped provider extension
point.

## Catalog and curation storage

The ingest DAG appends one row per episode to Parquet under the data root, and
[DuckDB](https://duckdb.org/) queries it directly: researcher-facing SQL (filter by task, robot,
success, sensor dropout, pinned versions) with zero additional services, and a curation query
writes a columnar manifest. This is the single-tenant collapse of the interface production
systems need at scale. Dyna's article reports that manifest-building by walking files took ~48
hours at 43M episodes; their fix was a transactional production DB CDC-replicated into an
analytical warehouse, making curation a SQL query whose columnar manifest is memory-mapped by
training ranks. The warehouse/CDC split solves contention between transactional load and
columnar scans at tens of millions of rows; a single-tenant deployment with thousands of
episodes doesn't have that problem. The manifest is Parquet; the
download-once/mmap/zero-copy-shard loading trick Dyna's article describes lives in a training
dataloader we don't ship, and is documented as [a recipe](./how-to/load-manifest-mmap.md) for
users who need it.

## Storage and durability

HFlow's small-scale durability model is drawn from production experience with
Pareto and targets the same requirements Dyna's article states: durable
externally-stored pipeline state, checkpointable multi-day runs, replay from
any step, and selective reprocessing (the article leaves the mechanisms
unspecified). Today, canonical outputs and catalog facts are durable,
the sync stage has a persisted completion marker, and run profiles can replay a
stage -- a later run (a relabel pass, a different profile) consumes the
canonical episode a previous run published, which is cross-DAG artifact
sharing in miniature, through the data root. A general content-addressed
checkpoint for every user step and replay from an arbitrary step remain design
targets rather than current guarantees:

- One local-directory-or-bucket **data root** holds everything durable; pipeline processes are stateless. Bucket roots download through an etag-validated per-worker mirror and publish canonical episodes and artifacts back to the store. Catalog appends use store-native create-if-absent writes.
- **Content-addressed artifacts (target)**: derived outputs will be addressed by
  a hash of exactly the inputs that determine their bytes (configuration,
  instrument identity, model id). Changing a prompt or a filter graph should
  create a parallel artifact directory, never a migration.
- **Completion markers (implemented narrowly)**: the sync stage publishes a
  completion marker after the canonical episode, so a later stage or run can
  prove the canonical it consumes was fully written for this source. Extending
  the same create-if-absent checkpoint contract to every user step is deferred.
- **Manifest-last publication (target)**: a build directory is sealed by
  writing its manifest once, last, so partial builds are unreachable by
  construction.
- **Two-axis failure taxonomy**: (data-scoped vs infrastructure) × (transient vs terminal). Transients retry in place; terminal data errors quarantine the episode within a budget (mass failure fails the run loudly); infrastructure trouble is never recorded as bad data.

## Language choice

**Pure Python for all first-party v1 code.** The compute-heavy work in this pipeline already runs in native code: video is FFmpeg (C, subprocess) and chunk decompression is C zstd (~2.6 GB/s measured through the pure-Python `mcap` package). Parallelism is process-level via Airflow tasks, so the interpreter is never the unit of concurrency. The contributor pool for a robotics data tool writes Python.

The one measured bottleneck is per-message decoding of state channels in pure Python (~38k msgs/s): roughly seconds per episode against minutes of H.264 encode, i.e. not v1's constraint. The mitigation is architectural now, native later: video bytes are passed opaque (remuxed by ffmpeg, never message-decoded in Python), and the episode reader is a **batch-oriented interface** (decoded batches in, Arrow/bytes out). That shape lets a Rust-backed reader drop in behind the same interface as a non-blocking parallel track: PyO3 bindings over the official [`mcap` Rust crate](https://docs.rs/mcap/latest/mcap/), shipped as a separate wheel-distributing package with the pure-Python backend as its parity-tested fallback. Nothing in the pipeline waits on it. The one API rule that cannot be retrofitted is the interface shape: batch iterators, never per-message callbacks in the hot loop.

The pinned ffmpeg build is treated as a measuring instrument: a static build is auto-downloaded (never whatever is on PATH), and its version is recorded in every measurement's identity, because different ffmpeg versions genuinely produce different measurements.

## Tenancy

The open-source deployment is **single-tenant everything**: one workspace, no user management, no RBAC, which is the norm across comparable OSS infrastructure (Airbyte, Dagster, Prefect, Temporal, Langfuse). A future hosted offering would be a multi-tenant *control plane* over per-customer or customer-run data planes (the Dagster+/Prefect hybrid pattern): raw sensor data never transits the vendor, and the OSS single-tenant engine is the hosted data-plane unit. The design consequence taken now, because it cannot be cheaply retrofitted: only metadata, states, and pointers cross the control boundary, never episode bytes. [HOSTING.md](./HOSTING.md) names that boundary concretely -- the workspace unit and its identity, the manifest artifacts a service exchanges, remote runtime addressing, the environment injection contract, the trust model, and the current limits.

## The scale path

What production scale needs that HFlow defers, and what replaces it here; Dyna's article is the
cited description of each mechanism proven at million-hour scale:

| Production-scale mechanism (per Dyna's article) | Why scale needs it | HFlow v1 |
|---|---|---|
| Joint optimizer tuning worker counts vs network/IO/DB limits | millions of concurrent runs | simple FFD bin-packing + deterministic start stagger (included); tuning deferred |
| Production DB + CDC + analytical warehouse | 50M+ row columnar scans vs transactional load | Parquet catalog + DuckDB, same SQL interface |
| Alluxio page-level NVMe cache + warm-before-launch orchestration | PB corpus, multi-vendor GPU clusters | out of scope; documented pointer for those who need it |
| mmap/zero-copy manifest loading across ranks | 2 TB node RAM ceilings | [docs recipe](./how-to/load-manifest-mmap.md) (Arrow memory-map); trivial at small scale |
| Topology-aware optimizer sharding, Slurm preflight gating, Ansible fleet provisioning | training-side, week-long runs on rented fleets | out of scope; this is a data pipeline, not a trainer |

A benchmark report (tracked in issues) will publish what the simple version achieves and where each limit is: storage vs per-frame JPEG (Dyna: ~68% reduction), topic-group vs default chunking read performance (Dyna: ~2.9× faster), and single-machine ingestion throughput.

## References

Cited throughout for mechanisms and measurements proven at million-hour scale:

- Dyna Robotics, [Training Dyna-2 at million-hour scale, repeatably](https://www.dyna.co/research/dyna-2-infrastructure) (Aug 2026)

Formats and infrastructure:

- [MCAP specification](https://mcap.dev/spec); [MCAP Python libraries](https://mcap.dev/docs/python/); [`mcap` Rust crate](https://docs.rs/mcap/latest/mcap/); [MCAP CLI](https://mcap.dev/guides/cli) (`doctor`, `recover`, `filter`)
- [foxglove.CompressedVideo schema](https://docs.foxglove.dev/docs/sdk/schemas/compressed-video); [H.264 support in Foxglove](https://foxglove.dev/blog/announcing-h264-support-in-foxglove)
- [Apache Airflow 3 architecture](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/overview.html); [dynamic task mapping](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/dynamic-task-mapping.html); [dependency-conflict guidance](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html); [object-storage XCom backend](https://airflow.apache.org/docs/apache-airflow-providers-common-io/stable/xcom_backend.html)
- [DuckDB](https://duckdb.org/); [Rerun](https://rerun.io/); [Foxglove](https://foxglove.dev/)
- [obstore](https://github.com/developmentseed/obstore); [blake3-py](https://github.com/oconnor663/blake3-py); [FFmpeg](https://ffmpeg.org/)
