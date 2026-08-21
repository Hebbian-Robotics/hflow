# Egocentric factory corpus

A complete corpus workflow on real data: download head-mounted factory footage
from Hugging Face, build a 96-episode MCAP corpus with **six deliberately
faulty episodes**, run the quality pipeline, and watch it catch exactly those
six, locally first and then under Airflow. Everything the run produces is
inspectable with standard tools: the DAGs in Airflow's UI, the evidence with
DuckDB, the episodes themselves in Foxglove.

The source is one pinned, hash-verified shard of
[`builddotai/Egocentric-10K`](https://huggingface.co/datasets/builddotai/Egocentric-10K)
(Apache-2.0): genuine factory footage spanning navigation, machine
interaction, material handling, component sorting, tray handling, and tool
setup. From it, [`prepare.py`](./prepare.py) deterministically generates 96
20-second, 640×360, 10 FPS input episodes:

- 90 unmodified excerpts;
- three with an injected three-second **camera blackout**
  (episodes 14, 29, 44, at 7-10 s);
- three with an injected four-second **camera freeze**
  (episodes 59, 74, 89, at 7-11 s).

The tracked [`manifest.json`](./manifest.json) pins the dataset revision, the
archive hash, all 11 source-video hashes, the excerpt plan, and the injected
faults, so every user prepares byte-identical inputs, and the faults are
declared data you can edit (see [inject your own faults](#inject-your-own-faults)).

## Prerequisites

- A Hugging Face account that has accepted the dataset's access terms, and the
  `hf` CLI, authenticated: `uv tool install -U huggingface_hub` then
  `hf auth login`.
- Network access and disk: a ~1 GB download, about 2.5 GB total once the
  sources are extracted and the 96 episodes (~640 MB) are written.
- Docker with Compose v2 (only for the [Airflow path](#run-it-under-airflow)).

All media stays under the gitignored `data/` directory; the repository tracks
only the manifest and the scripts.

## Prepare the corpus

```bash
hf auth whoami
uv run python examples/egocentric/prepare.py
```

The script downloads the pinned shard once, verifies every hash, extracts the
11 source videos, and writes the episodes:

```text
data/egocentric/
├── huggingface/            # pinned source archive (~1 GB, downloaded once)
├── source/                 # 11 hash-verified source videos
├── landing/                # 96 input MCAP episodes, 6 with injected faults
└── prepared-manifest.json  # what was generated, with per-file hashes
```

Re-running is safe: the download and extraction are skipped when the verified
files already exist, and the generated episodes are deterministic.

## Run the pipeline locally

```bash
uv run python examples/egocentric/pipeline.py data/egocentric/landing/*.mcap
```

[`pipeline.py`](./pipeline.py) registers two checks and one enrichment:
`timestamp_regularity`, a critical `camera_health` check (quarantine when
black frames exceed 5% or frozen time exceeds 2 s), and a `contact_sheet`
enrichment. Each episode is transformed to a canonical MCAP under
`data/egocentric/test-runs/`, checked, and recorded in the Parquet catalog at
`data/egocentric/catalog/`. The six fault episodes fail `camera_health` and
quarantine; the other 90 pass and get contact sheets under
`data/egocentric/artifacts/`.

Cut the training manifest with a SQL query over the catalog
([`curate.sql`](./curate.sql)):

```bash
uv run hflow curate \
  --catalog data/egocentric/catalog \
  --sql-file examples/egocentric/curate.sql \
  --output data/egocentric/manifest.parquet
```

Expect 90 rows and a coverage block showing `camera_health` ran on 96/96
episodes.

## Run it under Airflow

The same pipeline file, unchanged, as scheduled DAGs
(the [runtime guide](../../docs/RUNTIME.md) covers everything this section
uses):

```bash
uv run hflow up \
  --pipeline examples/egocentric/pipeline.py:app \
  --data-root data/egocentric

# episode URIs are data-root-relative, so trigger from the data root
(cd data/egocentric && uv run --project ../.. hflow ingest landing/*.mcap)
```

The sync stage plans four byte-balanced mapped batches of 24 episodes each,
with staggered starts. The `_meta` stage's quarantine gate tallies the six
quarantined episodes against the failure budget (`max(8, 1%)`), so the run
**succeeds**: bad data is recorded evidence, not a pipeline failure. On one
development machine a 24-episode batch took about 28 s; a full run lands
around a minute including scheduler overhead. Local runs and Airflow runs
share the same data root and therefore the same catalog: it is append-only
and content-addressed, so mixed runs coexist and repeating an
ingest over unchanged episodes is a recorded no-op.

## Inspect what happened

**Airflow: the runs.** `up` prints the UI URL and credentials. Click the
pipeline's tag in the DAG list to filter to its five DAGs (master + four
stages); each DAG's Docs tab explains itself, and each mapped
`process_batch` task carries per-episode logs. See
[watching runs in the Airflow UI](../../docs/RUNTIME.md#the-loop).

**DuckDB: the evidence.** Measurement keys are plain columns on the wide
`episodes` view (helpers and views:
[catalog guide](../../docs/CATALOG.md)):

```bash
uv run python -c "
import hflow
connection = hflow.open_catalog_connection('data/egocentric/catalog')
connection.sql('''
    SELECT episode_id, status, black_frame_pct, freeze_total_s
    FROM episodes ORDER BY status DESC, episode_id
''').show()
"
```

The six quarantined rows lead. The three blackouts show `black_frame_pct` ≈ 15;
all six show `freeze_total_s` ≈ 3, because a blacked-out camera is also a
frozen one and the detector brackets conservatively (a 4 s injected freeze
measures ≈ 3 s). The 90 `ok` rows sit below both thresholds.

**Foxglove: the episodes.** Every canonical episode is a standard MCAP file;
open one with a fault and scrub to the 7-second mark to see what the check
saw ([how-to](../../docs/how-to/inspect-episodes-in-foxglove.md)):

```text
data/egocentric/test-runs/factory_051_episode_0014-*/factory_051_episode_0014.canonical.mcap
```

(Run directories are named `<episode>-<source-hash>`, so the glob finds the
suffix. Airflow-run outputs land under
`data/egocentric/episodes/<episode>-<source-hash>/` instead.) The accepted
episodes' contact sheets under `data/egocentric/artifacts/` open in any
image viewer.

## Inject your own faults

The faults are data, not code. Add an entry to `episode_plan.faults` in
[`manifest.json`](./manifest.json), for example a two-second blackout in
episode 33:

```json
{ "episode_number": 33, "fault": "blackout", "fault_segment_s": [5.0, 7.0] }
```

then re-run `prepare.py` (regeneration is deterministic) and the pipeline.
`fault` is `blackout` or `freeze`; the segment must fit inside the 20-second
episode. Faults short enough to pass the thresholds in
[`pipeline.py`](./pipeline.py) are a good way to explore where your own
quality bar should sit. Or tighten the thresholds instead and watch the
quarantine count change. And nothing here is corpus-specific: the same
pipeline runs on any conforming MCAP recording you point it at.

## Provenance and privacy

The repository distributes the manifest and scripts, never the footage or
generated corpus. This footage shows real people working: before sharing any
generated artifact (contact sheets included), confirm the dataset terms
permit it and review for faces, screens, badges, or other sensitive content.
Keep quality statements about the *recordings* (camera health, sensor
coverage), never about the people who appear in them.
