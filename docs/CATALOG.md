# Query and curate Physical AI datasets

HFlow's answer to "which episodes go in the dataset?" is a SQL query over
plain files: every processed episode appends rows to **plain Parquet files
under your data root**, and [DuckDB](https://duckdb.org/) queries them
directly. Researcher-facing SQL, zero services (Dyna's article describes the
same query interface backed by a warehouse at their scale). A curation is a
query; its output is `manifest.parquet`.

Nothing on this page is a gate. The catalog is ordinary Parquet: point
DuckDB, pandas, or polars at the files and ignore our helpers entirely.

## Recording runs

Two entry points write the catalog, one operation underneath:

```python
report = app.test("episode_0001.mcap", record=True)  # dev loop: OFF by default
report = app.process("episode_0001.mcap")  # ingest path: ON by default
```

`app.test()` never records unless you ask; iterating on a check should not
pollute the corpus. `app.process()` is what the ingest DAG maps over (outputs
land under `<data_root>/episodes/<stem>-<source-identity-hash>/`, so equal
basenames from different source paths or bucket keys cannot collide), and recording
is the point. Either way, `report.catalog_entry` tells you what happened:

```python
entry = report.catalog_entry
entry.episode_id  # content address of the canonical file
entry.run_fingerprint  # content hash of versions + the observable run outcome
entry.written  # False when this exact run was already recorded
```

One append writes one Parquet file into each of five table directories under
`<data_root>/catalog/`, all named `<episode_id>-<run_fingerprint>.parquet`.
A sixth directory, `ingest_failures/`, records the attempts that produced no
append at all:

| Table | One row per | Carries |
|---|---|---|
| `episodes` | episode append | `uri`, `source_uri`, version stamps (`schema_version`, `pipeline_version`, `robot_software_version`, `ffmpeg_version`), promoted semantics (`task`, `operator`, `success`, `embodiment`), the rest of `episode/v1` as `metadata_json`, `quarantined` + `quarantine_tags_json`, `orchestrator_run_id`, `recorded_at` |
| `check_runs` | (episode, step) invocation | `check_name`, `check_version`, `critical`, `status`, `duration_s`, `error`; present even when a step produced nothing, which is what makes coverage countable |
| `measurements` | measurement key | `key`, typed value columns (`value_double`, `value_text`, `value_bool`), the producing `check_name`/`check_version` |
| `tags` | tag | `check_name`, `tag` |
| `intervals` | labeled time span | `label`, `start_ns`, `end_ns` |
| `ingest_failures` | attempt that produced NO episode row | `source_uri`, `stage`, `failure_kind` (`source-missing` / `source-unreadable` / `infrastructure`), `error_type`, `message`, `pipeline_version`, `orchestrator_run_id`, `recorded_at` |

`ingest_failures` is the complement of `episodes`, and exists because
`episode_id` is a hash of canonical bytes: a recording that never
canonicalized has nothing to hash and so cannot be an episode row at all. It
is keyed by source rather than by episode, which is why it is not one of the
tables above in any other sense. `failure_kind` is a heuristic and is stored
next to the verbatim `error_type` and `message` rather than in place of them;
anything unrecognized is `infrastructure`, never an accusation against the
recording.

Three durability rules govern writes:

- **Content-addressed**: `episode_id` is a sha256 of the canonical file's
  bytes. Re-ingesting an unchanged episode dedupes; a reprocessed episode
  (new `pipeline_version`) is a distinct fact.
- **Create-if-absent**: an append whose `(episode_id, run_fingerprint)` file
  already exists is a no-op (`written=False`). The fingerprint includes the
  observable outcome: exact retries deduplicate, while a successful retry
  after an error appends the repaired result.
- **Append, never overwrite**: when a check's results are no longer comparable,
  bump its explicit version. Re-running then adds rows under the new
  `check_version` next to the old ones. The corpus is assumed permanently
  mixed-version; curation picks.
- **Provenance, not identity**: `orchestrator_run_id` records which
  orchestrated run recorded the row (the generated Airflow DAGs pass their
  stage sub-DAG's own run id; a local run records NULL). It is deliberately
  outside `run_fingerprint`, or a rerun producing the same outcome would stop
  deduplicating. So it names the run that FIRST recorded an outcome, and since
  filters read the latest row per episode, selecting on it answers "whose work
  is the current answer" rather than "which runs ever touched this".

One timestamp per append: every file of one `(episode_id, run_fingerprint)`
carries the episodes file's `recorded_at` (its create-if-absent is the
atomic commit). The winner force-aligns the dependent tables after
committing, and a replayed append re-checks and heals any dependent a
crashed earlier attempt left with a stale timestamp -- so concurrent
duplicate appends and retried tasks converge instead of stitching two runs'
rows together.

## Querying

### Explore in DuckDB UI

Start DuckDB's browser UI over the default local catalog:

```bash
hflow catalog ui
```

The command works before the first ingest. It creates the empty catalog root,
opens the UI at `http://127.0.0.1:4213`, and waits for the first completed
append. When that append lands, HFlow replaces the empty in-memory relations
with the normal Parquet-backed views on the same DuckDB connection. The open UI
can then query `episodes`, `measurements`, `check_runs`, and the other views
listed below. Start the explorer first, then trigger the runtime in another
terminal.

Point it at another local catalog or keep it headless with:

```bash
hflow catalog ui --catalog data/egocentric/catalog
hflow catalog ui --catalog data/egocentric/catalog --no-browser
```

DuckDB UI listens on loopback. When HFlow is running on another machine, run
the second command there and forward the port from your laptop:

```bash
ssh -N -L 4213:127.0.0.1:4213 user@remote-host
```

Then open `http://127.0.0.1:4213` on the laptop. The first launch needs network
access so DuckDB can install its `ui` extension and the browser can load the UI
assets. DuckDB UI is an unrestricted local SQL explorer, so keep the port on
loopback and use a tunnel instead of exposing it publicly.

This command is separate from [`hflow serve`](./SERVE.md). `catalog ui` is
DuckDB's browser over the local catalog. `serve` is HFlow's optional workspace
REST API and does not ship a browser client.

### Query from Python or the CLI

```python
import hflow

report = hflow.curate(
    "data/catalog",
    "SELECT episode_id, uri FROM episodes WHERE status = 'ok'",
    output="data/manifest.parquet",
)
print(report.summary())
```

or on the command line:

```bash
hflow curate "SELECT episode_id, uri FROM episodes WHERE task = 'fold_napkin'" \
    --catalog data/catalog --output data/manifest.parquet
```

For a query stored in a file, pass `--sql-file` instead of the positional SQL
string:

```bash
hflow curate --sql-file query.sql \
    --catalog data/catalog --output data/manifest.parquet
```

Pass exactly one of the positional SQL string or `--sql-file`; passing both or
neither is an error.
The curate() function accepts exactly one SELECT statement

The manifest is written **manifest-last**: to a temp file, renamed into place
only after the query completed, so a partial manifest is unreachable.

The same API accepts object-store locations when the optional backend is
installed (`uv sync --extra bucket`):

```python
app = hflow.App("pipeline", data_root="s3://robot-data/production")
report = hflow.curate(
    "s3://robot-data/production/catalog",
    "SELECT episode_id, uri FROM episodes WHERE status = 'ok'",
    output="s3://robot-data/manifests/training.parquet",
)
```

Provider credentials come from the standard AWS, GCP, or Azure environment.
Workers process local files in an etag-validated mirror (override its base with
`HFLOW_MIRROR_DIR`), while canonical episodes, artifacts, completion markers,
catalog rows, and manifests publish to the bucket. Catalog table files are
append-only and content-named, so a mirror only downloads rows it does not yet
have.

Running SQL you did not write? Pass `constrained=True` to `curate()` or
`open_catalog_connection()`: the DuckDB connection's file access is limited
to the catalog (plus the manifest's own destination), extension
auto-install/auto-load is off, and the configuration is locked -- the
posture a service uses for tenant-supplied SQL
([docs/HOSTING.md](./HOSTING.md#trust-model)). The default stays
unrestricted for your own exploration.

### The view surface

`hflow.open_catalog_connection(catalog_root)` returns a DuckDB connection with
these views registered (it is what `curate()` uses; take it and explore):

- `episodes`: the wide view for everyday cuts. It has the latest row per
  episode, a `status` column (`'quarantined'` / `'unverified'` / `'ok'`), and **one numeric
  column per measurement key** (latest value; booleans as 0/1).
- `episodes_raw`, `check_runs`, `measurements`, `tags`, `intervals`,
  `ingest_failures`: the long tables, exactly as stored.
- `episodes_latest`, `measurements_latest`: one row per episode / per
  (episode, key), most recent append wins.

### Naming measurement keys

Because every numeric key becomes a column of the wide view, key names are a
queryable surface with no rename path: old rows keep the old name forever.
Three rules, in decreasing order of how much it hurts to get them wrong.

**One key, one owner.** Every step of a run shares that run's fingerprint and
timestamp, so two steps recording the same key on one episode is a tie
`measurements_latest` resolves arbitrarily -- one value silently disappears,
and the survivor is attributed to whichever step the reader assumes. The runner
refuses this at the point it happens, naming both steps. Prefixing keys with
their topic avoids it by construction; where a quantity is genuinely
episode-scoped, prefix it with the check instead.

**Name keys `<topic>/<metric>_<unit>`.** The topic prefix is what keeps two
cameras (or two checks) from colliding, and it also keeps a key from clashing
with an episode column like `task` or `uri`. Two namespaces are not yours to
take, and both are refused at append time rather than resolved silently: an
episode column name, refused naming the check, the key, and the column it
shadows; and anything starting with `artifact/`, which is reserved for the URIs
the framework publishes -- a snapshot export treats every key under that prefix
as media, so a user key there would ship as media the pipeline never produced.
Keys containing `/` need double quotes in SQL:

```sql
SELECT episode_id,
       "/wrist_cam/compressed/black_frame_pct" AS black_pct,
       "/wrist_cam/compressed/decode_deficit_pct" AS decode_deficit_pct
FROM episodes WHERE black_pct < 5.0 AND decode_deficit_pct = 0.0
```

The built-in `camera_frame_stats` check keeps two deficits separate:
`frame_deficit_pct` measures missing messages against the expected rate, while
`decode_deficit_pct` measures messages present whose frames did not decode.

**Omit the key rather than measuring NaN or infinity.** A non-finite float is
refused at append time, naming the check and the key. NaN compares false against
everything, so an episode carrying one falls out of both a threshold and its
complement: it is in neither `black_pct < 1.0` nor `black_pct >= 1.0`, and it
poisons any `avg()` over the corpus. A check with nothing to say about an episode
says nothing, and the coverage denominator already records that it ran.

**End the key with a unit the tooling knows.** The workspace UI labels a value
by the token after the key's last `_`, so `_s`, `_ms`, `_ns`, `_hz`, `_pct`,
`_ratio`, `_count`, `_bytes`, `_deg`, `_rad`, and `_m` render with their unit
and anything else renders as a bare number. Avoid the word `duration` in a key
that is not an episode length: it is read as a candidate for the episode's
timeline span, so a summed denominator named `..._duration_s` would claim to be
one.

### Worked queries

The everyday cut (this is the README example, running for real):

```sql
SELECT episode_id, uri FROM episodes
WHERE task = 'fold_napkin'
  AND status = 'ok'            -- not just un-quarantined; see the status table below
  AND black_pct < 1.0          -- measurement keys are columns; units are yours
```

For the everyday cut you do not have to write any of this. `hflow dataset
create` asks the pipeline itself:

```bash
hflow dataset create clean
hflow dataset create clean --print-sql   # see the policy, change nothing
```

It selects the current generation of every source recording that is not
quarantined, was produced by this pipeline's current transform, and has every
registered step recorded at its current version, then writes two immutable
files: `manifests/clean-<timestamp>.parquet` (the selection, which
`hflow export snapshot --manifest` reads) and `manifests/clean-<timestamp>.json`
(the effective SQL, the version stamps it required, the row count, and the
coverage). Nothing is hidden -- `--print-sql` gives you the query to edit into
a sharper one, and `--sql` runs yours instead while keeping the artifact and
the provenance record.

Two subtleties worth knowing if you write the equivalent yourself, because
both of them silently return an empty dataset that reads like a policy
decision:

- The policy asks whether each step **ran**, not whether it passed.
  Evidence-only checks offer no verdict and record `measured`, so a
  `status = 'passed'` filter selects nothing at all.
- It counts `superseded` as settled too, not only `passed`/`failed`/`measured`.
  A built-in you wrapped under a name of your own supersedes the automatic copy,
  which then records `superseded` on every episode by design -- reading that as
  an unfilled hole excludes the whole corpus. `skipped` is different and stays
  excluded: a step skipped because a critical check quarantined the episode has
  real work to do the moment that check is retuned. So does `error`: a crash is
  infrastructure, so it is a retry.

An episode inherits a term of its own from that last case. A crash is a retry
for the *check*, but the episode it was supposed to check is left in a third
state, and the `status` column on the `episodes` view names it:

| `status` | what it means |
|---|---|
| `quarantined` | a critical check ran and returned a False verdict |
| `unverified` | a critical check crashed, so no verdict was ever produced |
| `ok` | every critical check that ran answered |

`unverified` is not a milder `quarantined`. Nothing has said this episode is
bad; the point is that nobody knows, because the check that would have said so
never finished. That distinction is why the value exists at all. Before it,
`status = 'ok'` meant "not quarantined", which covered *verified clean* and
*never verified* alike, so a filter meaning "episodes I have checked for blur"
silently included the ones where the blur check crashed.

Only `error` earns it. A critical check can also record `skipped` (the episode
was already quarantined upstream) or `superseded` (your pipeline measures the
same thing itself); neither is a crash, and neither leaves the episode
unchecked, so both still read `ok`. A non-critical check that crashes does not
change the episode's status either, for the same reason it cannot quarantine
one.

Two consequences worth stating, because both are easy to trip over:

```sql
-- Keeps episodes nobody checked: their critical check crashed, so the
-- verdict this filter is standing in for was never produced.
SELECT episode_id FROM episodes WHERE status != 'quarantined'

-- What you almost always mean.
SELECT episode_id FROM episodes WHERE status = 'ok'
```

`hflow dataset create` applies `status = 'ok'` for you, so its default policy
does not need this filter written by hand. It also requires every registered
step to have a settled `check_runs` row, and `error` is not settled (see
`SETTLED_STATUSES`).

Those two rules overlap, and neither replaces the other. An episode whose
critical check *only ever* crashed has no settled row at all, so the step
requirement excludes it on its own. An episode that settled once and crashed on
a later run is different: the step requirement is satisfied by the older row,
and only the status column still reads `unverified`. That case is why the
default policy filters on status rather than leaning on the step rule alone.

Your own selection SQL gets neither rule for free. `status = 'ok'` is the one
worth carrying over.

Pinning a check version by hand (the mixed-version-corpus reality).
`check_version` is an opaque identifier declared by the pipeline, not an
ordered number, so pin exact values (or filter
`recorded_at`), never compare with `>=`:

```sql
SELECT episode_id, value_double AS max_velocity
FROM measurements
WHERE key = '/joint_states/max_abs_velocity'
  AND check_version = 'joint-motion-v2'   -- from the pipeline or check_runs
```

Tags: e.g. every episode a non-critical check failed (`failed:<check>` is
recorded automatically; quarantine tags live on the episode row instead):

```sql
SELECT episode_id, check_name, tag FROM tags WHERE tag LIKE 'failed:%'
```

Intervals: total recording-gap time per episode from the built-in timestamp
check (`gap:<topic>` labels; `joint_discontinuity:<topic>` works the same):

```sql
SELECT episode_id, sum(end_ns - start_ns) / 1e9 AS gap_seconds
FROM intervals WHERE label LIKE 'gap:%'
GROUP BY episode_id ORDER BY gap_seconds DESC
```

Exact duplicates, from the built-in `content_digest` check (`episode_id`
already dedupes byte-identical canonical files; the digest catches the same
recorded content landed under different names or provenance):

```sql
SELECT value_text AS digest, count(*) AS copies, list(episode_id) AS episodes
FROM measurements WHERE key = 'content_digest'
GROUP BY digest HAVING count(*) > 1
```

### Cohort statistics

Per-episode checks record evidence; cohort math (z-score, percentile) is a query over the catalog.

```sql
SELECT episode_id,
       "/joint_states/message_rate_hz" AS rate_hz,
       (rate_hz - AVG(rate_hz) OVER ()) / STDDEV(rate_hz) OVER () AS rate_hz_z,
       PERCENT_RANK() OVER (ORDER BY rate_hz) AS rate_hz_pct
FROM episodes
WHERE task = 'fold_napkin'
```

Alias a topic-prefixed key once, as above, and the rest of the query stays
readable.

Cohort statistics are corpus-relative. The z-score depends on which rows the query runs over, including any WHERE clause applied before the window, so compute the window over the same filtered cohort you intend to cut. Also note that STDDEV over a single-row cohort is NULL, which is the correct answer: one episode has no cohort to compare against.

### Finding stale episodes to reprocess

The corpus is assumed permanently mixed-version, so "reprocess everything" is
never the plan; the plan is to find exactly which sources are behind and
re-ingest only those. `hflow.stale_episodes()` compares each source's **latest**
cataloged run against the versions you pass:

```python
stale = hflow.stale_episodes("data/catalog", pipeline_version=app.pipeline_version)
for episode in stale:
    app.process(episode.source_uri)
```

or on the command line, where stdout is exactly the pipeable source-URI list:

```bash
hflow stale --catalog data/catalog --pipeline pipeline.py | xargs hflow ingest
```

`--pipeline` imports your pipeline file and compares against its current
`pipeline_version` plus the current episode format version; pass
`--pipeline-version <hash>` instead to compare against a known hash without
importing anything. Staleness follows the source recording, not the episode
id: reprocessing mints a new content-addressed `episode_id`, and a source
whose newest run already carries the current versions is not stale.

Add `--exit-code` to make the command exit 1 when any stale episode is found,
so a CI job or a pre-training checklist can fail on drift the way
`hflow doctor` already does:

```bash
hflow stale --catalog data/catalog --pipeline pipeline.py --exit-code
```

Without the flag the command always exits 0, so existing pipes into
`xargs hflow ingest` keep working.

### Coverage denominators

Every `CurationReport` carries (and `summary()` always prints) which checks
ran on what fraction of episodes:

```text
manifest: data/manifest.parquet (14 rows, from 20 cataloged episodes)
coverage (episodes each check ran on):
  camera_blackout: 20/20 (100%)
  joint_jumps: 20/20 (100%)
  vlm_labeling: 9/20 (45%)
```

A statistic over half a delivery must not look like a statistic over all of
it: steps skip when an episode quarantines upstream, so partial coverage is
normal, and it must be visible, not inferred. "Ran" means a `check_runs`
status of `passed`, `failed`, or `measured` (a failed verdict still ran;
`skipped` and `error` did not produce evidence).

Registration order therefore decides how much evidence a quarantine costs. A
critical check that quarantines skips every check registered **after** it, and
those skips are exactly what this block reports as missing coverage -- so
register gating checks last, after the checks whose measurements you still want
on a rejected episode. Gating early is how a corpus ends up unable to answer
why anything was rejected: the evidence that would explain it was never
gathered, and re-deciding means reprocessing the media rather than writing a
different query.

## Enrichment labels and artifacts

Enrichments (`@app.enrich`) record into the same tables, so their outputs
curate exactly like check evidence: `labels` become measurements (a text
caption is queryable via `value_text`), each artifact lands as a text
measurement `artifact/<name>` holding the file's URI, and tags flow to the
tags table:

```sql
SELECT episode_id, value_text AS caption
FROM measurements WHERE key = 'caption'
```

## No lock-in: it's just Parquet

The views are conveniences. The files answer to anything:

```python
import duckdb

duckdb.execute(
    "SELECT key, avg(value_double) FROM read_parquet('data/catalog/measurements/*.parquet') "
    "WHERE value_double IS NOT NULL GROUP BY key"
).fetchall()
```

`pandas.read_parquet("data/catalog/episodes")` and polars equivalents work
the same way. The only reserved file is the `format_version` marker at the
catalog root (currently `1`); a breaking layout change bumps it, and readers
refuse loudly on mismatch.

When a person or another tool needs a selected snapshot rather than the
append-only catalog history, use
[`hflow export snapshot`](./how-to/export-dataset-snapshot.md). It writes
samples, latest typed measurements, artifact media index, step runs, tags, and
intervals as a portable directory of Parquet tables. Artifact URIs can be
preserved or copied into the directory; no viewer-specific package is
required.

## Troubleshooting

**`<location> is not a catalog root`.** Pass the catalog location created
beneath the data root, commonly `<data_root>/catalog`, rather than the data root
itself. A catalog root contains the `format_version` marker.

## Current limits

- The wide `episodes` view pivots **numeric** measurements only (booleans as
  0/1). Text-valued measurements stay in the long `measurements` table;
  query them there.
- A measurement key that collides with an episode column name (`task`,
  `uri`, `pipeline_version`, ...) or with the derived `status` column is
  **refused at append time**. Pivoted beside the real column, DuckDB would
  rename it to `task_1`, so `SELECT task` would silently return the metadata
  instead of the measurement. Follow the
  [naming rules](#naming-measurement-keys) and the collision cannot arise;
  appends written before this check keep their renamed ghost columns -- query
  such values in the long `measurements` table.
- The wide view binds its measurement columns to the keys present when the
  connection was opened. `hflow catalog ui` refreshes this surface after an
  initially empty catalog's first append, and `curate()` reopens per call.
  If a later run introduces brand-new measurement keys into a long-lived UI
  or `open_catalog_connection()` session, restart that session to add those
  columns. New rows for existing keys appear without a restart.

## See also

- [Architecture](./ARCHITECTURE.md): "Catalog and curation storage" for the
  rationale behind this design
- [Porting guide](./PORTING.md): how measurements get produced in the first
  place
- [How HFlow fits the robotics data stack](./INTEGRATIONS.md): the boundary
  between HFlow, Parquet, DuckDB, object storage, and training loaders
- [Frequently asked questions](./FAQ.md): direct answers about outputs,
  infrastructure, and project scope
