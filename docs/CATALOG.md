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
`<data_root>/catalog/`, all named `<episode_id>-<run_fingerprint>.parquet`:

| Table | One row per | Carries |
|---|---|---|
| `episodes` | episode append | `uri`, `source_uri`, version stamps (`schema_version`, `pipeline_version`, `robot_software_version`, `ffmpeg_version`), promoted semantics (`task`, `operator`, `success`, `embodiment`), the rest of `episode/v1` as `metadata_json`, `quarantined` + `quarantine_tags_json`, `recorded_at` |
| `check_runs` | (episode, step) invocation | `check_name`, `check_version`, `critical`, `status`, `duration_s`, `error`; present even when a step produced nothing, which is what makes coverage countable |
| `measurements` | measurement key | `key`, typed value columns (`value_double`, `value_text`, `value_bool`), the producing `check_name`/`check_version` |
| `tags` | tag | `check_name`, `tag` |
| `intervals` | labeled time span | `label`, `start_ns`, `end_ns` |

Three durability rules govern writes:

- **Content-addressed**: `episode_id` is a sha256 of the canonical file's
  bytes. Re-ingesting an unchanged episode dedupes; a reprocessed episode
  (new `pipeline_version`) is a distinct fact.
- **Create-if-absent**: an append whose `(episode_id, run_fingerprint)` file
  already exists is a no-op (`written=False`). The fingerprint includes the
  observable outcome: exact retries deduplicate, while a successful retry
  after an error appends the repaired result.
- **Append, never overwrite**: change a check's source or config and its
  `check_version` changes, so re-running adds *new-version* rows next to the
  old ones. The corpus is assumed permanently mixed-version; curation picks.

One timestamp per append: every file of one `(episode_id, run_fingerprint)`
carries the episodes file's `recorded_at` (its create-if-absent is the
atomic commit). The winner force-aligns the dependent tables after
committing, and a replayed append re-checks and heals any dependent a
crashed earlier attempt left with a stale timestamp -- so concurrent
duplicate appends and retried tasks converge instead of stitching two runs'
rows together.

## Querying

```python
import hflow

report = hflow.curate(
    "data/catalog",
    "SELECT episode_id, uri FROM episodes WHERE status != 'quarantined'",
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
  episode, a `status` column (`'quarantined'` / `'ok'`), and **one numeric
  column per measurement key** (latest value; booleans as 0/1).
- `episodes_raw`, `check_runs`, `measurements`, `tags`, `intervals`: the
  long tables, exactly as stored.
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
with an episode column like `task` or `uri`, which makes the wide view error.
Keys containing `/` need double quotes in SQL:

```sql
SELECT episode_id, "/wrist_cam/compressed/black_frame_pct" AS black_pct
FROM episodes WHERE black_pct < 5.0
```

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
  AND status != 'quarantined'
  AND black_pct < 1.0          -- measurement keys are columns; units are yours
```

Pinning a check version (the mixed-version-corpus reality). `check_version`
is a content hash, not ordered, so pin exact values (or filter
`recorded_at`), never compare with `>=`:

```sql
SELECT episode_id, value_double AS max_velocity
FROM measurements
WHERE key = '/joint_states/max_abs_velocity'
  AND check_version = '1a2b3c4d5e6f'   -- from check_runs or a previous query
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
  connection was opened. `curate()` reopens per call, so this only matters
  if you hold a long-lived connection from `open_catalog_connection()` while
  new keys are being recorded.

## See also

- [Architecture](./ARCHITECTURE.md): "Catalog and curation storage" for the
  rationale behind this design
- [Porting guide](./PORTING.md): how measurements get produced in the first
  place
- [How HFlow fits the robotics data stack](./INTEGRATIONS.md): the boundary
  between HFlow, Parquet, DuckDB, object storage, and training loaders
- [Frequently asked questions](./FAQ.md): direct answers about outputs,
  infrastructure, and project scope
