# Run HFlow pipelines with Airflow

Your pipeline already works: `app.test("episode_0001.mcap")` runs every check
in-process, in seconds, with no Docker and no scheduler. This page is the
graduation step: running **the same pipeline, unchanged**, as an Airflow DAG.
`hflow up` provisions a local Airflow in Docker Compose, `hflow ingest`
triggers runs over batches of episodes, Airflow's own UI shows every step, and
`hflow curate` cuts datasets from what landed in the catalog.

Nothing here is required for development. Iterate with `app.test()`; come here
when you want a scheduler doing the running (mapped batches, retries, a UI
with per-task logs) instead of your terminal.

## Prerequisites

- **Docker with Compose v2** (the `docker compose` subcommand). That is the
  entire requirement: Airflow is never a pip dependency of HFlow. The SDK
  renders files and speaks the REST API; it never imports Airflow.
- **~2 GB of one-time downloads.** The first `up` pulls the Airflow and
  Postgres images and builds the task venv; later starts reuse both.
- **Windows: use WSL2.** Run HFlow inside a WSL2 distribution (with Docker
  Desktop's WSL2 backend), and keep the data root on the Linux filesystem:
  the bundle records your Unix UID for container file ownership and bind-mounts
  the data root, both of which assume a Linux-side path.

No cloud account, no billing, no telemetry: everything below is containers on
your machine, and the only recurring costs are disk and CPU.

### Global flags

The following options apply to the `hflow` command itself and must appear
before the subcommand:

- `--version` prints the installed HFlow version and exits.
- `-v`, `--verbose` enables verbose logging.

Every subcommand follows the same three-value exit-code convention (0 success,
1 ran and found something to report, 2 bad input and nothing happened), which
matters when scripting `up` or `ingest`: see
[Exit codes](./FORMAT.md#exit-codes).

## What `up` builds: anatomy of the bundle

```bash
hflow up --pipeline pipeline.py --data-root ./data
```

(or, equivalently, `app.run()` under an `if __name__ == "__main__":` guard in
the pipeline file itself; unguarded, the runtime's own workers would re-run
it when they import your pipeline.)

This renders a self-contained bundle at `<data-root>/runtime` (override with
`--bundle-dir`) that you can inspect, edit, and drive with plain
`docker compose`. The SDK provisions the runtime; it never *becomes* the
runtime:

| file | what it is |
|---|---|
| `docker-compose.yaml` | The official Airflow 3.3.1 reference compose reduced to LocalExecutor: `postgres`, `airflow-init`, `user-venv-init`, `airflow-apiserver`, `airflow-scheduler`, `airflow-dag-processor`, `airflow-triggerer`. No Redis, no Celery worker. The API binds to `127.0.0.1` only (widen deliberately via `API_BIND_HOST` in `.env`). Overwritten on every re-render. |
| `.env` | Generated secrets (JWT secret, admin password, Postgres password), the API port and bind host, image tags, your UID. **Create-if-absent**: written once at `0600`, never overwritten by a re-render; your secrets and edits survive, and what's on disk wins over config. |
| `dags/` | The five generated DAG files (below): the master `ingest.py` (`dag_id` defaults to `<pipeline stem>_ingest`) plus the four stage sub-DAGs `ingest_sync.py` / `ingest_meta.py` / `ingest_labels.py` / `ingest_media.py` (`<stem>_sync` etc.). |
| `user/` | A copy of your pipeline file and requirements, mounted read-only into the containers. Refreshed on every re-render; re-run `up` after editing your pipeline. |
| `logs/` | Airflow task logs, readable from the host. |
| `hflow-bundle.json` | The bundle described as data: manifest version, kind (`compose`/`deploy`), hflow version, master and sub-DAG ids, data root, pipeline filename, app variable, whether requirements were included, task queue, and the task venv's interpreter path. What tooling (and `load_bundle`) reads instead of parsing generated code; `hflow deploy` emits the same file. |

Two details exist because their absence bites:

- **A unique Compose project name.** Without one, Compose names every project
  after the bundle directory's basename (`runtime` for every default bundle),
  and two bundles on one machine would silently adopt each other's containers
  and share volumes (`down --volumes` in one would wipe the other's database).
  Each bundle gets a name derived from its absolute path.
- **The user venv.** The worker carries two Python environments: Airflow's
  own, and a venv built from your `--requirements` file inside a named
  volume. Every DAG task runs in *your* venv via `@task.external_python`, so
  your dependencies and Airflow's ~128 pins never meet. The venv is keyed to a
  content hash of your requirements and the exact HFlow install target:
  unchanged inputs skip the rebuild entirely; a changed requirement or HFlow
  version rebuilds exactly once. Provisioning also pre-warms HFlow's pinned
  ffmpeg into the same volume, so the download happens under your eyes at `up`
  time instead of stalling the first task (best-effort: air-gapped
  provisioning still succeeds, and the task-time fallback remains).

When launched from an installed wheel, the task venv installs the exact same
HFlow version from PyPI. An editable install instead infers its source checkout
automatically so local changes reach the runtime. Pass
`--hflow-source /path/to/hflow` only to override that development checkout.

## The loop

**1. `up`.** The first start takes minutes, and narrates so it never looks
hung: rendering the bundle, pulling images and building the venv, a ~15 s
heartbeat while Airflow's components come up, then waiting for all five
ingest DAGs (the master and its four sub-DAGs) to actually register. `up`
only declares victory once every one is triggerable, so an immediate `ingest`
never 404s and the master never fires at an unregistered sub-DAG. It ends by
printing what you need:

```text
Airflow UI:  http://127.0.0.1:8080
credentials: airflow / <generated password>
ingest DAG:  pipeline_ingest
bundle:      data/runtime
ingest with: hflow ingest <episode-uri> --bundle-dir data/runtime
```

**2. `ingest`.** Trigger the master DAG over episode files, by path **relative
to the data root** (they resolve against `/opt/airflow/data` inside the
containers, so absolute host paths and `../` escapes are rejected before
anything is triggered):

```bash
hflow ingest episodes-in/run_0001.mcap episodes-in/run_0002.mcap
```

Two flags select what runs and how:

- **`--profile`** picks the run profile: which stage sub-DAGs the master
  enables. `full` (default) runs everything; `metadata_backfill` re-runs only
  checks + catalog registration; `relabel` re-runs only enrichments (the
  canonical files are never rewritten). Same graph, different toggles. Note
  the catalog stays idempotent per episode content and step versions: a
  backfill or relabel pass appends rows when something actually changed (an
  updated check or enrichment is a new content-hash version); re-running
  identical steps over identical episodes is a recorded no-op, not duplicate
  evidence.
- **`--online`** selects the latency-first trigger lane: the sub-DAGs process
  the URIs as **one immediate batch**, no bin-packing, no stagger delays.
  Meant for one run per episode as it lands (a collection rig posting each
  finished file); the default batch lane staggers near-equal-byte batches for
  throughput over shards.

This wraps Airflow's REST API v2 `dagRuns` endpoint with
`{"uris": [...], "profile": "full", "mode": "batch"}` in the run conf. A
collection rig or upload script can call the same endpoint directly, no CLI
required (`"mode": "online"` is the `--online` lane).

**3. Watch it in the Airflow UI.** Open the printed URL, log in with the
printed credentials. HFlow ships no UI of its own: per-task status, logs,
retries, and re-runs are Airflow's screens, exposed on localhost.

**4. `curate`.** Every processed episode appended rows to the Parquet catalog
under the data root, the same files whether the run came from `app.test(...,
record=True)` or the DAG:

```bash
hflow curate "SELECT episode_id, uri FROM episodes WHERE status != 'quarantined'" \
    -o manifest.parquet
```

See [the catalog guide](./CATALOG.md) for the views and the query patterns.

## The five ingest DAGs: a master and four stage sub-DAGs

> **The ingest stage graph.** A master DAG resolves the run profile and
> triggers only the sub-DAGs it enables -- the ingestion shape Dyna's article
> describes. Two trigger lanes lead in: online (per-episode, latency-first)
> and batch (per-shard, staggered).

**The master (`<stem>_ingest`)** runs entirely in Airflow's own environment:
no user venv, no hflow import. A `resolve_profile` task validates
the conf's `profile` against the profile vocabulary (baked in from
`hflow.steps.RUN_PROFILES` at render time, so the runner and the DAGs can
never disagree); each stage's trigger then sits behind an `enabled_<stage>`
gate task that skips when the profile disables its stage, and one
`TriggerDagRunOperator` per **enabled** stage fires
its sub-DAG and waits for it (deferrable; the triggerer service carries the
wait), chained strictly `sync → meta → labels → media`. Disabled stages show
as skipped tasks in the UI; a failed sub-DAG stops the chain and fails the
master run.

**What you see in the Airflow UI.** Every generated DAG is tagged with your
pipeline's stem, so one click on that tag in the DAG list filters to exactly
this pipeline's five DAGs (plus a `master` / `stage:<name>` role tag each).
Display names read as `<stem> · ingest (master)` and `<stem> · <stage>`, each
DAG's **Docs** tab renders a generated explainer (the master's includes the
profile table, baked from the same vocabulary the runner uses), and profile
toggling is visible live: a disabled stage shows as skipped tasks, while a
teal **deferred** trigger task is intentionally waiting on its sub-DAG; its
display name says so. Direct per-stage reruns show up in that sub-DAG's own
run history, which is the demo view of "same graph, different sub-DAGs
enabled."

**The four sub-DAGs** are each `plan → process_batch (mapped) → gate`, every
task running in your venv, and each runs exactly its own stage via
`app.process(uri, stages={...})`:

| sub-DAG | stage | gate |
|---|---|---|
| `<stem>_sync` | Transform & sync: the canonical transform (critical path) | error budget |
| `<stem>_meta` | Metadata: checks + catalog registration | **quarantine budget** + error budget |
| `<stem>_labels` | Labels & artifacts: enrichments (non-critical) | error budget |
| `<stem>_media` | Media: derived media (contact sheets) | error budget |

1. **`plan`** bin-packs the conf's URIs into near-equal-*byte* batches
   (first-fit-decreasing over file sizes, which vary widely and unbalance
   workers otherwise) and assigns each batch a staggered start
   delay. Batch count defaults to `min(4, episodes)`; override per run with
   `"batch_count"` in the conf. In **online mode** it instead returns the
   URIs as one immediate batch: no packing, no stagger, `batch_count`
   ignored.
2. **`process_batch`**, dynamically mapped over the batches, sleeps its
   stagger delay, imports your pipeline, and calls `app.process(uri,
   stages={<stage>})` per episode with catalog recording on: the same code
   path as `app.test()`, same gate semantics, same quarantine tags. A
   per-episode exception or collected step error is counted and skipped,
   never batch-fatal, so one corrupt file cannot sink its batch-mates.
3. **The gate** sums the counts and fails the run loudly over the budget of
   `max(8, ⌈1% of total⌉)` (and a run where every episode errors always
   fails). Checks decide quarantine, so the **quarantine half of the budget
   lives only in `_meta`'s `quarantine_budget_gate`**; the other stages keep
   the `error_budget_gate` half. Quarantine is a tag, never a deletion. The
   gate does not undo any work; it makes mass failure visible instead of
   letting a run that quarantined half its input report green.

## The one rule: the App's data root is the runtime's data root

Your `--data-root` directory is mounted at `/opt/airflow/data` inside every
container, and that is where episode URIs resolve and outputs land. The
recommended shape is to construct the App **without** a `data_root` argument:

```python
app = hflow.App("my-pipeline")  # resolves HFLOW_DATA_ROOT, else ./data
```

The generated process task exports `HFLOW_DATA_ROOT=/opt/airflow/data` before
importing your pipeline, so the same file runs unedited in the dev loop
(where the root falls back to `./data`, or whatever you export) and inside
the runtime. An explicit `data_root=` literal still works, but then it must
be exactly the runtime's root:

```python
app = hflow.App("my-pipeline", data_root="/opt/airflow/data")
```

An App pointing anywhere else would silently write into the container
filesystem, so the rule is enforced twice: a **render-time warning** when `up`
sees a differing `data_root=` literal in your pipeline file (best-effort
textual scan, minutes earlier than the real check), and a **run-time
refusal**, where the process task raises before touching any episode when the
imported App's `data_root` differs. The mounted directory is shared both
ways: the catalog and canonical episodes the containers write are ordinary
files on your host.

Endpoint aliases resolve the same way: for a pipeline that declares
`endpoints={"judge": ...}` and `uses="judge"`, exporting
`HFLOW_ENDPOINT_<ALIAS>` (alias uppercased, non-alphanumerics as `_`, e.g.
`HFLOW_ENDPOINT_JUDGE`) before `hflow up` makes the renderer forward that
variable into every container, overriding -- or supplying -- the alias
without editing the pipeline file. Point the same check at your local Ollama
in the dev loop and at the team's vLLM box in the runtime, one file, zero
edits. Forwarding is by name only (values never land in the bundle) and is
captured at render time, so re-run `up` after changing which variables your
shell exports.

## Bucket data roots: `--data-root gs://bucket/prefix`

The Compose runtime also runs directly against an object store, the shape
production robot fleets actually have (robots upload to a bucket; nothing
lands on the machine running Airflow):

```bash
hflow up --pipeline pipeline.py --data-root gs://my-bucket/robot-data
hflow ingest landing/run_0001.mcap
```

What changes against local mode, all decided at render time:

- **No data mount.** Episodes never touch the host filesystem; the App is
  constructed with the bucket URL itself
  (`data_root="gs://my-bucket/robot-data"`; the same one-rule check applies,
  now against the URL). URIs in ingest conf are keys under the prefix.
- **The task venv gets the `[bucket]` extra** (obstore) and every task spools
  through an etag-validated local mirror under the containers'
  `XDG_CACHE_HOME` (the user-venv volume), so downloads persist across
  container restarts and are shared by all stages.
- **The bundle defaults to `./runtime`** (a bucket has no local directory to
  host it; override with `--bundle-dir`), and XCom's file store moves to the
  bundle-local `xcom/` directory. `ingest`/`status`/`down` look in
  `./data/runtime` and then `./runtime` automatically, so the commands above
  work without `--bundle-dir`.
- **Credentials pass through, never in.** GCS: on GCE/GKE the containers
  reach the instance metadata server and need nothing at all; elsewhere the
  renderer mounts your `GOOGLE_APPLICATION_CREDENTIALS` file (or gcloud's
  application-default-credentials file) read-only. S3/Azure: the standard
  environment variables (`AWS_ACCESS_KEY_ID`…, `AZURE_STORAGE_ACCOUNT_NAME`…)
  are wired through as `${VAR}` references for the variables set in your
  shell at render time (variable *names* only; values never land in the
  bundle). Export the credentials before `up` (and before any later
  `docker compose up`), and re-run `up` after changing *which* variables you
  use.

Canonical episodes, contact sheets, sync-completion markers, and catalog
appends all publish back to the bucket; `hflow curate --catalog
gs://my-bucket/robot-data/catalog` then works from any machine with read
access.

## Remote runtimes: `--airflow-url`

`ingest` and `status` can address a runtime that is not on this machine --
a team Airflow, or a hosted workspace whose operator handed you a URL, a DAG
id, and a credential ([docs/HOSTING.md](./HOSTING.md)). No bundle directory
is involved:

```bash
export HFLOW_AIRFLOW_TOKEN=...   # or HFLOW_AIRFLOW_USERNAME + HFLOW_AIRFLOW_PASSWORD
hflow ingest episodes-in/run_0001.mcap \
  --airflow-url https://workspace.example.com --dag-id kitchen_ingest
hflow status --airflow-url https://workspace.example.com --dag-id kitchen_ingest
```

- The URL and DAG id also resolve from `HFLOW_AIRFLOW_URL` and
  `HFLOW_AIRFLOW_DAG_ID`, so a shell profile (or a control plane's injected
  environment) can set the workspace once and the plain commands just work.
- Credentials come from the environment only, never flags -- command lines
  leak into process listings and shell history. A pre-issued bearer token
  (`HFLOW_AIRFLOW_TOKEN`) wins over username/password.
- An explicit `--bundle-dir` keeps a command local even when
  `HFLOW_AIRFLOW_URL` is exported; an explicit `--airflow-url` wins the other
  way.
- Remote `status` reports health, DAG registration, and recent run states
  over the REST API alone (the local `docker compose ps` half does not apply).

## `status`: diagnostics

```bash
hflow status
```

prints the API URL, Airflow's own per-component health, and the
`docker compose ps` service table, plus a plain-language hint for each
unhealthy component, because component names don't explain consequences:

```text
health:  UNHEALTHY (dag_processor=unhealthy, metadatabase=healthy, ...)
hint:    dag_processor is unhealthy: DAG files are not being parsed, so the
         ingest DAG never appears or updates -- check `docker compose logs
         airflow-dag-processor`
```

The SDK explains; Airflow's UI observes. For anything deeper, the bundle is
plain Compose: `docker compose --file data/runtime/docker-compose.yaml logs
<service>` is endorsed usage.

## `down`: what stops, what survives

```bash
hflow down              # stop containers; all state survives
hflow down --volumes    # full reset: also remove the named volumes
```

Plain `down` stops the containers; the next `up` is fast. `--volumes`
additionally discards the two named volumes:

- **the Postgres volume**, Airflow's metadata DB: run history, task states,
  the admin user (recreated from `.env` on the next `up`);
- **the user-venv volume**: the built venv and the pinned-ffmpeg cache; the
  next `up` rebuilds and re-downloads both.

Your **data root is not a volume** and is never touched by `down`: episodes,
the catalog, and manifests survive even a `--volumes` reset. So does the
bundle's `.env` (it's a file, and re-renders preserve it); delete the bundle
directory itself if you want fresh secrets.

## Bring-your-own Airflow: `hflow deploy`

If you already operate an Airflow 3 deployment (Astronomer, Amazon MWAA,
Cloud Composer, or self-managed), skip the Compose runtime entirely:

```bash
hflow deploy --pipeline pipeline.py --data-root-uri s3://my-bucket/robot-data
```

This emits plain files under `./deploy/` (or the directory passed with
`--output-dir`) and calls no platform API. Use `--requirements` to provide a
user requirements file for the task venv; it has the same meaning here as for
`hflow up`. The output contains the same
five ingest DAGs the Compose runtime generates (one set of templates, rendered
against your data root and your workers' venv interpreter, overridable with
`--venv-python`; each file is named after its dag id so synced dags/ folders
never collide),
your `user/` files, an `.airflowignore` so platforms that sync `user/` inside
the dags folder never parse your pipeline in Airflow's own environment, and a
**`DEPLOY.md`** with the concrete values already filled in: where `dags/` and
`user/` go per platform, the `HFLOW_USER_DIR` environment variable that
tells the DAG where `user/` landed, the task venv's package list, and the
trigger call.

The data root must be a supported object-store prefix (`s3://`, `gs://`, or
Azure) or an absolute filesystem path every worker mounts. Relative paths
cannot mean the same thing on remote workers, so they are refused at render
time. Bucket deployments install the `hflow[bucket]` extra in the task venv,
use the provider's standard credentials, and spool processing through an
etag-validated local mirror. Set `HFLOW_MIRROR_DIR` to place that cache on
appropriate worker-local storage. Canonical episodes, artifacts, completion
markers, catalog appends, and curation manifests publish back to the bucket.

## Troubleshooting

**An ingest DAG never appears / `ingest` keeps 404ing.** Health can be green
while a DAG file fails to import; component health says nothing about
parsing. Check the parser's logs:

```bash
docker compose --file data/runtime/docker-compose.yaml logs airflow-dag-processor
```

`up` itself waits for registration and, on timeout, points here; a 404 from
`ingest` prints the same hint (right after a fresh `up`, the DAG may still
be parsing; retry in a few seconds).

**Port 8080 is taken.** For a bundle that does not exist yet, pass the port:
`hflow up --api-port 9090` (1 to 65535; anything outside that is refused before
the bundle is rendered). For one already rendered, the `.env` on disk wins
(re-renders never overwrite a preserved `.env`, so `--api-port` does nothing
there and editing does stick): `hflow down`, edit `API_PORT`, `hflow up`
again. The API binds to `127.0.0.1` by default (`API_BIND_HOST` in `.env`
widens it deliberately), so two bundles on different ports coexist fine.

**`up` failed partway.** Whatever started is deliberately left running: the
state *is* the diagnosis. `hflow status` to look, `docker compose ... logs
<service>` to dig, `hflow down` to clear it.

**Edited the pipeline, runtime still runs the old code.** `user/` is a copy,
refreshed at render time: re-run `hflow up` (idempotent; running
containers pick up the new files, secrets survive).

**"Is this costing me anything?"** No. The stack is local containers with
local storage; there is no hosted service, account, or metered anything
behind it. The costs are ~2 GB of images plus the venv on disk, and your CPU
while runs execute.

## Runtime overhead

In the spirit of [the benchmark report](./BENCHMARKS.md): the runtime has a
latency floor. Know where it is before you attribute slowness to your own code.

Airflow schedules tasks in *seconds*, not milliseconds. Every sub-DAG task
also boots a fresh interpreter in your venv (the external-python isolation is
paid per task), each enabled stage is its own sub-DAG run of at minimum three
tasks (`plan`, one mapped `process_batch`, the gate) behind a master trigger
task, and batch starts add deliberate stagger delays (a full-profile run is
four sub-DAG runs in sequence; `--profile` trims it, `--online` removes the
stagger but not the per-task floor).
Ingesting one two-second episode therefore takes orders of magnitude longer
under the DAG than the sub-second transform inside it: pure orchestration
overhead. This is the design trade: the architecture keeps steps coarse (one
task processes a *batch* and runs for seconds to minutes, hot loops live
inside a task, never across tasks) so the per-task floor is amortized over
real work.

Two consequences to act on:

- **Never iterate on a check through the DAG.** That is what `app.test()` is
  for: the identical pipeline, in-process, seconds per cycle, no floor. The
  runtime exists for throughput over batches, retries, and observability,
  not for the edit-run loop.
- **Feed the DAG batches, not single files.** The floor is per *task*, so ten
  episodes in one `ingest` cost roughly the same overhead as one episode.
  The planner exists to spread real work evenly, not to make tiny runs fast.

## See also

- [Porting guide](./PORTING.md): wrapping your existing QC code; the
  `app.test()` dev loop this page graduates from
- [Architecture](./ARCHITECTURE.md): why the runtime is provisioned, not
  imported; deployment modes; the failure taxonomy behind the gate
- [Catalog and curation](./CATALOG.md): querying what the runs recorded
- [How HFlow fits the robotics data stack](./INTEGRATIONS.md): the boundary
  between HFlow, Airflow, object storage, and downstream tools
- [`references/airflow3-notes.md`](../references/airflow3-notes.md): the
  cited Airflow 3.3.1 facts the bundle encodes
