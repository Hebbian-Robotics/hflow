# Serve the workspace REST API: `hflow serve`

`hflow serve` is a local read-mostly server over one data root: it answers
questions about episodes and their quality evidence, compiles curation SQL,
pins manifests, monitors and triggers ingest runs, and describes the
registered pipeline. The server never rewrites or deletes an episode -- the
only files it writes are the manifests you pin and its own small state file.
It can trigger an ingest run, though, which the runtime then writes through
the normal pipeline; `--read-only` refuses that along with the other writes.

**The REST API is the product surface, not an implementation detail of some
frontend.** Every fact a browser could show is reachable from `/api/v1`, and
the OpenAPI schema at `/api/openapi.json` describes all of it -- so a
workspace UI is a *client*, and you can build or swap one without touching
this package. The server ships no frontend of its own; point
`HFLOW_UI_ASSETS` at a directory containing an `index.html` to serve one, or
install a wheel that packages assets under `hflow_server/static/`.

One such client lives in this repo at [`ui/`](../ui/README.md): a single canvas
that draws an ingest run and drills from the run into a stage, into the steps
that run inside a batch, and into the episodes that run recorded. It is built
separately (`cd ui && pnpm build`) and served through `HFLOW_UI_ASSETS`.

It ships as a separate package, `hflow-server`, on purpose: pipeline workers
install the `hflow` wheel into every task venv, and they should never carry a
web server. **It is not published to PyPI yet** -- until the first release,
run it from a clone:

```bash
git clone https://github.com/Hebbian-Robotics/hflow.git
cd hflow
uv sync                              # installs hflow and hflow-server
uv run hflow serve                      # browses $HFLOW_DATA_ROOT, else ./data
uv run hflow serve --data-root ./data --no-browser
```

Starting the server prints its URL (`http://127.0.0.1:4356/`) and opens your
browser. With no assets installed, that URL serves a page pointing at the
API. There is no login: see [Trust posture](#trust-posture) for what that
means and when it stops being appropriate.

## What the API exposes

These are API capabilities, not browser screens. A frontend supplied through
`HFLOW_UI_ASSETS` or any other HTTP client can use them:

- **Episodes data** -- the corpus over the catalog's
  wide `episodes` view (task, operator, status, quality measurements as
  columns). Filters compile to DuckDB SQL server-side, and the response
  includes SQL that can be used with `hflow curate`. `orchestrator_run_id`
  filters to what one orchestrated run recorded, which is the join from a run
  the runs endpoints are reporting on back to the corpus it produced.
- **Episode data** -- one recording's dossier: status and quarantine tags, contact
  sheets, every check run with its content-hash version, measurements with
  their producing step, intervals, tags, append history, and a canonical-MCAP
  download.
- **Curation** -- catalog schema and SQL execution with result rows,
  per-column statistics, and the coverage report (which checks ran over how
  much of the corpus). Pinning a manifest freezes a query's
  result as an immutable Parquet manifest under `<data_root>/manifests/`,
  recorded with its SQL, row count, and coverage in the Manifests registry.
- **Runs** -- the ingest runtime's health, recent runs with their trigger
  configuration, per-stage activity, and the same trigger operation used by
  `hflow ingest`. It addresses a rendered local bundle or a remote
  runtime (`HFLOW_AIRFLOW_URL` and friends); when neither is reachable the
  API reports which it looked for and why it failed.
- **Pipeline data** -- the generated DAG plus the registered steps by stage, with
  content-hash versions, critical flags, and endpoint aliases, and the
  versions actually observed in the catalog. The data nests each stage's
  steps inside its `process_batch` node, which is where they run, instead of
  inventing dependency edges between them. Requires
  `--pipeline path/to/pipeline.py[:app]`,
  which imports (executes) the pipeline file exactly like `hflow manifest` does.

## Flags

| flag | meaning |
|---|---|
| `--data-root` | workspace to browse (default `$HFLOW_DATA_ROOT`, else `./data`) |
| `--host` | bind address (default `127.0.0.1`; widening past loopback exposes your corpus) |
| `--port` | default `4356`, auto-retries upward when taken |
| `--no-browser` | do not open a browser (headless machines, tunnels) |
| `--read-only` | refuse manifest pinning, saved-query edits, and run triggering |
| `--pipeline` | pipeline file for the pipeline metadata endpoints (imported once at startup) |

## No frontend is shipped

Any UI is a strict client of a documented REST API (`/api/v1/...`; a running
server publishes its OpenAPI schema at `/api/openapi.json`, ready for a client
generator or any local OpenAPI viewer). Curation, runtime monitoring, and
pipeline metadata are thin calls into the same library functions the CLI uses; the
episode listing, facets, stats and timeline endpoints compile their own SQL
over the same [catalog views](./CATALOG.md) that `hflow curate` reads. The API
is reachable with `curl`, scriptable, and buildable-upon. There is currently
no reference browser client in this repository.

## Trust posture

**The server is unauthenticated.** There is no login, no token, and no
session: anyone who can reach the bound address can read your whole workspace
and trigger ingest runs. What protects it is the address it binds --
`127.0.0.1` by default, reachable only from your own machine. This is the
posture of every local developer tool that browses a working directory
(`mlflow ui`, TensorBoard, `dagster dev`, the DuckDB UI): a credential in
front of a single-user machine buys nothing but friction.

Passing `--host` past loopback is therefore a deliberate exposure, and it is
the only flag that changes who can reach the data. If you need the UI from
another machine, forward the port over SSH (`ssh -L 4356:127.0.0.1:4356
host`) rather than binding a network interface; if you must bind one, put a
reverse proxy that authenticates in front of it and firewall the port itself.
`--read-only` narrows what a reacher can *do* (no pins, no saved-query edits,
no triggering) but not what they can *read* -- it is a safety catch, not
access control.

Hosted, multi-user HFlow is a different problem and is solved elsewhere: the
control plane authenticates people and scopes them to workspaces
([HOSTING.md](./HOSTING.md)). That needs per-user identity and revocable
sessions, which one shared launch secret could never provide -- which is why
this server does not pretend to have a piece of it.

The rest of the posture is real and holds regardless. The server runs fully
local, and your data never leaves your machine unless a supplied frontend or
another API client sends it elsewhere. The server publishes the schema JSON
and no interactive Swagger page -- FastAPI's built-in one fetches
its JavaScript and CSS from a public CDN, which would break the promise and
run third-party script same-origin with your workspace's API. Media is
addressed by episode and artifact name rather than caller-chosen filesystem
paths, and the server refuses anything outside the data root. Airflow
credentials stay server-side, and curation SQL runs on a
[constrained DuckDB connection](./CATALOG.md) that cannot reach the catalog's
files or the network. What this server writes: `<data_root>/curation/state.json`
(saved queries and the manifest registry) and your pinned manifests -- nothing
else. Episodes, media and catalog rows are written by the ingest runtime, on
runs triggered through the API or CLI.

## See also

- [Catalog tables and curation API](./CATALOG.md) -- the views and SQL idioms
  the episode and curation endpoints use
- [Runtime guide](./RUNTIME.md) -- the Airflow runtime the run endpoints address
- [Hosting HFlow](./HOSTING.md) -- the data-plane contract for operating
  workspaces for other people, whose seams (bucket data roots, scoped
  credentials, constrained SQL, remote runtime addressing) are the ones this
  server reads through
