# Browse and curate a workspace in the browser: `hflow ui`

The workspace UI is a local web app over one data root: browse episodes and
their quality evidence, run curation SQL with instant previews, pin
manifests, monitor and trigger ingest runs, and inspect the registered
pipeline. The server never rewrites or deletes an episode -- the only files
it writes are the manifests you pin and its own small state file. It can
trigger an ingest run, though, which the runtime then writes through the
normal pipeline; `--read-only` refuses that along with the other writes.

The UI ships as a separate package, `hflow-ui`, on purpose: pipeline workers
install the `hflow` wheel into every task venv, and they should never carry
frontend assets. **It is not published to PyPI yet** -- until the first
release, run it from a clone (this also builds the frontend bundle, which the
published wheel will carry):

```bash
git clone https://github.com/Hebbian-Robotics/hflow.git
cd hflow
uv sync --all-extras                      # installs hflow and hflow-ui
(cd ui && pnpm install && pnpm build)     # the frontend bundle
HFLOW_UI_ASSETS=ui/dist uv run hflow ui   # browses $HFLOW_DATA_ROOT, else ./data
HFLOW_UI_ASSETS=ui/dist uv run hflow ui --data-root ./data --no-browser
```

Starting the server prints a one-time login URL
(`http://127.0.0.1:4356/?token=...`) and opens your browser.

## What it shows

- **Episodes** -- the corpus as a faceted, sortable table over the catalog's
  wide `episodes` view (task, operator, status, quality measurements as
  columns). Every filter you click compiles to DuckDB SQL server-side, and
  the exact SQL is always visible and copyable at the bottom of the screen --
  ready to paste into `hflow curate`.
- **Episode** -- one recording's dossier: status and quarantine tags, contact
  sheets, every check run with its content-hash version, measurements with
  their producing step, intervals, tags, append history, and a canonical-MCAP
  download.
- **Curate** -- a SQL studio over the catalog views: schema sidebar with
  per-table profiles, editor with run-selection, result preview with
  per-column statistics, and the coverage report (which checks ran over how
  much of the corpus) before you pin. **Pin manifest** freezes a query's
  result as an immutable Parquet manifest under `<data_root>/manifests/`,
  recorded with its SQL, row count, and coverage in the Manifests registry.
- **Runs** -- the ingest runtime's health, recent runs with their trigger
  configuration, per-stage activity, and a trigger form (`hflow ingest`'s
  wire shape, as a form). It addresses a rendered local bundle or a remote
  runtime (`HFLOW_AIRFLOW_URL` and friends); when neither is reachable the
  page says which it looked for and why it failed, rather than disappearing.
- **Pipeline** -- the registered steps by stage with content-hash versions,
  critical flags, and endpoint aliases, plus the versions actually observed
  in the catalog. Requires `--pipeline path/to/pipeline.py[:app]`, which
  imports (executes) the pipeline file exactly like `hflow manifest` does.

## Flags

| flag | meaning |
|---|---|
| `--data-root` | workspace to browse (default `$HFLOW_DATA_ROOT`, else `./data`) |
| `--host` | bind address (default `127.0.0.1`; widening past loopback exposes your corpus) |
| `--port` | default `4356`, auto-retries upward when taken |
| `--no-browser` | do not open a browser (headless machines, tunnels) |
| `--no-token` | disable the session token (trusted, loopback-only machines) |
| `--read-only` | viewer mode: hides and refuses manifest pinning, saved-query edits, and run triggering |
| `--pipeline` | pipeline file for the Pipeline page (imported once at startup) |

## Nothing is UI-only

The UI is a strict client of a documented JSON API (`/api/v1/...`; a running
server publishes its OpenAPI schema at `/api/openapi.json`, ready for a client
generator or any local OpenAPI viewer). Curation, the runs monitor and the
pipeline page are thin calls into the same library functions the CLI uses; the
episode listing, facets, stats and timeline endpoints compile their own
presentation-shaped SQL over the same [catalog views](./CATALOG.md) that
`hflow curate` reads. Either way, everything the UI can show or do is
reachable with `curl`, scriptable, and buildable-upon. If you want a different
frontend over your workspace, the API is the contract; the shipped UI is the
reference client.

## Trust posture

The UI runs fully local: all assets ship in the wheel (no CDN, no fonts, no
outbound requests), and your data never leaves your machine. That is why the
server publishes the schema JSON and no interactive Swagger page -- FastAPI's
built-in one fetches its JavaScript and CSS from a public CDN, which would
break the promise and run third-party script inside your logged-in session.
The server binds loopback with a random per-session token; the browser never
sees filesystem paths of its choosing (media is addressed by episode and
artifact name, and the server refuses anything outside the data root), Airflow
credentials stay server-side behind a proxy, and curation SQL runs on a
[constrained DuckDB connection](./CATALOG.md) that cannot reach the catalog's
files or the network. What this server writes: `<data_root>/ui/state.json`
(saved queries and the manifest registry) and your pinned manifests -- nothing
else. Episodes, media and catalog rows are written by the ingest runtime, on
runs you trigger from the Runs page.

## See also

- [Catalog tables and curation API](./CATALOG.md) -- the views and SQL idioms
  the Episodes and Curate screens are built on
- [Runtime guide](./RUNTIME.md) -- the Airflow runtime the Runs screen fronts
- [Hosting HFlow](./HOSTING.md) -- the data-plane contract for operating
  workspaces for other people, whose seams (bucket data roots, scoped
  credentials, constrained SQL, remote runtime addressing) are the ones this
  UI reads through
