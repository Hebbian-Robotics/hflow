# Browse and curate a workspace in the browser: `hflow ui`

The workspace UI is a local web app over one data root: browse episodes and
their quality evidence, run curation SQL with instant previews, pin
manifests, monitor ingest runs, and inspect the registered pipeline. It is
read-only toward your corpus by default-posture design: the only things it
ever writes are the manifests you pin and its own small state file.

```bash
uv add hflow-ui
hflow ui                       # browses $HFLOW_DATA_ROOT, else ./data
hflow ui --data-root ./data --no-browser
```

Starting the server prints a one-time login URL
(`http://127.0.0.1:4356/?token=...`) and opens your browser. The UI ships as
a separate package on purpose: pipeline workers install the `hflow` wheel
into every task venv, and they should never carry frontend assets.

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
- **Runs** -- the local Airflow runtime's health, recent ingest runs with
  their trigger configuration, per-stage activity, and a trigger form
  (`hflow ingest`'s wire shape, as a form). Appears when a rendered bundle or
  a remote runtime (`HFLOW_AIRFLOW_URL` and friends) is reachable; hidden
  otherwise.
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

The UI is a strict client of a documented JSON API (`/api/v1/...`; interactive
schema at `/api/docs` on a running server), and each endpoint is a thin call
into the same library functions the CLI uses -- so everything the UI can show
or do is reachable with `curl`, scriptable, and buildable-upon. If you want a
different frontend over your workspace, the API is the contract; the shipped
UI is the reference client.

## Trust posture

The UI runs fully local: all assets ship in the wheel (no CDN, no fonts, no
outbound requests), and your data never leaves your machine. The server binds
loopback with a random per-session token; the browser never sees filesystem
paths of its choosing (media is addressed by episode and artifact name, and
the server refuses anything outside the data root), Airflow credentials stay
server-side behind a proxy, and curation SQL runs on a
[constrained DuckDB connection](./CATALOG.md) that cannot reach the catalog's
files or the network. State on disk: `<data_root>/ui/state.json` (saved
queries and the manifest registry) and your pinned manifests -- nothing else.

## See also

- [Catalog tables and curation API](./CATALOG.md) -- the views and SQL idioms
  the Episodes and Curate screens are built on
- [Runtime guide](./RUNTIME.md) -- the Airflow runtime the Runs screen fronts
- [Hosting HFlow](./HOSTING.md) -- the same UI's place in an operated,
  multi-workspace deployment
