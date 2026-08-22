# Hosting HFlow: the data-plane contract

This page is for whoever operates HFlow for other people -- a platform team
running per-team workspaces, or the future managed offering README.md
gestures at. It names the contract this repository provides toward that
service: which seams exist, what crosses the control boundary, what the
trust model is, and where the current limits are. The control
plane itself (accounts, sign-up, billing, request authentication) is **not
in this repository** and is not a pre-v1 commitment; everything below is
about making that control plane an *addition*, never a rearchitecture.

The topology is the one [ARCHITECTURE.md](./ARCHITECTURE.md#tenancy)
commits to: a multi-tenant **control plane** over isolated per-customer
**workspaces**, where the open-source single-tenant engine is the
data-plane unit (the Dagster+/Prefect hybrid pattern). The design
consequence taken throughout: **only metadata, states, and pointers cross
the control boundary -- never episode bytes.**

## The workspace

A workspace is one data root plus everything the engine derives from it:
episodes, the Parquet catalog, test runs, and (for local roots) the rendered
runtime bundle. `hflow.workspace.Workspace` is the layout's one owner in
code, and `Workspace.ensure_identity()` mints a durable `workspace.json`
identity marker (create-if-absent) so a workspace keeps its id when its
storage moves -- a path is not an identity.

Two rules a hosted workspace must follow:

- **Bucket data roots only.** Local roots record host-absolute paths as
  durable catalog URIs; those are dead on arrival across any client/server
  split. Bucket roots record `gs://`/`s3://`/`az://` URLs, which any
  authorized reader can resolve.
- **Scoped credentials per workspace.** `BucketStorageRoot(url,
  store_options=...)` injects explicit per-root store configuration
  (credentials, region, endpoint) instead of mutating the process
  environment, so one process can hold several workspaces' distinct,
  least-privilege credentials. Storage-level isolation is only real when
  each workspace's credential reaches exactly its own prefix.

## What crosses the control boundary

| artifact | producer | contents |
|---|---|---|
| Pipeline manifest (`hflow manifest`, `App.manifest()`) | pipeline author's environment | step names, content-hash versions, gate flags, endpoint aliases, pipeline/schema versions -- what a service displays, diffs, and validates without holding the code. Producing it imports (executes) the pipeline, so treat it as the author's claims; the execution environment re-derives the same facts when it imports the code anyway. |
| Bundle manifest (`hflow-bundle.json`) | `hflow up` / `hflow deploy` | the rendered bundle as data: manifest version, kind, hflow version, DAG ids, data root, pipeline filename, app variable, requirements flag, task queue, venv interpreter path. The provisioning/upload contract. |
| Trigger conf | any REST caller | `{"uris": [...], "profile": ..., "mode": "batch"\|"online", "batch_count": ...}` on Airflow's `dagRuns` endpoint -- already the wire shape a control plane calls. |
| Run states | Airflow REST | `hflow status --airflow-url ...` (and `hflow.runtime.describe_remote_status`) report health, registration, and recent run states with no local files. |
| Catalog facts | curation | manifests, coverage reports, and stale lists carry URIs (pointers) and measurements -- not media. |

Episode bytes move only between the customer's collection systems and the
workspace's own bucket.

## Driving a workspace remotely

`hflow ingest`/`hflow status` address a runtime by URL instead of a local
bundle directory ([RUNTIME.md](./RUNTIME.md#remote-runtimes---airflow-url)).
`AirflowClient` authenticates with either the Compose bundle's
username/password flow or a **pre-issued bearer token**
(`hflow.runtime.BearerToken`) -- the shape where a control plane mints
scoped, expiring tokens per workspace and the SDK never sees an admin
password.

## The environment injection contract

Environment variables are how a deployment (Compose bundle, managed Airflow,
or a control plane provisioning a workspace) configures pipeline code it
must not edit. The full set:

| variable | read by | meaning |
|---|---|---|
| `HFLOW_DATA_ROOT` | `hflow.App` (when constructed without `data_root`) | the workspace's data root; the generated DAG tasks export it before importing the pipeline, so one file runs unedited at every vantage |
| `HFLOW_ENDPOINT_<ALIAS>` | `App` endpoints at run start | overrides (or supplies) the endpoint alias `<alias>` (uppercased, non-alphanumerics as `_`; aliases whose names collide under that mapping are refused at preflight); how per-workspace model endpoints are injected without touching customer code. The Compose renderer forwards variables exported at render time into the containers by name, like bucket credentials; deploy-mode workers set them directly (DEPLOY.md). |
| `HFLOW_AIRFLOW_URL` / `HFLOW_AIRFLOW_DAG_ID` | `hflow ingest` / `hflow status` | the remote runtime to address when no `--airflow-url`/`--dag-id` flag is given |
| `HFLOW_AIRFLOW_TOKEN`, `HFLOW_AIRFLOW_USERNAME`, `HFLOW_AIRFLOW_PASSWORD` | remote-endpoint resolution | credentials for the remote runtime (environment only, never argv; token wins) |
| `HFLOW_USER_DIR` | generated DAG tasks | where the pipeline file lives on the workers (default `/opt/user`) |
| `HFLOW_MIRROR_DIR` | `hflow.storage` | base directory for bucket-root spool mirrors; point it at per-job ephemeral disk on shared workers so tenant bytes do not accumulate in a machine-global cache |
| `HFLOW_FFMPEG` / `HFLOW_FFPROBE` | `hflow.ffmpeg` | operator-managed binaries instead of the pinned auto-download; bake these into hosted worker images (see licensing below) |

Step credentials (API keys for model endpoints) stay in the customer's own
client code via the environment, exactly like the examples: the seam a
deployment controls is *which* environment those step processes receive.

## Run semantics live in the library

`hflow.stage_execution` owns the lane planning (batch vs online), the
pipeline-loading contract, the per-episode accounting loop, and the
error/quarantine budgets (`max(8, ceil(1% of total))`, all-errors always
fails). The generated Airflow DAGs are thin callers -- so any other
execution backend (a hosted executor, a different scheduler, a plain worker
loop around `app.process()`) reuses the same semantics instead of copying
generated code. Bundles pin `hflow==<renderer's version>`, so rendered DAGs
and the library they call cannot skew inside one bundle.

Routing seams for shared or multi-pool schedulers, all rendered per bundle:

- `dag_id` prefixes namespace pipelines on a shared scheduler
  (`acme__kitchen_ingest` prefixes every sub-DAG id and UI tag).
- `task_queue` stamps every stage task with an Airflow queue, so a
  workspace's tasks route to its own worker pool. LocalExecutor ignores it.
- `xcom_objectstorage_url` replaces the single-host `file://` XCom store
  with a bucket URL. Note the failure mode without it: payloads under the
  4 KB threshold ride the metadata DB and *work* across machines, so a
  multi-machine executor with the file store breaks only on the first
  above-threshold batch plan -- in production, not in the demo.
- `api_bind_host` widens the api-server bind past loopback for a data plane
  fronted by its own gateway; the Postgres password is generated per bundle.

## Trust model

**The workspace is the trust boundary.** Inside one workspace, pipeline code
is trusted: the two-venv split is *dependency* isolation, not a security
boundary. Today, task processes can reach the runtime's ambient environment
-- including the workspace's storage credentials -- and have unrestricted
network egress (the documented enrichment pattern requires egress). The
generated-code path treats user-influenced strings as untrusted (identifier
validation, repr-literal injection), and tenant SQL has a dedicated posture:
`open_catalog_connection(..., constrained=True)` /
`curate(..., constrained=True)` lock DuckDB's file access to the catalog and
the manifest destination, disable extension auto-loading, and lock the
configuration -- use it for any curation surface that executes SQL the
operator did not write.

Consequences for a hosted deployment:

- Isolation between customers is **infrastructure per workspace** --
  separate compute, separate bucket (or rigorously scoped per-prefix
  credentials), separate Airflow -- not anything inside the engine.
- Per-workspace isolation does **not** contain a supply-chain compromise
  *within* a workspace: customer `requirements.txt` installs run arbitrary
  code at provision time, task processes hold the workspace's storage
  credentials, and egress is open -- one malicious dependency can exfiltrate
  that workspace's corpus. Mitigations (dependency lockfiles/scanning,
  short-lived scoped storage credentials, an egress policy) are the
  operator's infrastructure to provide today; see the limits below.

## Current limits

Stated plainly, in the spirit of [ARCHITECTURE.md](./ARCHITECTURE.md)'s
implemented/simplified/deferred matrix, so an operator sizes their
deployment against facts:

- **No metering or quotas.** `check_runs.duration_s` records per-step wall
  time, but everything in the catalog is written by the pipeline's own code
  -- treat it as informational, and meter from your orchestrator and storage
  inventory instead.
- **No deletion, retention, or erasure operation.** The catalog is
  append-only and quarantine never deletes. Robot episodes are video of
  real places and people: if your workspaces hold personal data, plan
  retention and erasure obligations at the storage layer, because the
  engine offers no episode-level erasure today.
- **No result-submission API.** Workers write the catalog directly, so
  every machine that executes steps needs credentials for the workspace's
  store.
- **No tenant-facing log or metrics API.** Observability is Airflow's own
  UI and task logs on the workspace.
- **The workspace UI (`hflow serve`) authenticates nobody.** It is a local
  developer tool bound to `127.0.0.1`, deliberately without a login; it is
  not a tenant-facing surface, and serving it to anyone but the workspace's
  own operator means putting an authenticating proxy in front of it. Signing
  people in and scoping them to a workspace is the control plane's job --
  per-user identity and revocable sessions, which no shared launch secret
  could stand in for.
- **Task processes share the runtime's environment**, including the
  workspace's storage credentials, and the venv build runs as root at
  provision time -- isolation between principals must come from your
  infrastructure (one workspace per trust domain), not from the engine.
- **A workspace's Airflow stack idles at several GB of RAM** across five
  long-running services (the compose file defines seven; two are one-shot
  init containers).
- **Engine upgrades re-version a corpus only when processing changed.** An
  hflow release no longer moves any identity by itself: `pipeline_version`
  folds in `hflow.behavior.TRANSFORM_BEHAVIOR_VERSION` (bumped deliberately,
  only when the transform would write different bytes) instead of the release
  number, the canonical file's header carries no release number, and step
  versions record the modules they reference by name rather than by version.
  A byte-identical input therefore keeps its `episode_id` across upgrades, so
  content-addressed dedupe holds. The flip side is a real one: an engine
  change that alters processing without a behavior bump is invisible to
  `hflow stale`, so operators upgrading across a behavior bump should expect
  exactly one corpus-wide re-version and plan reprocessing then. The corpus is
  designed to be permanently mixed-version, so curation pins keep working.
- **ffmpeg licensing**: the pinned build is BtbN's **GPL** variant (it
  carries the H.264 encoder the canonical transform needs). GPL source
  obligations attach to **redistribution** -- shipping worker images or
  bundles that contain it triggers them, not running it server-side --
  and a commercial deployment encoding H.264/H.265 at scale should review
  codec patent posture. Bake an operator-managed binary via
  `HFLOW_FFMPEG` when that review says so.

## See also

- [ARCHITECTURE.md](./ARCHITECTURE.md#tenancy) -- the tenancy decision this
  page implements the consequences of
- [RUNTIME.md](./RUNTIME.md) -- the Compose runtime, bucket data roots, and
  remote runtime addressing
- [CATALOG.md](./CATALOG.md) -- the catalog's tables, views, and concurrency
  idioms
