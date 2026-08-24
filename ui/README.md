# hflow workspace UI

One canvas over an ingest run. It draws the run's graph, and every node you can
open leads one level further in:

```
run  ->  stage  ->  steps          the orchestration, and the code inside it
run  ->  episodes  ->  episode     the data that run produced
```

- **run** -- the ingest DAG as a chain: resolve the profile, then each stage.
- **stage** -- one stage's sub-DAG: plan the batches, fan `process_batch` out
  over them, close on a budget gate.
- **steps** -- what one `process_batch` does to each episode: the pipeline's
  registered checks and enrichments, plus the engine's own work.
- **episodes** -- the episodes whose current catalog row came out of this run.
- **episode** -- every check recorded for it, with its verdict, its gate, and
  the measurements it was judged on.

A node with a `>` opens; the inspector on the right explains whatever is
selected; Escape walks back out.

This is a **client of the `hflow-server` REST API** and holds no knowledge the
server does not serve. It is not published as a package: build it and point the
server at the output.

## Running it

```bash
pnpm install
pnpm dev                      # http://localhost:5173, proxying /api to :4356
```

`pnpm dev` needs a server to talk to. In another terminal:

```bash
uv run hflow serve --no-browser --pipeline path/to/pipeline.py
```

`--pipeline` is what makes the **steps** level non-empty: without it the server
does not know which checks run inside a batch, and the canvas says so rather
than guessing.

To serve the built bundle from the API server itself:

```bash
pnpm build
HFLOW_UI_ASSETS=$PWD/dist uv run hflow serve --no-browser
```

## Checks

```bash
pnpm check     # tsc --noEmit, biome check, vitest
pnpm format    # biome check --write
pnpm gen:api   # regenerate src/apiSchema.ts from the server's OpenAPI schema
```

CI runs `pnpm check`, `pnpm build`, and re-runs `pnpm gen:api` to verify the
generated types are not stale.

## How it is put together

Five files carry the whole thing, and only one of them has decisions in it:

| file | what it owns |
| --- | --- |
| `src/canvas/buildGraph.ts` | focus + server payloads -> nodes and edges. Pure, and where every judgement about what is honest to draw lives. |
| `src/canvas/focus.ts` | where the canvas is pointed, and the breadcrumb derived from it |
| `src/canvas/layout.ts` | dagre positions, left to right |
| `src/api.ts` | every request, typed against the generated schema |
| `src/App.tsx` | the screen, and what a click does |

`src/apiSchema.ts` is **generated** by `pnpm gen:api` from the server's own
OpenAPI declaration -- do not hand-edit it. Nothing else in `src/` restates a
payload field name, so a contract change surfaces as a TypeScript error rather
than as an `undefined` at runtime.

`buildGraph` is tested (`pnpm test`) because it is pure and because its rules
matter: **an edge means a real dependency.** The server is explicit that a
pipeline's registered steps have no dependency edges on each other, so the
steps level groups them into tier columns and draws arrows only at the
boundaries that are real.

`src/tones.ts` is the one owner of "what colour does this outcome read as", for
two separate vocabularies that must not be confused: Airflow's task states and
hflow's own recorded check statuses.

## Constraints it keeps

- **No network beyond the API.** No CDN, no fonts, no telemetry. The workspace
  server makes an offline promise (`docs/SERVE.md`, "Trust posture") and a
  frontend that phones home would break it.
- **No theme toggle.** Both palettes are in `styles.css` under
  `prefers-color-scheme`, so there is nothing stored and nothing to keep in
  sync with a pre-paint script.
- **TypeScript stays on 5.x.** `openapi-typescript` drives the TypeScript
  compiler API through `ts.factory`, which TypeScript 7's native port does not
  expose; on 7 `pnpm gen:api` dies before emitting anything.
