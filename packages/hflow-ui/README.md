# hflow-ui

The HFlow workspace UI: a local, read-only web app over one HFlow data root —
browse episodes, quality evidence, and the Parquet catalog in a browser.

```bash
uv add hflow-ui
hflow ui --data-root ./data
```

The UI is a strict client of the same surfaces the `hflow` CLI uses (the
DuckDB-queryable catalog, episode files, and manifests): everything it shows
is reachable with `curl` against its documented JSON API (`/api/docs` on the
running server), and nothing is UI-only. It runs fully offline — all assets
ship in this wheel, and your data never leaves your machine.

Every endpoint publishes a typed response schema, so `/api/openapi.json` is a
usable contract to generate a client from rather than a list of paths
returning "object". One module — `hflow_ui/_contract.py` — owns those payload
shapes; the routes construct its models instead of hand-building dicts.

This package is deliberately separate from the `hflow` SDK wheel so that
pipeline worker environments (which install `hflow` into every task venv)
never carry frontend assets. Frontend source lives in the repository's `ui/`
directory; built assets are bundled here under `hflow_ui/static/`.
