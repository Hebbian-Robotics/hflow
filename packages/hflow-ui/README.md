# hflow-ui

The HFlow workspace UI: a local web app over one HFlow data root — browse
episodes, quality evidence, and the Parquet catalog in a browser. It writes
nothing but your pinned manifests and its own small state file.

```bash
hflow ui --data-root ./data
```

It binds `127.0.0.1` and authenticates nobody, like other local developer
tools that browse a working directory: anyone who can reach the bound address
can read the workspace and trigger runs, so binding past loopback is a
deliberate exposure. `docs/UI.md` ("Trust posture") has the details.

This package is not on PyPI yet. Until the first release, run it from a clone
of the [repository](https://github.com/Hebbian-Robotics/hflow); `docs/UI.md`
there has the exact steps, including the frontend build.

The UI is a strict client of the same surfaces the `hflow` CLI uses (the
DuckDB-queryable catalog, episode files, and manifests): everything it shows
is reachable with `curl` against its documented JSON API, and nothing is
UI-only. It runs fully offline — all assets ship in this wheel, and your data
never leaves your machine. There is deliberately no Swagger page: FastAPI's
built-in one would load its JavaScript and CSS from a CDN.

Every endpoint publishes a typed response schema, so `/api/openapi.json` — the
schema the running server serves — is a usable contract to generate a client
from rather than a list of paths returning "object". One module —
`hflow_ui/_contract.py` — owns those payload shapes; the routes construct its
models instead of hand-building dicts.

This package is deliberately separate from the `hflow` SDK wheel so that
pipeline worker environments (which install `hflow` into every task venv)
never carry frontend assets. Frontend source lives in the repository's `ui/`
directory; built assets are bundled here under `hflow_ui/static/`.
