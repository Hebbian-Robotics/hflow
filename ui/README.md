# HFlow workspace UI (frontend)

Read-only React SPA for browsing an hflow catalog. Served in production by the
`hflow-ui` FastAPI package (`packages/hflow-ui/`), which copies the built
assets from `ui/dist` into `hflow_ui/static/` at integration time.

## Development

Requires Node >= 20.19 and pnpm.

```bash
cd ui
pnpm install
pnpm dev        # Vite dev server; proxies /api to http://127.0.0.1:4356
```

Start the API in another terminal (`hflow ui` binds 127.0.0.1:4356 by
default), then open the Vite URL with the `?token=...` query string that
`hflow ui` printed — the SPA remembers the token for the session.

## Checks and build

```bash
pnpm check      # tsc --noEmit && biome check .
pnpm format     # biome check --write .
pnpm build      # type-checks, then emits ui/dist
```

The frontend must stay fully offline: no CDN scripts, no web fonts, no remote
assets of any kind. Everything ships in the bundle.
