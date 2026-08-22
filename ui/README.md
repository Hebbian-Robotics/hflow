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

`pnpm preview` serves the built bundle with the same `/api` proxy — use it when
you need to see production behaviour, such as the pre-paint theme script, which
dev cannot show because dev injects the stylesheet from JS.

## Checks and build

```bash
pnpm check      # tsc --noEmit && biome check .
pnpm format     # biome check --write .
pnpm build      # type-checks, then emits ui/dist
```

The frontend must stay fully offline: no CDN scripts, no web fonts, no remote
assets of any kind. Everything ships in the bundle.

## What this UI is built out of

Prefer a well-maintained library over a hand-rolled one, and prefer the
headless kind — take the behaviour, leave the visual language. The look is
ours: one hand-written `src/styles.css` over CSS custom properties, no
framework, no utility classes.

- **[radix-ui](https://www.radix-ui.com/primitives)** (the umbrella package —
  one dependency, tree-shakes to the same bytes as the ~55 scoped ones) for
  every overlay and composite widget: `Dialog`, `Popover`, `DropdownMenu`,
  `Tabs`, `RadioGroup`. It brings focus traps, floating-ui positioning,
  portalling, dismiss layers, roving tabindex and the ARIA wiring. Style Radix
  parts with the existing classes and their `data-state` / `data-highlighted`
  / `data-disabled` attributes.
  - `components/CatalogTree.tsx` stays hand-rolled on purpose: it is a correct
    disclosure pattern and deliberately does not claim `role="tree"`.
- **[lucide-react](https://lucide.dev)** for every glyph. Import icons
  directly — `import { Search } from "lucide-react"` — and let the icon-scale
  block in `styles.css` size them; do not pass `size` at call sites. Icons are
  decorative and lucide marks them `aria-hidden` unless you give them an
  `aria-*`, `role` or `title` prop. `components/BrandMark.tsx` is the one
  hand-drawn SVG: it is HFlow's identity and must match the favicon in
  `index.html`.
- **[@tanstack/react-query](https://tanstack.com/query)** for every server
  read and write, **[@tanstack/react-table](https://tanstack.com/table)** for
  the grids, **[react-router](https://reactrouter.com)** for routing, and
  **[CodeMirror 6](https://codemirror.net)** for the SQL editor.

### Theme

Three states — `"system"` (the default), `"light"`, `"dark"` — modelled in
`src/theme.ts` and persisted in `localStorage` under `hflow-ui-theme`. The
choice is stamped on `<html>` as `data-theme`; `"system"` sets no attribute and
lets the stylesheet's `prefers-color-scheme` block answer, with a `matchMedia`
listener keeping the control's label honest as the OS flips.

`styles.css` is authoritative for all three: every colour token has a value in
the base `:root` block, and the two dark blocks (OS dark, explicit dark) are
the same palette twice and must stay in sync. The inline script at the top of
`index.html` applies the stored choice before the first paint — it duplicates
the storage key and the attribute contract, so change the two together.
