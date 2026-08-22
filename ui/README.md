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
default), then open the Vite URL. The API server authenticates nobody, so the
proxied calls just work — there is no credential to carry over.

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
  `Tabs`, `Tooltip`. It brings focus traps, floating-ui positioning,
  portalling, dismiss layers, roving tabindex and the ARIA wiring. Style Radix
  parts with the existing classes and their `data-state` / `data-highlighted`
  / `data-disabled` attributes.
  - `components/CatalogTree.tsx` stays hand-rolled on purpose: it is a correct
    disclosure pattern and deliberately does not claim `role="tree"`.
- **[lucide-react](https://lucide.dev)** for every glyph. Import icons
  directly — `import { Search } from "lucide-react"` — and let the icon-scale
  block in `styles.css` size them; do not pass `size` at call sites. Icons are
  decorative and lucide marks them `aria-hidden` unless you give them an
  `aria-*`, `role` or `title` prop. `components/BrandMark.tsx` is the one glyph
  from outside lucide: it is the real Hebbian Robotics mark, and
  `docs/assets/hebbian-logo-on-black.svg` is the source of truth for the
  artwork. Its five paths are copied verbatim into both `BrandMark.tsx` and the
  favicon data URI in `index.html`, so re-cutting the logo changes all three.
  The component drops the artboard's black rect, paints with `currentColor` so
  the mark takes the accent in both themes, and tightens the viewBox to the
  glyph's own bounds; the mark is 0.77 : 1, so `.brand-mark` sizes it by height
  rather than taking the square `.lucide` box.
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

`components/ThemeControl.tsx` drives those three states from one quiet icon
button in the rail's footer that cycles `system → light → dark`, showing the
glyph of the state it is in. Because the glyph alone says neither what the
state is nor what a press will do, the accessible name says both ("Theme:
following system (light). Switch to light.") and a visually-hidden
`role="status"` region announces the new state after each press.

`styles.css` is authoritative for all three: every colour token has a value in
the base `:root` block, and the two dark blocks (OS dark, explicit dark) are
the same palette twice and must stay in sync. The inline script at the top of
`index.html` applies the stored choice before the first paint — it duplicates
the storage key and the attribute contract, so change the two together.

### Nav rail width

The rail collapses to icons only. Two states — `"expanded"` (the default),
`"collapsed"` — modelled in `src/railLayout.ts`, persisted in `localStorage`
under `hflow-ui-rail`, and stamped on `<html>` as `data-rail` (expanded sets no
attribute). The same inline script in `index.html` applies it before the first
paint, so a reload never shows the wrong width for a frame.

Three rules the markup keeps, and CSS cannot:

- every destination keeps its label in the DOM in both states — collapsed,
  `styles.css` clips it rather than removing it, so a link is never an unnamed
  icon;
- collapsed, each destination and each rail control grows a Radix `Tooltip`
  (not `title`, which no keyboard reaches and no stylesheet can touch). The
  Radix wrapper is mounted in both states with a controlled `open`, so
  collapsing does not rebuild the element under the reader's focus;
- the collapse control names the action rather than the state ("Collapse the
  sidebar to icons only") and carries `aria-expanded` + `aria-controls`.

At and below 780px the rail is already a horizontal bar, so the collapse is
suspended: the labels come back, the control leaves the layout, and the stored
choice is untouched and returns when the window widens. `railLayout.ts`
duplicates that breakpoint so the markup agrees with the stylesheet about
whether the preference is in force — change the two together.

### Colour

One rule: **chrome is neutral, colour means something.**

Backgrounds, ink, borders, the accent and the focus ring are greys, near-black
and near-white. `--accent` is the far end of that ramp and *inverts* between
themes (near-black on light, near-white on dark), which is why
`--accent-contrast` exists — "the ink that reads on a solid accent fill" cannot
be spelled at the call site. `--boundary` is the one grey held at 3:1 against
every surface it lands on: the outline of anything a reader must be able to
find (fields, buttons, graph nodes, graph connectors), as distinct from the
decorative `--border` / `--border-strong` hairlines.

The hues left in the file all answer a question about the data:
`--ok` / `--warn` / `--err` for status, `--viz-1..3` for interval kinds (a
categorical encoding; `--viz-other` is a true grey because "everything else"
should read as the absence of an identity). Work in flight — `running`,
`deferred` — is the `--run` tone, aliased to the neutral accent: it is the most
separable slot left beside green, amber, red and faint grey, and it separates
by lightness so it survives every form of colour blindness. `src/runState.ts`
owns which Airflow state maps to which tone.

If you need a new colour, first check whether what you are building is chrome.
If it is, it does not get one.
