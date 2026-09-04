# Contributing

Thanks for helping make robotics data infrastructure easier to use. This
project is pre-v1: focused bug reports, documentation fixes, compatibility
work, and small changes with clear outcomes are especially useful.

New to open-source contribution? GitHub's
[contributing to a project quickstart](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)
covers the fork, branch, and pull request flow this project uses. For
questions at any point, ask in our [Discord](https://discord.gg/vacepQvjmg).

By submitting a contribution, you agree that it may be distributed under the
repository's [Apache-2.0 license](./LICENSE).

## Development setup

You need:

- Git and [uv](https://docs.astral.sh/uv/)
- Python 3.11 or newer
- ffmpeg and ffprobe on `PATH` for the test suite
- Docker with Compose v2 only when working on the Airflow runtime integration
- [lychee](https://github.com/lycheeverse/lychee) when editing Markdown links

## Platform support

Linux and macOS both work for native development. CI runs on Linux only, so
run the quality checks yourself before opening a pull request on macOS.

Native Windows does not work: HFlow imports `fcntl` for file locking
(`src/hflow/storage.py`), and that module does not exist on Windows, so
`import hflow` fails before any test can run. Work inside WSL2 with an Ubuntu
distribution instead, following the setup steps below from the WSL2 shell and
keeping both the clone and your data root on the Linux filesystem. The
[runtime prerequisites](./docs/RUNTIME.md#prerequisites) explain why the data
root has to live there.

Clone the repository and create the locked development environment:

```bash
git clone https://github.com/Hebbian-Robotics/hflow.git
cd hflow
uv sync --locked
```

Every root-project extra the suite exercises is mirrored into the `dev` group,
so a plain sync is the whole root development environment.

Examples that need a substantial or specialized dependency stack are separate
uv workspace projects. Each owns a `pyproject.toml` and colocated tests, while
using the repository's shared lockfile and workspace copy of HFlow. Select one
with `--project` only when working on it:

```bash
uv sync --locked --project examples/path_to_example
```

The `mediapipe` extra also stays opt-in because it is a model that adds ~430 MB
and a second OpenCV distribution. Add it with `--extra mediapipe` when you are
working on it.

Confirm the package and CLI are available:

```bash
uv run python -c "import hflow; print(hflow.__version__)"
uv run hflow --help
```

Run the smallest end-to-end development loop without Docker:

```bash
uv run python examples/quickstart.py
```

Generated episodes, catalogs, and runtime bundles live under `data/`, which is
gitignored. Never commit robot recordings, generated media, credentials, or
runtime `.env` files.

## Tests

Run the default suite after every behavior change:

```bash
uv run pytest -q
```

Run a workspace example's colocated tests against its own environment:

```bash
uv run --locked --project examples/path_to_example \
  pytest -q examples/path_to_example/tests
```

CI runs the suite on Python 3.11 and 3.14. When changing compatibility-sensitive
code, exercise both ends locally:

```bash
uv run --python 3.11 --locked pytest -q
uv run --python 3.14 --locked pytest -q
```

Four integration test suites are intentionally opt-in because they need network
access, Docker, a writable object-store prefix, or a model outside the default
environment:

```bash
HFLOW_NETWORK_TESTS=1 uv run pytest tests/test_ffmpeg.py -q
HFLOW_DOCKER_TESTS=1 uv run pytest tests/test_runtime_integration.py -q
HFLOW_TEST_BUCKET_URL=gs://your-bucket/tmp-prefix uv run pytest tests/test_storage.py -q
HFLOW_MEDIAPIPE_TESTS=1 uv run --extra mediapipe pytest tests/test_mediapipe_hands.py -q
```

The MediaPipe one brings its own OpenCV, and the OpenCV wheels share one
`cv2/` directory, so syncing back out can leave `import cv2` broken while
`uv sync` still calls the environment correct. One command puts it back:

```bash
uv sync --locked --reinstall-package opencv-python-headless
```

Run the Docker test when changing the runtime bundle, DAG templates, REST
client, runtime lifecycle, or task-venv behavior. It pulls large images on the
first run, starts an isolated Compose project, and removes its containers and
volumes afterward.

Tests must verify HFlow's observable business outcomes. Do not add tests
whose only purpose is to assert mock interactions or third-party library
behavior.

Prefer one test for each distinct contract or boundary. Before adding a
regression test, search for existing coverage and extend the closest behavioral
test when it already exercises the same setup and outcome. A test double is
appropriate at a network, process, clock, or plugin boundary, but assertions
should still describe what the caller observes rather than which private helper
was invoked.

## Documentation and examples

The [documentation home](./docs/README.md) uses the Diátaxis categories:
tutorial, how-to guide, reference, and explanation. Give a new page one primary
purpose, link it from that index, and link task guides to the corresponding
[runnable example](./examples/README.md) when code is involved.

Examples are executable documentation. Each example must state its
prerequisites and external side effects, provide an exact command from the
repository root, and name the observable result. Keep examples on public APIs;
tests belong to business logic and boundary behavior, not to checking that a
documentation snippet copied a third-party SDK correctly.

Give an example its own workspace project when it has a substantial dependency
stack, multiple entry points, or colocated tests that need dependencies the
root suite should not install. Its `pyproject.toml` should set
`tool.uv.package = false`, depend on the workspace copy of `hflow`, and declare
its own development tools. Register the directory in the root workspace and in
CI's `workspace-example-checks` matrix. Keep small examples that only use HFlow
or one optional client in the root project.

## Changing how episodes are processed

Identities in HFlow are content hashes, and one of them -- `pipeline_version`
-- is stamped inside the canonical bytes that `episode_id` hashes. A release
number deliberately does **not** feed any of them: a CLI fix or a docs bump
must never invalidate somebody's corpus. What does feed them is
`TRANSFORM_BEHAVIOR_VERSION` in [`src/hflow/behavior.py`](./src/hflow/behavior.py).

**Bump it in the same commit whenever your change makes the transform write
different bytes for the same input** -- encoder settings or defaults,
chunking and grouping, timestamp handling, the provenance record's shape, or
a bugfix to any of those. Bumping re-versions every existing corpus exactly
once, which is the honest cost; not bumping when behavior changed silently
mixes two behaviors under one version, which is worse. When in doubt, bump,
and say so in the pull request.

Before HFlow 1.0, canonical MCAP files are derived artifacts and exact byte
compatibility is not guaranteed. Retain the raw/source recording and regenerate
canonical outputs after a behavior bump. The behavior version exists to make
that rewrite explicit and detectable, not to require adapters for every old
layout.

Changes to checks and enrichments do not require a transform behavior bump.
Instead, bump the step's explicit version when its new results are no longer
comparable with the old ones. Derived channels follow the same author-owned
rule, but their versions feed `pipeline_version` because their samples are
written into canonical episode bytes. `tests/test_identity_stability.py` pins
the transform rules.

## Quality checks

Run the Python quality gate and fix every reported issue:

```bash
uv run ruff check --fix
uv run ruff format
uv run ty check
```

Workspace examples have the same gate, run against their own environments:

```bash
uv run --locked --project examples/path_to_example \
  ruff check --fix examples/path_to_example
uv run --locked --project examples/path_to_example \
  ruff format examples/path_to_example
uv run --locked --project examples/path_to_example \
  ty check --project examples/path_to_example --extra-search-path . \
  examples/path_to_example
```

### Optional: Pre-commit hooks

To catch style issues before you commit, install pre-commit hooks (optional; CI will catch anything you miss):

```bash
uvx pre-commit install
# or: pipx run pre-commit install
```

Hooks will run automatically on `git commit`. To run them manually on all files:

```bash
uvx pre-commit run --all-files
```

Hooks enforce the same style gates documented above: `ruff check --fix` and `ruff format`.

When Markdown changes, run the local link check. It accesses external sites, so
it is deliberately not part of CI:

```bash
lychee --no-progress --include-fragments \
  --exclude '^https://github\.com/Hebbian-Robotics/hflow/(issues|security/advisories/new)$' \
  --exclude-path references/mcap-spec.md \
  --exclude-path references/foxglove-CompressedVideo.proto .
```

`references/` contains pinned third-party source material. Do not edit a mirror
as if it were project documentation: update it from the authoritative upstream
URL, record the new retrieval time, and preserve its license notice in
[references/LICENSE](./references/LICENSE).

## Pull requests

- Keep each pull request focused on one problem.
- Explain the user-visible outcome and why the change is needed.
- Include the exact validation commands you ran.
- Add or update documentation whenever behavior, flags, formats, or operational
  requirements change.
- Add outcome-focused regression coverage for bug fixes and business logic.
- Never delete or corrupt raw/source recordings. For pre-1.0 derived outputs,
  document the behavior bump and regeneration path instead of preserving old
  byte layouts by default.
- Verify `git status` does not include recordings, generated artifacts, caches,
  credentials, or runtime bundles.

Use the issue tracker for reproducible bugs and concrete feature requests. For
security-sensitive reports, follow [SECURITY.md](./SECURITY.md) instead of
opening a public issue.

## Code Style & Philosophy

### Typing & Pattern Matching

- Prefer **explicit types** over raw dicts -- make invalid states unrepresentable where practical
- Prefer **typed variants over string literals** when the set of valid values is known
- Use **exhaustive pattern matching** (`match` in Python and Rust, `ts-pattern` in TypeScript) so the type checker can verify all cases are handled
- Structure types to enable exhaustive matching when handling variants
- Prefer **shared internal functions over factory patterns** when extracting common logic from hooks or functions -- keep each export explicitly defined for better IDE navigation and readability

### Forward Compatibility

- **Unknown values**: Parse to an explicit `Unknown*` variant (never `None`), log at warn level, preserve raw data, gracefully ignore instead of raising exception

### Self-Documenting Code

- **Verbose naming**: Variable and function naming should read like documentation
- **Strategic comments**: Only for non-obvious logic or architectural decisions; avoid restating what code shows
