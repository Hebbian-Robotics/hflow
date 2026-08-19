# Contributing

Thanks for helping make robotics data infrastructure easier to use. This
project is pre-v1: focused bug reports, documentation fixes, compatibility
work, and small changes with clear outcomes are especially useful.

By submitting a contribution, you agree that it may be distributed under the
repository's [Apache-2.0 license](./LICENSE).

## Development setup

You need:

- Git and [uv](https://docs.astral.sh/uv/)
- Python 3.11 or newer
- ffmpeg and ffprobe on `PATH` for the test suite
- Docker with Compose v2 only when working on the Airflow runtime integration
- [lychee](https://github.com/lycheeverse/lychee) when editing Markdown links

Clone the repository and create the locked development environment:

```bash
git clone https://github.com/Hebbian-Robotics/hflow.git
cd hflow
uv sync --locked --all-extras
```

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

CI runs the suite on Python 3.11 and 3.14. When changing compatibility-sensitive
code, exercise both ends locally:

```bash
uv run --python 3.11 --locked --all-extras pytest -q
uv run --python 3.14 --locked --all-extras pytest -q
```

Two integration tests are intentionally opt-in because they need network or
Docker access:

```bash
HFLOW_NETWORK_TESTS=1 uv run pytest tests/test_ffmpeg.py -q
HFLOW_DOCKER_TESTS=1 uv run pytest tests/test_runtime_integration.py -q
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

## Quality checks

Run the Python quality gate and fix every reported issue:

```bash
uv run ruff check --fix
uv run ruff format
uv run ty check
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

When Markdown changes, run the same link check as CI:

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
- Preserve backward compatibility for stored data unless the change explicitly
  introduces a new versioned format.
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
