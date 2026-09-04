# Compile Python implementation modules for deployment

Use a native overlay when a runtime deployment should contain Cython
extensions for Python implementation modules while retaining normal wheel
installation semantics. The overlay replaces selected `.py` files with
target-bound `.so` files; it does not bundle Python, dependencies, models, or
a container image. The applied tree is a runtime-only deployment artifact, not
a wheel to publish or a package tree to put on a type checker's search path.

This is useful for reducing casual source inspection and may reduce import or
pure-Python wrapper overhead. It is not an isolation boundary: module names,
strings, native code, process memory, and runtime behavior remain inspectable.
FFmpeg, NumPy, and model execution are already native, so measure performance
instead of assuming their Python wrappers became faster.

## Why an extension overlay instead of a standalone executable

Cython extensions retain normal Python package imports, entry points, dynamic
step registration, and selective wrapper compilation. That makes them a small
deployment-layer change and lets one ordinary wheel remain the source of
dependency, version, and license metadata.

A Nuitka standalone build instead freezes a chosen application entry point
together with a Python runtime and discovered dependencies. It can reduce more
of the visible Python surface, but produces a larger application-specific
bundle and makes dynamic imports, plugins, and multiple worker entry points
more involved. Use it when shipping a closed executable is the actual product
boundary; it is not the default for HFlow's library and worker-wrapper use
case.

## What the overlay preserves

Apply an overlay only to a disposable installation of the exact package tree
that built it. The operation deliberately leaves these wheel-owned files in
place:

- package and subpackage `__init__.py` adapters;
- `__main__.py`, console-script entry points, and `py.typed`;
- the distribution's `.dist-info`, including version metadata and licenses;
- every source or resource not explicitly selected for compilation.

By default, `package build` selects every `.py` below `--package-root` except
`__init__.py` and `__main__.py`. Use repeatable `--module` arguments for a
smaller explicit set.

`py.typed` remains because the overlay changes a disposable runtime tree
rather than repackaging the wheel. It does not imply that a type checker can
recover declarations from the compiled modules. Run static analysis against
the original wheel or source tree, never the applied deployment tree.

## Build and apply in a container stage

Install the ordinary runtime wheel without the build extra. Install the build
tool in a separate virtual environment so Cython and setuptools do not enter
the final image:

```dockerfile
FROM python:3.12-slim AS builder

ARG HFLOW_VERSION
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN test -n "${HFLOW_VERSION}" && \
    uv pip install --system "hflow==${HFLOW_VERSION}"
RUN uv venv /opt/native-builder && \
    uv pip install --python /opt/native-builder/bin/python \
      "hflow[native-build]==${HFLOW_VERSION}"

RUN /opt/native-builder/bin/hflow package build \
      --package-root /usr/local/lib/python3.12/site-packages/hflow \
      --output-dir /opt/hflow-overlay && \
    /opt/native-builder/bin/hflow package apply /opt/hflow-overlay \
      --target-package-root /usr/local/lib/python3.12/site-packages/hflow && \
    /opt/native-builder/bin/hflow package verify /opt/hflow-overlay \
      --target-package-root /usr/local/lib/python3.12/site-packages/hflow

FROM python:3.12-slim AS runtime
COPY --from=builder /usr/local /usr/local
```

Set `HFLOW_VERSION` to a release that includes the `hflow package` command;
HFlow 0.2.4 predates it. The unreleased-checkout flow below is the path until
that release exists.

Pin production base images by digest rather than using `latest`; it is shown
above only to keep the mechanics readable. Build and run stages must use the
same CPython ABI, operating system, architecture, and compatible system
libraries. Raw Cython extensions are not portable wheels and are not a
cross-compilation mechanism.

To compile only application wrappers, point `--package-root` at that package
and name selected modules explicitly:

```bash
hflow package build \
  --package-root ./src/robot_quality \
  --module robot_quality.camera_checks \
  --module robot_quality.result_writer \
  --output-dir ./dist/robot-quality-overlay
```

The target installation must hold byte-for-byte matching source files when
the overlay is first applied. For a wheel installation, `package apply`
unambiguously finds the one `.dist-info/RECORD` that owns every selected
source. It atomically moves that RECORD through a prepared state which owns
both source and native paths, installs and verifies the native artifacts and
overlay receipt, removes only proven sources and corresponding bytecode, then
atomically finalizes RECORD. Pip or uv can therefore remove the native files
on uninstall or upgrade, and retrying the same overlay can finish an
interrupted prepared application.

Application refuses ambiguous or malformed ownership and a wheel containing
`RECORD.jws` or `RECORD.p7s`, because changing RECORD would invalidate its
signature. A source checkout is still supported when its installation parent
contains no wheel RECORD at all; no package-manager ownership metadata is
invented in that mode. Do not run `package apply` concurrently with pip, uv,
or another overlay application against the same installation.

## Build from an unreleased checkout

Build one ordinary wheel from the checkout, then use that same wheel as both
the runtime installation and the build-tool installation. Do not apply an
overlay from an unreleased checkout to a differently sourced PyPI wheel, even
when their version strings happen to match.

```bash
uv build --wheel --out-dir ./dist/local
hflow_wheel="$(printf '%s\n' ./dist/local/hflow-*.whl)"
test -f "${hflow_wheel}"
uv venv --python 3.12 ./dist/builder-venv
uv pip install --python ./dist/builder-venv/bin/python \
  "${hflow_wheel}" \
  "Cython>=3.3.0" "setuptools>=84.0.0"
uv venv --python 3.12 ./dist/runtime-venv
uv pip install --python ./dist/runtime-venv/bin/python \
  "${hflow_wheel}"
```

Use the builder environment's `hflow package` command against the runtime
environment's `site-packages/hflow` directory. Keep the wheel SHA-256, overlay
`bundle_digest`, and final OCI image digest together as the deployment receipt.
The manifest is canonical JSON and records source and artifact hashes plus the
CPython ABI and platform. Builds from identical inputs on the same pinned
toolchain and target are tested to reproduce the same manifest and extension
bytes; portability across compilers or operating-system images is neither
claimed nor expected. Integrity still depends on pinning the manifest digest
outside the workload; an operator able to replace both an artifact and its
manifest can redefine the bundle.

## Licensing and image reduction

Do not remove `.dist-info/licenses`, model licenses, notices, or other required
attribution when minimizing the final image. HFlow's installed metadata is also
how `hflow.__version__` resolves the distribution version. Cython and
setuptools can remain build-stage-only, but redistribution obligations for
runtime dependencies and model assets still apply. HFlow's pinned FFmpeg build
has additional GPL redistribution considerations described in the
[hosting guide](../HOSTING.md#current-limits).

Use separate images for materially different model or check families so a
worker does not pull assets it cannot use. After correctness equivalence is
established, compare source and native variants using the same immutable input
and exact base image. Record at least:

- compressed image size and cold image-pull time;
- process start to package import and application-ready time;
- model-ready time separately from Python import time;
- steady-state episode throughput and peak RAM/VRAM;
- median and tail latency across repeated fresh processes.

Treat identical pipeline manifests and real check outputs as the correctness
gate. A speedup in a synthetic wrapper loop does not justify a deployment whose
recorded evidence changed.
