# Environment variables

Every `HFLOW_*` environment variable HFlow reads, in one place. Each is also
documented where it applies (linked below); this page exists to answer "how
do I point at my own ffmpeg *and* relocate the mirror cache?" without
crossing pages.

| Variable | Purpose | Default | Applies to |
|---|---|---|---|
| `HFLOW_FFMPEG` | Explicit override for the ffmpeg binary path -- skips the pinned auto-download entirely. | Unset: the pinned static build is auto-downloaded and sha256-verified (Linux); the PATH binary with a loud warning on platforms with no pinned build (macOS, native Windows). | Any pipeline run; also how the test suite pins a known ffmpeg (see [CONTRIBUTING.md](../CONTRIBUTING.md)). |
| `HFLOW_FFPROBE` | Explicit override for the ffprobe binary path. | Unset with `HFLOW_FFMPEG` also unset: same pinned/PATH resolution as ffmpeg. Unset with `HFLOW_FFMPEG` set: the `ffprobe` sibling next to the overridden ffmpeg, falling back to PATH with a warning. | Same as `HFLOW_FFMPEG`. |
| `HFLOW_MIRROR_DIR` | Overrides the base directory for the local mirror a bucket data root spools through. | `$XDG_CACHE_HOME/hflow/mirrors`, else `~/.cache/hflow/mirrors`. | Bucket data roots (local runs and Airflow workers) -- see [RUNTIME.md](./RUNTIME.md) and [CATALOG.md](./CATALOG.md). |
| `HFLOW_USER_DIR` | Absolute path where a deployed Airflow worker finds `user/` (your pipeline file), for platforms that cannot mount it at the default path. | `/opt/user` | Runtime deployments only, set on the Airflow components -- see [RUNTIME.md](./RUNTIME.md#bring-your-own-airflow-hflow-deploy) and the generated `DEPLOY.md`. |
| `HFLOW_NETWORK_TESTS` | Opts in to the real-download ffmpeg integration test. | Unset (test skipped). | Development only -- see [CONTRIBUTING.md](../CONTRIBUTING.md#tests). |
| `HFLOW_DOCKER_TESTS` | Opts in to the Docker-based Airflow runtime integration test. | Unset (test skipped). | Development only -- see [CONTRIBUTING.md](../CONTRIBUTING.md#tests). |
| `HFLOW_TEST_BUCKET_URL` | Live object-store URL (e.g. `gs://bucket/tmp-prefix`) that opts in to a real round-trip test against that bucket. | Unset (test skipped). | Development only, when changing `hflow.storage`. |
