# Environment-variable reference

HFlow uses environment variables for deployment-specific paths, remote-runtime
credentials, optional measurement assets, and opt-in integration tests. This
page is a lookup index; follow the linked guides for the surrounding workflow.

## Runtime configuration

| Variable or family | Purpose | Default or unset behavior | Scope | Read by |
|---|---|---|---|---|
| `HFLOW_DATA_ROOT` | Selects the workspace root for commands and an `App` that has no explicit data root. | The nearest ancestor `hflow.toml` `data_root`, then `./data`. An explicit `--data-root` or `data_root=` wins over the environment. | CLI and SDK | `hflow` CLI defaults and `hflow.App`; see [runtime data roots](./RUNTIME.md#the-one-rule-the-apps-data-root-is-the-runtimes-data-root) |
| `HFLOW_AIRFLOW_URL` | Selects a remote Airflow API base URL when `--airflow-url` is absent. | No remote endpoint is selected; `hflow ingest` runs in-process and runtime commands use their local-bundle path. | CLI | `hflow ingest` and remote runtime commands; see [remote runtimes](./RUNTIME.md#remote-runtimes---airflow-url) |
| `HFLOW_AIRFLOW_DAG_ID` | Names the ingest DAG when `--dag-id` is absent. | Ignored until a remote URL is selected; then omission is an error. | CLI | Remote-endpoint resolution |
| `HFLOW_AIRFLOW_TOKEN` | Supplies bearer-token credentials for a remote runtime. | If unset, username/password credentials are tried. A token wins when both forms are present. | CLI secret | Remote-endpoint resolution |
| `HFLOW_AIRFLOW_USERNAME` / `HFLOW_AIRFLOW_PASSWORD` | Supply basic-auth credentials for a remote runtime. | Both must be set when a remote URL is selected and no token is present; otherwise resolution fails. | CLI secrets | Remote-endpoint resolution |
| `HFLOW_USER_DIR` | Relocates the directory containing the user's pipeline file on Airflow workers. | `/opt/user` | Worker runtime | Generated DAG tasks; see [deployment paths](./RUNTIME.md#bring-your-own-airflow-hflow-deploy) |
| `HFLOW_MIRROR_DIR` | Chooses the base directory for local mirrors of object-store roots. | `$XDG_CACHE_HOME/hflow/mirrors`, or `~/.cache/hflow/mirrors` when `XDG_CACHE_HOME` is unset. Each storage URL receives a hashed subdirectory. | Storage runtime | `hflow.storage`; see [bucket data roots](./RUNTIME.md#bucket-data-roots---data-root-gsbucketprefix) |
| `HFLOW_FFMPEG` | Pins an operator-managed `ffmpeg` executable. | Linux uses HFlow's verified pinned build. Other platforms fall back to `ffmpeg` on `PATH` with a warning, or fail if none exists. | Measurement runtime | `hflow.ffmpeg` |
| `HFLOW_FFPROBE` | Pins an operator-managed `ffprobe` executable. | Normally follows the `HFLOW_FFMPEG` policy. If only `HFLOW_FFMPEG` is set, its sibling `ffprobe` is preferred, then `PATH` with a warning. | Measurement runtime | `hflow.ffmpeg` |
| `HFLOW_HAND_LANDMARKER_MODEL` | Selects an operator-managed MediaPipe Hand Landmarker model. | HFlow downloads and verifies its pinned float16 model in the user cache. | Optional MediaPipe check | `hflow.mediapipe_hands`; see [hand-presence measurement](./how-to/measure-hand-presence-with-mediapipe.md) |
| `HFLOW_UI_ASSETS` | Points the workspace server at a frontend directory containing `index.html`. | An explicit `ServerSettings.assets_dir` wins; otherwise a packaged `hflow_server/static/` directory is used when present, then the server shows its API placeholder page. | Workspace server | `hflow_server`; see [serving a workspace](./SERVE.md) |

Airflow credentials are environment-only by design so they do not appear in
process listings or shell history. Step-specific credentials, such as model API
keys, belong to the step's client code and are not HFlow-owned variables.

## Test-only gates

The default test suite does not enable external network, Docker, object-store,
or model-dependent tests.

| Variable | Purpose | Default or unset behavior | Scope | Read by |
|---|---|---|---|---|
| `HFLOW_NETWORK_TESTS` | Enables the real pinned-FFmpeg download test when set to `1`. | Test is skipped. | Test only | `tests/test_ffmpeg.py` |
| `HFLOW_DOCKER_TESTS` | Enables the Docker Compose runtime integration suite when set to `1`. | Suite is skipped. | Test only | `tests/test_runtime_integration.py` |
| `HFLOW_TEST_BUCKET_URL` | Supplies a writable object-store prefix for live storage round-trip tests. Any non-empty value enables the suite. | Suite is skipped. | Test only | `tests/test_storage.py` |
| `HFLOW_MEDIAPIPE_TESTS` | Enables the real MediaPipe model test when set to `1`; install the `mediapipe` extra as well. | Test is skipped. | Test only | `tests/test_mediapipe_hands.py` |

The exact commands and prerequisites for these gates are in
[Contributing](../CONTRIBUTING.md#tests).
