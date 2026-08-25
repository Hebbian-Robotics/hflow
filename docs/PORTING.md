# Add existing robotics quality checks to HFlow

You already have a script that checks joint limits, or flags dark frames, or asks a VLM whether the gripper reached the towel. This page shows how to run it inside HFlow **without rewriting it**: your function stays untouched, and two or three of our lines extract the input it already expects and record what it returns.

The accessor surface exists because robotics QC scripts consume a small set of input dialects: numpy arrays, an MP4 path, JPEG frames, a metadata dict. Each section below is one dialect: your original function first, then the wrapper.

## The model: evidence, not verdicts

A check returns `hflow.CheckResult`: **measurements** (named numbers/strings), **intervals** (labeled time spans), and **tags**. Every field is recorded regardless of pass or fail. Thresholds are not baked into the corpus, because quality heuristics are known to *invert* on real defects (smoothness metrics have scored an early-gripper-release defect *better* than clean demos); a stored measurement lets you re-decide with a query, a stored verdict bakes in the wrong call.

A check *may* also declare a `verdict`, a boolean you compute from your own thresholds. On a check registered with `critical=True`, a `False` verdict **quarantines** the episode: it gets a `quarantined:<check>` tag and its downstream steps are skipped, so an episode with a dead camera never runs expensive enrichment. Quarantine is a tag, never a deletion. On a non-critical check, a `False` verdict records a `failed:<check>` tag and the run proceeds. A check that *crashes* is treated as infrastructure failure, not bad data: it is reported as an error and never recorded as a quality outcome.

## The pattern

```python
import hflow
from your_existing_qc import your_function  # untouched

app = hflow.App("my-pipeline")


@app.check()
def my_check(ep: hflow.Episode) -> hflow.CheckResult:
    inputs = ep.<accessor>(...)          # our line: extract the dialect
    result = your_function(inputs, ...)  # your line: unchanged
    return hflow.CheckResult(measurements=result)  # our line: record
```

`@app.check()` takes `name=` (defaults to the function name), `critical=`, `requires={...}` (capability set, e.g. `{"gpu"}`), `uses="alias"` (a named endpoint; see the VLM section), `gate=` (a declarative accept policy the runner evaluates over what the check returned), and optional `version=`. By default a step's identity is derived: its source, its defaults and captured stable configuration such as numeric thresholds, and then transitively the helpers it calls and the constants those read, across your own package and hflow's but never into a dependency. Passing `version=` takes the identity over instead. The function is then not inspected at all, so you can refactor freely and your rows stay comparable, at the cost of remembering to bump the number when behaviour really changes; what you declared beside it (`critical`, `requires`, `uses`, `gate`) still counts. Use it for an opaque client object the SDK cannot inspect deterministically, or for a step you would rather promise about than have measured. Checks that declare no resources run before checks that do, so cheap integrity checks gate expensive model calls. Today `requires`/`uses` record intent, order steps, and let preflight verify named endpoints are configured; they do **not** route the step to a particular worker or GPU pool (per-step compute routing is [deferred](./ARCHITECTURE.md#implementation-status)); a bring-your-own Airflow deployment arranges those resources itself.

Every step is called with exactly one argument, an `Episode`, so `@app.check()`, `@app.enrich()`, and `@app.derive()` refuse a function the runtime could never call: one with a required parameter beyond the episode, or with no positional slot to receive it. That refusal happens at registration, not once per episode, and the error shows the wrapper form to use instead. To pass configuration, bind it in a wrapper (`return action_rate(ep, topics=[...])`) rather than adding a parameter.

## Dialect 1: numpy arrays

Your years-old script takes an `(N, joints)` array:

```python
# your_existing_qc.py -- untouched
def check_joint_smoothness(joints: np.ndarray, rate_hz: float) -> dict[str, float]:
    velocities = np.abs(np.diff(joints, axis=0)) * rate_hz
    return {"max_velocity_rad_s": float(velocities.max())}
```

The wrapper:

```python
@app.check()
def joint_smoothness(ep: hflow.Episode) -> hflow.CheckResult:
    joints = ep.channel("/joint_states").to_numpy()
    result = check_joint_smoothness(joints, rate_hz=100)
    return hflow.CheckResult(measurements=result)
```

`to_numpy()` picks the field automatically: the numeric array field named `position` when present (the `JointState` case), otherwise the only numeric field. When the choice is ambiguous it raises with the candidate list. Pass the field explicitly for anything else: `to_numpy(field="effort")`.

Time-aware checks take the timestamps too: `ep.channel(topic).timestamps` is an int64 array of nanosecond log times, aligned index-for-index with the rows of `to_numpy()`:

```python
stamps_ns = ep.channel("/joint_states").timestamps
dt_s = np.diff(stamps_ns) / 1e9
```

There is also `.to_arrow()` (a `pyarrow.Table` of `log_time_ns` plus the primitive fields; requires the `arrow` extra) and `.messages` (the decoded message objects) when your script wants structured records rather than one array.

## Dialect 2: an MP4 path

Your script takes a video file, via OpenCV or your own ffmpeg subprocess:

```python
# your_existing_qc.py -- untouched
def measure_sharpness(video_path: str) -> dict[str, float]:
    capture = cv2.VideoCapture(video_path)
    ...
```

The wrapper:

```python
@app.check()
def sharpness(ep: hflow.Episode) -> hflow.CheckResult:
    mp4 = ep.video("wrist_cam")  # lossless remux of the in-band H.264, cached
    return hflow.CheckResult(measurements=measure_sharpness(str(mp4)))
```

`ep.video()` remuxes the camera's in-band H.264 into a plain MP4 with **no re-encode**: the pixels your check measures are the pixels in the episode. Camera names resolve by full topic or unique substring: `"wrist_cam"` finds `/wrist_cam/compressed`; with a single camera you can omit the argument.

For the common camera integrity questions -- blackout, freeze, exposure, frame
count versus the rate the stream claims -- do not write this one. The packaged
check already measures every camera in a single decode pass and records freeze
spans as `freeze:<topic>` intervals, so registering it is the whole job:

```python
from hflow.checks import camera_frame_stats

app.check()(camera_frame_stats)
```

Import the built-in as a function rather than reaching it as
`hflow.checks.camera_frame_stats` inside a wrapper. A step's version
content-hashes the functions it *names*, and transitively the hflow code those
call, so an import by name means a change to what the check measures shows up
as a new step version. A module reached by attribute contributes only its name,
and the change would append rows under the old version instead.

To make it *reject* episodes rather than only measure them, attach a gate at
registration instead of computing a verdict inside the check --
[enabling the built-in checks](./how-to/enable-built-in-checks.md#gate-on-one)
covers the shipped gates, writing your own, and why registration order matters.
Gating that way keeps the evidence: a verdict computed inside the check and
returned as a fresh `CheckResult(verdict=...)` throws away the measurements the
instrument already produced.

The format-independent instrument is incubating behind a private package while
its result model settles. Use the built-in check when you want a stable HFlow
API. If you are helping develop the measurement package, call the private API
with an explicit toolchain:

```python
from hflow._video_measurement_toolchain import resolved_video_measurement_toolchain
from hflow._video_measurements import measure_video_frame_statistics

statistics = measure_video_frame_statistics(
    ep.video("wrist_cam"),
    toolchain=resolved_video_measurement_toolchain(),
)
# statistics.black_frame_percent, statistics.freeze_intervals, ...
```

## Dialect 3: JPEG frames and your own VLM client

There is deliberately **no bundled VLM client**. Most models behind OpenAI-compatible endpoints do not take video, so the honest unit in v1 is the frame: you declare the sampling rate, you call your own client, and you own how per-frame answers aggregate into an episode-level answer. Any OpenAI-compatible endpoint works: hosted, or vLLM/Ollama you run yourself.

For a complete executable pipeline using OpenAI's Responses API and a
timestamped contact sheet, use
[Call an OpenAI vision endpoint from a step](./how-to/call-openai-vision.md).
The lower-level example below uses Chat Completions because that interface is
widely implemented by locally hosted OpenAI-compatible servers.

Name the endpoint once on the App, declare it on the check with `uses=`, and read it back from `app.endpoints` inside your own client code. The declaration is what lets preflight verify the endpoint is configured before any episode is processed:

```python
import base64

app = hflow.App(
    "my-pipeline",
    endpoints={"judge": "http://localhost:8000/v1"},  # vLLM, Ollama, or hosted
)


@app.check(uses="judge")
def gripper_reached_target(ep: hflow.Episode) -> hflow.CheckResult:
    from openai import OpenAI  # your client, your dependency

    client = OpenAI(base_url=app.endpoints["judge"], api_key="not-needed-locally")
    answers: list[bool] = []
    for frame in ep.frames("wrist_cam", fps=0.5):  # you declare the rate
        image_b64 = base64.b64encode(frame.path.read_bytes()).decode()
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is the gripper touching the towel? yes/no"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        answers.append("yes" in response.choices[0].message.content.lower())

    reached_fraction = sum(answers) / len(answers)  # aggregation is yours
    return hflow.CheckResult(measurements={"gripper_reached_fraction": reached_fraction})
```

Each `ExtractedFrame` carries its `path` and its `log_time_ns`, so a per-frame answer can become an interval, not just a ratio.

**Episode-level questions on single-image models**: composite the frames into one timestamped grid. It works even on models that accept a single image, and it cuts vision tokens from N images to one:

```python
sheet = hflow.ffmpeg.contact_sheet(
    ep.frames("wrist_cam", fps=0.5),
    ep.workdir / "sheet.jpg",
    columns=4,
    max_tiles=24,  # more frames than this are sampled evenly, reported on the result
)
# send sheet.path as ONE image; sheet.tile_log_times_ns maps tiles back to times
```

`sheet.timestamps_burned` tells you whether relative timestamps were drawn into the tiles (it requires a system font); when `False`, pass `sheet.tile_log_times_ns` in the prompt instead.

## Dialect 4: the metadata dict

Your script keys off task labels or collection outcomes:

```python
# your_existing_qc.py -- untouched
def validate_labels(meta: dict[str, str]) -> dict[str, bool]:
    return {"has_task": bool(meta.get("task")), "labeled_success": "success" in meta}
```

The wrapper:

```python
@app.check()
def labels(ep: hflow.Episode) -> hflow.CheckResult:
    return hflow.CheckResult(measurements=validate_labels(ep.metadata))
```

`ep.metadata` is a flat `dict[str, str]`: the episode semantics record (task, operator, success, embodiment, robot_software_version, plus whatever your rig recorded) merged with the version stamps (schema_version, pipeline_version, ffmpeg_version). Values are strings: `success` is `"true"`/`"false"`. `ep.metadata_records` exposes every MCAP Metadata record in the file, keyed by record name, when you need more than the merged view.
* Use `ep.attachments` to access episode-scoped files (like URDFs and calibration data) preserved by the format.

## Dialect 5: the escape hatch

The episode is standard MCAP; the accessors are conveniences, not a lock-in layer. If your script already speaks `mcap`, hand it the path:

```python
from mcap.reader import make_reader


@app.check()
def my_raw_check(ep: hflow.Episode) -> hflow.CheckResult:
    with ep.path.open("rb") as stream:
        reader = make_reader(stream)
        ...  # your existing reader code, untouched
    return hflow.CheckResult(measurements={...})
```

The same is true outside checks entirely: canonical episodes open in Foxglove, Rerun, and any conforming MCAP tooling.

## What your check returns

| Field | Type | Meaning |
|---|---|---|
| `measurements` | `dict[str, float \| int \| str \| bool]` | Named facts that become catalog columns (record a run with `app.test(..., record=True)`, then query with `hflow.curate()` or any Parquet reader). Name them `<topic>/<metric>_<unit>`: the topic prefix keeps two steps from claiming one key, and the unit suffix lets downstream tools label the value without guessing. |
| `intervals` | `list[hflow.Interval]` | Labeled time spans; `Interval(start_ns, end_ns, label)` in nanoseconds of log time (the same clock as `ChannelData.timestamps`). |
| `tags` | `list[str]` | Free-form labels routed to the catalog. |
| `verdict` | `bool \| None` | Optional, user-owned. `None` means evidence only. `False` on a `critical` check quarantines the episode. Prefer `@app.check(gate=...)` over computing this inline: a gate is evaluated over the measurements you already returned, so a threshold aimed at a missing key cannot cost you the evidence. |

One measurement key may have only one owner, and the runner refuses a run where
two steps claim the same one --
[the naming rules](./CATALOG.md#naming-measurement-keys) explain why a shared
key is unrecoverable rather than merely untidy.

Every recorded result also carries the check's version: a content hash of its source and configuration, of the first-party code it calls, and of any gate you attached. Re-running a changed check, or one whose threshold you retuned, appends new-version rows instead of silently overwriting old ones.

## The dev loop

```python
report = app.test("episode_0001.mcap")
```

`app.test()` runs the whole registered pipeline on one episode **in-process**, with no Docker and no scheduler. It transforms the input into a canonical episode under `<data_root>/test-runs/`, runs every check with the ordering and gate semantics described above, prints a summary, and returns the full `TestReport` (per-check status, measurements, durations, quarantine tags). Iterate on a check in seconds; the canonical file it writes opens directly in Foxglove or Rerun for eyeballing.

### Test one check directly

For the tightest loop on one check, open an existing canonical episode and call
the registered function directly:

```python
import hflow

from my_pipeline import camera_blackout

with hflow.Episode("episode_0001.canonical.mcap") as episode:
    result = camera_blackout(episode)

print(result.measurements)
print(result.intervals)
print(result.tags)
print(result.verdict)
```

The decorator returns the original function, so a check defined with
`@app.check()` remains directly callable. Use a canonical episode when you want
the same input shape the pipeline supplies at run time.

This tests the check function only. It does not apply its registered gate,
produce run status or timing, enforce quarantine and ordering, resolve endpoint
overrides, or write catalog rows. Follow it with `app.test(...)` when you need to
verify those pipeline behaviors; there is currently no check-name selector for
`app.test()`.

When the in-process loop is stable, `app.run()` or `hflow up` executes the
same pipeline as an Airflow DAG in the local Docker Compose runtime. For an
existing Airflow 3 deployment, `hflow deploy` emits the DAG and user-venv
bundle without calling a platform API. See the [runtime guide](./RUNTIME.md) for
both paths.

## See also

- [Documentation home](./README.md): choose a tutorial, how-to guide, reference, or explanation
- [Enable the built-in quality checks](./how-to/enable-built-in-checks.md): the sixteen packaged checks, and how to gate on one
- [Query quality evidence and create a manifest](./CATALOG.md): where measurements land, and the naming rules for keys
- [Runnable examples](../examples/README.md): commands, prerequisites, and expected output
- [OpenAI vision how-to](./how-to/call-openai-vision.md): a complete Responses API check
- [How HFlow fits the robotics data stack](./INTEGRATIONS.md): which responsibilities stay in HFlow and which stay in surrounding tools
- [Frequently asked questions](./FAQ.md): direct answers about formats, infrastructure, and project scope
- [Architecture](./ARCHITECTURE.md): the full design, with per-decision provenance
- [`examples/quickstart.py`](../examples/quickstart.py): the pattern on a synthetic episode, runnable today
