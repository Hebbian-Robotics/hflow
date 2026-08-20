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

app = hflow.App("my-pipeline", data_root="./data")


@app.check()
def my_check(ep: hflow.Episode) -> hflow.CheckResult:
    inputs = ep.<accessor>(...)          # our line: extract the dialect
    result = your_function(inputs, ...)  # your line: unchanged
    return hflow.CheckResult(measurements=result)  # our line: record
```

`@app.check()` takes `name=` (defaults to the function name), `critical=`, `requires={...}` (capability set, e.g. `{"gpu"}`), `uses="alias"` (a named endpoint; see the VLM section), and optional `version=`. Step identity automatically includes source, defaults, and captured stable configuration such as numeric thresholds. Pass an explicit `version=` for opaque client objects or external model configuration the SDK cannot inspect deterministically. Checks that declare no resources run before checks that do, so cheap integrity checks gate expensive model calls. Today `requires`/`uses` record intent, order steps, and let preflight verify named endpoints are configured; they do **not** route the step to a particular worker or GPU pool (per-step compute routing is [deferred](./ARCHITECTURE.md#what-is-different-from-dyna)) -- a bring-your-own Airflow deployment arranges those resources itself.

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

Running your own ffmpeg against that file is endorsed usage, not a workaround:

```python
subprocess.run(["ffmpeg", "-i", str(ep.video("wrist_cam")), ...], check=True)
```

For the common camera integrity questions (blackout, freeze, exposure) the built-in single-decode instrument already exists (one pass, one shared frame denominator):

```python
stats = hflow.ffmpeg.frame_stats(ep.video("wrist_cam"))
# stats.black_frame_pct, stats.freeze_intervals, stats.luma_avg_mean, ...
```

The packaged check `hflow.checks.camera_frame_stats(ep)` wraps that instrument
over every camera, adds the frame-count-vs-expected-rate comparison, and
returns the results (freeze spans included, as `freeze:<topic>` intervals) as
a ready-to-register `CheckResult`.

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
    data_root="./data",
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
| `measurements` | `dict[str, float \| int \| str \| bool]` | Named facts. Keys are flat strings and become catalog columns (record a run with `app.test(..., record=True)`, then query with `hflow.curate()` or any Parquet reader); a `topic/metric` convention keeps them queryable. |
| `intervals` | `list[hflow.Interval]` | Labeled time spans; `Interval(start_ns, end_ns, label)` in nanoseconds of log time (the same clock as `ChannelData.timestamps`). |
| `tags` | `list[str]` | Free-form labels routed to the catalog. |
| `verdict` | `bool \| None` | Optional, user-owned. `None` means evidence only. `False` on a `critical` check quarantines the episode. |

Every recorded result also carries the check's version (a content hash of its configuration and source), so re-running a changed check appends new-version rows instead of silently overwriting old ones.

## The dev loop

```python
report = app.test("episode_0001.mcap")
```

`app.test()` runs the whole registered pipeline on one episode **in-process**, with no Docker and no scheduler. It transforms the input into a canonical episode under `<data_root>/test-runs/`, runs every check with the ordering and gate semantics described above, prints a summary, and returns the full `TestReport` (per-check status, measurements, durations, quarantine tags). Iterate on a check in seconds; the canonical file it writes opens directly in Foxglove or Rerun for eyeballing.

When the in-process loop is stable, `app.run()` or `hflow up` executes the
same pipeline as an Airflow DAG in the local Docker Compose runtime. For an
existing Airflow 3 deployment, `hflow deploy` emits the DAG and user-venv
bundle without calling a platform API. See the [runtime guide](./RUNTIME.md) for
both paths.

## See also

- [Documentation home](./README.md): choose a tutorial, how-to guide, reference, or explanation
- [Runnable examples](../examples/README.md): commands, prerequisites, and expected output
- [OpenAI vision how-to](./how-to/call-openai-vision.md): a complete Responses API check
- [How HFlow fits the robotics data stack](./INTEGRATIONS.md): which responsibilities stay in HFlow and which stay in surrounding tools
- [Frequently asked questions](./FAQ.md): direct answers about formats, infrastructure, and project scope
- [Architecture](./ARCHITECTURE.md): the full design, with per-decision provenance
- [`examples/quickstart.py`](../examples/quickstart.py): the pattern on a synthetic episode, runnable today
