# Enable the built-in quality checks

**Goal:** run HFlow's packaged checks over your episodes, and gate on the ones
you want to reject episodes for, without writing the checks yourself.

HFlow ships its checks as plain functions in
[`hflow.checks`](../../src/hflow/checks.py). They are written in exactly the
shape you would write your own, so reading one is the fastest way to learn the
[porting pattern](../PORTING.md) -- and registering one is a single line.

## Some of them already run

A handful run on every episode without being registered, so a new pipeline
records a baseline of evidence before anyone has opted into anything:

```python
app = hflow.App("my-pipeline")  # already measuring, before any @app.check
```

That set is `hflow.checks.DEFAULT_CHECKS`: `episode_duration`,
`timestamp_regularity`, `camera_frame_stats`, `keyframe_interval`,
`content_digest`, and `media_digest`. Each is zero-configuration, meaningful
on any corpus (a recording with no cameras simply gets fewer keys), and cheap
enough not to think about -- `camera_frame_stats` is one ffmpeg decode per
camera and blackout, freeze, exposure and the frame count all come out of it.
HFlow owns explicit versions for these automatic registrations. Once you
register or wrap a check yourself, the version in your pipeline is the one
that controls compatibility.

To change the set, pass it:

```python
app = hflow.App("my-pipeline", default_checks=())  # no automatic baseline

app = hflow.App(  # everything except one you configure yourself, below
    "my-pipeline",
    default_checks=[c for c in hflow.checks.DEFAULT_CHECKS if c is not camera_frame_stats],
)
```

Registering one of them yourself replaces the automatic copy rather than
colliding with it, which is how a default gets a gate or `critical=True`. And
if you wrap one under a name of your own, the automatic copy stands down: it
is recorded as `superseded`, naming the step that superseded it, so the catalog
never shows a check version claiming measurements it did not supply. That is a
status of its own rather than `skipped`, because standing down this way is
permanent while a quarantine skip lifts as soon as its critical check is
retuned -- and `hflow dataset create` and re-ingest planning both have to tell
those apart.

## Register the rest

A check is registered by calling `app.check(version="1")` with the function. No wrapper is
needed when the defaults suit you:

```python
import hflow
from hflow.checks import (
    action_integrity,
    camera_frame_stats,
    idle_fraction,
    joint_discontinuity,
    required_topics,
    timestamp_regularity,
)

app = hflow.App("my-pipeline")

# timestamp_regularity, camera_frame_stats, episode_duration, keyframe_interval
# and media_digest already run; these are the ones that do not.
app.check(version="1")(joint_discontinuity)
app.check(version="1")(idle_fraction)
app.check(version="1")(action_integrity)
```

To pass configuration, bind it -- either with `functools.partial` or a wrapper,
whichever reads better to you. The version is explicit, so bump it when a
retuned number makes the new measurements or verdicts incompatible with the
old ones. These replace the bare registration of the same check rather than
adding to it:

```python
import functools

# Instead of `app.check(version="1")(timestamp_regularity)` above:
app.check(version="1", name="timestamps")(
    functools.partial(timestamp_regularity, tolerance_s=0.005)
)


# Instead of `app.check(version="1")(camera_frame_stats)`:
@app.check(version="1")
def camera_health(ep: hflow.Episode) -> hflow.CheckResult:
    return camera_frame_stats(ep, expected_hz={"/wrist_cam/compressed": 30.0})


@app.check(version="1")
def topic_inventory(ep: hflow.Episode) -> hflow.CheckResult:
    return required_topics(ep, topics=["/joint_states", "/imu"])
```

Registering two steps of your own under one name is refused, because both
copies would record the same measurement keys and the catalog would keep only
one of them. A default is the exception: registering or wrapping one of those
supersedes it, as above.

One cost worth knowing before you wrap `camera_frame_stats` specifically.
Supersession of a default happens before the default runs when the pipeline's
wrapper is registered in the same run: HFlow knows the keys each default would
emit, and any default whose predicted key set overlaps the wrapper's emitted
keys is recorded as `superseded` without ever calling ffmpeg. The wrapper
alone decodes the video, the default does no work, and the catalog records
both rows -- the wrapper's measurements and the `superseded` notice on the
default, naming the keys the wrapper covered. Every other built-in is cheap
enough that this never mattered; the doc only called `camera_frame_stats` out
because it is the one whose cost is ffmpeg.

If you prefer to drop the default and own the measurement yourself, that works
the same as before:

```python
app = hflow.App(
    "my-pipeline",
    default_checks=[c for c in hflow.checks.DEFAULT_CHECKS if c is not camera_frame_stats],
)
```

`functools.partial` costs the same, and so does registering it bare under its
own name -- that is a replacement, not a wrapper, so the default does not
exist to be superseded in the first place.

[`examples/stress/synthetic.py`](../../examples/stress/synthetic.py) registers
several this way over a generated corpus, if you want a running reference.

## What each one measures

Checks record evidence, never verdicts, so all of these produce measurements you
query later rather than pass/fail decisions baked into your corpus.

| Check | Answers |
|---|---|
| `timestamp_regularity` | Are message intervals regular, and are the camera and state streams aligned with each other? |
| `camera_frame_stats` | Blackout, freeze, exposure, and stored frame count versus the rate the stream claims -- all from one decode pass. |
| `joint_discontinuity` | Does any joint move faster than a limit you set? |
| `idle_fraction` | How much of the episode had nothing moving? |
| `camera_signal_quality` | Coding range, range-gated exposure, impulse noise, and stillness. Shares the same one ffmpeg decode pass per camera per episode as `camera_frame_stats` -- the instrument's raw output is cached in the workdir, so registering both checks against one episode pays a single decode, not two. |
| `camera_stability` | How much footage is shaky rather than deliberately moving. Needs the `motion` extra. |
| `action_integrity` | Are the recorded values sound -- no NaNs, no publisher stalls repeating a sample, no dimension that never moves? |
| `trajectory_metrics` | How far and fast did it move, how much stood still, was it still settling when recording stopped? |
| `trajectory_segments` | When inside the episode did it hold still, change direction sharply, or hit peak speed? |
| `keyframe_interval` | How seekable is the footage, and can it be cut without re-encoding? |
| `camera_fps_conformance` | Did the camera run at the rate the corpus says it should? |
| `required_topics` | Did the recording declare every topic the rig should produce, and how many messages did each carry? |
| `action_rate` | What message rate did each action topic run at? |
| `episode_duration` | How long is it, for corpus-relative length cuts. |
| `content_digest` | Did this same recording arrive twice? |
| `media_digest` | Did this same *footage* arrive twice, even re-stamped or with different telemetry? |

Three need a topic or a rate you have to supply: `required_topics` and
`action_rate` take `topics=`, and `camera_fps_conformance` needs `nominal_fps=`
to compare against. The joint-stream checks default to `/joint_states`; pass
`topic=` for anything else, and skip them entirely on a corpus with no state
streams (human egocentric video, for instance -- see
[the egocentric example](../../examples/egocentric/)).

One check needs a dependency the core install does not carry:
`camera_stability` uses optical flow and so needs OpenCV, which ships in the
optional `motion` extra (`pip install 'hflow[motion]'`). Everything else here
runs on the core install. Enabling it without the extra raises at the first
episode with the install command in the message, rather than failing obscurely.

The trajectory checks report in the stream's own units. If your dimensions share
no unit -- a gripper width beside a shoulder angle -- pass `dimension_scales=`,
one positive divisor per dimension, and `{topic}/scale_source` will record that
you did. There is deliberately no auto-scaling from the episode's own observed
range: a near-still episode has a tiny range, so dividing by it would inflate
that episode's sensor jitter into apparent motion, inverting the metric on
exactly the episodes worth catching.

## Gate on one

Recording evidence is the default because the same measurements should support
different policies later. When you do want an episode *rejected*, attach a gate
at registration rather than rewriting the check. HFlow ships a recommended one:

```python
@app.check(version="1", critical=True, gate=hflow.checks.RECOMMENDED_CAMERA_INTEGRITY)
def camera_health(ep: hflow.Episode) -> hflow.CheckResult:
    return camera_frame_stats(ep)
```

`critical=True` is what makes a failing gate quarantine the episode and skip its
downstream steps; without it, a failure records a `failed:<check>` tag and the
run continues. Quarantine is a tag, never a deletion.

To use your own numbers, build a gate of your own -- same shape:

```python
STRICTER = hflow.Gate(
    accept_when=(
        hflow.Threshold("*black_frame_pct", hflow.Comparison.AT_MOST, 5.0),
        hflow.Threshold("*freeze_total_s", hflow.Comparison.AT_MOST, 1.0),
    )
)
```

Patterns are globs over measurement keys, which carry their topic, so one
threshold covers every camera. By default *every* matching key must hold; pass
`hflow.Aggregation.ANY_KEY` for "at least one camera is usable" instead.

Three things worth knowing before you gate:

- **Register gating checks last.** A quarantine skips every check after it, and
  skipped steps do not count toward coverage -- so gating early costs you the
  evidence that would have explained the rejection.
- **A gate cannot read a key that is not there.** If a threshold matches no
  measurement, the gate abstains rather than passing, and records a
  `gate-unevaluated:<pattern>` tag so the mistake is one query away. A threshold
  that *does* fail still rejects, even if another could not be read.
- **Bump the check version when you retune a threshold.** Two policies must
  never share one version, or curation cannot pin either.

## Confirm it ran

```bash
uv run python my_pipeline.py
```

The dev loop prints every measurement per check, with `*` for evidence-only,
`+` for a passing gate, and `x` for a failing one. Record a run and the same
facts land in the catalog for querying:

```python
report = app.process(episode_path, record=True)
```

Then query them, keeping in mind that measurement keys carry their topic and so
need double quotes in SQL:

```sql
SELECT episode_id, "/wrist_cam/compressed/black_frame_pct" AS black_pct
FROM episodes WHERE status = 'ok' AND black_pct < 5.0
```

## One packaged check that is not in this list

`hflow.mediapipe_hands.mediapipe_hand_detection` also ships with HFlow, and is
deliberately not one of them: it runs a model rather than computing a
signal statistic. Everything above -- evidence-only results, explicit
versions, gates you attach -- works the same way for it, but it needs an opt-in
extra and a downloaded model asset, and what it records is what MediaPipe
*detected*, which is a weaker claim than what was in the frame. See
[Measure hand presence with MediaPipe](./measure-hand-presence-with-mediapipe.md),
particularly the limitations.

## See also

- [Add existing robotics quality checks](../PORTING.md) -- for the checks HFlow
  does not ship, including the accessors your own code will want.
- [Query quality evidence and create a manifest](../CATALOG.md) -- the view
  surface, the measurement-key naming rules, and coverage denominators.
- [Architecture: quality checks and curation](../ARCHITECTURE.md) -- why checks
  record evidence and gates are optional policy.
