# Enable the built-in quality checks

**Goal:** run HFlow's packaged checks over your episodes, and gate on the ones
you want to reject episodes for, without writing the checks yourself.

HFlow ships thirteen checks as plain functions in
[`hflow.checks`](../../src/hflow/checks.py). They are written in exactly the
shape you would write your own, so reading one is the fastest way to learn the
[porting pattern](../PORTING.md) -- and registering one is a single line.

## Register them

A check is registered by calling `app.check()` with the function. No wrapper is
needed when the defaults suit you:

```python
import hflow
from hflow.checks import (
    action_integrity,
    camera_frame_stats,
    episode_duration,
    idle_fraction,
    joint_discontinuity,
    keyframe_interval,
    media_digest,
    timestamp_regularity,
)

app = hflow.App("my-pipeline", data_root="./data")

app.check()(timestamp_regularity)
app.check()(joint_discontinuity)
app.check()(camera_frame_stats)
app.check()(idle_fraction)
app.check()(episode_duration)
app.check()(action_integrity)
app.check()(keyframe_interval)
app.check()(media_digest)
```

To pass configuration, bind it -- either with `functools.partial` or a wrapper,
whichever reads better to you. Both land the configuration inside the check's
content-hash version, so a retuned number appends new-version rows rather than
silently overwriting the old ones. These replace the bare registration of the
same check rather than adding to it:

```python
import functools

# Instead of `app.check()(timestamp_regularity)` above:
app.check(name="timestamps")(functools.partial(timestamp_regularity, tolerance_s=0.005))


# Instead of `app.check()(camera_frame_stats)`:
@app.check()
def camera_health(ep: hflow.Episode) -> hflow.CheckResult:
    return camera_frame_stats(ep, expected_hz={"/wrist_cam/compressed": 30.0})
```

Registering the same check twice is refused, because both copies would record
the same measurement keys and the catalog would keep only one of them.

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
| `camera_signal_quality` | Coding range, range-gated exposure, impulse noise, and stillness -- from the same decode pass. |
| `action_integrity` | Are the recorded values sound -- no NaNs, no publisher stalls repeating a sample, no dimension that never moves? |
| `trajectory_metrics` | How far and fast did it move, how much stood still, was it still settling when recording stopped? |
| `trajectory_segments` | When inside the episode did it hold still, change direction sharply, or hit peak speed? |
| `keyframe_interval` | How seekable is the footage, and can it be cut without re-encoding? |
| `camera_fps_conformance` | Did the camera run at the rate the corpus says it should? |
| `action_rate` | What message rate did each action topic run at? |
| `episode_duration` | How long is it, for corpus-relative length cuts. |
| `content_digest` | Did this same recording arrive twice? |
| `media_digest` | Did this same *footage* arrive twice, even re-stamped or with different telemetry? |

Two need a topic or a rate you have to supply: `action_rate` takes `topics=`,
and `camera_fps_conformance` needs `nominal_fps=` to compare against. The
joint-stream checks default to `/joint_states`; pass `topic=` for anything else,
and skip them entirely on a corpus with no state streams (human egocentric video,
for instance -- see [the egocentric example](../../examples/egocentric/)).

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
@app.check(critical=True, gate=hflow.checks.RECOMMENDED_CAMERA_INTEGRITY)
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
- **Retuning a threshold is a new check version.** That is deliberate: two
  policies must never share one version, or curation cannot pin either.

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
FROM episodes WHERE status != 'quarantined' AND black_pct < 5.0
```

## See also

- [Add existing robotics quality checks](../PORTING.md) -- for the checks HFlow
  does not ship, including the accessors your own code will want.
- [Query quality evidence and create a manifest](../CATALOG.md) -- the view
  surface, the measurement-key naming rules, and coverage denominators.
- [Architecture: quality checks and curation](../ARCHITECTURE.md) -- why checks
  record evidence and gates are optional policy.
