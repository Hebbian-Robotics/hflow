# Measure hand presence with MediaPipe

**Goal:** record how often an operator's hands appear in each camera's footage,
using Google's MediaPipe Hand Landmarker, and know exactly what that number
does and does not tell you.

Read the [limitations](#limitations-you-are-accepting) before you use the
result for anything. They are not caveats about this implementation; they are
properties of the model, and the most important one is that **a gloved hand is
not detected**.

This is deliberately **not** one of HFlow's
[built-in checks](./enable-built-in-checks.md). Those are deterministic signal
statistics. This one runs a model, so it lives in its own module, names the
vendor at every call site, ships no recommended gate, and stays outside the
core install.

## What it measures

`mediapipe_hand_detection` samples frames from each camera, runs the Hand
Landmarker over them, and records what the model *detected*:

| Measurement | Answers |
|---|---|
| `{topic}/hand_detected_frame_share` | What share of sampled frames had at least one hand detected. |
| `{topic}/two_hand_detected_frame_share` | What share had two. |
| `{topic}/hand_detection_frame_count` | How many frames were sampled -- the denominator the shares are over. |
| `{topic}/left_hand_detected_frame_count` | How many frames had a hand MediaPipe labelled left; likewise right. |
| `{topic}/hand_sample_fps` | The rate the frames were sampled at. |
| `{topic}/mediapipe_version`, `{topic}/hand_model_digest` | Which build and which weights produced the row. |

It also records `no_hand_detected:{topic}` intervals: the spans of footage
where the model found nothing, so "when did the hands leave frame" is a query
rather than a re-run.

A camera that could not be sampled records the frame count alone and **no
share**. Nothing measured is not a share of zero: an episode whose camera
failed did not have hands absent from it.

Landmark coordinates are not recorded. The model returns 21 image and 21 world
landmarks per hand, which is a nested shape the flat catalog cannot hold, and
the counts are what the shares need.

## 1. Install the extra

```bash
uv sync --locked --extra mediapipe
```

That is the whole install. This extra already covers everything the `motion`
extra provides, so use it instead of `hflow[motion]` rather than alongside it,
and `camera_stability` keeps working.

## 2. Let it fetch the model

Nothing to do. MediaPipe ships no weights, so the first episode downloads the
7.8 MB Hand Landmarker into your user cache and checks it against a pinned
hash, the same way HFlow already handles its ffmpeg build.

To point it at your own asset instead:

```bash
export HFLOW_HAND_LANDMARKER_MODEL=/path/to/hand_landmarker.task
```

The override wins outright and its real digest is recorded, so a different
model is a visible fact in the catalog rather than a silent substitution.

## 3. Register it

```python
import hflow
from hflow.mediapipe_hands import mediapipe_hand_detection

app = hflow.App("my-pipeline", data_root="./data")

app.check()(mediapipe_hand_detection)
```

To bind configuration, use `functools.partial` or a wrapper, exactly as with
the built-ins:

```python
import functools

app.check(name="hands")(
    functools.partial(
        mediapipe_hand_detection,
        sample_fps=2.0,
        inference_long_edge_pixels=912,
    )
)
```

`sample_fps` defaults to 1.0, which is what makes the answer comparable to a
frames-only VLM reading the same footage, and costs about 11 ms of CPU
inference per frame. `inference_long_edge_pixels` resizes each frame before
inference; it changes which hands are found at all on small footage, so it is
part of the question rather than a performance knob.

Re-tuning either one appends new-version rows rather than mixing two
behaviours under one version, and so does a change of model weights. Upgrading
MediaPipe itself does not, which is why every row also carries
`mediapipe_version`.

## Limitations you are accepting

**These come from the model, and HFlow has not measured any of them.** They are
documented from MediaPipe's training domain and from the behaviour of the
instrument, not from a validation run on robot footage. Treat them as things to
verify on your own data.

**Gloves.** The Hand Landmarker is trained on in-the-wild images of bare
hands. Gloved hands -- nitrile, work gloves, anything that changes hand colour
and texture -- are the known failure case. If your operators wear gloves, this
check may report near-zero on footage that is full of hands, and it cannot
tell you that is what happened. The predecessor system that this check was
ported from never asked the landmarker about gloves; it asked a
vision-language model a separate categorical question (bare / gloved / mixed /
other), which is the honest way to get that answer.

**It detects hands, not the camera wearer's hands.** A coworker in frame, a
bystander, a hand on a monitor -- all count. On footage with other people in
it, the share reads higher than "the operator's hands were visible".

**Two hands is a hard cap.** The model returns at most two detections per
frame. Extra hands in frame do not raise the count; they compete for the two
slots, so `two_hand_detected_frame_share` can describe two people's hands
rather than one person's pair.

**The handedness score is not a detection confidence.** It says how sure the
model is that a hand it already found is a left hand. HFlow uses it only to
pick between labels and never records it as a grade on the detection, and
neither should you.

**Ordinary vision failures suppress detections**: motion blur, low light,
hands far from the camera, and hands occluded by the object being held. An
egocentric camera looking down at a grasp is a hard case for all four at once.

**Hand order is not an identity track.** Frame-to-frame, the model does not
promise that hand 0 is the same hand it was. The measurements here are counts
precisely because they do not depend on that.

**The count is a floor, never a ceiling.** Every limitation above pushes
detections down, except the bystander case, which pushes them up. A high share
is fairly strong evidence hands were visible; a low share is weak evidence of
anything.

**It is not bit-reproducible across machines.** Float16 inference on CPU can
differ slightly between architectures, so two runs of the same version on
different hardware may not produce identical numbers. The check's version
covers the weights and the settings, not the arithmetic of the host.

This is also why nothing here ships a recommended gate, and why
`mediapipe_hand_detection` is not `critical=True` in any example. Quarantining
an episode because a model did not see hands in it is exactly the mistake the
gloves limitation sets up.

## Sanity-check it on your own footage

Because a zero share is both the correct answer for footage without hands and
what a broken setup looks like, confirm a positive detection once, by hand, on
a clip you know contains hands:

```python
import hflow
from hflow.mediapipe_hands import mediapipe_hand_detection

with hflow.Episode("path/to/episode.canonical.mcap") as episode:
    result = mediapipe_hand_detection(episode)

for key, value in sorted(result.measurements.items()):
    print(f"{key} = {value}")
```

If the share is 0.0 on footage you can see hands in, the limitations above are
the place to look first -- gloves, then frame size, then lighting -- before
suspecting the plumbing.

Expect MediaPipe to write a dozen or so native `W0000 ...` warning lines to
stderr per run, about feedback tensors and `NORM_RECT`. They come from the
compiled runtime, are not suppressible through the usual logging environment
variables, and do not indicate a problem.

## Query the result

```sql
SELECT
  episode_id,
  "/wrist_cam/compressed/hand_detected_frame_share" AS hands_share,
  "/wrist_cam/compressed/hand_detection_frame_count" AS frames
FROM episodes
WHERE frames > 0
ORDER BY hands_share
```

Read the share next to its denominator, always. Ten sampled frames and six
hundred sampled frames produce shares that look alike and mean very different
things.

## See also

- [Enable the built-in quality checks](./enable-built-in-checks.md) -- the
  fifteen deterministic checks, which this is not one of.
- [Call an OpenAI vision endpoint from a step](./call-openai-vision.md) -- the
  other way to ask about hands, and the one that can answer questions a
  landmark detector cannot (whose hands, gloved or bare, holding what).
- [Architecture: model-based checks](../ARCHITECTURE.md#model-based-checks) --
  why the two lanes both exist.
