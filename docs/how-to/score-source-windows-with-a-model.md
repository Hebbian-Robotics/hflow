# Score source windows with a model while the next ones are prepared

Use this when one long recording is split into windows and each window is
prepared locally (frames sampled, a clip re-encoded) and then sent to a model
endpoint. In a plain loop the endpoint waits during every preparation and the
CPU waits during every request. `hflow.asyncio_utils.prefetch` overlaps the two:
while the model scores one window, the next windows are already being prepared.

The [runnable example](../../examples/score_source_windows.py) plans complete
windows, samples frames for upcoming windows ahead of time, and asks an
OpenAI-compatible chat completions endpoint one question per window.

## Run the example

From the repository root, with the normal uv environment, a local video, and an
endpoint that accepts image inputs (for example a local vLLM server):

```bash
uv run python examples/score_source_windows.py recording.mp4 \
  --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-VL-7B-Instruct \
  --question "Are human hands visible? Answer yes or no." --lookahead 2
```

It prints one JSON record per window, in window order, with the sampled frame
timestamps and the model's answer. A window with no eligible frames gets
`"answer": null` and no request; that is not a negative answer. The source is
read only and sampled JPEGs are deleted once their window is scored. Frames are
sent to the endpoint. If it needs a key, export the variable named by
`--api-key-environment-variable` (`OPENAI_API_KEY` by default).

## Prefetch the preparation

```python
from hflow.asyncio_utils import prefetch


def sample_window(window, directory):
    return sample_source_frames(source, directory / "frames", window=window, settings=settings)


async with prefetch(
    windows, sample_window, working_directory=scratch, lookahead=2
) as sampled_windows:
    async for sampled in sampled_windows:
        answer = await ask_model(sampled.value)  # sampled.item is the window
```

- `prepare(item, directory)` is ordinary blocking code. It runs through
  `run_blocking` in a new, empty directory under `working_directory`.
- Items arrive in input order, even when several preparations run at once.
- `lookahead` is how many later items are prepared, or kept ready, while you use
  the current one. At most `max(lookahead, 1)` preparations run at once and at
  most `lookahead + 1` directories exist. The next preparation starts when an
  item is handed to you. `lookahead=0` prepares each item only when you ask for
  it, which is the plain loop.
- An item's directory is removed when you advance to the next item or leave the
  `async with` block. Copy anything you need to keep.
- A failed preparation raises when you reach that item. Earlier items are still
  delivered, and leaving the block cleans up the rest.

Choose `lookahead` from how long a preparation takes relative to a request. If
preparing a window takes about as long as scoring it, `1` keeps both busy.
Raise it when preparation is slower than the endpoint: more windows then prepare
in parallel. Each running preparation uses its own CPU time and each held item
its own directory, so size it to the machine.

## Stop outside readers before cleanup

A local model server may still be reading the current item's files when the
caller is cancelled, and a preparation may be waiting on that server. Pass
`cancel_hook=` to stop it. When the block is left while it still holds items,
because of cancellation, an error, or an early `break`, `prefetch` runs the hook
to completion, then waits for started preparations, then removes their
directories. Repeated cancellation cannot interrupt that sequence. Running out
of items does not call the hook.

For a single blocking call with the same guarantee, use
`run_blocking_with_cancel_hook(stop_server, operation, ...)`.

## When not to use it

`App.process_many(max_workers=...)` already runs several episodes concurrently,
so preparation for one episode overlaps model calls for another. Use `prefetch`
inside one episode or one source, where windows must be handled in order and
their preparation dominates. It does not retry, persist progress, or limit
request concurrency; the caller owns those policies.
