"""Score every source window with a vision model while the next windows are sampled.

Prerequisites: the root uv environment, FFmpeg/ffprobe, a local video, and an
OpenAI-compatible chat completions endpoint that accepts image inputs, such as
a local vLLM server. Run from the repository root::

    uv run python examples/score_source_windows.py recording.mp4 \\
        --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --question "Are human hands visible? Answer yes or no." --lookahead 2

Prints one JSON record per window with the model's answer. The source is read
only. Sampled JPEGs live in a temporary directory and are removed after each
window is scored. Sends the frames to the endpoint; set the environment
variable named by --api-key-environment-variable (OPENAI_API_KEY by default)
when the endpoint requires a key. HFlow may download its managed FFmpeg.
"""

import argparse
import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path

import httpx2

from hflow import (
    SourceFrameSamples,
    SourceFrameSampling,
    SourceWindow,
    plan_source_windows,
    sample_source_frames,
)
from hflow.asyncio_utils import prefetch, run_blocking
from hflow.media import UnreadableVideo, UnsupportedVideo, VideoProperties, probe_video

REQUEST_TIMEOUT_SECONDS = 120.0


def encode_image_parts(samples: SourceFrameSamples) -> list[dict[str, object]]:
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,"
                + base64.b64encode(frame.path.read_bytes()).decode("ascii")
            },
        }
        for frame in samples.frames
    ]


async def ask_about_frames(
    client: httpx2.AsyncClient,
    *,
    endpoint: str,
    model: str,
    question: str,
    samples: SourceFrameSamples,
) -> str:
    # File reads and encoding are blocking; keep them off the event loop.
    image_parts = await run_blocking(encode_image_parts, samples)
    response = await client.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": [*image_parts, {"type": "text", "text": question}]}
            ],
        },
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"]).strip()


async def score_windows(arguments: argparse.Namespace) -> None:
    settings = SourceFrameSampling(
        maximum_frames=arguments.maximum_frames,
        maximum_window_millis=arguments.maximum_window_millis,
    )
    match probe_video(arguments.source):
        case VideoProperties() as properties:
            windows = plan_source_windows(
                properties.duration_millis,
                maximum_window_millis=settings.maximum_window_millis,
            )
        case UnreadableVideo():
            raise SystemExit("Source video is unreadable")
        case UnsupportedVideo():
            raise SystemExit("Source video exceeds the inspection limits")

    def sample_window(window: SourceWindow, directory: Path) -> SourceFrameSamples:
        # Blocking FFmpeg work; prefetch runs it off the event loop.
        return sample_source_frames(
            arguments.source, directory / "frames", window=window, settings=settings
        )

    api_key = os.environ.get(arguments.api_key_environment_variable)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with tempfile.TemporaryDirectory(prefix="score-source-windows-") as scratch_directory:
        async with (
            httpx2.AsyncClient(headers=headers, timeout=REQUEST_TIMEOUT_SECONDS) as client,
            prefetch(
                windows,
                sample_window,
                working_directory=scratch_directory,
                lookahead=arguments.lookahead,
            ) as sampled_windows,
        ):
            async for sampled in sampled_windows:
                samples = sampled.value
                # An empty window has nothing to judge; it is not a "no" answer.
                answer = (
                    await ask_about_frames(
                        client,
                        endpoint=arguments.endpoint,
                        model=arguments.model,
                        question=arguments.question,
                        samples=samples,
                    )
                    if samples.frames
                    else None
                )
                print(
                    json.dumps(
                        {
                            "start_millis": sampled.item.start_millis,
                            "end_millis": sampled.item.end_millis,
                            "frame_timestamps_seconds": [
                                float(frame.timestamp_seconds) for frame in samples.frames
                            ],
                            "answer": answer,
                        }
                    ),
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--maximum-window-millis", type=int, default=8_000)
    parser.add_argument("--maximum-frames", type=int, default=8)
    parser.add_argument(
        "--lookahead", type=int, default=2, help="windows sampled ahead of the one being scored"
    )
    parser.add_argument("--api-key-environment-variable", default="OPENAI_API_KEY")
    asyncio.run(score_windows(parser.parse_args()))


if __name__ == "__main__":
    main()
