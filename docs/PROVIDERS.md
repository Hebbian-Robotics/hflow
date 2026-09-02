# Add native-video providers to HFlow

How to add support for a VLM server that accepts video natively, as a separate package, without touching HFlow core.

## Why frames-only is the core default

Most models, and the OpenAI-compatible protocol itself, do not take video. The honest unit in v1 is therefore the frame: you extract frames explicitly (`ep.frames(fps=...)`), call your own client, and own how per-frame answers aggregate (see the [porting guide](./PORTING.md), Dialect 3). Baking a video-in abstraction into core would pretend a capability most endpoints lack. Protocol honesty wins over convenience here.

But some servers *do* speak a native-video protocol (vLLM's `video_url` content parts, for example), and the knowledge worth packaging is exactly that protocol trivia: **how this server wants video attached to a request**. That is what a provider encodes, nothing more. Core stays frames-only; providers are the contributor-shaped extension point for everything else.

## The contract

A provider is any object satisfying `hflow.providers.NativeVideoProvider`:

| Member | Meaning |
|---|---|
| `name: str` | Registry key, e.g. `"vllm"`. The provider owns this fact. |
| `protocol: str` | Wire-protocol identifier the payload conforms to, e.g. `"vllm-video"`. |
| `prepare_video_request(video_path, prompt) -> dict[str, object]` | Builds the JSON-serializable request body for one video + prompt. |

`prepare_video_request` returns a **payload, not a response**: you post it with your own HTTP client. HFlow never ships a client, providers included.

The protocol is `@runtime_checkable`, so `isinstance(obj, NativeVideoProvider)` works as a duck-check. Standard `Protocol` caveat: `isinstance` verifies the three members *exist*, not their signatures. Run a type checker over your provider package for the rest.

## What a provider package looks like

A minimal package (`hflow-vllm-video`, say) is one class and one entry point.

`src/hflow_vllm_video/provider.py`:

```python
import base64
from pathlib import Path


class VllmVideoProvider:
    """Builds chat-completions payloads with vLLM's video_url content part."""

    name = "vllm"
    protocol = "vllm-video"

    def __init__(self, model: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
        self.model = model

    def prepare_video_request(self, video_path: Path, prompt: str) -> dict[str, object]:
        video_b64 = base64.b64encode(video_path.read_bytes()).decode()
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                        },
                    ],
                }
            ],
        }
```

`pyproject.toml`:

```toml
[project.entry-points."hflow.video_providers"]
vllm = "hflow_vllm_video.provider:VllmVideoProvider"
```

The entry point may reference a provider **class** (discovery instantiates it with no arguments) or an already-constructed module-level **instance**. Keep the entry-point name equal to the provider's `name`: discovery registers under `provider.name` and warns when the two drift.

## Using a provider in a check

Discovery is one call; the HTTP request stays yours:

```python
import httpx  # your client, your dependency
import os

import hflow
from hflow.providers import discover_providers


@app.check(version="1", requires=("vision-model",))
def grasp_succeeded(ep: hflow.Episode) -> hflow.CheckResult:
    provider = discover_providers()["vllm"]
    payload = provider.prepare_video_request(
        ep.video("wrist_cam"),  # lossless remux, cached
        "Did the gripper grasp the towel? yes/no",
    )
    endpoint = os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1")
    response = httpx.post(f"{endpoint}/chat/completions", json=payload)
    answer = response.json()["choices"][0]["message"]["content"]
    return hflow.CheckResult(measurements={"grasp_succeeded": "yes" in answer.lower()})
```

`discover_providers()` returns every installed provider keyed by `name`. Malformed entry points (import errors, failing constructors, objects missing the protocol members) are skipped with a warning, never fatal: one broken plugin cannot take down discovery of the rest.

## Stability of the entry-point group

The group name `hflow.video_providers` is a fixed literal: it lives in *your* package's metadata, so it never changes silently. Any change would be a documented breaking change for plugin authors, announced in release notes.

## Non-goals

- **No bundled client.** Providers build payloads; sending them is your code, with your retries, timeouts, and auth.
- **No orchestration of model servers.** Starting, scaling, or health-checking vLLM/Ollama/hosted endpoints is out of scope. A step owns the full client configuration it consumes. ("Provider" in this page means a protocol extension package, like Airflow's provider packages, not an endpoint URL.)
- **No aggregation policy.** Whether one video-level answer or many frame-level answers make an episode verdict is yours, same as in the frames-only flow.

## See also

- [Architecture](./ARCHITECTURE.md): see "Model-based checks" for the frames-only decision and its provenance
- [Porting guide](./PORTING.md): Dialect 3 (JPEG frames and your own VLM client)
