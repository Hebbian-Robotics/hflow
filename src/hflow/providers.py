"""Provider extension point for native-video VLM protocols.

v1 core is deliberately frames-only: most models and the OpenAI-compatible
protocol do not take video, so the honest unit is the frame, and there is no
generic VLM client abstraction (docs/ARCHITECTURE.md, "Model-based checks"). Some model
servers *do* speak a native-video protocol, and the knowledge worth packaging
is exactly the protocol trivia: how that server wants video attached to a
request. This module is the seam where contributors ship that trivia as
separate packages, without touching core (the no-obfuscation tenet: core
never grows a client wrapper, and plugins never need a core patch to exist).

The contract is narrow on purpose:

- A provider package implements :class:`NativeVideoProvider` and registers it
  under the :data:`PROVIDER_ENTRY_POINT_GROUP` entry-point group.
- :func:`discover_providers` finds every installed provider.
- ``prepare_video_request()`` returns the request *payload*; the user posts it
  with their own HTTP client. Shipping a client here would re-wrap what users
  already know, so we never do -- same rule as the frames-only core.

See docs/PROVIDERS.md for the contributor guide.
"""

import logging
from importlib.metadata import EntryPoint, entry_points
from inspect import isclass
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# The entry-point group third-party provider packages register under. A
# fixed literal: it is written into OTHER packages' pyproject.toml, so
# changing it is a documented breaking change for plugin authors, announced
# in the release notes, not an accident.
PROVIDER_ENTRY_POINT_GROUP = "hflow.video_providers"


@runtime_checkable
class NativeVideoProvider(Protocol):
    """One server's knowledge of how video is attached to a VLM request.

    Implementations live in third-party packages (never in core) and are
    registered under :data:`PROVIDER_ENTRY_POINT_GROUP`. A provider builds
    request payloads; it never sends them -- the user owns the HTTP client,
    exactly as in the frames-only flow.

    ``@runtime_checkable`` makes ``isinstance(obj, NativeVideoProvider)`` a
    usable duck-check, with the standard Protocol caveat: it verifies that the
    three members *exist*, not their signatures or return types. Discovery
    uses it to reject obviously-wrong entry points; static checking of a
    provider package is the plugin author's job.
    """

    @property
    def name(self) -> str:
        """Registry key: a short identifier, e.g. ``"vllm"``."""
        ...

    @property
    def protocol(self) -> str:
        """Wire-protocol identifier the payload conforms to, e.g. ``"vllm-video"``."""
        ...

    def prepare_video_request(self, video_path: Path, prompt: str) -> dict[str, object]:
        """Build the request payload this server expects for one video + prompt.

        Returns the JSON-serializable body the *user* passes to their own HTTP
        client (e.g. ``httpx.post(url, json=payload)``). Encoding the video
        into the payload (base64 data URI, file-upload reference, ...) is the
        protocol knowledge the provider exists to encode.
        """
        ...


def discover_providers() -> dict[str, NativeVideoProvider]:
    """Find every installed native-video provider, keyed by ``provider.name``.

    Scans the :data:`PROVIDER_ENTRY_POINT_GROUP` entry points. Each entry
    point must reference either a provider *class* (instantiated with no
    arguments) or an already-constructed provider *instance*. Malformed entry
    points -- import errors, failing constructors, objects that do not satisfy
    :class:`NativeVideoProvider` -- are skipped with a warning, never fatal
    (forward-compat rule: one broken plugin must not take down discovery of
    the rest, or the pipeline importing this module).
    """
    discovered: dict[str, NativeVideoProvider] = {}
    for entry_point in entry_points(group=PROVIDER_ENTRY_POINT_GROUP):
        provider = _load_provider(entry_point)
        if provider is None:
            continue
        if provider.name != entry_point.name:
            # Not fatal: the provider owns its name (one owner per fact); the
            # entry-point name is just packaging metadata. But drift misleads
            # anyone reading the plugin's pyproject.toml, so surface it.
            logger.warning(
                "video provider entry point %r resolves to provider named %r; "
                "registering under %r (the provider's own name)",
                entry_point.name,
                provider.name,
                provider.name,
            )
        if provider.name in discovered:
            logger.warning(
                "duplicate video provider name %r (entry point %r from %r); keeping the first",
                provider.name,
                entry_point.name,
                entry_point.value,
            )
            continue
        discovered[provider.name] = provider
    return discovered


def _load_provider(entry_point: EntryPoint) -> NativeVideoProvider | None:
    """Load one entry point into a provider, or None (with a warning) if malformed.

    The broad ``except Exception`` blocks are deliberate: entry-point metadata
    is external input from arbitrary installed packages, and any failure mode
    it produces is that package's bug, not grounds to crash discovery.
    """
    try:
        loaded = entry_point.load()
    except Exception:
        logger.warning(
            "skipping video provider entry point %r: %r failed to load",
            entry_point.name,
            entry_point.value,
            exc_info=True,
        )
        return None

    if isclass(loaded):
        try:
            candidate: object = loaded()
        except Exception:
            logger.warning(
                "skipping video provider entry point %r: instantiating %r failed "
                "(provider classes must construct with no arguments)",
                entry_point.name,
                entry_point.value,
                exc_info=True,
            )
            return None
    else:
        candidate = loaded

    if not isinstance(candidate, NativeVideoProvider):
        logger.warning(
            "skipping video provider entry point %r: %r does not satisfy "
            "NativeVideoProvider (needs name, protocol, prepare_video_request)",
            entry_point.name,
            entry_point.value,
        )
        return None
    return candidate
