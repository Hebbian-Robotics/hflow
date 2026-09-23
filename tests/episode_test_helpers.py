"""Shared helper for tests that need a canonical episode from a synthetic spec."""

from __future__ import annotations

from pathlib import Path

from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import TransformConfig, write_canonical_episode


def synthesize_canonical_episode(
    directory: Path,
    spec: SyntheticEpisodeSpec,
    *,
    config: TransformConfig | None = None,
    stem: str = "episode",
) -> Path:
    """Synthesize ``spec`` as ``<stem>.mcap`` in ``directory`` and canonicalize it.

    Returns the canonical path, ``<stem>.canonical.mcap`` beside the source.
    ``config=None`` means the default :class:`TransformConfig`.
    """
    source = synthesize_episode(directory / f"{stem}.mcap", spec)
    canonical = directory / f"{stem}.canonical.mcap"
    write_canonical_episode(source, canonical, config)
    return canonical
