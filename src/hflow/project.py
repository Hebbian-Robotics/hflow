"""``hflow.toml``: which workspace and which pipeline a directory works with.

Deliberately NOT the workspace's own file. :mod:`hflow.workspace` owns one
data root -- its layout and its durable identity -- and that unit lives
wherever the data lives, which for a hosted deployment is a bucket nobody
edits by hand. This file is the other side: a small, hand-edited marker at
the root of the user's PROJECT saying which workspace that project's
commands act on and which file holds its pipeline, so ``hflow curate`` and
``hflow stale`` stop needing ``--catalog`` and ``--pipeline`` spelled out on
every invocation.

Everything in it is optional and only ever supplies a DEFAULT. Resolution
order, so that a shell or a control plane can always override a file
committed to a repository:

1. an explicit command-line flag
2. the environment (``HFLOW_DATA_ROOT``)
3. the nearest ancestor ``hflow.toml``
4. the built-in default (``./data``)

Remote runtime addressing and credentials are deliberately absent: those
stay environment-injected (docs/HOSTING.md, "The environment injection
contract"), because a file committed beside the pipeline is the wrong place
for a per-deployment URL and exactly the wrong place for a token.

Relative paths resolve against the file's OWN directory rather than the
working directory. That is what makes the ancestor walk worth having: `hflow
curate` from a subdirectory of a project addresses the same workspace as
`hflow curate` from its root.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from hflow.storage import StorageRoot, is_bucket_url, parse_storage_root

PROJECT_CONFIG_FILE_NAME = "hflow.toml"

# Versions the FILE's shape, not the episode schema and not the catalog
# layout -- hence ``config_version`` rather than the ``schema_version`` that
# already means "episode format" in provenance, in the catalog, and in
# staleness queries. Same convention as ``identity_version`` in
# workspace.json and ``manifest_version`` in hflow-bundle.json.
PROJECT_CONFIG_VERSION = 1

_CONFIG_VERSION_KEY = "config_version"
_DATA_ROOT_KEY = "data_root"
_PIPELINE_KEY = "pipeline"
_KNOWN_KEYS = frozenset({_CONFIG_VERSION_KEY, _DATA_ROOT_KEY, _PIPELINE_KEY})


@dataclass(frozen=True)
class ProjectConfig:
    """A parsed ``hflow.toml``. Absent keys stay ``None`` and defer.

    Both settings are optional because the file is a set of defaults, not a
    description: a project that only wants to stop typing ``--catalog``
    writes one line.
    """

    config_file: Path
    storage_root: StorageRoot | None = None
    pipeline_file: Path | None = None


@dataclass(frozen=True)
class NoProjectConfig:
    """No ``hflow.toml`` in the starting directory or any ancestor.

    The ordinary case, and a recoverable one: every caller falls back to the
    environment and then to the built-in defaults. It is a return value
    rather than an exception precisely because it is not a failure -- unlike
    a file that exists and cannot be read, which raises.
    """


ProjectConfigLookup = ProjectConfig | NoProjectConfig


def _require_string(raw_value: object, *, key: str, source_description: str) -> str:
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(
            f"invalid {PROJECT_CONFIG_FILE_NAME} at {source_description}: "
            f"{key!r} must be a non-empty string"
        )
    return raw_value


def _resolve_data_root(raw_value: str, config_directory: Path) -> StorageRoot:
    """Parse the configured data root into a StorageRoot, at the boundary.

    Resolving here rather than at each use is what keeps ``"./data"`` from
    being compared as a string later: ``str(Path("./data"))`` is ``"data"``,
    so a raw value and a resolved one never match even when they name one
    directory.
    """
    if is_bucket_url(raw_value):
        return parse_storage_root(raw_value)
    return parse_storage_root((config_directory / raw_value).resolve())


def parse_project_config(
    raw_payload: bytes, config_file: Path, config_directory: Path
) -> ProjectConfig:
    """Parse one ``hflow.toml`` payload, loudly.

    A file that exists but cannot be understood is never silently skipped:
    falling back to ``./data`` because of a typo would write a corpus into a
    directory the user did not choose, and they would find out later.
    """
    source_description = str(config_file)
    try:
        parsed = tomllib.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(
            f"invalid {PROJECT_CONFIG_FILE_NAME} at {source_description}: {error}"
        ) from error

    config_version = parsed.get(_CONFIG_VERSION_KEY, PROJECT_CONFIG_VERSION)
    if config_version != PROJECT_CONFIG_VERSION:
        raise ValueError(
            f"{PROJECT_CONFIG_FILE_NAME} at {source_description} has "
            f"{_CONFIG_VERSION_KEY} {config_version!r}; this build reads version "
            f"{PROJECT_CONFIG_VERSION!r}"
        )
    unknown_keys = sorted(set(parsed) - _KNOWN_KEYS)
    if unknown_keys:
        # A typo in a settings file is silent by nature: the setting simply
        # never takes effect, and the user concludes the feature is broken.
        raise ValueError(
            f"invalid {PROJECT_CONFIG_FILE_NAME} at {source_description}: unknown "
            f"{'keys' if len(unknown_keys) > 1 else 'key'} "
            f"{', '.join(repr(key) for key in unknown_keys)}; known keys are "
            f"{', '.join(repr(key) for key in sorted(_KNOWN_KEYS))}"
        )

    storage_root: StorageRoot | None = None
    if _DATA_ROOT_KEY in parsed:
        raw_data_root = _require_string(
            parsed[_DATA_ROOT_KEY], key=_DATA_ROOT_KEY, source_description=source_description
        )
        try:
            storage_root = _resolve_data_root(raw_data_root, config_directory)
        except ValueError as error:
            raise ValueError(
                f"invalid {PROJECT_CONFIG_FILE_NAME} at {source_description}: "
                f"{_DATA_ROOT_KEY} {raw_data_root!r} is not a usable data root: {error}"
            ) from error

    pipeline_file: Path | None = None
    if _PIPELINE_KEY in parsed:
        raw_pipeline = _require_string(
            parsed[_PIPELINE_KEY], key=_PIPELINE_KEY, source_description=source_description
        )
        pipeline_file = (config_directory / raw_pipeline).resolve()

    return ProjectConfig(
        config_file=config_file, storage_root=storage_root, pipeline_file=pipeline_file
    )


def find_project_config(start_directory: Path | str | None = None) -> ProjectConfigLookup:
    """The nearest ``hflow.toml`` at or above ``start_directory``.

    The same ancestor walk ``hflow.runtime.infer_hflow_source`` already does
    for a source checkout, and the same one every other project-scoped tool
    does, so running a command from a subdirectory works.
    """
    directory = Path(start_directory) if start_directory is not None else Path.cwd()
    directory = directory.resolve()
    for candidate_directory in (directory, *directory.parents):
        config_file = candidate_directory / PROJECT_CONFIG_FILE_NAME
        if config_file.is_file():
            return parse_project_config(config_file.read_bytes(), config_file, candidate_directory)
    return NoProjectConfig()
