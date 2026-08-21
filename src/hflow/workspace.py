"""The workspace: one data root's layout and durable identity.

A workspace is the single-tenant unit the whole engine operates on -- one
data root holding episodes, the catalog, dev-loop test runs, and (for local
roots) the rendered runtime bundle. docs/ARCHITECTURE.md's "Tenancy" section
commits to scaling HFlow as isolated per-customer workspaces behind an
external control plane, which makes the workspace the thing such a control
plane creates, enumerates, meters, and deletes. This module gives that unit
one owner in code:

- **Layout**: the child-directory names that used to live as string literals
  inside ``App`` and the CLI are defined once here, so every component (and
  the future control plane) agrees on where things are under a data root.
- **Identity**: :meth:`Workspace.ensure_identity` mints a durable workspace
  id into ``workspace.json`` at the data root (create-if-absent, so the
  first writer wins and re-runs are no-ops). A path is not an identity -- a
  workspace moved to a new bucket or mounted at a different vantage keeps
  its id. Identity is opt-in: the open-source single-tenant flow never needs
  it, so nothing writes the marker unless asked.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hflow.storage import StorageRoot, parse_storage_root

# Layout facts, one owner each. These names are stored-data conventions:
# changing any of them is a breaking layout change for existing data roots.
EPISODES_DIRECTORY_NAME = "episodes"
CATALOG_DIRECTORY_NAME = "catalog"
TEST_RUNS_DIRECTORY_NAME = "test-runs"
RUNTIME_BUNDLE_DIRECTORY_NAME = "runtime"

WORKSPACE_IDENTITY_FILE_NAME = "workspace.json"

# Versions the marker's own shape (not the data-root layout): bump when the
# JSON grows or changes meaning, so readers can refuse loudly.
WORKSPACE_IDENTITY_VERSION = 1


@dataclass(frozen=True)
class WorkspaceIdentity:
    """The durable identity stored in a workspace's ``workspace.json``."""

    workspace_id: str
    created_at: str  # ISO-8601 UTC timestamp of first initialization

    def to_json_dict(self) -> dict[str, str | int]:
        return {
            "identity_version": WORKSPACE_IDENTITY_VERSION,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
        }


def _parse_identity_payload(raw_payload: bytes, source_description: str) -> WorkspaceIdentity:
    """Parse a ``workspace.json`` payload at the boundary, loudly."""
    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid workspace identity at {source_description}: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(
            f"invalid workspace identity at {source_description}: expected a JSON object"
        )
    identity_version = parsed.get("identity_version")
    if identity_version != WORKSPACE_IDENTITY_VERSION:
        raise ValueError(
            f"workspace identity at {source_description} has identity_version "
            f"{identity_version!r}; this build reads version {WORKSPACE_IDENTITY_VERSION!r}"
        )
    workspace_id = parsed.get("workspace_id")
    created_at = parsed.get("created_at")
    if not isinstance(workspace_id, str) or not workspace_id:
        raise ValueError(
            f"invalid workspace identity at {source_description}: "
            "'workspace_id' must be a non-empty string"
        )
    if not isinstance(created_at, str) or not created_at:
        raise ValueError(
            f"invalid workspace identity at {source_description}: "
            "'created_at' must be a non-empty string"
        )
    return WorkspaceIdentity(workspace_id=workspace_id, created_at=created_at)


@dataclass(frozen=True)
class Workspace:
    """One data root, addressed through its layout.

    Construct from anything a data root accepts (local path, bucket URL, or
    an existing :data:`~hflow.storage.StorageRoot`) via :meth:`parse`.
    """

    storage_root: StorageRoot

    @classmethod
    def parse(cls, data_root: "Path | str | StorageRoot") -> "Workspace":
        return cls(storage_root=parse_storage_root(data_root))

    def __str__(self) -> str:
        return str(self.storage_root)

    @property
    def episodes_root(self) -> StorageRoot:
        """Where processed runs land: ``<data_root>/episodes/``."""
        return self.storage_root.child(EPISODES_DIRECTORY_NAME)

    @property
    def catalog_root(self) -> StorageRoot:
        """Where the Parquet catalog lives: ``<data_root>/catalog/``."""
        return self.storage_root.child(CATALOG_DIRECTORY_NAME)

    @property
    def test_runs_root(self) -> StorageRoot:
        """Where dev-loop runs land: ``<data_root>/test-runs/``."""
        return self.storage_root.child(TEST_RUNS_DIRECTORY_NAME)

    def identity(self) -> WorkspaceIdentity | None:
        """The stored identity, or ``None`` when none was ever minted."""
        try:
            raw_payload = self.storage_root.read_bytes(WORKSPACE_IDENTITY_FILE_NAME)
        except FileNotFoundError:
            return None
        return _parse_identity_payload(
            raw_payload, self.storage_root.uri_for(WORKSPACE_IDENTITY_FILE_NAME)
        )

    def ensure_identity(self) -> WorkspaceIdentity:
        """Mint the durable workspace id if absent; return what is stored.

        Create-if-absent: concurrent initializers race safely (the storage
        layer arbitrates, exactly like catalog appends) and every caller
        reads back the one identity that won.
        """
        candidate_identity = WorkspaceIdentity(
            workspace_id=uuid.uuid4().hex,
            created_at=datetime.now(UTC).isoformat(),
        )
        candidate_payload = (
            json.dumps(candidate_identity.to_json_dict(), sort_keys=True) + "\n"
        ).encode()
        self.storage_root.write_bytes_if_absent(WORKSPACE_IDENTITY_FILE_NAME, candidate_payload)
        stored_identity = self.identity()
        if stored_identity is None:
            # Not the recoverable "never minted" case identity() reports as
            # None: the storage accepted the write and then lost it, which is
            # an integrity failure, not a missing file.
            raise RuntimeError(
                f"workspace identity vanished after initialization at {self.storage_root}"
            )
        return stored_identity
