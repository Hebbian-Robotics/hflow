"""The pipeline manifest: a JSON-able description of one App's registrations.

The manifest is the metadata a pipeline submission crosses a control
boundary as (docs/ARCHITECTURE.md "Tenancy": only metadata, states, and
pointers cross -- never episode bytes, and a service should not need to
execute customer code just to display or diff a pipeline). It carries the
pipeline's identity facts: step names and author-declared versions, gate
configuration, resource requirements, and the pipeline/schema versions the
catalog will stamp.

Honesty boundary: producing a manifest REQUIRES importing the pipeline file,
because registrations are executable SDK calls even though their versions are
explicit. ``hflow manifest`` therefore runs in the pipeline author's own
environment (or a sandbox the operator trusts), and a consumer treats the
result as the author's claims.
"""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum

from hflow.steps import Gate, RegisteredCheck, RegisteredEnrichment, StepVersion

# Versions the manifest's own JSON shape; bump on any change so consumers
# can refuse loudly instead of misreading.
# 2: steps carry their declarative gate, so a service can show WHICH policy
#    rejected an episode rather than only that some critical check did.
# 3: step versions are explicit pipeline-author promises rather than
#    engine-derived identifiers.
# 4: endpoint aliases were removed; external-service configuration belongs to
#    the step that calls the service rather than the App orchestration layer.
PIPELINE_MANIFEST_VERSION = 4


class StepKind(StrEnum):
    """Which registration surface a step came from."""

    CHECK = "check"
    ENRICHMENT = "enrichment"


@dataclass(frozen=True)
class StepManifest:
    """One registered step's identity and gate configuration."""

    name: str
    # Restates which PipelineManifest list holds this entry -- deliberate,
    # so a serialized step stays self-describing when consumers flatten or
    # diff entries out of their list context.
    kind: StepKind
    version: StepVersion
    critical: bool
    requires: tuple[str, ...]
    # The policy this step gates on, when it declares one. ``critical`` alone
    # says a gate exists; this says what it is, which is the difference between
    # a service reporting "a critical check failed" and reporting which
    # threshold on which measurement rejected the episode.
    gate: Gate | None = None

    @classmethod
    def from_registered_check(cls, registered: RegisteredCheck) -> "StepManifest":
        return cls(
            name=registered.name,
            kind=StepKind.CHECK,
            version=registered.version,
            critical=registered.critical,
            requires=tuple(sorted(registered.requires)),
            gate=registered.gate,
        )

    @classmethod
    def from_registered_enrichment(cls, registered: RegisteredEnrichment) -> "StepManifest":
        # Enrichments never gate, so there is nothing to report rather than a
        # gate that is merely absent.
        return cls(
            name=registered.name,
            kind=StepKind.ENRICHMENT,
            version=registered.version,
            critical=False,
            requires=tuple(sorted(registered.requires)),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "version": self.version,
            "critical": self.critical,
            "requires": list(self.requires),
            "gate": _gate_json(self.gate),
        }


def _gate_json(gate: Gate | None) -> dict[str, object] | None:
    """A gate as plain JSON, derived from the dataclass rather than restated.

    ``asdict`` keeps this from drifting out of step with the fields, and the
    enum members serialize as their own values because they are ``StrEnum``.
    """
    return asdict(gate) if gate is not None else None


@dataclass(frozen=True)
class DerivedChannelManifest:
    """One registered derived channel's identity."""

    topic: str
    version: StepVersion

    def to_json_dict(self) -> dict[str, object]:
        return {"topic": self.topic, "version": self.version}


@dataclass(frozen=True)
class PipelineManifest:
    """Everything a service needs to display, diff, and validate a pipeline
    without holding the code: names, explicit versions, gate flags, and
    resource requirements."""

    pipeline_name: str
    hflow_version: str
    schema_version: str
    pipeline_version: str
    checks: tuple[StepManifest, ...]
    enrichments: tuple[StepManifest, ...]
    derived_channels: tuple[DerivedChannelManifest, ...]
    has_transform_override: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "manifest_version": PIPELINE_MANIFEST_VERSION,
            "pipeline_name": self.pipeline_name,
            "hflow_version": self.hflow_version,
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "checks": [step.to_json_dict() for step in self.checks],
            "enrichments": [step.to_json_dict() for step in self.enrichments],
            "derived_channels": [channel.to_json_dict() for channel in self.derived_channels],
            "has_transform_override": self.has_transform_override,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=False) + "\n"
