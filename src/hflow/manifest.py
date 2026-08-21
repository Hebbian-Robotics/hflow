"""The pipeline manifest: a JSON-able description of one App's registrations.

The manifest is the metadata a pipeline submission crosses a control
boundary as (docs/ARCHITECTURE.md "Tenancy": only metadata, states, and
pointers cross -- never episode bytes, and a service should not need to
execute customer code just to display or diff a pipeline). It carries the
pipeline's identity facts: step names and content-hash versions, gate
configuration, endpoint aliases, and the pipeline/schema versions the
catalog will stamp.

Honesty boundary: producing a manifest REQUIRES importing the pipeline file,
because step versions are content hashes of the live functions
(``hflow.steps.compute_check_version``). ``hflow manifest`` therefore runs
in the pipeline author's own environment (or a sandbox the operator trusts),
and a consumer treats the result as the author's claims -- verified, if
needed, by re-deriving it inside the execution environment, where the same
code must be imported anyway.
"""

import json
from dataclasses import dataclass
from enum import StrEnum

from hflow.steps import RegisteredCheck, RegisteredEnrichment

# Versions the manifest's own JSON shape; bump on any change so consumers
# can refuse loudly instead of misreading.
PIPELINE_MANIFEST_VERSION = 1


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
    version: str
    critical: bool
    requires: tuple[str, ...]
    uses: str | None

    @classmethod
    def from_registered_check(cls, registered: RegisteredCheck) -> "StepManifest":
        return cls(
            name=registered.name,
            kind=StepKind.CHECK,
            version=registered.version,
            critical=registered.critical,
            requires=tuple(sorted(registered.requires)),
            uses=registered.uses,
        )

    @classmethod
    def from_registered_enrichment(cls, registered: RegisteredEnrichment) -> "StepManifest":
        return cls(
            name=registered.name,
            kind=StepKind.ENRICHMENT,
            version=registered.version,
            critical=False,
            requires=tuple(sorted(registered.requires)),
            uses=registered.uses,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "version": self.version,
            "critical": self.critical,
            "requires": list(self.requires),
            "uses": self.uses,
        }


@dataclass(frozen=True)
class DerivedChannelManifest:
    """One registered derived channel's identity."""

    topic: str
    version: str

    def to_json_dict(self) -> dict[str, object]:
        return {"topic": self.topic, "version": self.version}


@dataclass(frozen=True)
class PipelineManifest:
    """Everything a service needs to display, diff, and validate a pipeline
    without holding the code: names, content-hash versions, gate flags, and
    the endpoint aliases the steps declare."""

    pipeline_name: str
    hflow_version: str
    schema_version: str
    pipeline_version: str
    checks: tuple[StepManifest, ...]
    enrichments: tuple[StepManifest, ...]
    derived_channels: tuple[DerivedChannelManifest, ...]
    endpoint_aliases: tuple[str, ...]
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
            "endpoint_aliases": list(self.endpoint_aliases),
            "has_transform_override": self.has_transform_override,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=False) + "\n"
