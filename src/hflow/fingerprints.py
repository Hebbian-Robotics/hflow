"""Stable fingerprints for author-owned, JSON-compatible contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import NewType, TypeAlias

from hflow.steps import StepVersion, parse_step_version

ContractFingerprint = NewType("ContractFingerprint", str)
ContractFingerprint.__module__ = __name__
ContractScalar: TypeAlias = str | int | float | bool | None
NormalizedContractValue: TypeAlias = (
    ContractScalar | list["NormalizedContractValue"] | dict[str, "NormalizedContractValue"]
)


def _normalize_contract_value(value: object, *, path: str) -> NormalizedContractValue:
    """Parse one external contract value into the canonical JSON domain."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"contract value at {path} must be finite, got {value!r}")
        return value
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, NormalizedContractValue] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"contract key at {path} must be a string, got {type(key).__name__}"
                )
            normalized_mapping[key] = _normalize_contract_value(
                nested_value,
                path=f"{path}.{key}",
            )
        return normalized_mapping
    if isinstance(value, list | tuple):
        return [
            _normalize_contract_value(nested_value, path=f"{path}[{index}]")
            for index, nested_value in enumerate(value)
        ]
    raise TypeError(f"contract value at {path} must be JSON-compatible, got {type(value).__name__}")


def fingerprint_contract(contract: Mapping[str, object]) -> ContractFingerprint:
    """Return the full SHA-256 of a canonical JSON-compatible mapping.

    Mapping order and list-versus-tuple representation do not affect the
    fingerprint. Non-string keys, non-finite floats, and values outside the JSON
    domain are refused before hashing so two callers cannot accidentally rely on
    serializer-specific fallbacks.
    """

    normalized_contract = _normalize_contract_value(contract, path="$")
    if not isinstance(normalized_contract, dict):
        raise AssertionError("a mapping contract must normalize to a dictionary")
    serialized_contract = json.dumps(
        normalized_contract,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ContractFingerprint(hashlib.sha256(serialized_contract.encode()).hexdigest())


def step_version_from_contract(
    version_namespace: str,
    contract: Mapping[str, object],
) -> StepVersion:
    """Build a step version from its compatibility namespace and full contract.

    The namespace remains the author's human-readable compatibility promise. The
    digest makes prompt, model, threshold, and other serialized configuration
    changes visible without hand-rolling canonical JSON at each registration site.
    """

    parsed_namespace = parse_step_version(version_namespace)
    contract_digest = str(fingerprint_contract(contract))[:16]
    return parse_step_version(f"{parsed_namespace}-{contract_digest}")
