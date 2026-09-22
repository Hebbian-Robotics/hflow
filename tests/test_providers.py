"""The native-video provider extension point.

Hermetic strategy: fake plugin modules are injected into ``sys.modules`` and
real ``importlib.metadata.EntryPoint`` objects point at them, so discovery
exercises the genuine ``EntryPoint.load()`` path without installing any
distribution.
"""

import logging
import sys
import types
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from hflow import providers
from hflow.providers import (
    PROVIDER_ENTRY_POINT_GROUP,
    NativeVideoProvider,
    discover_providers,
)

FAKE_PLUGIN_MODULE = "fake_video_provider_plugin"


class DummyVllmVideoProvider:
    """A well-formed provider, as a plugin package would ship it."""

    name = "dummy-vllm"
    protocol = "vllm-video"

    def prepare_video_request(self, video_path: Path, prompt: str) -> dict[str, object]:
        return {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
                    ],
                }
            ],
        }


class MissingMethodProvider:
    """Has the identity properties but no prepare_video_request."""

    name = "incomplete"
    protocol = "nothing"


class ExplodingConstructorProvider:
    name = "exploding"
    protocol = "boom"

    def __init__(self) -> None:
        raise RuntimeError("constructor requires configuration we do not have")

    def prepare_video_request(self, video_path: Path, prompt: str) -> dict[str, object]:
        return {}


class MisnamedProvider(DummyVllmVideoProvider):
    name = "actual-name"


# A plugin may also register a module-level instance rather than a class.
PREBUILT_INSTANCE = DummyVllmVideoProvider()


def _install_fake_plugin_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType(FAKE_PLUGIN_MODULE)
    exported_plugin_objects: dict[str, object] = {
        "DummyVllmVideoProvider": DummyVllmVideoProvider,
        "MissingMethodProvider": MissingMethodProvider,
        "ExplodingConstructorProvider": ExplodingConstructorProvider,
        "MisnamedProvider": MisnamedProvider,
        "PREBUILT_INSTANCE": PREBUILT_INSTANCE,
    }
    for attribute_name, attribute_value in exported_plugin_objects.items():
        setattr(module, attribute_name, attribute_value)
    monkeypatch.setitem(sys.modules, FAKE_PLUGIN_MODULE, module)


def _register_entry_points(
    monkeypatch: pytest.MonkeyPatch, *name_value_pairs: tuple[str, str]
) -> None:
    """Make discovery see exactly these entry points in the provider group."""
    _install_fake_plugin_module(monkeypatch)
    fake_entry_points = [
        EntryPoint(name=name, value=value, group=PROVIDER_ENTRY_POINT_GROUP)
        for name, value in name_value_pairs
    ]

    def entry_points_stub(*, group: str) -> list[EntryPoint]:
        assert group == PROVIDER_ENTRY_POINT_GROUP
        return fake_entry_points

    monkeypatch.setattr(providers, "entry_points", entry_points_stub)


def test_registered_provider_class_is_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_entry_points(
        monkeypatch, ("dummy-vllm", f"{FAKE_PLUGIN_MODULE}:DummyVllmVideoProvider")
    )
    discovered = discover_providers()
    assert set(discovered) == {"dummy-vllm"}
    provider = discovered["dummy-vllm"]
    assert provider.protocol == "vllm-video"
    assert isinstance(provider, NativeVideoProvider)


def test_module_level_instance_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_entry_points(monkeypatch, ("dummy-vllm", f"{FAKE_PLUGIN_MODULE}:PREBUILT_INSTANCE"))
    discovered = discover_providers()
    assert discovered["dummy-vllm"] is PREBUILT_INSTANCE


def test_unimportable_entry_point_warns_and_spares_the_rest(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _register_entry_points(
        monkeypatch,
        ("broken", "package_that_does_not_exist:Provider"),
        ("dummy-vllm", f"{FAKE_PLUGIN_MODULE}:DummyVllmVideoProvider"),
    )
    with caplog.at_level(logging.WARNING, logger="hflow.providers"):
        discovered = discover_providers()
    assert set(discovered) == {"dummy-vllm"}  # the broken plugin never blocks the healthy one
    assert any("failed to load" in record.message for record in caplog.records)


def test_object_not_satisfying_protocol_warns_and_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _register_entry_points(
        monkeypatch, ("incomplete", f"{FAKE_PLUGIN_MODULE}:MissingMethodProvider")
    )
    with caplog.at_level(logging.WARNING, logger="hflow.providers"):
        discovered = discover_providers()
    assert discovered == {}
    assert any("does not satisfy" in record.message for record in caplog.records)


def test_failing_constructor_warns_and_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _register_entry_points(
        monkeypatch, ("exploding", f"{FAKE_PLUGIN_MODULE}:ExplodingConstructorProvider")
    )
    with caplog.at_level(logging.WARNING, logger="hflow.providers"):
        discovered = discover_providers()
    assert discovered == {}
    assert any("instantiating" in record.message for record in caplog.records)


def test_duplicate_provider_name_keeps_the_first_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _register_entry_points(
        monkeypatch,
        ("dummy-vllm", f"{FAKE_PLUGIN_MODULE}:PREBUILT_INSTANCE"),
        ("dummy-vllm", f"{FAKE_PLUGIN_MODULE}:DummyVllmVideoProvider"),
    )
    with caplog.at_level(logging.WARNING, logger="hflow.providers"):
        discovered = discover_providers()
    assert discovered["dummy-vllm"] is PREBUILT_INSTANCE
    assert any("duplicate video provider name" in record.message for record in caplog.records)


def test_entry_point_name_drift_warns_but_registers_under_provider_name(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _register_entry_points(
        monkeypatch, ("stale-metadata-name", f"{FAKE_PLUGIN_MODULE}:MisnamedProvider")
    )
    with caplog.at_level(logging.WARNING, logger="hflow.providers"):
        discovered = discover_providers()
    # The provider owns its name; the entry-point name is packaging metadata.
    assert set(discovered) == {"actual-name"}
    assert any("registering under" in record.message for record in caplog.records)
