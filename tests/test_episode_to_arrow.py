"""Regression tests for ChannelData.to_arrow."""

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from hflow.episode import ChannelData
from hflow.reader import TopicInfo


def _json_channel(
    topic: str,
    messages: list[dict[str, object]],
    *,
    log_times: list[int] | None = None,
    decoder: Callable[[bytes], Any] | None = None,
) -> ChannelData:
    raw_payloads = [json.dumps(message).encode() for message in messages]
    info = TopicInfo(
        topic=topic,
        channel_id=1,
        schema_name="test_schema",
        schema_encoding="jsonschema",
        message_encoding="json",
        message_count=len(raw_payloads),
        schema_data=b"{}",
    )
    times = np.asarray(log_times if log_times is not None else range(len(messages)), dtype=np.int64)
    return ChannelData(
        topic=topic,
        channel_id=1,
        info=info,
        log_times=times,
        publish_times=times.copy(),
        raw=raw_payloads,
        decoder=decoder or (lambda payload: json.loads(payload.decode())),
    )


def test_to_arrow_rejects_payload_field_that_shadows_log_time() -> None:
    pytest.importorskip("pyarrow")
    channel = _json_channel(
        "/state",
        [{"log_time_ns": 7}, {"log_time_ns": 3}],
        log_times=[1_000_000_000, 2_000_000_000],
    )

    with pytest.raises(ValueError) as excinfo:
        channel.to_arrow()
    message = str(excinfo.value)
    assert "log_time_ns" in message
    assert "/state" in message
    assert "reserved Arrow column" in message


def test_to_arrow_includes_fields_missing_from_the_first_message() -> None:
    pytest.importorskip("pyarrow")
    channel = _json_channel(
        "/state",
        [{"pos": 1}, {"pos": 2, "enabled": True}, {"pos": 3, "enabled": False}],
    )

    table = channel.to_arrow().to_pydict()

    assert table["pos"] == [1, 2, 3]
    assert table["enabled"] == [None, True, False]


def test_to_arrow_keeps_a_column_when_the_first_value_is_null() -> None:
    pytest.importorskip("pyarrow")
    channel = _json_channel("/state", [{"pos": None}, {"pos": 2}, {"pos": 3}])

    assert channel.to_arrow().to_pydict()["pos"] == [None, 2, 3]


def test_to_arrow_keeps_a_numeric_list_when_the_first_value_is_empty() -> None:
    pytest.importorskip("pyarrow")
    channel = _json_channel("/state", [{"pos": []}, {"pos": [1.0, 2.0]}, {"pos": [3.0, 4.0]}])

    assert channel.to_arrow().to_pydict()["pos"] == [[], [1.0, 2.0], [3.0, 4.0]]


def test_to_arrow_includes_numpy_bool_scalars() -> None:
    pytest.importorskip("pyarrow")

    def decode(payload: bytes) -> dict[str, object]:
        decoded = json.loads(payload.decode())
        decoded["flag"] = np.bool_(decoded["flag"])
        return decoded

    channel = _json_channel(
        "/state",
        [{"flag": True}, {"flag": False}],
        decoder=decode,
    )

    assert channel.to_arrow().to_pydict()["flag"] == [True, False]


def test_to_arrow_skips_nested_fields_and_untyped_nulls() -> None:
    pytest.importorskip("pyarrow")
    channel = _json_channel(
        "/state",
        [{"nested": {"a": 1}, "unknown": None}, {"nested": {"a": 2}, "unknown": None}],
    )

    assert channel.to_arrow().column_names == ["log_time_ns"]


def test_to_arrow_skips_a_field_that_starts_empty_then_turns_nested() -> None:
    pytest.importorskip("pyarrow")
    # An empty list carries no element type; a later list of objects stays
    # nested. The field is skipped for the whole channel, not an error.
    channel = _json_channel("/state", [{"items": []}, {"items": [{"a": 1}]}])

    assert channel.to_arrow().column_names == ["log_time_ns"]


def test_to_arrow_keeps_an_empty_numeric_array_before_typed_samples() -> None:
    pytest.importorskip("pyarrow")

    def decode(payload: bytes) -> dict[str, object]:
        decoded = json.loads(payload.decode())
        if decoded["pos"] == "empty":
            decoded["pos"] = np.array([], dtype=np.float64)
        return decoded

    channel = _json_channel(
        "/state",
        [{"pos": "empty"}, {"pos": [1.0, 2.0]}],
        decoder=decode,
    )

    assert channel.to_arrow().to_pydict()["pos"] == [[], [1.0, 2.0]]


def test_to_arrow_rejects_a_field_that_mixes_primitives_and_nested_values() -> None:
    pytest.importorskip("pyarrow")
    channel = _json_channel("/state", [{"pos": 1}, {"pos": {"x": 1}}])

    with pytest.raises(ValueError, match=r"field 'pos' of topic '/state'") as excinfo:
        channel.to_arrow()
    assert "nested" in str(excinfo.value)


def test_to_arrow_rejects_a_field_that_changes_shape() -> None:
    pytest.importorskip("pyarrow")
    channel = _json_channel("/state", [{"pos": 1}, {"pos": [1.0, 2.0]}])

    with pytest.raises(ValueError, match=r"field 'pos' of topic '/state'") as excinfo:
        channel.to_arrow()
    assert "scalar" in str(excinfo.value)
    assert "list" in str(excinfo.value)
