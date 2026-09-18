"""Refusal-branch tests for :meth:`hflow.episode.ChannelData.to_numpy`.

``to_numpy`` stacks one message field into a ``(n_messages, ...)`` array,
selecting the field automatically when ``field=None`` through a priority
ladder -- ``position`` array, lone numeric array, lone numeric scalar -- and
refusing with a specific error when the choice is empty or ambiguous.

These refusals cannot be produced by :func:`hflow.testing.synthesize_episode`,
which only generates well-formed happy-path channels, so the fixtures build
``ChannelData`` objects directly from raw JSON payloads whose decodes yield
exactly the fields each scenario needs.
"""

import json

import numpy as np
import pytest

from hflow.episode import ChannelData
from hflow.reader import TopicInfo


def _json_channel(
    topic: str, messages: list[dict[str, object]], *, schema_name: str = "test_schema"
) -> ChannelData:
    """A ``ChannelData`` (no file) whose JSON payloads decode to ``messages``."""
    raw_payloads = [json.dumps(message).encode() for message in messages]
    info = TopicInfo(
        topic=topic,
        channel_id=1,
        schema_name=schema_name,
        schema_encoding="jsonschema",
        message_encoding="json",
        message_count=len(raw_payloads),
        schema_data=b"{}",
    )
    return ChannelData(
        topic=topic,
        channel_id=1,
        info=info,
        log_times=np.asarray([], dtype=np.int64),
        publish_times=np.asarray([], dtype=np.int64),
        raw=raw_payloads,
        decoder=lambda payload: json.loads(payload.decode()),
    )


def test_empty_channel_refuses() -> None:
    channel = _json_channel("/empty", [])

    with pytest.raises(ValueError, match=r"topic '/empty' has no messages"):
        channel.to_numpy()


def test_ambiguous_array_fields_list_candidates() -> None:
    channel = _json_channel(
        "/multi_array",
        [{"field_a": [1.0, 2.0], "field_b": [3.0, 4.0]} for _ in range(2)],
    )

    with pytest.raises(
        ValueError, match=r"topic '/multi_array' has multiple numeric array fields"
    ) as excinfo:
        channel.to_numpy()
    message = str(excinfo.value)
    assert "'field_a'" in message
    assert "'field_b'" in message
    assert "pass field=" in message


def test_ambiguous_scalar_fields_list_candidates() -> None:
    channel = _json_channel(
        "/multi_scalar",
        [{"field_a": 1.0, "field_b": 2.0} for _ in range(2)],
    )

    with pytest.raises(
        ValueError, match=r"topic '/multi_scalar' has multiple numeric fields"
    ) as excinfo:
        channel.to_numpy()
    message = str(excinfo.value)
    assert "'field_a'" in message
    assert "'field_b'" in message
    assert "pass field=" in message


def test_no_numeric_fields_lists_available_fields() -> None:
    channel = _json_channel(
        "/no_numeric",
        [{"name": "joint", "enabled": True} for _ in range(2)],
    )

    with pytest.raises(
        ValueError,
        match=r"topic '/no_numeric' \([^)]*\) has no numeric fields; available fields:",
    ) as excinfo:
        channel.to_numpy()
    message = str(excinfo.value)
    assert "'name'" in message
    assert "'enabled'" in message


def test_invalid_explicit_field_lists_available_fields() -> None:
    channel = _json_channel(
        "/jpos",
        [{"position": [1.0, 2.0], "velocity": [3.0, 4.0]} for _ in range(2)],
    )

    with pytest.raises(KeyError, match=r"field 'nope' not in topic '/jpos'; available:") as excinfo:
        channel.to_numpy(field="nope")
    message = str(excinfo.value)
    assert "'position'" in message
    assert "'velocity'" in message


def test_ragged_field_is_rejected() -> None:
    channel = _json_channel(
        "/ragged",
        [{"path": [1.0, 2.0]}, {"path": [3.0, 4.0, 5.0]}],
    )

    with pytest.raises(
        ValueError,
        match=r"field 'path' of topic '/ragged' is ragged \(per-message lengths differ\)",
    ):
        channel.to_numpy(field="path")


def test_non_numeric_field_is_rejected() -> None:
    channel = _json_channel(
        "/non_numeric",
        [{"label": "a"}, {"label": "b"}],
    )

    with pytest.raises(
        ValueError, match=r"field 'label' of topic '/non_numeric' is not numeric \(dtype"
    ):
        channel.to_numpy(field="label")


def test_position_field_wins_over_other_arrays() -> None:
    """The first selection clause: ``position`` beats an ambiguous array field."""
    channel = _json_channel(
        "/joint",
        [
            {"position": [1.0, 2.0], "velocity": [3.0, 4.0]},
            {"position": [5.0, 6.0], "velocity": [7.0, 8.0]},
        ],
    )

    result = channel.to_numpy()
    np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [5.0, 6.0]]))


def test_lone_array_field_is_selected() -> None:
    channel = _json_channel(
        "/lone_array",
        [{"joint_angles": [1.0, 2.0, 3.0]}, {"joint_angles": [4.0, 5.0, 6.0]}],
    )

    result = channel.to_numpy()
    np.testing.assert_array_equal(result, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_lone_scalar_field_is_selected() -> None:
    channel = _json_channel(
        "/lone_scalar",
        [{"temperature": 20.5}, {"temperature": 21.0}],
    )

    result = channel.to_numpy()
    np.testing.assert_array_equal(result, np.array([20.5, 21.0]))
