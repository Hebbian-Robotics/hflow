"""Regression tests for ChannelData.to_arrow."""

import json

import numpy as np
import pytest

from hflow.episode import ChannelData
from hflow.reader import TopicInfo


def _json_channel(
    topic: str,
    messages: list[dict[str, object]],
    *,
    log_times: list[int],
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
    times = np.asarray(log_times, dtype=np.int64)
    return ChannelData(
        topic=topic,
        channel_id=1,
        info=info,
        log_times=times,
        publish_times=times.copy(),
        raw=raw_payloads,
        decoder=lambda payload: json.loads(payload.decode()),
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
