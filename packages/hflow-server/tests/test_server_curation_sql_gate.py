"""Preview and pin agree on which SQL the curation gate accepts (#450).

DuckDB types introspection statements (``PRAGMA``, ``DESCRIBE``,
``SUMMARIZE``) as StatementType.SELECT, so a statement-type-only gate let
them through: pin ran them as-is while preview's ``DESCRIBE SELECT * FROM
(<sql>)`` wrapper failed to parse and blamed the caller for the wrapper's
syntax error. The gate now also requires the statement to READ as a SELECT
(leading keyword SELECT/WITH/FROM/VALUES), so both routes refuse the same
inputs with the same sentence, and no refusal ever echoes the rewrite.
"""

import duckdb
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from hflow_server import _curation

_WRAPPER_TEXT = "DESCRIBE SELECT * FROM"
_REFUSAL_SENTENCE = "sql must be exactly one read-only SELECT statement"

# The issue's measured table, plus the query spellings DuckDB supports that
# must stay accepted: FROM-first, VALUES, WITH, and comment-prefixed SELECTs.
AGREEMENT_TABLE = [
    ("SELECT episode_id FROM episodes", True),
    ("WITH ok AS (SELECT episode_id FROM episodes) SELECT * FROM ok", True),
    ("FROM episodes", True),
    ("VALUES (1), (2)", True),
    ("-- keep the good runs\nSELECT episode_id FROM episodes", True),
    ("/* leading block comment */ SELECT 1 AS n", True),
    ("(SELECT episode_id FROM episodes)", True),
    ("PRAGMA database_list", False),
    ("PRAGMA show_tables", False),
    ("PRAGMA version", False),
    ("DESCRIBE SELECT 1", False),
    ("SUMMARIZE SELECT 1", False),
    ("SHOW TABLES", False),
    ("/* SELECT in a comment fools nobody */ PRAGMA version", False),
    ("SELECT 1; SELECT 2", False),
]


@pytest.mark.parametrize(("sql", "accepted"), AGREEMENT_TABLE)
def test_preview_and_pin_agree_on_what_the_gate_accepts(
    writable_api: TestClient, sql: str, accepted: bool
) -> None:
    preview = writable_api.post("/api/v1/curation/preview", json={"sql": sql})
    pin = writable_api.post("/api/v1/curation/pin", json={"sql": sql, "name": "agreement case"})
    if accepted:
        assert preview.status_code == 200, preview.text
        assert pin.status_code == 200, pin.text
    else:
        # Both routes refuse, with the SAME fixed sentence -- never DuckDB's
        # diagnostic for the preview wrapper, and never the rewrite itself.
        for response in (preview, pin):
            assert response.status_code == 400, response.text
            assert response.json()["detail"] == _REFUSAL_SENTENCE
            assert _WRAPPER_TEXT not in response.text


def test_a_wrapper_parse_failure_is_never_blamed_on_the_caller() -> None:
    # Belt and braces behind the gate: if a statement that cannot be
    # subqueried ever reaches the DESCRIBE wrapper again, the 400 is the
    # gate's fixed sentence, not DuckDB's parse diagnostic for OUR rewrite
    # (which names a ')' the caller never typed and echoes the wrapper).
    connection = duckdb.connect()
    try:
        with pytest.raises(HTTPException) as raised:
            _curation._described_columns(connection, "PRAGMA database_list")
    finally:
        connection.close()
    assert raised.value.status_code == 400
    assert raised.value.detail == _REFUSAL_SENTENCE
    assert _WRAPPER_TEXT not in str(raised.value.detail)
