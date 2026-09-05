"""Unit coverage for native-overlay manifest field validators.

These helpers sit at the external JSON boundary. Keeping their exact refusal
messages pinned makes schema drift visible without rebuilding an overlay for
every scalar/path shape.
"""

from __future__ import annotations

import pytest

import hflow.packaging as packaging
from hflow.packaging import CythonOverlayManifestError


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "field must be an object"),
        ({1: "value"}, "field must be an object"),
    ],
)
def test_require_mapping_refuses_non_string_keyed_objects(value: object, message: str) -> None:
    with pytest.raises(CythonOverlayManifestError, match=f"^{message}$"):
        packaging._require_mapping(value, "field")


def test_require_exact_fields_refuses_schema_drift() -> None:
    with pytest.raises(CythonOverlayManifestError, match="^field has unexpected fields$"):
        packaging._require_exact_fields({"a": 1, "extra": 2}, {"a"}, "field")


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_require_integer_refuses_non_integer_values(value: object) -> None:
    with pytest.raises(CythonOverlayManifestError, match="^field must be an integer$"):
        packaging._require_integer(value, "field")


@pytest.mark.parametrize("value", [0, -1])
def test_require_positive_integer_refuses_zero_and_negative(value: int) -> None:
    with pytest.raises(CythonOverlayManifestError, match="^field must be positive$"):
        packaging._require_positive_integer(value, "field")


def test_require_nonnegative_integer_refuses_negative_values() -> None:
    with pytest.raises(CythonOverlayManifestError, match="^field must not be negative$"):
        packaging._require_nonnegative_integer(-1, "field")


@pytest.mark.parametrize("value", ["", "line\nbreak", 7, None])
def test_require_nonempty_string_refuses_non_printable_or_non_string_values(value: object) -> None:
    with pytest.raises(
        CythonOverlayManifestError,
        match="^field must be a non-empty printable string$",
    ):
        packaging._require_nonempty_string(value, "field")


@pytest.mark.parametrize("value", ["not-valid!", "class", "pkg.class", ".pkg"])
def test_require_dotted_name_refuses_invalid_python_identifiers(value: str) -> None:
    with pytest.raises(
        CythonOverlayManifestError,
        match="^field must be a dotted Python identifier$",
    ):
        packaging._require_dotted_name(value, "field")


def test_require_relative_path_refuses_windows_separators() -> None:
    with pytest.raises(CythonOverlayManifestError, match="^field must use POSIX separators$"):
        packaging._require_relative_path(r"pkg\module.py", "field")


@pytest.mark.parametrize("value", ["/absolute/path", "../escape", "pkg/../escape", "./pkg/file"])
def test_require_relative_path_refuses_paths_outside_root(value: str) -> None:
    with pytest.raises(CythonOverlayManifestError, match="^field must remain within its root$"):
        packaging._require_relative_path(value, "field")


@pytest.mark.parametrize(
    "value",
    [
        "sha256:ABCDEF" + "0" * 58,
        "sha1:" + "0" * 40,
        "sha256:" + "0" * 63,
        "0" * 64,
    ],
)
def test_require_sha256_refuses_noncanonical_digests(value: str) -> None:
    with pytest.raises(CythonOverlayManifestError, match="^field must be a lowercase SHA-256$"):
        packaging._require_sha256(value, "field")


def test_manifest_field_validators_accept_canonical_values() -> None:
    assert packaging._require_mapping({"name": "value"}, "field") == {"name": "value"}
    packaging._require_exact_fields({"name": "value"}, {"name"}, "field")
    assert packaging._require_integer(0, "field") == 0
    assert packaging._require_positive_integer(1, "field") == 1
    assert packaging._require_nonnegative_integer(0, "field") == 0
    assert packaging._require_nonempty_string("value", "field") == "value"
    assert packaging._require_dotted_name("pkg.module", "field") == "pkg.module"
    assert packaging._require_relative_path("pkg/module.py", "field") == "pkg/module.py"
    digest = "sha256:" + "a" * 64
    assert packaging._require_sha256(digest, "field") == digest
