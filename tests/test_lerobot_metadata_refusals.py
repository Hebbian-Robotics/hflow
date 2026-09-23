"""Regression coverage for LeRobot metadata refusal boundaries (#405)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import Request, Response
from huggingface_hub.errors import RemoteEntryNotFoundError
from lerobot_test_helpers import (
    V3_DATA_PATH_TEMPLATE,
    V3_VIDEO_PATH_TEMPLATE,
    exactly,
    stub_hub_repo_info,
)

import hflow.importers.lerobot as prep

_REPO = "fake/repo"
_SHA = "abcdef1234567890"


def _info(**overrides: object) -> dict[str, object]:
    info: dict[str, object] = {
        "fps": 30,
        "data_path": V3_DATA_PATH_TEMPLATE,
        "video_path": V3_VIDEO_PATH_TEMPLATE,
        "features": {},
    }
    info.update(overrides)
    return info


def _assert_no_dataset_output(output_dir: Path) -> None:
    assert not (output_dir / "landing").exists()
    assert not (output_dir / "prepared-manifest.json").exists()


def _import(output_dir: Path) -> None:
    prep.import_lerobot_dataset(dataset_repo=_REPO, output_dir=output_dir)


def test_import_refuses_repository_without_lerobot_v3_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_hub_repo_info(monkeypatch, resolved_sha=_SHA)

    def missing_info(*_args: object, **_kwargs: object) -> str:
        raise RemoteEntryNotFoundError(
            "missing meta/info.json",
            response=Response(404, request=Request("GET", "https://huggingface.co/missing")),
        )

    monkeypatch.setattr(prep, "hf_hub_download", missing_info)
    output_dir = tmp_path / "out"

    with pytest.raises(
        RuntimeError, match=exactly("meta/info.json not found; not a LeRobot v3 repository")
    ):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_info_json_that_is_not_an_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_hub_repo_info(monkeypatch, resolved_sha=_SHA)
    metadata_file = tmp_path / "info.json"
    metadata_file.write_text("[]")
    monkeypatch.setattr(prep, "hf_hub_download", lambda *_args, **_kwargs: str(metadata_file))
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=exactly("LeRobot meta/info.json is not a JSON object")):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("data_path", "", "LeRobot meta/info.json must define a non-empty data_path template"),
        ("video_path", "", "LeRobot meta/info.json must define a non-empty video_path template"),
    ],
)
def test_import_refuses_empty_path_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    stub_hub_repo_info(monkeypatch, resolved_sha=_SHA)
    info = _info(**{field: value})
    monkeypatch.setattr(prep, "_fetch_info_json", lambda _repo, _revision, _cache: info)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=exactly(message)):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


@pytest.mark.parametrize(
    ("field", "template", "detail"),
    [
        ("data_path", "data/{episode_index:03d}/f.parquet", "unknown field 'episode_index'"),
        ("data_path", "data/{chunk_index:03d/f.parquet", "unmatched '{' in format spec"),
        (
            "data_path",
            "data/{chunk_index:qq}/f.parquet",
            "Invalid format specifier 'qq' for object of type 'int'",
        ),
        (
            "data_path",
            "data/{0}/f.parquet",
            "positional fields are not supported, name the field instead",
        ),
        ("data_path", "data/}chunk/f.parquet", "Single '}' encountered in format string"),
        # Subscripting a field raises TypeError rather than the other three,
        # so without it in the caught set this one escapes the boundary raw.
        ("data_path", "data/{chunk_index[0]}/f.parquet", "'int' object is not subscriptable"),
        ("video_path", "videos/{episode_index}/f.mp4", "unknown field 'episode_index'"),
        (
            "video_path",
            "videos/{0}/f.mp4",
            "positional fields are not supported, name the field instead",
        ),
    ],
)
def test_import_refuses_a_template_the_converter_could_not_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    template: str,
    detail: str,
) -> None:
    """Each of these used to pass the boundary and raise at ``str.format``.

    The second assertion is the point. The old failure arrived after
    ``meta/episodes`` had been listed and every episode metadata parquet
    downloaded; asserting only the message would pass either way.
    """
    stub_hub_repo_info(monkeypatch, resolved_sha=_SHA)
    monkeypatch.setattr(
        prep, "_fetch_info_json", lambda _repo, _revision, _cache: _info(**{field: template})
    )
    listed_repositories: list[str] = []

    def recording_metadata_files(repo: str, _revision: str) -> list[str]:
        listed_repositories.append(repo)
        return []

    monkeypatch.setattr(prep, "_hf_episode_metadata_files", recording_metadata_files)
    output_dir = tmp_path / "out"

    message = f"LeRobot meta/info.json has an invalid {field} template {template!r}: {detail}"
    with pytest.raises(ValueError, match=exactly(message)):
        _import(output_dir)

    assert listed_repositories == []
    _assert_no_dataset_output(output_dir)


def test_the_two_templates_are_checked_against_different_field_sets() -> None:
    """``camera_key`` is legal in ``video_path`` and not in ``data_path``.

    ``_convert_single_episode`` formats ``data_path`` with ``chunk_index`` and
    ``file_index`` only, and ``video_path`` with ``video_key`` and
    ``camera_key`` as well. Checking both against the union would accept a
    ``data_path`` naming ``camera_key``, which is one of the failures this
    boundary is meant to catch.
    """
    prep._parse_dataset_information(_info(video_path="videos/{camera_key}/f.mp4"))

    with pytest.raises(ValueError, match="unknown field 'camera_key'"):
        prep._parse_dataset_information(_info(data_path="data/{camera_key}/f.parquet"))


def test_import_refuses_info_json_without_features_before_listing_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal fires at the parse boundary, not after a shard download.

    #433 moved this ahead of episode discovery. Asserting only the message
    would pass either way, so the point of the test is the second assertion:
    a corpus with no ``features`` must not cost a listing of ``meta/episodes``.
    """
    stub_hub_repo_info(monkeypatch, resolved_sha=_SHA)
    monkeypatch.setattr(
        prep,
        "_fetch_info_json",
        lambda _repo, _revision, _cache: {
            key: value for key, value in _info().items() if key != "features"
        },
    )
    listed_repositories: list[str] = []

    def recording_metadata_files(repo: str, _revision: str) -> list[str]:
        listed_repositories.append(repo)
        return []

    monkeypatch.setattr(prep, "_hf_episode_metadata_files", recording_metadata_files)
    output_dir = tmp_path / "out"

    with pytest.raises(
        ValueError, match=exactly("LeRobot meta/info.json must define a features object")
    ):
        _import(output_dir)

    assert listed_repositories == []
    _assert_no_dataset_output(output_dir)


def test_import_refuses_repository_without_episode_parquets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_hub_repo_info(monkeypatch, resolved_sha=_SHA)
    monkeypatch.setattr(prep, "_fetch_info_json", lambda _repo, _revision, _cache: _info())

    def missing_episode_tree(*_args: object, **_kwargs: object) -> list[object]:
        raise RemoteEntryNotFoundError(
            "missing meta/episodes",
            response=Response(404, request=Request("GET", "https://huggingface.co/missing")),
        )

    monkeypatch.setattr(
        prep,
        "HfApi",
        lambda: SimpleNamespace(list_repo_tree=missing_episode_tree),
    )
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match=exactly("no meta/episodes parquet files found")):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_wraps_error_from_later_episode_tree_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_hub_repo_info(monkeypatch, resolved_sha=_SHA)
    monkeypatch.setattr(prep, "_fetch_info_json", lambda *_args: _info())

    def failing_episode_tree(*_args: object, **_kwargs: object) -> Iterator[object]:
        yield prep.RepoFile(
            path="meta/episodes/chunk-000/file-000.parquet",
            size=0,
            oid="deadbeef",
        )
        raise RemoteEntryNotFoundError(
            "missing later page",
            response=Response(
                404,
                request=Request("GET", "https://huggingface.co/missing-later-page"),
            ),
        )

    monkeypatch.setattr(
        prep,
        "HfApi",
        lambda: SimpleNamespace(list_repo_tree=failing_episode_tree),
    )
    output_dir = tmp_path / "out"

    with pytest.raises(
        ValueError,
        match=rf"^Hugging Face files for {re.escape(_REPO)}@{_SHA}:",
    ):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


@pytest.mark.parametrize("filename", [None, "", 123])
def test_import_refuses_inventory_with_invalid_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: object
) -> None:
    stub_hub_repo_info(monkeypatch, resolved_sha=_SHA)
    monkeypatch.setattr(prep, "_fetch_info_json", lambda *_args: _info())
    monkeypatch.setattr(
        prep,
        "HfApi",
        lambda: SimpleNamespace(
            list_repo_tree=lambda *_args, **_kwargs: [
                prep.RepoFile(path=filename, size=0, oid="deadbeef")
            ]
        ),
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="Hugging Face listed an invalid filename"):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)
