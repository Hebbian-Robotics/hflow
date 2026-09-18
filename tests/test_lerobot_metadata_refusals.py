"""Regression coverage for LeRobot metadata refusal boundaries (#405)."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import Request, Response
from huggingface_hub.errors import RemoteEntryNotFoundError

import hflow.importers.lerobot as prep

_REPO = "fake/repo"
_SHA = "abcdef1234567890"


def _exactly(message: str) -> str:
    """A ``match=`` pattern pinning the whole message, metacharacters and all.

    #405 asks for these to be byte-identical, and several contain a ``.``
    (``meta/info.json``), which unescaped would also match ``meta/infoXjson``.
    """
    return rf"^{re.escape(message)}$"


def _stub_repo_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prep,
        "_hf_repo_info",
        lambda _repo, _revision: {"sha": _SHA, "license": "apache-2.0"},
    )


def _assert_no_dataset_output(output_dir: Path) -> None:
    assert not (output_dir / "landing").exists()
    assert not (output_dir / "prepared-manifest.json").exists()


def _import(output_dir: Path) -> None:
    prep.import_lerobot_dataset(dataset_repo=_REPO, output_dir=output_dir)


def test_import_refuses_repository_without_lerobot_v3_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)

    def missing_info(*_args: object, **_kwargs: object) -> str:
        raise RemoteEntryNotFoundError(
            "missing meta/info.json",
            response=Response(404, request=Request("GET", "https://huggingface.co/missing")),
        )

    monkeypatch.setattr(prep, "hf_hub_download", missing_info)
    output_dir = tmp_path / "out"

    with pytest.raises(
        RuntimeError, match=_exactly("meta/info.json not found; not a LeRobot v3 repository")
    ):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_info_json_that_is_not_an_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    metadata_file = tmp_path / "info.json"
    metadata_file.write_text("[]")
    monkeypatch.setattr(prep, "hf_hub_download", lambda *_args, **_kwargs: str(metadata_file))
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=_exactly("LeRobot meta/info.json is not a JSON object")):
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
    _stub_repo_info(monkeypatch)
    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {},
    }
    info[field] = value
    monkeypatch.setattr(prep, "_fetch_info_json", lambda _repo, _revision, _cache: info)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=_exactly(message)):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


_VALID_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VALID_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _info(**overrides: object) -> dict[str, object]:
    info: dict[str, object] = {
        "fps": 30,
        "data_path": _VALID_DATA_PATH,
        "video_path": _VALID_VIDEO_PATH,
        "features": {},
    }
    info.update(overrides)
    return info


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
    _stub_repo_info(monkeypatch)
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
    with pytest.raises(ValueError, match=_exactly(message)):
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
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(
        prep,
        "_fetch_info_json",
        lambda _repo, _revision, _cache: {
            "fps": 30,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        },
    )
    listed_repositories: list[str] = []

    def recording_metadata_files(repo: str, _revision: str) -> list[str]:
        listed_repositories.append(repo)
        return []

    monkeypatch.setattr(prep, "_hf_episode_metadata_files", recording_metadata_files)
    output_dir = tmp_path / "out"

    with pytest.raises(
        ValueError, match=_exactly("LeRobot meta/info.json must define a features object")
    ):
        _import(output_dir)

    assert listed_repositories == []
    _assert_no_dataset_output(output_dir)


def test_import_refuses_repository_without_episode_parquets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(
        prep,
        "_fetch_info_json",
        lambda _repo, _revision, _cache: {
            "fps": 30,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {},
        },
    )
    monkeypatch.setattr(prep, "_hf_episode_metadata_files", lambda _repo, _revision: [])
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match=_exactly("no meta/episodes parquet files found")):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


@pytest.mark.parametrize("filename", [None, "", 123])
def test_import_refuses_inventory_with_invalid_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: object
) -> None:
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(prep, "_fetch_info_json", lambda *_args: _info())
    monkeypatch.setattr(
        prep,
        "HfApi",
        lambda: SimpleNamespace(
            dataset_info=lambda *_args, **_kwargs: SimpleNamespace(
                siblings=[SimpleNamespace(rfilename=filename)]
            )
        ),
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="Hugging Face listed an invalid filename"):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)
