"""Regression coverage for LeRobot metadata refusal boundaries (#405)."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request

import hflow.importers.lerobot as prep
import pytest

_REPO = "fake/repo"
_SHA = "abcdef1234567890"


class _Response:
    def __init__(self, body: bytes, *, link: str | None = None) -> None:
        self._body = body
        self.headers = {} if link is None else {"Link": link}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


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
    monkeypatch.setattr(prep, "_hf_tree", lambda _repo, _revision, _path: [])
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match=r"^meta/info.json not found; not a LeRobot v3 repository$"):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_info_json_that_is_not_an_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda _repo, _revision, _path: [{"path": "meta/info.json", "type": "file"}],
    )
    monkeypatch.setattr(prep.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(b"[]"))
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=r"^LeRobot meta/info.json is not a JSON object$"):
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

    with pytest.raises(ValueError, match=f"^{message}$"):
        _import(output_dir)

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
    monkeypatch.setattr(prep, "_hf_tree", lambda _repo, _revision, _path: [])
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match=r"^no meta/episodes parquet files found$"):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_tree_response_that_is_not_a_list_of_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(prep.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(b"{}"))
    output_dir = tmp_path / "out"

    with pytest.raises(
        ValueError,
        match=r"^Hugging Face tree response for 'meta' is not a list of objects$",
    ):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_repeated_pagination_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    initial_url = f"https://huggingface.co/api/datasets/{_REPO}/tree/{_SHA}/meta?recursive=true"

    def fake_urlopen(request: Request, **_kwargs: object) -> _Response:
        assert request.full_url == initial_url
        return _Response(b"[]", link=f'<{initial_url}>; rel="next"')

    monkeypatch.setattr(prep.urllib.request, "urlopen", fake_urlopen)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="repeated an already fetched pagination URL"):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_invalid_pagination_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    invalid_url = "https://[invalid"

    def fake_urlopen(_request: Request, **_kwargs: object) -> _Response:
        return _Response(b"[]", link=f'<{invalid_url}>; rel="next"')

    monkeypatch.setattr(prep.urllib.request, "urlopen", fake_urlopen)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="contains an invalid pagination URL"):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)