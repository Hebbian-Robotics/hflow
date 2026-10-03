"""Combine pinned Build AI teacher-labelled frames, deduplicate, then split.

This is the data preparation stage of the example. It does not train or run a
model. See README.md for label and independence limitations.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from pathlib import Path

import duckdb
import PIL
import pyarrow
import pyarrow.parquet as parquet
import typer
from hflow import (
    ManifestDeduplicationSettings,
    ManifestSplitSettings,
    __version__,
    deduplicate_manifest,
    split_manifest,
)
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps


class Corpus(StrEnum):
    BUILD = "build"
    EGO4D = "ego4d"
    EPIC_KITCHENS = "epic-kitchens"


class Release(StrEnum):
    TEN_K = "10k"
    HUNDRED_K = "100k"


class SourceOrigin(StrEnum):
    PINNED_DOWNLOAD = "pinned-hub-download"
    LOCAL_CACHE = "local-file"


class SourceVerification(StrEnum):
    PUBLISHED_SHA256 = "matches-pinned-revision-sha256"
    UNVERIFIED = "local-file-revision-unverified"


class HandCount(IntEnum):
    NONE_VISIBLE = 0
    ONE_VISIBLE = 1
    TWO_VISIBLE = 2


@dataclass(frozen=True)
class PublishedSource:
    release: Release
    repository: str
    revision: str
    corpus: Corpus
    filename: str
    expected_sha256: str | None = None

    @property
    def cache_key(self) -> Path:
        return Path(self.release) / self.filename


PUBLISHED_FILE_DIGESTS = {
    (
        Release.TEN_K,
        "egocentric_10k.parquet",
    ): "db9016153d5991f84a1d66b65b2d89938c41dd9a3f33c6597310e9cc7fbf8f02",
    (
        Release.TEN_K,
        "ego4d.parquet",
    ): "9ea179782be618d887cb10cccd3cff3dc10f1f431b5e4aa871b1cc4c6cbc225e",
    (
        Release.TEN_K,
        "epic_kitchens.parquet",
    ): "8786900485e1327d787fd59b46bdf3767d4ebe70a2e2570448e0015d4a9b395b",
    (
        Release.HUNDRED_K,
        "egocentric_100k.parquet",
    ): "9a7a03a057447bfa077d0ef12a744b7d9aee5327bb8ae88535ab065048627b75",
    (
        Release.HUNDRED_K,
        "ego4d.parquet",
    ): "c5352cdf17df5a591e7f041930a6c169fbed9d45fe36dd3889ed1b861ffbe781",
    (
        Release.HUNDRED_K,
        "epic_kitchens.parquet",
    ): "c533c2f585f90a64701f60c52ac1db05a5effd995b9ea5ffa6083371f8a406f0",
}

PUBLISHED_SOURCES = tuple(
    PublishedSource(
        release, repository, revision, corpus, filename, PUBLISHED_FILE_DIGESTS[(release, filename)]
    )
    for release, repository, revision, build_filename in (
        (
            Release.TEN_K,
            "builddotai/Egocentric-10K-Evaluation",
            "d74b7883c998dd360e3f051830fcc792a83985e6",
            "egocentric_10k.parquet",
        ),
        (
            Release.HUNDRED_K,
            "builddotai/Egocentric-100K-Evaluation",
            "d0f69a56b0525c1bead80d918dc57ef83dcac899",
            "egocentric_100k.parquet",
        ),
    )
    for corpus, filename in (
        (Corpus.BUILD, build_filename),
        (Corpus.EGO4D, "ego4d.parquet"),
        (Corpus.EPIC_KITCHENS, "epic_kitchens.parquet"),
    )
)


@dataclass(frozen=True)
class SourceFile:
    specification: PublishedSource
    path: Path
    origin: SourceOrigin = SourceOrigin.LOCAL_CACHE


@dataclass(frozen=True)
class TeacherFrame:
    image_bytes: bytes
    hand_count: HandCount
    source_row_index: int
    upstream_frame_id: str | None


@dataclass(frozen=True)
class SourceReference:
    release: Release
    corpus: Corpus
    filename: str
    row_index: int
    frame_id: str | None
    encoded_sha256: str
    hand_count: HandCount


@dataclass(frozen=True)
class FrameOccurrence:
    sample_id: str
    pixel_sha256: str
    image_path: str
    image_sha256: str
    width: int
    height: int
    hand_count: HandCount
    source_reference_json: str


@dataclass(frozen=True)
class PreparedDataset:
    input_rows: int
    unique_images: int
    retained_images: int
    conflicting_images: int
    output_directory: Path


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write_new_json(path: Path, value: object) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def _parse_teacher_frame(row: Mapping[str, object], row_index: int) -> TeacherFrame:
    image_value = row.get("image")
    if isinstance(image_value, Mapping):
        image_value = image_value.get("bytes")
    if not isinstance(image_value, bytes) or not image_value:
        raise ValueError(f"row {row_index} has no embedded image bytes")
    hand_count = row.get("hand_count")
    if (
        isinstance(hand_count, bool)
        or not isinstance(hand_count, int)
        or hand_count not in (0, 1, 2)
    ):
        raise ValueError(f"row {row_index} hand_count must be an integer 0, 1, or 2")
    frame_id = row.get("frame_id")
    if frame_id is not None and (not isinstance(frame_id, str) or not frame_id):
        raise ValueError(f"row {row_index} frame_id must be a nonempty string when supplied")
    return TeacherFrame(image_value, HandCount(hand_count), row_index, frame_id)


def _iter_teacher_frames(path: Path, limit: int | None) -> Iterator[TeacherFrame]:
    with parquet.ParquetFile(path) as source:
        available_columns = set(source.schema_arrow.names)
        if not {"image", "hand_count"} <= available_columns:
            raise ValueError(f"{path.name} must contain image and hand_count columns")
        selected_columns = ["image", "hand_count"]
        if "frame_id" in available_columns:
            selected_columns.append("frame_id")
        row_index = 0
        # Embedded image row groups can be gigabytes; retain only small batches.
        for batch in source.iter_batches(batch_size=8, columns=selected_columns):
            for row in batch.to_pylist():
                if limit is not None and row_index >= limit:
                    return
                yield _parse_teacher_frame(row, row_index)
                row_index += 1


def _pixel_identity(image_bytes: bytes) -> tuple[str, int, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.width * image.height > 32_000_000:
            raise ValueError("image exceeds the 32-million-pixel preparation budget")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("animated or multi-frame inputs are unsupported")
        canonical_image = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(f"RGB:{canonical_image.width}:{canonical_image.height}:".encode())
        digest.update(canonical_image.tobytes())
        return digest.hexdigest(), canonical_image.width, canonical_image.height


def prepare_dataset(
    sources: Sequence[SourceFile],
    output_directory: Path,
    *,
    row_limit_per_source: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    """Freeze local sources, omit contradictory teacher labels, and split pixels.

    Re-encoded images with identical oriented RGB pixels collapse to one row.
    This is exact pixel deduplication, not near-duplicate or recording grouping.
    All references survive deduplication; conflicting hand labels are quarantined.
    """
    if not sources or len({source.specification.cache_key for source in sources}) != len(sources):
        raise ValueError("sources must be nonempty with distinct release/file identities")
    if row_limit_per_source is not None and (
        isinstance(row_limit_per_source, bool)
        or not isinstance(row_limit_per_source, int)
        or row_limit_per_source <= 0
    ):
        raise ValueError("row_limit_per_source must be positive")
    settings = ManifestSplitSettings("sample_id", ("pixel_sha256",), seed=seed)
    preparation_code_sha256 = _file_sha256(Path(__file__))
    output_directory.mkdir(parents=True, exist_ok=False)
    try:
        images_directory = output_directory / "images"
        images_directory.mkdir()
        image_metadata: dict[str, tuple[str, str, int, int]] = {}
        frame_occurrences: list[FrameOccurrence] = []
        encoded_identities: dict[str, str] = {}
        source_receipts: list[dict[str, object]] = []
        for source_file in sorted(sources, key=lambda item: str(item.specification.cache_key)):
            source, path = source_file.specification, source_file.path
            original_digest = _file_sha256(path)
            if source.expected_sha256 is not None and original_digest != source.expected_sha256:
                raise ValueError(f"source does not match the pinned revision: {source.cache_key}")
            selected_rows = 0
            for frame in _iter_teacher_frames(path, row_limit_per_source):
                encoded_digest = hashlib.sha256(frame.image_bytes).hexdigest()
                pixel_digest = encoded_identities.get(encoded_digest)
                if pixel_digest is None:
                    pixel_digest, width, height = _pixel_identity(frame.image_bytes)
                    encoded_identities[encoded_digest] = pixel_digest
                    if pixel_digest not in image_metadata:
                        image_path = f"images/{pixel_digest}.image"
                        (output_directory / image_path).write_bytes(frame.image_bytes)
                        image_metadata[pixel_digest] = (image_path, encoded_digest, width, height)
                reference = SourceReference(
                    source.release,
                    source.corpus,
                    source.filename,
                    frame.source_row_index,
                    frame.upstream_frame_id,
                    encoded_digest,
                    frame.hand_count,
                )
                occurrence_identity = json.dumps(
                    [source.release, source.filename, frame.source_row_index],
                    separators=(",", ":"),
                )
                image_path, image_digest, width, height = image_metadata[pixel_digest]
                frame_occurrences.append(
                    FrameOccurrence(
                        occurrence_identity,
                        pixel_digest,
                        image_path,
                        image_digest,
                        width,
                        height,
                        frame.hand_count,
                        json.dumps(asdict(reference), sort_keys=True),
                    )
                )
                selected_rows += 1
            if _file_sha256(path) != original_digest:
                raise ValueError(f"source changed while preparing: {source.cache_key}")
            source_receipts.append(
                {
                    **asdict(source),
                    "sha256": original_digest,
                    "selected_rows": selected_rows,
                    "origin": source_file.origin,
                    "verification": SourceVerification.PUBLISHED_SHA256
                    if source.expected_sha256 is not None
                    else SourceVerification.UNVERIFIED,
                }
            )
        with duckdb.connect() as connection:
            connection.execute(
                "CREATE TABLE occurrences (sample_id VARCHAR, pixel_sha256 VARCHAR, image_path VARCHAR, image_sha256 VARCHAR, width INTEGER, height INTEGER, hand_count INTEGER, source_reference_json VARCHAR)"
            )
            if frame_occurrences:
                connection.executemany(
                    "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [tuple(asdict(occurrence).values()) for occurrence in frame_occurrences],
                )
            source_manifest = output_directory / "source-samples.parquet"
            connection.sql("SELECT * FROM occurrences ORDER BY sample_id").write_parquet(
                str(source_manifest)
            )
        deduplication_report = deduplicate_manifest(
            source_manifest,
            output_directory / "deduplication",
            settings=ManifestDeduplicationSettings(
                sample_id_column="sample_id",
                identity_columns=("pixel_sha256",),
                conflict_columns=("hand_count",),
            ),
        )
        with duckdb.connect() as connection:
            connection.read_parquet(
                str(output_directory / "deduplication/samples.parquet")
            ).create_view("representatives")
            connection.read_parquet(
                str(output_directory / "deduplication/members.parquet")
            ).create_view("members")
            connection.execute(
                "CREATE TEMP TABLE provenance AS SELECT retained_sample_id, to_json(list(source_reference_json::JSON ORDER BY sample_id)) AS source_references_json FROM members GROUP BY retained_sample_id"
            )
            manifest_path = output_directory / "samples.parquet"
            # Excluding contradictory teacher labels is this example's policy.
            connection.sql(
                "SELECT representatives.pixel_sha256 AS sample_id, representatives.pixel_sha256, image_path, image_sha256, width, height, hand_count, source_references_json FROM representatives JOIN provenance ON representatives.sample_id = provenance.retained_sample_id WHERE len(deduplication_conflicts) = 0 ORDER BY sample_id"
            ).write_parquet(str(manifest_path))
            conflicts = [
                {"pixel_sha256": pixel_digest, "references": json.loads(references_json)}
                for pixel_digest, references_json in connection.execute(
                    "SELECT pixel_sha256, source_references_json FROM representatives JOIN provenance ON representatives.sample_id = provenance.retained_sample_id WHERE len(deduplication_conflicts) > 0 ORDER BY pixel_sha256"
                ).fetchall()
            ]
        for conflict in conflicts:
            (output_directory / f"images/{conflict['pixel_sha256']}.image").unlink()
        _write_new_json(output_directory / "conflicts.json", conflicts)
        split_report = split_manifest(manifest_path, output_directory / "splits", settings=settings)
        report = PreparedDataset(
            deduplication_report.input_rows,
            deduplication_report.unique_samples,
            deduplication_report.unique_samples - deduplication_report.conflicting_samples,
            deduplication_report.conflicting_samples,
            output_directory,
        )
        if _file_sha256(Path(__file__)) != preparation_code_sha256:
            raise ValueError("preparation code changed during the run")
        _write_new_json(
            output_directory / "preparation.json",
            {
                "schema_version": 2,
                "preparation_code_sha256": preparation_code_sha256,
                "runtime": {
                    "hflow": __version__,
                    "pillow": PIL.__version__,
                    "pyarrow": pyarrow.__version__,
                    "duckdb": duckdb.__version__,
                },
                "task": "wearer-hand-count",
                "label_source": "published-gemini-teacher-labels",
                "independence_scope": "exact-pixel-frame-only",
                "deduplication": "sha256-exif-oriented-rgb-v1",
                "row_limit_per_source": row_limit_per_source,
                "sources": source_receipts,
                "source_manifest_sha256": deduplication_report.input_sha256,
                "deduplication_receipt_sha256": _file_sha256(
                    output_directory / "deduplication/receipt.json"
                ),
                "input_rows": report.input_rows,
                "unique_images": report.unique_images,
                "retained_images": report.retained_images,
                "conflicting_images": report.conflicting_images,
                "manifest_sha256": _file_sha256(manifest_path),
                "conflicts_sha256": _file_sha256(output_directory / "conflicts.json"),
                "split_receipt_sha256": _file_sha256(output_directory / "splits/receipt.json"),
                "partitions": [asdict(partition) for partition in split_report.partitions],
            },
        )
        return report
    except Exception:
        shutil.rmtree(output_directory)
        raise


def main(
    output: Path,
    cache: Path = Path("data/build-ai-autoresearch/downloads"),
    download: bool = False,
    limit: int | None = None,
    seed: int = 42,
) -> None:
    """Prepare both releases. --download explicitly permits public downloads."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    resolved_sources: list[SourceFile] = []
    for source in PUBLISHED_SOURCES:
        if download:
            source_path = Path(
                hf_hub_download(
                    repo_id=source.repository,
                    repo_type="dataset",
                    revision=source.revision,
                    filename=source.filename,
                    local_dir=cache / source.release,
                )
            )
        else:
            source_path = cache / source.cache_key
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"missing {source.cache_key}; use --download or provide --cache"
                )
        resolved_sources.append(
            SourceFile(
                source,
                source_path,
                SourceOrigin.PINNED_DOWNLOAD if download else SourceOrigin.LOCAL_CACHE,
            )
        )
    report = prepare_dataset(resolved_sources, output, row_limit_per_source=limit, seed=seed)
    print(
        f"{report.input_rows} rows → {report.unique_images} unique images; {report.conflicting_images} conflicts omitted; {report.retained_images} retained"
    )
    print(f"Receipt: {output / 'preparation.json'}")


if __name__ == "__main__":
    typer.run(main)
