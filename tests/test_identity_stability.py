"""A release must not re-version a corpus that processed nothing differently.

Until the identity epoch in :mod:`hflow.behavior`, ``hflow.__version__`` was
folded into ``pipeline_version`` (and thence, via ``provenance/v1``, into the
canonical bytes and so into ``episode_id``), into the MCAP header's library
string (also inside those bytes), and into any step version whose function
referenced a module. A CLI-only patch release therefore invalidated an entire
corpus: ``hflow stale`` listed everything, and re-ingesting byte-identical
sources minted new episode identities instead of deduping.

These tests fail if any of those couplings comes back.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy
from mcap.reader import make_reader

import hflow
from hflow.behavior import TRANSFORM_BEHAVIOR_VERSION
from hflow.mcap_writer import _default_library_identifier
from hflow.steps import CheckResult, compute_check_version
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import (
    TransformConfig,
    compute_pipeline_version,
    write_canonical_episode,
)

FAKE_RELEASE = "9.9.9"


def _step_version(function: Callable[..., object]) -> str:
    return compute_check_version(
        name="probe",
        function=function,
        critical=False,
        requires=frozenset(),
        uses=None,
    )


def _with_faked_release(compute: Callable[[], str]) -> tuple[str, str]:
    """Return (value now, value under a faked hflow release)."""
    before = compute()
    original = hflow.__version__
    hflow.__version__ = FAKE_RELEASE
    try:
        after = compute()
    finally:
        hflow.__version__ = original
    return before, after


def test_pipeline_version_survives_a_release() -> None:
    before, after = _with_faked_release(lambda: compute_pipeline_version(TransformConfig()))
    assert before == after, "a release must not mark every episode stale"


def test_pipeline_version_still_tracks_transform_configuration() -> None:
    baseline = compute_pipeline_version(TransformConfig())
    assert compute_pipeline_version(TransformConfig(crf=30)) != baseline
    assert compute_pipeline_version(TransformConfig(), {"/derived/speed": "abc"}) != baseline


def test_pipeline_version_tracks_the_transform_behavior_version() -> None:
    """The deliberate lever still works: bumping behavior re-versions."""
    baseline = compute_pipeline_version(TransformConfig())
    import hflow.transform as transform_module

    original = transform_module.TRANSFORM_BEHAVIOR_VERSION
    # Any value but the current one; a literal here would silently stop
    # testing anything the moment the constant caught up with it.
    transform_module.TRANSFORM_BEHAVIOR_VERSION = f"{original}-probe"
    try:
        assert compute_pipeline_version(TransformConfig()) != baseline
    finally:
        transform_module.TRANSFORM_BEHAVIOR_VERSION = original


def test_pipeline_version_tracks_the_resample_policy_only_when_derived_channels_exist() -> None:
    """The resample policy decides derived samples and no step hash sees it.

    A step version folds in a referenced MODULE's name only, so a pipeline
    whose derive function calls ``hflow.resample.to_grid`` cannot notice the
    policy changing. ``compute_pipeline_version`` folds it in instead -- but
    only for episodes that actually have derived channels, so bumping the
    policy never churns a corpus that resampled nothing.
    """
    import hflow.transform as transform_module

    derived = {"/derived/joint_grid": "abc123"}
    original = transform_module.RESAMPLE_POLICY_VERSION
    without_derived = compute_pipeline_version(TransformConfig())
    with_derived = compute_pipeline_version(TransformConfig(), derived)

    transform_module.RESAMPLE_POLICY_VERSION = f"{original}-probe"
    try:
        assert compute_pipeline_version(TransformConfig(), derived) != with_derived, (
            "a resample policy bump must make derived-channel episodes stale"
        )
        assert compute_pipeline_version(TransformConfig()) == without_derived, (
            "a corpus with no derived channels must not move"
        )
    finally:
        transform_module.RESAMPLE_POLICY_VERSION = original


def test_canonical_bytes_carry_no_release_number() -> None:
    """The MCAP header's library string is inside the hashed bytes."""
    identifier = _default_library_identifier()
    assert hflow.__version__ not in identifier
    assert TRANSFORM_BEHAVIOR_VERSION in identifier

    before, after = _with_faked_release(_default_library_identifier)
    assert before == after, "a release must not change episode_id"


def test_step_referencing_the_hflow_module_survives_a_release() -> None:
    def check_via_module(episode: object) -> object:
        return hflow.CheckResult(measurements={"ok": 1.0})

    before, after = _with_faked_release(lambda: _step_version(check_via_module))
    assert before == after


def test_step_referencing_a_third_party_module_survives_its_upgrade() -> None:
    """The defect was never hflow-specific: numpy upgrades churned too."""

    def check_via_numpy(values: list[float]) -> float:
        return float(numpy.mean(values))

    before = _step_version(check_via_numpy)
    original = numpy.__version__
    numpy.__version__ = "99.0.0"
    try:
        after = _step_version(check_via_numpy)
    finally:
        numpy.__version__ = original
    assert before == after


def test_step_version_still_tracks_what_the_author_wrote() -> None:
    """Author-owned facts must still re-version -- that is the point."""

    def original_threshold(episode: object) -> CheckResult:
        return CheckResult(measurements={"limit": 1.0})

    def changed_threshold(episode: object) -> CheckResult:
        return CheckResult(measurements={"limit": 2.0})

    assert _step_version(original_threshold) != _step_version(changed_threshold)

    captured_limit = 1.0

    def uses_closure(episode: object) -> CheckResult:
        return CheckResult(measurements={"limit": captured_limit})

    with_first_capture = _step_version(uses_closure)
    captured_limit = 2.0
    assert _step_version(uses_closure) != with_first_capture


def test_canonical_write_order_is_total_so_identity_cannot_ride_on_append_order(
    tmp_path: Path,
) -> None:
    """Messages sharing a timestamp must be ordered by something in the DATA.

    ``episode_id`` is a hash of the canonical bytes, and those bytes are
    written in the order the transform sorted its messages into. Sorting on
    ``log_time`` alone leaves ties -- the fixture below has hundreds -- which
    a stable sort then settles by the order the transform happened to append
    them: passthrough during the read, then transcoded video, then derived
    channels. That is a property of the code, so any reordering of those
    appends would silently mint new identities for an unchanged corpus.

    Asserting the ORDER rather than a golden digest is deliberate: the
    canonical embeds transcoded video, so a pinned hash would fail on a
    different ffmpeg build for reasons that have nothing to do with ordering.
    """
    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(
            duration_s=6.0,
            cameras=("wrist_cam", "overhead_cam"),
            image_hz=30.0,
            joint_hz=100.0,
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical)

    written: list[tuple[int, str]] = []
    with canonical.open("rb") as stream:
        for _schema, channel, message in make_reader(stream).iter_messages():
            written.append((message.log_time, channel.topic))

    timestamp_counts = Counter(log_time for log_time, _topic in written)
    tied_messages = sum(count for count in timestamp_counts.values() if count > 1)
    assert tied_messages > 0, "fixture has no simultaneous messages; it proves nothing"

    assert written == sorted(written), (
        "canonical messages must be written in (log_time, topic) order -- "
        f"{tied_messages} of {len(written)} share a timestamp, and without a "
        "total order their arrangement is decided by append order"
    )
