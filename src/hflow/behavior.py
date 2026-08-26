"""Processing-behavior versions: what actually changes a corpus's identity.

HFlow's identities are content hashes, and until identity epoch 2 they
folded in ``hflow.__version__`` -- the RELEASE number. That made every
release, however unrelated to processing, change three things at once:
``pipeline_version`` (so ``hflow stale`` listed the whole corpus),
``episode_id`` (because the pipeline version is stamped inside the canonical
bytes the id hashes, breaking content-addressed dedupe), and every step
version whose function happened to reference the ``hflow`` module. A CLI fix
or a docs-driven version bump invalidated a petabyte.

The release number is a bad proxy for "does this build process data
differently". This module holds the honest answer instead: a version that a
maintainer bumps DELIBERATELY when canonicalization semantics change -- when
the same input would now produce different canonical bytes.

**Bump ``TRANSFORM_BEHAVIOR_VERSION`` when, and only when, a change makes the
transform write different bytes for the same input**: encoder settings or
defaults, chunking and grouping, timestamp handling, the provenance record's
shape, or a fix to any of them. Bumping it re-versions every corpus exactly
once, which is the cost of telling the truth; NOT bumping it when behavior
changed silently mixes two behaviors under one version, which is worse. When
in doubt, bump.

Deliberately NOT here: an "analysis" behavior version covering what a check
observes. Step versions must not fold in an engine-wide constant, because
``@app.derive`` channel versions flow into ``compute_pipeline_version`` and
therefore into ``episode_id`` -- one engine-wide analysis bump would churn
every derived-channel user's episode identities, reintroducing the defect
this module exists to remove.

A change to a built-in check's algorithm is caught in the step's own content
hash, because a step hash folds in the source of every function the step
NAMES and then keeps descending through first-party code: the helpers those
functions call, and the constants they read, down to the leaves (see
``_IdentityScope`` in :mod:`hflow.steps`). That is what keeps composition
honest here. Checks compose by sharing library code (two built-ins reading one
video-measurement definition, including camera motion) rather than by
depending on each other's results. Without the descent, tuning a threshold
inside the shared instrument changed what every one of them measured while
their versions stood still.

What the descent cannot see is a module reached by ATTRIBUTE: a step whose
body calls ``hflow.checks.timestamp_regularity`` captures the module, and a
module contributes only its NAME. Registering a built-in bare
(``app.check()(timestamp_regularity)``) or importing it by name avoids this
entirely, which is what the docs and examples teach; a wrapper that reaches
through the module does not, and is the one remaining spelling where a
built-in's change goes unversioned. The boundary is deliberate in the other
direction too: the descent stops at hflow's own code and the pipeline's, never
entering a dependency. Folding numpy's source into a step version would
re-version a corpus on a release that changed nothing a step observes, which
is this module's whole subject.

Two further exceptions live in
:func:`hflow.transform.compute_pipeline_version`, which folds in
``RESAMPLE_POLICY_VERSION`` for episodes that have derived channels: the
resample policy decides those samples and no step hash can see it.

The release number is not lost, only demoted from identity to provenance:
:attr:`hflow.PipelineManifest.hflow_version` and the rendered bundle's
``hflow-bundle.json`` both record which build produced a pipeline.

Identity epochs live here as prose, not as a constant: no build stamps an
epoch into a corpus, so a module attribute nothing reads would be a comment
wearing a type annotation. Epoch 1 folded ``hflow.__version__`` into
``pipeline_version``, ``episode_id``, and step versions; epoch 2 folded in
only author-owned facts plus ``TRANSFORM_BEHAVIOR_VERSION``; epoch 3, what
this build produces, widened a step hash from the functions a step names to
the first-party code those transitively call. Written down so a reader can
explain the one-time re-version between them rather than guess at it.

Epoch 3 moves the version of every step that reaches shared code, which is
most of them. Where such a step is a ``@app.derive`` channel the move carries
further: derived-channel versions reach ``compute_pipeline_version``, which is
stamped inside the bytes ``episode_id`` hashes, so corpora WITH derived
channels re-mint episode identities once. That is the honest outcome rather
than an accident, because those steps really were versioned as though the code
they call could not change. It is still a re-ingest, so it belongs in release
notes and not only here.
"""

# Canonicalization semantics: bump when the transform would write different
# bytes for the same input. See the module docstring for the rule. Annotated
# as ``str`` rather than inferred as a literal: the whole point is that it
# changes.
#
# "3": bulk non-camera channels (point clouds, occupancy grids -- anything
# averaging over BULK_MESSAGE_BYTES per message) now get their own chunk
# group instead of sharing chunks with proprio-sized telemetry, and the
# resolved topic-to-group map is written into provenance/v1. Both change the
# bytes. docs/BENCHMARKS.md measured the old behavior dragging 230 MB through
# an /imu scan because /imu shared chunks with lidar, against 4 MB grouped by
# read pattern; a recording with no bulk channels is laid out exactly as
# before and only re-versions because this constant moved.
#
# "4": chunk targets are now derived per group from that group's own byte rate
# and the preset's read window, instead of one flat 800 KB for every group.
# Measured on the reference recording (docs/BENCHMARKS.md): 3.79 chunk fetches
# per training sample fetching 11.21 MB, against 9.18 fetching 6.65 MB at the
# flat 800 KB this replaced -- half the round trips for 1.7x the bytes, the
# trade worth making when round trips dominate. It is NOT the measured minimum
# of fetches x bytes: a flat 5.41 MB beat it there on that recording, and the
# default stands anyway because 5.41 MB is that recording's camera rate rather
# than a principle. Groups whose rate falls below the floor -- every state
# group there, every synthetic fixture -- keep their previous layout and
# re-version only because this constant moved.
#
# "5": the low-level grouped writer moved behind a private internal boundary
# and now stamps its own neutral library identifier. Canonical episodes are
# pre-1.0 derived artifacts, so the old bytes are intentionally not preserved;
# regenerate them from their source recordings when upgrading.
#
# "6": pass-through H.264 video messages that lack an access-unit delimiter
# now receive one losslessly. Existing slice data stays byte-identical, but the
# canonical MCAP bytes change, so old and new transforms must not share a
# pipeline identity.
TRANSFORM_BEHAVIOR_VERSION: str = "6"
