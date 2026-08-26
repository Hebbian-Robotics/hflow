"""Processing-behavior versions: what actually changes a corpus's identity.

HFlow keeps release versions, transform behavior, and pipeline-authored step
behavior separate. A package release is provenance, never evidence that an
episode was processed differently.

**Bump ``TRANSFORM_BEHAVIOR_VERSION`` when, and only when, a change makes the
default transform write different bytes for the same input**: encoder settings
or defaults, chunking and grouping, timestamp handling, the provenance shape,
or a fix to any of them. Bumping re-versions every corpus once, which is the
cost of telling the truth; not bumping silently mixes two transform behaviors.

Checks, enrichments, and derived channels use a different rule. Every
registration declares an explicit version owned by its pipeline author. HFlow
does not inspect Python functions or derive a version from source code. Bump a
step's version when its implementation, configuration, gate, or dependency
changes in a way that makes its old and new results no longer comparable. Keep
the version when a refactor preserves that contract.

Explicit versions make the identity portable across SDK languages and keep the
compatibility decision with the person who understands the pipeline. They also
make forgetting a bump possible; that is a deliberate, visible author promise,
not something HFlow pretends it can infer reliably from a dynamic program.

Derived-channel versions flow into
:func:`hflow.transform.compute_pipeline_version`, because those samples are
written into canonical episode bytes. Changing a derived-channel version
therefore re-mints affected episode identities. Check and enrichment versions
only version their catalog rows.

The HFlow release remains available as provenance through
:attr:`hflow.PipelineManifest.hflow_version` and the rendered bundle's
``hflow-bundle.json``.

Identity epochs live here as prose so release notes can explain intentional
one-time changes. Epoch 1 coupled identities to ``hflow.__version__``. Epoch 2
removed that release coupling. Earlier pre-1.0 builds derived step identities
from Python implementations. This build uses only explicit, author-owned step
versions. Pre-1.0 derived outputs may be regenerated rather than adapted
across these changes.
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
#
# "7": a bare data-root-relative source key now records the same root-relative
# source_uri as the data-root-prefixed and absolute spellings. The corrected
# provenance changes canonical bytes for recordings previously processed under
# the cwd-relative absolute identity, so they must not share a pipeline identity.
TRANSFORM_BEHAVIOR_VERSION: str = "7"
