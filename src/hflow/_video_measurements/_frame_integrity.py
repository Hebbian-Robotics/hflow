"""Frame-level integrity helpers for canonical no-B-frame video streams."""

from collections.abc import Collection


def cleanly_decoded_frame_indices(
    decoded_indices: Collection[int], *, keyframe_indices: Collection[int], frame_count: int
) -> set[int]:
    """Return decoded frames whose complete reference chain was decoded.

    Canonical video has no B-frames, so a picture depends on every picture since
    its preceding keyframe. After a lost keyframe the decoder may still output a
    dependent picture, concealed from an unrelated reference and carrying no
    error flag. That picture is trustworthy again only from the next keyframe.
    """
    clean_indices: set[int] = set()
    reference_chain_intact = False
    for frame_index in range(frame_count):
        if frame_index in keyframe_indices:
            reference_chain_intact = frame_index in decoded_indices
        else:
            reference_chain_intact = reference_chain_intact and frame_index in decoded_indices
        if reference_chain_intact:
            clean_indices.add(frame_index)
    return clean_indices
