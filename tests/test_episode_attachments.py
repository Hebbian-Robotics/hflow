from pathlib import Path

from hflow.episode import Episode
from hflow.mcap_writer import CanonicalMcapWriter


def test_episode_attachments_round_trip(tmp_path: Path) -> None:
    mcap_path = tmp_path / "test_attachments.mcap"
    attachment_name = "calibration.txt"
    attachment_data = b"calibration data"
    media_type = "text/plain"

    # 1. Write it to a temporary MCAP file using CanonicalMcapWriter
    with CanonicalMcapWriter(mcap_path) as writer:
        writer.add_attachment(
            attachment_name,
            media_type,
            attachment_data,
        )

    # 2. Read the MCAP file back into an Episode object
    ep = Episode(mcap_path)

    # 3. Assert that the attachment's name and content survived the round-trip
    assert len(ep.attachments) == 1
    attachment = ep.attachments[0]

    assert attachment.name == attachment_name
    assert attachment.data == attachment_data
    assert attachment.media_type == media_type
