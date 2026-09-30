"""The only file an autoresearch agent edits: the model-input preparation budget."""

from hflow.importers import VideoImportConfig


def preparation_configuration() -> VideoImportConfig:
    return VideoImportConfig(duration_s=1.0, image_hz=8.0, image_width=64, image_height=48)
