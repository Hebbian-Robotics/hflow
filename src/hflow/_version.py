"""Installed HFlow package version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hflow")
except PackageNotFoundError:  # Running directly from an unpackaged source tree.
    __version__ = "0.0.0"
