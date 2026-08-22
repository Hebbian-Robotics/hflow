"""HFlow workspace server: a read-mostly HTTP API over one data root.

The public surface is deliberately tiny: :class:`ServerSettings` (parsed launch
configuration) and :func:`serve` (runs the server). The CLI's ``hflow serve``
subcommand is a thin caller of exactly these two names.
"""

from hflow_server._settings import ServerSettings
from hflow_server.server import create_app, serve

__version__ = "0.1.0"

__all__ = ["ServerSettings", "__version__", "create_app", "serve"]
