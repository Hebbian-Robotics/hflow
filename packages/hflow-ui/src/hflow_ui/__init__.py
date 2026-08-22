"""HFlow workspace UI: a read-only local web app over one data root.

The public surface is deliberately tiny: :class:`UiSettings` (parsed launch
configuration) and :func:`serve` (runs the server). The CLI's ``hflow ui``
subcommand is a thin caller of exactly these two names.
"""

from hflow_ui._settings import UiSettings
from hflow_ui.server import create_app, serve

__version__ = "0.1.0"

__all__ = ["UiSettings", "__version__", "create_app", "serve"]
