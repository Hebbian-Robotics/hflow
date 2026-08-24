"""Print the workspace server's OpenAPI schema to stdout.

The generated TypeScript in ``src/apiSchema.ts`` is derived from this, so the
browser's idea of every payload comes from the server's own declaration
rather than from a hand-copied interface. Run it through ``pnpm gen:api``.

No server is started and no workspace is read: ``create_app`` builds the
routes from ``hflow_server._contract``, and the schema is a property of those
routes. The data root only has to exist, so a temporary directory does.
"""

import json
import sys
import tempfile
from pathlib import Path

from hflow_server import ServerSettings, create_app

with tempfile.TemporaryDirectory() as temporary_root:
    # assets_dir is pinned empty so the schema never depends on whether
    # someone happens to have a frontend built locally.
    settings = ServerSettings(
        data_root=temporary_root,
        assets_dir=Path(temporary_root) / "no-assets",
    )
    json.dump(create_app(settings).openapi(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
