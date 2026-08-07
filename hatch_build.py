"""Custom Hatch build hook that embeds the build/release date.

At build time this captures the committer date of the current git ``HEAD``
(the commit the release is built from) and writes it to a generated
``src/ezpit/_build_date.py`` module so the date is available at runtime even
from an installed wheel.
"""

from __future__ import annotations

import datetime
import pathlib
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        try:
            # Committer date of HEAD in ISO 8601 (the commit the release is built from).
            iso = (
                subprocess.check_output(
                    ["git", "log", "-1", "--format=%cI"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            iso = datetime.datetime.now().astimezone().isoformat()

        out = pathlib.Path("src/ezpit/_build_date.py")
        out.write_text(f'# file generated at build time; don\'t track in version control\n__build_date__ = "{iso}"\n')
        build_data["artifacts"].append("src/ezpit/_build_date.py")
