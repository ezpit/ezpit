"""
"""

from datetime import datetime

from ._version import __version__  # Import version from _version.py

try:
    from ._build_date import __build_date__  # Generated at build time (ISO 8601)

    # Reformat the ISO date into e.g. "August 07, 2026".
    __build_date__ = datetime.fromisoformat(__build_date__).strftime("%B %d, %Y")
except ImportError:  # Running from a source checkout without a build
    __build_date__ = None