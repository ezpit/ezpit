from ._version import __version__ as __version__  # Import version from _version.py

try:
    from ._build_date import __build_date__  # Generated at build time (ISO 8601)
except ImportError:  # Running from a source checkout without a build
    __build_date__ = None
