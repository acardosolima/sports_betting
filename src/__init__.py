# This imports the 'version' variable from the _version.py file that setuptools_scm generates.
# The `_version.py` file will be created by setuptools_scm when package is built.
# It ensures package's runtime version matches Git tag.
try:
    from ._version import version as __version__
except ImportError:
    # Fallback for local development if _version.py isn't generated yet (e.g., in IDE)
    # or if the package is not installed editable.
    __version__ = "unknown"
