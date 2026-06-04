"""Research code for LLM-augmented LightGCN recommendation experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lightgcnrec")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
