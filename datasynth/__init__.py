"""datasynth: agentic AI CLI that builds models from a natural-language ask over enterprise data."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _resolve_version() -> str:
    try:
        return version("datasynth")
    except PackageNotFoundError:
        pass
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


__version__ = _resolve_version()
