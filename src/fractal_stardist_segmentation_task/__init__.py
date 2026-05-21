"""Package description."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal_stardist_segmentation_task")
except PackageNotFoundError:
    __version__ = "uninstalled"
