"""Package description."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal_cellpose_2_segmentation_task")
except PackageNotFoundError:
    __version__ = "uninstalled"
