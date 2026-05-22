"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Fabio Steffen & Joel Luethi"


DOCS_LINK = (
    "https://github.com/fractal-analytics-platform/fractal-cellpose-2-segmentation-task"
)


TASK_LIST = [
    ParallelTask(
        name="Cellpose Segmentation",
        executable="cellpose_segmentation_task.py",
        meta={"cpus_per_task": 1, "mem": 8000, "needs_gpu": True},
        category="Segmentation",
        tags=["Instance Segmentation", "Cellpose", "Deep Learning"],
        docs_info="file:docs_info/cellpose_segmentation_task.md",
    ),
]
