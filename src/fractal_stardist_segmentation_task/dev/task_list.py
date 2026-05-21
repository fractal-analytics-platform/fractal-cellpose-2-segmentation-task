"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Fabio Steffen & Joel Luethi"


DOCS_LINK = (
    "https://github.com/fractal-analytics-platform/fractal-stardist-segmentation-task"
)


TASK_LIST = [
    ParallelTask(
        name="StarDist Segmentation",
        executable="stardist_segmentation_task.py",
        # Modify the meta according to your task requirements
        # If the task requires a GPU, add "needs_gpu": True
        meta={"cpus_per_task": 1, "mem": 4000, "needs_gpu": True},
        category="Segmentation",
        tags=["Instance Segmentation", "StarDist", "Deep Learning"],
        docs_info="file:docs_info/stardist_segmentation_task.md",
    ),
]
