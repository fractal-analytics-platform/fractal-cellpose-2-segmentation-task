from pathlib import Path

import numpy as np
import pytest
from fractal_tasks_utils.segmentation import IteratorConfig
from fractal_tasks_utils.segmentation._models import MaskingConfig
from ngio import ChannelSelectionModel, create_synthetic_ome_zarr

from fractal_cellpose_2_segmentation_task.cellpose_segmentation_task import (
    cellpose_segmentation_task,
)

# 2D shapes with different axis configurations.
# Expected object counts are determined empirically by running Cellpose 2
# (cyto2, diameter=30) on the synthetic OME-Zarr data. 
_SHAPES = [
    ((64, 64), "yx", 4),
    ((1, 64, 64), "cyx", 4),
    ((3, 64, 64), "cyx", 4),
    ((4, 64, 64), "tyx", 16),
    ((1, 3, 64, 64), "tcyx", 4),
    ((1, 10, 128, 128), "czyx", 15)
]
_SHAPES_masked = [
    ((128, 128), "yx", 11),
]


@pytest.mark.parametrize("shape, axes, expected_objects", _SHAPES)
def test_cellpose_segmentation_task(
    tmp_path: Path, shape: tuple[int, ...], axes: str, expected_objects: int | None
):
    """Base test for the Cellpose segmentation task."""
    test_data_path = tmp_path / "data.zarr"


    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )
    channel_1 = ChannelSelectionModel(identifier="DAPI_0", mode="label")

    cellpose_segmentation_task(
        zarr_url=str(test_data_path),
        channel_1=channel_1,
        model_type="cyto2",
        overwrite=True,
    )

    expected_label = "DAPI_0_cellpose_segmented"
    assert expected_label in ome_zarr.list_labels()

    label = ome_zarr.get_label(expected_label)
    label_data = label.get_as_numpy()
    # Cellpose finds objects in the first channel, independent of axis setup.
    # For timeseries the count is multiplied by number of timepoints.
    assert np.max(label_data) > 0
    if expected_objects is not None:
        assert np.max(label_data) == expected_objects


@pytest.mark.parametrize("shape, axes, expected_objects", _SHAPES_masked)
def test_cellpose_segmentation_task_masked(
    tmp_path: Path, shape: tuple[int, ...], axes: str, expected_objects: int | None
):
    """Test the Cellpose segmentation task with a masking configuration."""
    test_data_path = tmp_path / "data.zarr"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )
    channel_1 = ChannelSelectionModel(identifier="DAPI_0", mode="label")

    iter_config = IteratorConfig(
        masking=MaskingConfig(masking_source="Label Name", identifier="nuclei_mask"),
    )
    cellpose_segmentation_task(
        zarr_url=str(test_data_path),
        channel_1=channel_1,
        model_type="cyto2",
        overwrite=True,
        iterator_configuration=iter_config,
    )

    expected_label = "DAPI_0_cellpose_segmented"
    assert expected_label in ome_zarr.list_labels()

    label = ome_zarr.get_label(expected_label)
    label_data = label.get_as_numpy()
    # Masked segmentation reduces the object count vs the unmasked run.
    assert np.max(label_data) > 0
    if expected_objects is not None:
        assert np.max(label_data) == expected_objects
