from pathlib import Path

import numpy as np
import pytest
from fractal_tasks_utils.segmentation import IteratorConfig
from fractal_tasks_utils.segmentation._models import MaskingConfig
from ngio import ChannelSelectionModel, create_synthetic_ome_zarr

from fractal_stardist_segmentation_task.segmentation_utils import StarDistModelPreset
from fractal_stardist_segmentation_task.stardist_segmentation_task import (
    stardist_segmentation_task,
)

# 2D shapes with different axis configurations and expected object counts
# based on the synthetic data generation
_2D_SHAPES = [
    ((256, 256), "yx", 40),
    ((1, 256, 256), "cyx", 40),
    ((3, 256, 256), "cyx", 40),
    ((4, 256, 256), "tyx", 160),
    ((1, 3, 256, 256), "tcyx", 40),
]
_2D_SHAPES_masked = [
    ((256, 256), "yx", 37),
    ((1, 256, 256), "cyx", 37),
    ((3, 256, 256), "cyx", 37),
    ((4, 256, 256), "tyx", 148),
    ((1, 3, 256, 256), "tcyx", 37),
]


@pytest.mark.parametrize("shape, axes, expected_objects", _2D_SHAPES)
def test_stardist_segmentation_task(
    tmp_path: Path, shape: tuple[int, ...], axes: str, expected_objects: int
):
    """Base test for the StarDist segmentation task."""
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
    channel = ChannelSelectionModel(identifier="DAPI_0", mode="label")

    stardist_segmentation_task(
        zarr_url=str(test_data_path),
        channel=channel,
        stardist_model=StarDistModelPreset.versatile_fluo_2d,
        overwrite=True,
    )

    expected_label = "DAPI_0_stardist_segmented"
    assert expected_label in ome_zarr.list_labels()

    label = ome_zarr.get_label(expected_label)
    label_data = label.get_as_numpy()
    # In the synthethic data above, StarDist finds 40 objects in the
    # first channel, independent of the axis setup. For timeseries, this gets
    # multiplied by number of timepoints (as the same synthetic data is
    # repeated across time).
    assert np.max(label_data) == expected_objects


@pytest.mark.parametrize("shape, axes, expected_objects", _2D_SHAPES_masked)
def test_stardist_segmentation_task_masked(
    tmp_path: Path, shape: tuple[int, ...], axes: str, expected_objects: int
):
    """Test the StarDist segmentation task with a masking configuration."""
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
    channel = ChannelSelectionModel(identifier="DAPI_0", mode="label")

    iter_config = IteratorConfig(
        masking=MaskingConfig(masking_source="Label Name", identifier="nuclei_mask"),
    )
    stardist_segmentation_task(
        zarr_url=str(test_data_path),
        channel=channel,
        stardist_model=StarDistModelPreset.versatile_fluo_2d,
        overwrite=True,
        iterator_configuration=iter_config,
    )

    expected_label = "DAPI_0_stardist_segmented"
    assert expected_label in ome_zarr.list_labels()

    label = ome_zarr.get_label(expected_label)
    label_data = label.get_as_numpy()
    # In the synthethic data above, StarDist finds 37 objects in the
    # first channel using the mask, independent of the axis setup.
    # For timeseries, this gets multiplied by number of timepoints (as the
    # same synthetic data is repeated across time).
    assert np.max(label_data) == expected_objects
