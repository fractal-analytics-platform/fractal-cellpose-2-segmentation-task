"""Cellpose 2 segmentation task for Fractal."""

import logging

from fractal_tasks_utils.segmentation import (
    IteratorConfig,
    compute_segmentation,
    setup_segmentation_iterator,
)
from fractal_tasks_utils.segmentation._transforms import SegmentationTransformConfig
from ngio import ChannelSelectionModel, open_ome_zarr_container
from pydantic import Field, validate_call

from fractal_cellpose_2_segmentation_task.segmentation_utils import (
    CellposeCustomNormalizer,
    CellposeModelName,
    CellposeModelParams,
    load_cellpose_model,
    segment_image,
)
from fractal_cellpose_2_segmentation_task.utils import (
    AnyCreateRoiTableModel,
    CreateMaskingRoiTable,
    SkipCreateMaskingRoiTable,
)

logger = logging.getLogger("cellpose_segmentation_task")


@validate_call
def cellpose_segmentation_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Channel selection
    channel_1: ChannelSelectionModel,
    channel_2: ChannelSelectionModel | None = None,
    label_name: str = "{channel_identifier}_cellpose_segmented",
    level_path: str | None = None,
    # Cellpose model parameters
    model_type: CellposeModelName = "cyto2",  # type: ignore[assignment]
    custom_model_path: str | None = None,
    diameter: float = 30.0,
    # Normalization (one per channel)
    normalize: CellposeCustomNormalizer = Field(  # noqa: B008
        default_factory=CellposeCustomNormalizer
    ),
    normalize_2: CellposeCustomNormalizer = Field(  # noqa: B008
        default_factory=CellposeCustomNormalizer
    ),
    # Advanced Cellpose eval parameters
    advanced_model_params: CellposeModelParams = Field(  # noqa: B008
        default_factory=CellposeModelParams
    ),
    # Iterator / infrastructure parameters
    iterator_configuration: IteratorConfig | None = None,
    pre_post_process: SegmentationTransformConfig = Field(  # noqa: B008
        default_factory=SegmentationTransformConfig
    ),
    create_masking_roi_table: AnyCreateRoiTableModel = Field(  # noqa: B008
        default_factory=SkipCreateMaskingRoiTable
    ),
    overwrite: bool = True,
) -> None:
    """Segment an image using Cellpose 2.

    Runs Cellpose 2 instance segmentation on an OME-Zarr dataset.
    Supports single-channel and two-channel (cytoplasm + nuclear) input,
    pretrained models from Cellpose's model registry, and custom model files.

    Args:
        zarr_url (str): URL to the OME-Zarr container.
        channel_1 (ChannelSelectionModel): Primary channel for segmentation
            (typically cytoplasm). Required.
        channel_2 (ChannelSelectionModel | None): Optional secondary channel
            (typically nuclear). When provided, Cellpose is run in two-channel
            mode (channels=[1, 2]).
        label_name (str): Name of the resulting label image. The placeholder
            "{channel_identifier}" is replaced by the identifier of channel_1.
        level_path (str | None): Resolution level to use. If not provided, the
            highest resolution level is used.
        model_type (CellposeModelName): Pretrained Cellpose model name
            (e.g. "cyto2", "nuclei", "cyto3"). Ignored when
            custom_model_path is provided. Default: "cyto2".
        custom_model_path (str | None): Path to a custom Cellpose model file.
            When provided, takes precedence over model_type.
        diameter (float): Expected object diameter in pixels. 
            Calculated in pixels => if you change the level_path (=> image 
            resolution), also adapt the diameter accordingly.
        normalize (CellposeCustomNormalizer): Normalization settings for the
            primary channel.
        normalize_2 (CellposeCustomNormalizer): Normalization settings for the
            secondary channel. Only used when channel_2 is provided.
        advanced_model_params (CellposeModelParams): Advanced Cellpose eval
            hyperparameters (flow_threshold, cellprob_threshold, etc.).
        iterator_configuration (IteratorConfig | None): Advanced configuration
            to control masked and ROI-based iteration.
        pre_post_process (SegmentationTransformConfig): Configuration for pre-
            and post-processing transforms applied by the iterator.
        create_masking_roi_table (AnyCreateRoiTableModel): Configuration to
            create a masking ROI table after segmentation.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    # Format the label name using the primary channel identifier
    label_name = label_name.format(channel_identifier=channel_1.identifier)
    logger.info(f"Formatted label name: {label_name=}")

    # Load the Cellpose model (with retry logic for cluster race conditions)
    model = load_cellpose_model(
        model_type=model_type,
        custom_model_path=custom_model_path,
        use_gpu=advanced_model_params.use_gpu,
    )

    # Build channel list: include channel_2 only when provided
    channels = [channel_1] if channel_2 is None else [channel_1, channel_2]

    # Set up the segmentation iterator
    iterator = setup_segmentation_iterator(
        zarr_url=zarr_url,
        channels=channels,
        output_label_name=label_name,
        level_path=level_path,
        iterator_configuration=iterator_configuration,
        segmentation_transform_config=pre_post_process,
        overwrite=overwrite,
    )

    # Run the core segmentation loop
    compute_segmentation(
        segmentation_func=lambda x: segment_image(
            image=x,
            model=model,
            diameter=diameter,
            normalize=normalize,
            normalize2=normalize_2,
            advanced_model_params=advanced_model_params,
        ),
        iterator=iterator,
    )
    logger.info(f"label {label_name} successfully created at {zarr_url}")

    # Build a masking ROI table if configured
    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(label_name=label_name)
        label = ome_zarr.get_label(name=label_name, path=level_path)
        masking_roi_table = label.build_masking_roi_table()
        ome_zarr.add_table(
            name=table_name, table=masking_roi_table, overwrite=overwrite
        )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=cellpose_segmentation_task)
