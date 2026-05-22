"""Segmentation utils for Cellpose 2."""

import logging
import random
import time
from collections.abc import Callable
from typing import Literal, Optional

import numpy as np
from cellpose import models
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


# Dynamically built from Cellpose's own model registry so we never need to
# maintain a manual list.
CellposeModelName = Literal[tuple(models.MODEL_NAMES)]  # type: ignore[valid-type]


class CellposeCustomNormalizer(BaseModel):
    """Validator to handle different normalization scenarios for Cellpose models.

    If `type="default"`, then Cellpose default normalization is used and no
    other parameters can be specified.
    If `type="no_normalization"`, then no normalization is used and no other
    parameters can be specified.
    If `type="custom"`, then either percentiles or explicit integer bounds can
    be applied.

    Attributes:
        type: One of `default`, `custom`, or `no_normalization`.
        lower_percentile: Custom lower-bound percentile (0-100). Only when
            type="custom". Cannot be combined with bounds.
        upper_percentile: Custom upper-bound percentile (0-100). Only when
            type="custom". Cannot be combined with bounds.
        lower_bound: Explicit lower integer value for rescaling. Only when
            type="custom". Cannot be combined with percentiles.
        upper_bound: Explicit upper integer value for rescaling. Only when
            type="custom". Cannot be combined with percentiles.
    """

    type: Literal["default", "custom", "no_normalization"] = "default"
    lower_percentile: Optional[float] = Field(None, ge=0, le=100)
    upper_percentile: Optional[float] = Field(None, ge=0, le=100)
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None

    @model_validator(mode="after")
    def validate_conditions(self: Self) -> Self:
        """Validate that custom params are only set when type='custom'."""
        t = self.type
        lp, up = self.lower_percentile, self.upper_percentile
        lb, ub = self.lower_bound, self.upper_bound

        if t != "custom":
            for name, val in [
                ("lower_percentile", lp),
                ("upper_percentile", up),
                ("lower_bound", lb),
                ("upper_bound", ub),
            ]:
                if val is not None:
                    raise ValueError(
                        f"Type='{t}' but {name}={val}. Hint: set type='custom'."
                    )

        are_percentiles_set = (lp is not None, up is not None)
        are_bounds_set = (lb is not None, ub is not None)
        if len(set(are_percentiles_set)) != 1:
            raise ValueError(
                "Both lower_percentile and upper_percentile must be set together."
            )
        if len(set(are_bounds_set)) != 1:
            raise ValueError(
                "Both lower_bound and upper_bound must be set together."
            )
        if lp is not None and lb is not None:
            raise ValueError(
                "Cannot set both explicit bounds and percentile bounds at the "
                "same time. Use only one of the two options."
            )
        return self

    @property
    def cellpose_normalize(self) -> bool:
        """Whether Cellpose should apply its internal normalization."""
        return self.type == "default"


class CellposeModelParams(BaseModel):
    """Advanced Cellpose model evaluation parameters.

    Attributes:
        cellprob_threshold: Valid values -6 to 6. Decrease to return more ROIs,
            increase to return fewer.
        flow_threshold: Valid values 0.0-1.0. Increase to return more ROIs,
            decrease to remove ill-shaped ROIs.
        anisotropy: Ratio of pixel sizes along Z and XY (3D only). Inferred
            from OME-NGFF metadata if unset.
        min_size: Minimum object size in pixels. Use -1 to disable size filter.
        augment: Whether to tile images with overlap for augmentation.
        net_avg: Whether to average the 4 built-in networks (useful for
            nuclei/cyto/cyto2).
        use_gpu: Use GPU if available; fall back to CPU otherwise.
        batch_size: Number of 224x224 patches run simultaneously on GPU/CPU.
        invert: Invert image pixel intensity before running network.
        tile: Tile image to limit GPU/CPU memory usage (recommended).
        tile_overlap: Fraction of overlap between tiles when computing flows.
        resample: Run dynamics at original image size (slower, more accurate).
        interp: Interpolate during 2D dynamics (not available in 3D).
        stitch_threshold: If >0.0, stitch masks in 3D across Z slices.
            Only used when do_3D=False.
        do_3D: Whether to run true 3D segmentation. Only relevant when the
            input has Z > 1. If True, Cellpose processes XY, XZ and YZ planes
            and averages the flows (true volumetric segmentation). If False,
            each XY plane is processed independently and `stitch_threshold`
            is used to connect masks across Z slices. Has no effect on 2D
            (single-plane) data.
    """

    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    anisotropy: Optional[float] = None
    min_size: int = 15
    augment: bool = False
    net_avg: bool = False
    use_gpu: bool = True
    batch_size: int = 8
    invert: bool = False
    tile: bool = True
    tile_overlap: float = 0.1
    resample: bool = True
    interp: bool = True
    stitch_threshold: float = 0.0
    do_3D: bool = True


def _load_with_retry(
    loader: Callable[[], models.CellposeModel],
    description: str,
    max_attempts: int = 10,
) -> models.CellposeModel:
    """Load a Cellpose model with retry logic for cluster safety.

    Multiple parallel workers may attempt to download the same model
    simultaneously. This retries with random backoff to avoid race conditions.

    Args:
        loader: Zero-argument callable that returns the loaded model.
        description: Human-readable description for log/error messages.
        max_attempts: Maximum number of attempts before raising.

    Returns:
        Loaded CellposeModel.

    Raises:
        RuntimeError: If the model cannot be loaded after max_attempts.
    """
    model = None
    for attempt in range(1, max_attempts + 1):
        try:
            model = loader()
            if model is not None:
                break
        except Exception:
            logger.warning(
                f"Failed to load {description} (attempt {attempt}/{max_attempts})."
                " Retrying..."
            )
            time.sleep(random.uniform(2, 7))

    if model is None:
        raise RuntimeError(
            f"Could not load {description} after {max_attempts} attempts."
        )
    return model


def load_cellpose_model(
    model_type: CellposeModelName = "cyto2",  # type: ignore[assignment]
    custom_model_path: Optional[str] = None,
    use_gpu: bool = True,
) -> models.CellposeModel:
    """Load a Cellpose model, either from a preset name or a custom local path.

    When custom_model_path is provided it takes precedence over model_type.
    Both loading paths use retry logic with random backoff to handle race
    conditions when multiple cluster workers access the model simultaneously.

    Args:
        model_type: Pretrained model name. Must be one of the names in
            cellpose.models.MODEL_NAMES (e.g. "cyto2", "nuclei", "cyto3").
            Default: "cyto2".
        custom_model_path: Path to a custom Cellpose model file. When provided,
            takes precedence over model_type.
        use_gpu: Whether to use GPU acceleration.

    Returns:
        Loaded CellposeModel instance.

    Raises:
        RuntimeError: If the model cannot be loaded after 10 attempts.
    """
    if custom_model_path is not None:
        logger.info(f"Loading custom Cellpose model from {custom_model_path}")

        def _load_custom() -> models.CellposeModel:
            return models.CellposeModel(gpu=use_gpu, pretrained_model=custom_model_path)

        return _load_with_retry(
            _load_custom, description=f"custom model at {custom_model_path}"
        )

    logger.info(f"Loading pretrained Cellpose model '{model_type}'")

    def _load_pretrained() -> models.CellposeModel:
        return models.CellposeModel(gpu=use_gpu, model_type=model_type)

    model = _load_with_retry(
        _load_pretrained, description=f"pretrained model '{model_type}'"
    )
    logger.info(f"Successfully loaded Cellpose model '{model_type}'")
    return model


def _normalize_cellpose_channels(
    x: np.ndarray,
    channels: list[int],
    normalize: CellposeCustomNormalizer,
    normalize2: CellposeCustomNormalizer,
) -> np.ndarray:
    """Normalize a Cellpose input array by channel.

    Args:
        x: Input numpy array.
        channels: Which channels to use. [0, 0] for single channel; [1, 2]
            for two channels (x[0] = cyto, x[1] = nuclear).
        normalize: Normalization config for the primary channel.
        normalize2: Normalization config for the secondary channel. Only used
            when channels == [1, 2].
    """
    if channels == [1, 2]:
        if (normalize.type == "default") != (normalize2.type == "default"):
            raise ValueError(
                f"Normalization mismatch: {normalize.type=} and "
                f"{normalize2.type=}. Either both must be 'default' or neither."
            )
        if normalize.type == "custom":
            x[channels[0] - 1 : channels[0]] = _normalized_img(
                x[channels[0] - 1 : channels[0]],
                lower_p=normalize.lower_percentile,
                upper_p=normalize.upper_percentile,
                lower_bound=normalize.lower_bound,
                upper_bound=normalize.upper_bound,
            )
        if normalize2.type == "custom":
            x[channels[1] - 1 : channels[1]] = _normalized_img(
                x[channels[1] - 1 : channels[1]],
                lower_p=normalize2.lower_percentile,
                upper_p=normalize2.upper_percentile,
                lower_bound=normalize2.lower_bound,
                upper_bound=normalize2.upper_bound,
            )
    else:
        if normalize.type == "custom":
            x = _normalized_img(
                x,
                lower_p=normalize.lower_percentile,
                upper_p=normalize.upper_percentile,
                lower_bound=normalize.lower_bound,
                upper_bound=normalize.upper_bound,
            )
    return x


def _normalized_img(
    img: np.ndarray,
    axis: int = -1,
    invert: bool = False,
    lower_p: Optional[float] = 1.0,
    upper_p: Optional[float] = 99.0,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
) -> np.ndarray:
    """Normalize image so 0.0 = lower bound and 1.0 = upper bound.

    Args:
        img: ND-array (at least 3 dimensions).
        axis: Channel axis to loop over.
        invert: Invert image after normalization.
        lower_p: Lower percentile for rescaling.
        upper_p: Upper percentile for rescaling.
        lower_bound: Lower fixed value for rescaling.
        upper_bound: Upper fixed value for rescaling.

    Returns:
        Normalized float32 image of same shape.
    """
    if img.ndim < 3:
        raise ValueError("Image needs to have at least 3 dimensions")

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if lower_p is not None:
            i99 = np.percentile(img[k], upper_p)
            i1 = np.percentile(img[k], lower_p)
            if i99 - i1 > 1e-3:
                img[k] = _normalize_percentile(img[k], lower=lower_p, upper=upper_p)
                if invert:
                    img[k] = -1 * img[k] + 1
            else:
                img[k] = 0
        elif lower_bound is not None:
            if upper_bound - lower_bound > 1e-3:
                img[k] = _normalize_bounds(
                    img[k], lower=lower_bound, upper=upper_bound
                )
                if invert:
                    img[k] = -1 * img[k] + 1
            else:
                img[k] = 0
        else:
            raise ValueError("No normalization method specified")
    img = np.moveaxis(img, 0, axis)
    return img


def _normalize_percentile(
    Y: np.ndarray, lower: float = 1, upper: float = 99
) -> np.ndarray:
    """Normalize so 0.0 = lower percentile and 1.0 = upper percentile."""
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    return (X - x01) / (x99 - x01)


def _normalize_bounds(Y: np.ndarray, lower: int = 0, upper: int = 65535) -> np.ndarray:
    """Normalize so 0.0 = lower value and 1.0 = upper value."""
    X = Y.copy()
    return (X - lower) / (upper - lower)


def segment_image(
    image: np.ndarray,
    model: models.CellposeModel,
    diameter: float = 30.0,
    normalize: Optional[CellposeCustomNormalizer] = None,
    normalize2: Optional[CellposeCustomNormalizer] = None,
    advanced_model_params: Optional[CellposeModelParams] = None,
) -> np.ndarray:
    """Run Cellpose 2 instance segmentation on a single image.

    Cellpose 2 accepts 4D arrays directly; no stripping of leading dims is
    needed. The iterator passes (Z, Y, X) or (2, Z, Y, X) arrays.

    Args:
        image: Input image as numpy array. Shape (Z, Y, X) for single channel
            or (2, Z, Y, X) for two channels (cyto + nuclear).
        model: Loaded CellposeModel instance.
        diameter: Expected object diameter in pixels.
        normalize: Normalization config for the primary (or only) channel.
        normalize2: Normalization config for the secondary channel.
        advanced_model_params: Advanced Cellpose eval hyperparameters.
            Includes `do_3D` which controls 3D segmentation behaviour
            (only relevant when the input has Z > 1).

    Returns:
        Instance segmentation label array, dtype uint32. Shape (Z, Y, X).
    """
    print(image.shape)
    if normalize is None:
        normalize = CellposeCustomNormalizer()
    if normalize2 is None:
        normalize2 = CellposeCustomNormalizer()
    if advanced_model_params is None:
        advanced_model_params = CellposeModelParams()

    input_ndim = image.ndim

    # Determine channel mode: two-channel if first dim is exactly 2
    if image.ndim >= 4 and image.shape[0] == 2:
        channels = [1, 2]
    else:
        channels = [0, 0]

    logger.info(
        f"[segment_image] START | shape={image.shape} | {channels=} | "
        f"do_3D={advanced_model_params.do_3D} | {diameter=} | "
        f"normalize.type={normalize.type}"
    )

    image = _normalize_cellpose_channels(image, channels, normalize, normalize2)

    mask, _, _ = model.eval(
        image,
        channels=channels,
        do_3D=advanced_model_params.do_3D,
        net_avg=advanced_model_params.net_avg,
        augment=advanced_model_params.augment,
        diameter=diameter,
        anisotropy=advanced_model_params.anisotropy,
        cellprob_threshold=advanced_model_params.cellprob_threshold,
        flow_threshold=advanced_model_params.flow_threshold,
        normalize=normalize.cellpose_normalize,
        min_size=advanced_model_params.min_size,
        batch_size=advanced_model_params.batch_size,
        invert=advanced_model_params.invert,
        tile=advanced_model_params.tile,
        tile_overlap=advanced_model_params.tile_overlap,
        resample=advanced_model_params.resample,
        interp=advanced_model_params.interp,
        stitch_threshold=advanced_model_params.stitch_threshold,
    )

    while mask.ndim < input_ndim:
        mask = np.expand_dims(mask, axis=0)

    logger.info(
        f"[segment_image] END | shape={mask.shape} | max_label={np.max(mask)}"
    )
    print(mask.shape)

    return mask.astype(np.uint32)
