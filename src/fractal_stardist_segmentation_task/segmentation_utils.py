"""Segmentation utils for StarDist."""

import json
import logging
import random
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Optional

import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D, StarDist3D

logger = logging.getLogger(__name__)


class StarDistModelPreset(StrEnum):
    """Pretrained StarDist model presets.

    2D models are suitable for single-plane fluorescence or histology images.
    3D models are suitable for volumetric fluorescence data.
    """

    versatile_fluo_2d = "2D_versatile_fluo"
    versatile_he_2d = "2D_versatile_he"
    paper_dsb2018_2d = "2D_paper_dsb2018"
    demo_3d = "3D_demo"


def _load_with_retry(
    loader: Callable[[], StarDist2D | StarDist3D],
    description: str,
    max_attempts: int = 10,
) -> StarDist2D | StarDist3D:
    """Load a StarDist model with retry logic for cluster safety.

    Multiple parallel workers may attempt to load or download the same model
    simultaneously. This retries with random backoff to avoid race conditions.

    Args:
        loader: Zero-argument callable that returns the loaded model.
        description: Human-readable description used in log/error messages.
        max_attempts: Maximum number of attempts before raising.

    Returns:
        Loaded StarDist2D or StarDist3D model.

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


def _loader_from_pretrained(model_name: str) -> Callable[[], StarDist2D | StarDist3D]:
    """Return a loader callable for a pretrained StarDist model by name."""

    def _load() -> StarDist2D | StarDist3D:
        if "3D" in model_name:
            return StarDist3D.from_pretrained(model_name)
        return StarDist2D.from_pretrained(model_name)

    return _load


def _loader_from_path(model_path: str) -> Callable[[], StarDist2D | StarDist3D]:
    """Return a loader callable for a StarDist model stored in a local directory.

    Reads config.json to determine whether the model is 2D or 3D.

    Args:
        model_path: Full path to the StarDist model directory.

    Returns:
        Callable that loads and returns the model.

    Raises:
        FileNotFoundError: If config.json is not found in the model directory.
    """
    model_dir = Path(model_path)
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No config.json found in model directory: {model_path}"
        )

    with open(config_file) as f:
        config = json.load(f)
    n_dim = config.get("n_dim", 2)
    basedir = str(model_dir.parent)
    name = model_dir.name

    def _load() -> StarDist2D | StarDist3D:
        if n_dim == 3:
            return StarDist3D(config=None, name=name, basedir=basedir)
        return StarDist2D(config=None, name=name, basedir=basedir)

    return _load


def load_stardist_model(
    stardist_model: StarDistModelPreset,
    custom_model_path: Optional[str] = None,
) -> StarDist2D | StarDist3D:
    """Load a StarDist model, either from a preset or a custom local path.

    When custom_model_path is provided it takes precedence over stardist_model.
    Both loading paths use retry logic with random backoff to handle race
    conditions when multiple cluster workers access the model simultaneously.

    Args:
        stardist_model: Pretrained model preset to use if no custom path is given.
        custom_model_path: Full path to a custom StarDist model directory.
            The directory must contain a config.json file. If provided, this
            takes precedence over stardist_model.

    Returns:
        Loaded StarDist2D or StarDist3D model.

    Raises:
        RuntimeError: If the model cannot be loaded after 10 attempts.
        FileNotFoundError: If custom_model_path is given but config.json is missing.
    """
    if custom_model_path is not None:
        logger.info(f"Loading custom StarDist model from {custom_model_path}")
        loader = _loader_from_path(custom_model_path)
        return _load_with_retry(
            loader, description=f"custom model at {custom_model_path}"
        )

    model_name = stardist_model.value
    logger.info(f"Loading pretrained StarDist model '{model_name}'")
    loader = _loader_from_pretrained(model_name)
    model = _load_with_retry(loader, description=f"pretrained model '{model_name}'")
    logger.info(f"Successfully loaded StarDist model '{model_name}'")
    return model


def segment_image(
    image: np.ndarray,
    model: StarDist2D | StarDist3D,
    prob_thresh: Optional[float] = None,
    nms_thresh: Optional[float] = None,
    normalize_perc_low: float = 1.0,
    normalize_perc_high: float = 99.8,
) -> np.ndarray:
    """Run StarDist instance segmentation on a single image.

    Handles both 2D and 3D images. The Fractal iterator passes channel images
    that may have extra leading singleton dimensions (e.g. shape (1, H, W) for
    a 2D image). This function strips those extra dims, normalizes the image,
    runs prediction, then restores the original leading shape.

    Args:
        image: Input image as numpy array. May be 2D (H, W), 3D (Z, H, W), or
            have extra leading singleton dims like (1, H, W) or (1, Z, H, W).
        model: Loaded StarDist2D or StarDist3D model.
        prob_thresh: Probability threshold for instance detection. If None, the
            model's default is used.
        nms_thresh: Non-maximum suppression threshold for overlap removal.
            If None, the model's default is used.
        normalize_perc_low: Lower percentile for input normalization.
            Pixels at or below this percentile are mapped to 0.
        normalize_perc_high: Upper percentile for input normalization.
            Pixels at or above this percentile are mapped to 1.

    Returns:
        Instance segmentation label array of same shape as input, dtype uint32.
    """
    is_3d = isinstance(model, StarDist3D)
    n_spatial = 3 if is_3d else 2
    axes = "ZYX" if is_3d else "YX"

    # Strip leading singleton dimensions, remembering how many to restore
    extra_dims = image.shape[:-n_spatial]
    spatial_image = image.reshape(image.shape[-n_spatial:])

    # Normalize to float32 using percentile clipping (matches StarDist's own default)
    spatial_image = normalize(
        spatial_image, normalize_perc_low, normalize_perc_high
    ).astype(np.float32)

    predict_kwargs: dict = {}
    if prob_thresh is not None:
        predict_kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        predict_kwargs["nms_thresh"] = nms_thresh

    logger.info(
        f"Running StarDist prediction: shape={spatial_image.shape}, "
        f"axes={axes}, {predict_kwargs=}"
    )

    labels, _ = model.predict_instances(
        spatial_image, axes=axes, normalizer=None, **predict_kwargs
    )

    logger.info(f"Generated {labels.max()} instances, shape={labels.shape}")

    # Restore leading dimensions so the ngio writer can handle them correctly
    for _ in extra_dims:
        labels = labels[np.newaxis]

    return labels.astype(np.uint32)
