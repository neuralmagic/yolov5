import os
from typing import Any, Dict, Optional, Union

import torch
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import download_framework_model_by_recipe_type
from sparsezoo import Model

from models.yolo import Model as Yolov5Model
from utils.general import LOGGER, colorstr
from utils.neuralmagic.quantization import update_model_bottlenecks
from utils.torch_utils import ModelEMA

__all__ = [
    "sparsezoo_download",
    "ToggleableModelEMA",
    "load_ema",
    "load_sparsified_model",
]


RANK = int(os.getenv("RANK", -1))


class ToggleableModelEMA(ModelEMA):
    """
    Subclasses YOLOv5 ModelEMA to enabled disabling during QAT
    """

    def __init__(self, enabled, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = enabled

    def update(self, *args, **kwargs):
        if self.enabled:
            super().update(*args, **kwargs)


def sparsezoo_download(path: str, sparsification_recipe: Optional[str] = None) -> str:
    """
    Loads model from the SparseZoo and override the path with the new download path
    """
    return download_framework_model_by_recipe_type(
        Model(path), sparsification_recipe, "pt"
    )


def load_ema(
    ema_state_dict: Dict[str, Any],
    model: torch.nn.Module,
    enabled: bool = True,
    **ema_kwargs,
) -> ToggleableModelEMA:
    """
    Loads a ToggleableModelEMA object from a ModelEMA state dict and loaded model
    """
    ema = ToggleableModelEMA(enabled, model, **ema_kwargs)
    ema.ema.load_state_dict(ema_state_dict)
    return ema


def load_sparsified_model(
    ckpt: Union[Dict[str, Any], str], device: Union[str, torch.device] = "cpu"
) -> torch.nn.Module:
    """
    From a sparisifed checkpoint, loads a model with the saved weights and
    sparsification recipe applied

    :param ckpt: either a loaded checkpoint or the path to a saved checkpoint
    :param device: device to load the model onto
    """
    nm_log_console("Loading sparsified model")

    # Load checkpoint if not yet loaded
    ckpt = ckpt if isinstance(ckpt, dict) else torch.load(ckpt, map_location=device)

    # Construct randomly initialized model model and apply sparse structure modifiers
    model = Yolov5Model(ckpt.get("yaml"))
    model = update_model_bottlenecks(model).to(device)
    checkpoint_manager = ScheduledModifierManager.from_yaml(ckpt["checkpoint_recipe"])
    checkpoint_manager.apply_structure(
        model, ckpt["epoch"] if ckpt["epoch"] >= 0 else float("inf")
    )

    # Load state dict
    model.load_state_dict(ckpt["ema"] or ckpt["model"], strict=True)
    return model


def nm_log_console(self, message: str, logger: "Logger" = None, level: str = "info"):
    """
    Log sparsification-related messages to the console

    :param message: message to be logged
    :param level: level to be logged at
    """
    # default to global logger if none provided
    logger = logger or LOGGER

    if RANK in [0, -1]:
        if level == "warning":
            logger.warning(
                f"{colorstr('Neural Magic: ')}{colorstr('yellow', 'warning - ')}"
                f"{message}"
            )
        else:  # default to info
            logger.info(f"{colorstr('Neural Magic: ')}{message}")
