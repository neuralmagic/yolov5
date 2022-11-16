import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy
import torch
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import download_framework_model_by_recipe_type
from sparsezoo import Model

from models.yolo import Model as Yolov5Model
from utils.general import LOGGER, colorstr
from utils.dataloaders import LoadImages
from utils.general import check_dataset, check_yaml
from utils.neuralmagic.quantization import update_model_bottlenecks
from utils.torch_utils import ModelEMA

__all__ = [
    "ALMOST_ONE",
    "sparsezoo_download",
    "ToggleableModelEMA",
    "load_ema",
    "load_sparsified_model",
    "get_sample_data",
    "export_sample_inputs_outputs",
]


RANK = int(os.getenv("RANK", -1))
ALMOST_ONE = 1 - 1e-9  # for incrementing epoch to be applied to recipe


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
        model, ckpt["epoch"] + ALMOST_ONE if ckpt["epoch"] >= 0 else float("inf")
    )

    # Load state dict
    model.load_state_dict(ckpt["ema"] or ckpt["model"], strict=True)
    return model


def nm_log_console(message: str, logger: "Logger" = None, level: str = "info"):
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
def get_sample_data(
    image: torch.Tensor, data: Union[str, Path], number_samples: int = 20
) -> List[numpy.ndarray]:
    """
    Extracts number_samples of samples from the given dataset, with each sample as a
    numpy array

    :param image: sample image to determine image size
    :param data: path to dataset
    :number_samples: number of samples to extract
    """
    _, _, *imgsz = list(image.shape)
    dataset = LoadImages(
        check_dataset(check_yaml(data))["train"], img_size=imgsz, auto=False
    )

    samples = []
    for i, image in enumerate(dataset):
        if i >= number_samples:
            break
        samples.append(image[1])

    return samples


def export_sample_inputs_outputs(
    dataset: Union[str, Path],
    model: torch.nn.Module,
    save_dir: Path,
    number_export_samples=100,
    image_size: int = 640,
    save_inputs_as_uint8: bool = False,
):
    """
    Export sample model input and output for testing with the DeepSparse Engine
    """

    dataloader = LoadImages(
        check_dataset(check_yaml(dataset))["train"], img_size=image_size, auto=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exported_samples = 0

    sample_in_dir = save_dir / "sample_inputs"
    sample_out_dir = save_dir / "sample_outputs"

    sample_in_dir.mkdir(exist_ok=True)
    sample_out_dir.mkdir(exist_ok=True)

    for _, image, _, _, _ in dataloader:
        # uint8 to float32, 0-255 to 0.0-1.0
        image = (torch.from_numpy(image).float() / 255).expand(1, *image.shape)
        image = image.to(device, non_blocking=True)
        model_out = model(image)

        # Move to cpu for exporting
        image = image.detach().to("cpu")  # TODO: need to assign?

        if isinstance(model_out, tuple) and len(model_out) > 1:
            # flatten into a single list
            model_out = [model_out[0], *model_out[1]]

        model_out = [elem.detach().to("cpu") for elem in model_out]
        outs_gen = zip(*model_out)

        for sample_in, sample_out in zip(image, outs_gen):
            sample_out = list(sample_out)
            file_idx = f"{exported_samples}".zfill(4)

            sample_input_filename = sample_in_dir / f"inp-{file_idx}.npz"
            if save_inputs_as_uint8:
                sample_in = (255 * sample_in).to(dtype=torch.uint8)
            numpy.savez(sample_input_filename, sample_in)

            sample_output_filename = sample_out_dir / f"out-{file_idx}.npz"
            numpy.savez(sample_output_filename, *sample_out)
            exported_samples += 1

            if exported_samples >= number_export_samples:
                break

        if exported_samples >= number_export_samples:
            break
