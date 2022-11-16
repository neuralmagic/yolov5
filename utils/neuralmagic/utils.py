import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy
import torch
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import download_framework_model_by_recipe_type
from sparsezoo import Model

from models.yolo import Model as Yolov5Model
from utils.dataloaders import create_dataloader
from utils.general import LOGGER, check_dataset, check_yaml, colorstr
from utils.neuralmagic.quantization import update_model_bottlenecks
from utils.torch_utils import ModelEMA

__all__ = [
    "ALMOST_ONE",
    "sparsezoo_download",
    "ToggleableModelEMA",
    "load_ema",
    "load_sparsified_model",
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


def export_sample_inputs_outputs(
    dataset: Union[str, Path],
    model: torch.nn.Module,
    save_dir: Path,
    number_export_samples=100,
    image_size: int = 640,
    onnx_path: Union[str, Path, None] = None,
):
    """
    Export sample model input and output for testing with the DeepSparse Engine

    :param dataset: path to dataset to take samples from
    :param model: model to be exported. Used to generate outputs
    :param save_dir: directory to save samples to
    :param number_export_samples: number of samples to export
    :param image_size: image size
    :param onnx_path: Path to saved onnx model. Used to check if it uses uints8 inputs
    """

    # Create dataloader
    data_dict = check_dataset(dataset)
    dataloader, _ = create_dataloader(
        path=data_dict["train"],
        imgsz=image_size,
        batch_size=1,
        stride=max(int(model.stride.max()), 32),
        hyp=model.hyp,
        augment=True,
        prefix=colorstr("train: "),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exported_samples = 0

    # Sample export directories
    sample_in_dir = save_dir / "sample_inputs"
    sample_out_dir = save_dir / "sample_outputs"
    sample_in_dir.mkdir(exist_ok=True)
    sample_out_dir.mkdir(exist_ok=True)

    save_inputs_as_uint8 = _graph_has_uint8_inputs(onnx_path) if onnx_path else False

    for images, _, _, _ in dataloader:
        # uint8 to float32, 0-255 to 0.0-1.0
        images = (images.float() / 255).to(device, non_blocking=True)
        model_out = model(images)

        if isinstance(model_out, tuple) and len(model_out) > 1:
            # Flatten into a single list
            model_out = [model_out[0], *model_out[1]]

        # Move to cpu for exporting
        images = images.detach().to("cpu")
        model_out = [elem.detach().to("cpu") for elem in model_out]

        outs_gen = zip(*model_out)

        for sample_in, sample_out in zip(images, outs_gen):

            sample_out = list(sample_out)

            file_idx = f"{exported_samples}".zfill(4)

            # Save inputs as numpy array
            sample_input_filename = sample_in_dir / f"inp-{file_idx}.npz"
            if save_inputs_as_uint8:
                sample_in = (255 * sample_in).to(dtype=torch.uint8)
            numpy.savez(sample_input_filename, sample_in)

            # Save outputs as numpy array
            sample_output_filename = sample_out_dir / f"out-{file_idx}.npz"
            numpy.savez(sample_output_filename, *sample_out)
            exported_samples += 1

            if exported_samples >= number_export_samples:
                break

        if exported_samples >= number_export_samples:
            break

    if exported_samples < number_export_samples:
        nm_log_console(
            f"Could not export {number_export_samples} samples. Exhausted dataloader "
            f"and exported {exported_samples} samples",
            level="warning",
        )


def _graph_has_uint8_inputs(onnx_path: Union[str, Path]) -> bool:
    """
    Load onnx model and check if it's input is type 2 (unit8)
    """
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    return onnx_model.graph.input[0].type.tensor_type.elem_type == 2
