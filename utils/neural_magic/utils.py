from re import search

from sparsezoo import Model

from utils.torch_utils import ModelEMA

__all__ = ["attempt_sparsezoo_download", "ToggleableModelEMA", "load_ema"]


class ToggleableModelEMA(ModelEMA):
    def __init__(self, enabled, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = enabled

    def update(self, *args, **kwargs):
        if self.enabled:
            super().update(*args, **kwargs)


def attempt_sparsezoo_download(path):
    # load model from the SparseZoo and override the path with the new download
    model = Model(path)
    file = _get_model_framework_file(model, path)
    path = file.path
    return path


def _get_model_framework_file(model, path):
    available_files = model.training.default.files
    transfer_request = search("recipe(.*)transfer", path)
    checkpoint_available = any([".ckpt" in file.name for file in available_files])
    final_available = any([".ckpt" not in file.name for file in available_files])

    if transfer_request and checkpoint_available:
        # checkpoints are saved for transfer learning use cases,
        # return checkpoint if available and requested
        return [file for file in available_files if ".ckpt" in file.name][0]
    elif final_available:
        # default to returning final state, if available
        return [file for file in available_files if ".ckpt" not in file.name][0]

    raise ValueError(f"Could not find a valid framework file for {path}")


def load_ema(ema_state_dict, model, enabled=True, **ema_kwargs):
    ema = ToggleableModelEMA(enabled, model, **ema_kwargs)
    ema.ema.load_state_dict(ema_state_dict)
    return ema
