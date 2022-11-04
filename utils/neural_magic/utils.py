from sparsezoo import Model
from re import search

__all__ = ["check_download_sparsezoo_weights"]

def check_download_sparsezoo_weights(path):
    # load model from the SparseZoo and override the path with the new download
    model = Model(path)
    file = _get_model_framework_file(model, path)
    path = file.path
    return path

def _get_model_framework_file(model, path):
    available_files = model.training.default.files
    transfer_request = search("recipe(.*)transfer", path)
    checkpoint_available = any([".ckpt" in file.name for file in available_files])
    final_available = any([not ".ckpt" in file.name for file in available_files])

    if transfer_request and checkpoint_available:
        # checkpoints are saved for transfer learning use cases,
        # return checkpoint if available and requested
        return [file for file in available_files if ".ckpt" in file.name][0]
    elif final_available:
        # default to returning final state, if available
        return [file for file in available_files if ".ckpt" not in file.name][0]

    raise ValueError(f"Could not find a valid framework file for {path}")