from sparseml.pytorch.optim import ScheduledModifierManager
import torch
from typing import Optional, Union, Dict, Any, Tuple

from utils.neural_magic.utils import load_ema
from utils.torch_utils import ModelEMA

__all__ = ["SparseTrainManager", "maybe_load_sparse_model"]


class SparseTrainManager(object):
    """
    Class for managing train state during sparse training with Neural Magic

    :param model: model to be trained
    :param train_recipe: yaml string or path to recipe to apply during training
    :param checkpoint_recipe: yaml string or path to recipe previously used to create 
        loaded model, if any
    :param last_epoch: last training epoch run for loaded model, relative to checkpoint
        recipe
    """
    def __init__(
        self, model: torch.nn.Module, train_recipe: str, recipe_args: Optional[Union[Dict[str, Any], str]], checkpoint_recipe: str=None, last_epoch: int=0
    ):
        # Recipes can be sensitive to module names, target correct submodule if parallel
        self.model = (
            model.module
            if (
                type(model)
                in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)
            )
            else model
        )

        # Training manager created from checkpoint recipe, if any 
        self.checkpoint_manager = (
            ScheduledModifierManager.from_yaml(checkpoint_recipe)
            if checkpoint_recipe
            else None
        )

        # Training manager for current training run 
        self.train_manager = (
            ScheduledModifierManager.from_yaml(
                file_path=train_recipe, recipe_variables=recipe_args
            )
        )

        # Apply recipe structure from checkpoint recipe. Can include QAT and layer
        # thinning
        if self.checkpoint_manager:
            self.checkpoint_manager.apply_structure(
                self.model, last_epoch if last_epoch > -1 else float("inf")
            )

    def initialize(
        self,
        scaler: torch.cuda.amp.GradScaler,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        ema: ModelEMA,
        dataloader: torch.utils.data.DataLoader,
        start_epoch: int,
        epochs: int,
        ema_kwargs: Dict[str, Any]={},
        resume: bool =False,
    ) -> Tuple[torch.cuda.amp.GradScaler, torch.optim.lr_scheduler._LRScheduler, ModelEMA, int]:
        """
        Update objects controlling the training process for sparse training
        """

        # If resumed run, apply recipe structure up to last epoch run. Structure can
        # include QAT and layer thinning
        if resume:
            self.train_manager.apply_structure(self.model, start_epoch-1)

        # Wrap model for sparse training modifiers from recipe
        self.train_manager.initialize(module=self.model, epoch=start_epoch)

        # Wrap the scaler for sparse training modifiers from recipe
        scaler = self.train_manager.modify(
            self.model, optimizer, steps_per_epoch=len(dataloader), wrap_optim=scaler
        )

        # If recipe contains lr modifiers, turn off native lr scheduler
        if self.train_manager.learning_rate_modifiers:
            scheduler = None

        # If recipe contains epoch range modifiers, overwrite epoch range
        if self.train_manager.epoch_modifiers and self.train_manager.max_epochs:
            epochs = self.train_manager.max_epochs

        # construct a ToggleableModelEMA from ModelEMA, allowing for disabling for QAT
        ema = load_ema(ema.ema.state_dict(), self.model, **ema_kwargs)

        return scaler, scheduler, ema, epochs

    def qat_active(self, epoch):
        if self.train_manager.quantization_modifiers:
            qat_start = min(
                [mod.start_epoch for mod in self.train_manager.quantization_modifiers]
            )
            return qat_start < epoch + 1
        else:
            return False

    def is_qat_recipe(self):
        return bool(self.train_manager.quantization_modifiers)

    def turn_off_scaler(self, scaler):
        scaler._enabled = False

    def update_state_dict_for_saving(self, ckpt, final_epoch, ema_enabled):
        if final_epoch:
            checkpoint_recipe = (
                ScheduledModifierManager.compose_staged(
                    self.checkpoint_manager, self.train_manager
                )
                if self.checkpoint_manager
                else self.train_manager
            )
        else:
            checkpoint_recipe = None

        sparseml_dict_update = {
            "model": ckpt["model"].state_dict(),
            "yaml": ckpt["model"].yaml,
            "ema": ckpt["ema"].state_dict() if ema_enabled else None,
            "updates": ckpt["updates"] if ema_enabled else None,
            "checkpoint_recipe": str(checkpoint_recipe),
        }
        ckpt.update(sparseml_dict_update)

        return ckpt


def maybe_load_sparse_model(model, ckpt, train_recipe, recipe_args):
    if ckpt.get("checkpoint_recipe") or train_recipe:

        if not train_recipe:
            print("warning here")

        # reconstruct ToggleableModelEMA from saved state dictionary
        if ckpt["ema"]:
            ckpt["ema"] = load_ema(ckpt["ema"], model)

        sparse_manager = SparseTrainManager(
            model,
            train_recipe,
            recipe_args,
            ckpt.get("checkpoint_recipe"),
            ckpt["epoch"],
        )
        return sparse_manager

    else:
        return None
