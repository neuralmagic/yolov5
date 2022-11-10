import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import torch
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import SparsificationGroupLogger

from utils.autobatch import check_train_batch_size

from utils.general import colorstr
from utils.loggers import Loggers

from utils.neural_magic.utils import load_ema
from utils.torch_utils import ModelEMA, de_parallel

__all__ = ["SparseTrainManager", "maybe_load_sparse_model"]

RANK = int(os.getenv("RANK", -1))

class SparseTrainManager(object):
    """
    Class for managing train state during sparse training with Neural Magic

    :param model: model to be trained
    :param train_recipe: yaml string or path to recipe to apply during training
    :param recipe_args: additional arguments to override any root variables
        in the recipe with (i.e. num_epochs, init_lr)
    :param checkpoint_recipe: yaml string or path to recipe previously used to create
        loaded model, if any
    :param last_epoch: last training epoch run for loaded model, relative to checkpoint
        recipe
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_recipe: str,
        recipe_args: Optional[Union[Dict[str, Any], str]],
        checkpoint_recipe: str = None,
        last_epoch: int = 0,
    ):
        # Recipes can be sensitive to module names, target correct submodule if parallel
        self.model = (
            model.module
            if (
                type(model)
                in (
                    torch.nn.parallel.DataParallel,
                    torch.nn.parallel.DistributedDataParallel,
                )
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
        self.train_manager = ScheduledModifierManager.from_yaml(
            file_path=train_recipe, recipe_variables=recipe_args
        )

        # Apply recipe structure from checkpoint recipe. Can include QAT and layer
        # thinning
        if self.checkpoint_manager:
            self.checkpoint_manager.apply_structure(
                self.model, last_epoch if last_epoch >= 0 else float("inf")
            )

    def initialize(
        self,
        loggers: Loggers,
        scaler: torch.cuda.amp.GradScaler,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        ema: ModelEMA,
        dataloader: torch.utils.data.DataLoader,
        start_epoch: int,
        epochs: int,
        ema_kwargs: Dict[str, Any] = {},
        resume: bool = False,
    ) -> Tuple[
        torch.cuda.amp.GradScaler, torch.optim.lr_scheduler._LRScheduler, ModelEMA, int
    ]:
        """
        Update objects controlling the training process for sparse training
        """
        # Wrap model for sparse training modifiers from recipe
        self.train_manager.initialize(module=self.model, epoch=start_epoch)

        # initialize SparseML loggers, including recipe modifier loggers
        self.initialize_loggers(loggers)
        self.log_console_info(
            "Sparse training detected. Wrapping training process with SparseML"
        )

        # If resumed run, apply recipe structure up to last epoch run. Structure can
        # include QAT and layer thinning
        if resume:
            self.train_manager.apply_structure(self.model, start_epoch - 1)

        # Wrap the scaler for sparse training modifiers from recipe
        scaler = self.train_manager.modify(
            self.model, optimizer, steps_per_epoch=len(dataloader), wrap_optim=scaler
        )

        # If recipe contains lr modifiers, turn off native lr scheduler
        if self.train_manager.learning_rate_modifiers:
            scheduler = None
            self.log_console_info(
                "Disabling LR scheduler, managing LR using SparseML recipe"
            )

        # If recipe contains epoch range modifiers, overwrite epoch range
        if self.train_manager.epoch_modifiers and self.train_manager.max_epochs:
            epochs = self.train_manager.max_epochs
            self.log_console_info(
                "Overriding total number of training epochs with value from recipe: "
                f"{epochs}"
            )

        # construct a ToggleableModelEMA from ModelEMA, allowing for disabling for QAT
        ema = load_ema(ema.ema.state_dict(), self.model, **ema_kwargs)

        return scaler, scheduler, ema, epochs

    def initialize_loggers(self, loggers: Loggers):
        """
        Initialize SparseML console, wandb, and tensorboard loggers from YOLOv5 loggers
        """
        # Console logger
        self.logger = loggers.logger

        # For logging sparse training values (e.g. sparsity %, custom lr schedule, etc.)
        def _logging_lambda(tag, value, values, step, wall_time, level):
            if not loggers.wandb or not loggers.wandb.wandb:
                return

            if value is not None:
                loggers.wandb.log({tag: value})

            if values:
                loggers.wandb.log(values)

        self.train_manager.initialize_loggers(
            [
                SparsificationGroupLogger(
                    lambda_func=_logging_lambda,
                    tensorboard=loggers.tb,
                )
            ]
        )

        # Attach recipe to wandb log
        if loggers.wandb and loggers.wandb.wandb:
            artifact = loggers.wandb.wandb.Artifact("recipe", type="recipe")
            with artifact.new_file("recipe.yaml") as file:
                file.write(str(self.train_manager))
            loggers.wandb.wandb.log_artifact(artifact)

    def log_console_info(self, message: str):
        if RANK in [0, -1]:
            self.logger.info(f"{colorstr('Neural Magic: ')}{message}")

    def qat_active(self, epoch: int) -> bool:
        """
        Returns true if QAT is turned on for the given epoch

        :param epoch: epoch to check QAT status for
        """
        if self.train_manager.quantization_modifiers:
            qat_start = min(
                [mod.start_epoch for mod in self.train_manager.quantization_modifiers]
            )
            return qat_start < epoch + 1
        else:
            return False

    def is_qat_recipe(self) -> bool:
        """
        Returns true if the training recipe contains a QAT modifier
        """
        return bool(self.train_manager.quantization_modifiers)

    def turn_off_scaler(self, scaler: torch.cuda.amp.GradScaler):
        """
        Turns off grad scaler

        :param scaler: scaler to run off
        """
        scaler._enabled = False

    def rescale_gradient_accumulation(
        self, batch_size: int, accumulate: int, image_size: int
    ) -> Tuple[int, int]:
        """
        Used when autobatch and QAT are both enabled. Training with QAT adds additional
        overhead which can cause OOM errors if autobatch is enabled. This function
        rescales batch size and gradient accumulation to fit into memory with QAT while
        maintaining the original effective batch size
        """
        # Temporary copy of the model with QAT applied
        quant_model_copy = deepcopy(de_parallel(self.model))
        self.train_manager.apply_structure(quant_model_copy, float("inf"))

        # Calculate maximum batch size that will fit in memory
        new_batch_size = check_train_batch_size(quant_model_copy, image_size, False)

        # Calculate batch size closest to maximum that can be accumulated to maintain
        # the original effective batch size. Note that if the original batch size is odd
        # then the effective batch size will be incremented by 1
        batch_size = batch_size if batch_size % 2 == 0 else batch_size + 1
        batch_size_ratio = math.floor(batch_size / new_batch_size)
        closest_divisor = next(
            divisor
            for divisor in range(batch_size_ratio, 1, -1)
            if batch_size % divisor == 0
        )
        new_batch_size = batch_size // closest_divisor
        new_accumulate = math.floor((batch_size * accumulate) // new_batch_size)
        new_batch_size = batch_size // new_accumulate

        self.log_console_info(
            f"Batch size rescaled to {new_batch_size} with {new_accumulate} gradient "
            "accumulation steps for QAT"
        )

        if new_accumulate * new_batch_size != batch_size * accumulate:
            raise RuntimeError(
                "New effective batch size doesn't match previous effective batch size. "
                f"Previous batch size and accumulation: {[batch_size, accumulate]}. "
                f"New batch size and accumulation: {[new_batch_size, new_accumulate]}"
            )

        return new_batch_size, new_accumulate

    def update_state_dict_for_saving(
        self, ckpt: Dict[str, Any], final_epoch: bool, ema_enabled: bool
    ) -> Dict[str, Any]:
        """
        Update checkpoint dictionary to be compatible with sparse model saving

        :param ckpt: original checkpoint dictionary
        :param final_epoch: True if called after last training epoch
        :param ema_enabled: True if ema is turned on
        """
        if final_epoch:
            # save model with a checkpoint recipe representing all recipes applied to
            # model, allowing for multiple stages of sparse training
            checkpoint_recipe = (
                ScheduledModifierManager.compose_staged(
                    self.checkpoint_manager, self.train_manager
                )
                if self.checkpoint_manager
                else self.train_manager
            )
        else:
            checkpoint_recipe = None

        # Pickling is not supported for quantized models for a subset of the supported
        # torch versions, thus all sparse models are saved via their state dict
        sparseml_dict_update = {
            "model": ckpt["model"].state_dict(),
            "yaml": ckpt["model"].yaml,
            "ema": ckpt["ema"].state_dict() if ema_enabled else None,
            "updates": ckpt["updates"] if ema_enabled else None,
            "checkpoint_recipe": str(checkpoint_recipe),
            "epoch": -1 if final_epoch else ckpt["epoch"],
        }
        ckpt.update(sparseml_dict_update)

        return ckpt


def maybe_load_sparse_model(
    model: torch.nn.Module,
    ckpt: Dict[str, Any],
    train_recipe: str,
    recipe_args: Optional[Union[Dict[str, Any], str]],
):
    """
    If sparse training or checkpoint detected, load sparse model and return
    SparseTrainManager object. Otherwise do nothing.

    :param model: skeleton model
    :param ckpt: loaded checkpoint
    :param train_recipe: yaml string or path to recipe to apply during training
    :param recipe_args: additional arguments to override any root variables
        in the recipe with (i.e. num_epochs, init_lr)
    """
    if ckpt.get("checkpoint_recipe") or train_recipe:

        # reconstruct ToggleableModelEMA from state dictionary
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
