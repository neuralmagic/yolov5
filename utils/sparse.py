import math
from copy import deepcopy
import random

from sparsezoo import Zoo
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import SparsificationGroupLogger
from sparseml.pytorch.utils import GradSampler

from utils.torch_utils import is_parallel
import torch.nn as nn

def _get_model_framework_file(model, path):
    transfer_request = 'recipe_type=transfer' in path
    checkpoint_available = any([file.checkpoint for file in model.framework_files])
    final_available = any([not file.checkpoint for file in model.framework_files])

    if transfer_request and checkpoint_available:
        # checkpoints are saved for transfer learning use cases,
        # return checkpoint if avaiable and requested
        return [file for file in model.framework_files if file.checkpoint][0]
    elif final_available:
        # default to returning final state, if available
        return [file for file in model.framework_files if not file.checkpoint][0]

    raise ValueError(f"Could not find a valid framework file for {path}")


def check_download_sparsezoo_weights(path):
    if isinstance(path, str):
        if path.startswith("zoo:"):
            # load model from the SparseZoo and override the path with the new download
            model = Zoo.load_model_from_stub(path)
            file = _get_model_framework_file(model, path)
            path = file.downloaded_path()

        return path

    if isinstance(path, list):
        return [check_download_sparsezoo_weights(p) for p in path]

    return path


class SparseMLWrapper(object):
    def __init__(self, model, checkpoint_recipe, train_recipe, steps_per_epoch=-1):
        self.enabled = bool(train_recipe)
        self.model = model.module if is_parallel(model) else model
        self.checkpoint_manager = ScheduledModifierManager.from_yaml(checkpoint_recipe) if checkpoint_recipe else None
        self.manager = ScheduledModifierManager.from_yaml(train_recipe) if train_recipe else None
        self.logger = None
        self.start_epoch = None
        self.steps_per_epoch = steps_per_epoch

    def state_dict(self):
        manager = (ScheduledModifierManager.compose_staged(self.checkpoint_manager, self.manager) 
        if self.checkpoint_manager and self.enabled else self.manager)

        return {
            'recipe': str(manager) if self.enabled else None,
        }

    def apply_checkpoint_structure(self):
        if self.checkpoint_manager:
            self.checkpoint_manager.apply_structure(self.model, math.inf)

    def initialize(
        self, 
        start_epoch, 
        compute_loss, 
        train_loader, 
        device, 
        **train_loader_kwargs
    ):
        if not self.enabled:
            return

        grad_sampler = GradSampler(
            self._mfac_data_loader(train_loader, device, **train_loader_kwargs), 
            lambda pred, target: compute_loss([p for p in pred[1]], target.to(device))[0]
        )

        self.manager.initialize(self.model, start_epoch, grad_sampler=grad_sampler)
        self.start_epoch = start_epoch

    def initialize_loggers(self, logger, tb_writer, wandb_logger):
        self.logger = logger

        if not self.enabled:
            return

        def _logging_lambda(tag, value, values, step, wall_time, level):
            if not wandb_logger or not wandb_logger.wandb:
                return

            if value is not None:
                wandb_logger.log({tag: value})

            if values:
                wandb_logger.log(values)

        self.manager.initialize_loggers([
            SparsificationGroupLogger(
                lambda_func=_logging_lambda,
                tensorboard=tb_writer,
            )
        ])

        if wandb_logger and wandb_logger.wandb:
            artifact = wandb_logger.wandb.Artifact('recipe', type='recipe')
            with artifact.new_file('recipe.yaml') as file:
                file.write(str(self.manager))
            wandb_logger.wandb.log_artifact(artifact)

    def modify(self, scaler, optimizer, model, dataloader):
        if not self.enabled:
            return scaler

        self.steps_per_epoch = self.steps_per_epoch if self.steps_per_epoch > 0 else len(dataloader)
        return self.manager.modify(model, optimizer, steps_per_epoch=self.steps_per_epoch, wrap_optim=scaler)

    def check_lr_override(self, scheduler, rank):
        # Override lr scheduler if recipe makes any LR updates
        if self.enabled and self.manager.learning_rate_modifiers:
            if rank in [0,-1]:
                self.logger.info('Disabling LR scheduler, managing LR using SparseML recipe')
            scheduler = None

        return scheduler

    def check_epoch_override(self, epochs, rank):
        # Override num epochs if recipe explicitly modifies epoch range
        if self.enabled and self.manager.epoch_modifiers and self.manager.max_epochs:
            if rank in [0,-1]:
                self.logger.info(f'Overriding number of epochs from SparseML manager to {epochs}')
            epochs = self.manager.max_epochs + self.start_epoch or epochs  # override num_epochs

        return epochs

    def qat_active(self, epoch):
        if not self.enabled or not self.manager.quantization_modifiers:
            return False

        qat_start = min([mod.start_epoch for mod in self.manager.quantization_modifiers])

        return qat_start < epoch + 1

    def reset_best(self, epoch):
        if not self.enabled:
            return False

        # if pruning is active or quantization just started, need to reset best checkpoint
        # this is in case the pruned and/or quantized model do not fully recover
        pruning_start = math.floor(max([mod.start_epoch for mod in self.manager.pruning_modifiers])) \
            if self.manager.pruning_modifiers else -1
        pruning_end = math.ceil(max([mod.end_epoch for mod in self.manager.pruning_modifiers])) \
            if self.manager.pruning_modifiers else -1
        qat_start = math.floor(max([mod.start_epoch for mod in self.manager.quantization_modifiers])) \
            if self.manager.quantization_modifiers else -1

        return (pruning_start <= epoch <= pruning_end) or epoch == qat_start

    def _mfac_data_loader(self, train_loader, device, multi_scale, imgsz, gs):
        def dataloader():
            loader_type = type(train_loader)
            mfac_dataloader = loader_type(
                dataset=train_loader.dataset,
                batch_size=train_loader.batch_size // 2,
                sampler=deepcopy(train_loader.sampler),
                num_workers=train_loader.num_workers,
                collate_fn=train_loader.collate_fn,
                pin_memory=train_loader.pin_memory,
            )

            for imgs, targets, *_ in mfac_dataloader:
                imgs = imgs.to(device, non_blocking=True).float() / 255
                if multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                yield [imgs], {}, targets
        return dataloader
