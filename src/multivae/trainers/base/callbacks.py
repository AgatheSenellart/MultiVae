"""Training Callbacks for training monitoring integrated in `pythae` (inspired from
https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_callback.py).
"""

import importlib
import json
import logging
import os
import warnings
from typing import Literal

import numpy as np
from pythae.models.base.base_config import BaseConfig
from tqdm.auto import tqdm

from .base_trainer_config import BaseTrainerConfig

logger = logging.getLogger(__name__)


def wandb_is_available():
    """Check if wandb logger is available."""
    return importlib.util.find_spec("wandb") is not None


def load_wandb_path_from_folder(path):
    """To load the wandb_path from a trained model."""
    with open(os.path.join(path, "wandb_info.json")) as fp:
        wandb_info = json.load(fp)

        return wandb_info["path"]


def rename_logs(logs):
    """Renames the logs train_metric to train/metric, which is more
    suited for wandb.
    """
    train_prefix = "train_"
    eval_prefix = "eval_"

    clean_logs = {}

    for metric_name in logs.keys():
        if metric_name.startswith(train_prefix):
            clean_logs[metric_name.replace(train_prefix, "train/")] = logs[metric_name]

        if metric_name.startswith(eval_prefix):
            clean_logs[metric_name.replace(eval_prefix, "eval/")] = logs[metric_name]

    return clean_logs


class TrainingCallback:
    """Base class for creating training callbacks."""

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called at the end of the initialization of the [`Trainer`]."""

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called at the beginning of training."""

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called at the end of training."""

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called at the beginning of an epoch."""

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called at the end of an epoch."""

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called at the beginning of a training step."""

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called at the end of a training step."""

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called at the beginning of a evaluation step."""

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called at the end of a evaluation step."""

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called after an evaluation phase."""

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called after a prediction phase."""

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called after a checkpoint save."""

    def on_save_checkpoint(self, training_config: BaseTrainerConfig, **kwargs):
        """Event called after a checkpoint save."""

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        """Event called after logging the last logs."""


class CallbackHandler:
    """Class to handle list of Callback."""

    def __init__(self, callbacks, model):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks but one is already used."
                f" The current list of callbacks is\n: {self.callback_list}"
            )
        self.callbacks.append(cb)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_init_end", training_config, **kwargs)

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_step_begin", training_config, **kwargs)

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_step_end", training_config, **kwargs)

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_eval_step_begin", training_config, **kwargs)

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_eval_step_end", training_config, **kwargs)

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_begin", training_config, **kwargs)

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_end", training_config, **kwargs)

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_epoch_begin", training_config, **kwargs)

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_epoch_end", training_config)

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_evaluate", **kwargs)

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_save", training_config, **kwargs)

    def on_save_checkpoint(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_save_checkpoint", training_config, **kwargs)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        self.call_event("on_log", training_config, logs=logs, **kwargs)

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_prediction_step", training_config, **kwargs)

    def call_event(self, event, training_config, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event)(
                training_config,
                model=self.model,
                **kwargs,
            )


class MetricConsolePrinterCallback(TrainingCallback):
    """A :class:`TrainingCallback` printing the training logs in the console."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # make it print to the console.
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        logger = kwargs.pop("logger", self.logger)
        rank = kwargs.pop("rank", -1)

        if logger is not None and (rank == -1 or rank == 0):
            epoch_train_loss = logs.get("train_epoch_loss", None)
            epoch_eval_loss = logs.get("eval_epoch_loss", None)

            logger.info(
                "--------------------------------------------------------------------------"
            )
            if epoch_train_loss is not None:
                logger.info(f"Train loss: {np.round(epoch_train_loss, 4)}")
            if epoch_eval_loss is not None:
                logger.info(f"Eval loss: {np.round(epoch_eval_loss, 4)}")
            logger.info(
                "--------------------------------------------------------------------------"
            )


class ProgressBarCallback(TrainingCallback):
    """A :class:`TrainingCallback` printing the training progress bar."""

    def __init__(self):
        self.train_progress_bar = None
        self.eval_progress_bar = None

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        epoch = kwargs.pop("epoch", None)
        train_loader = kwargs.pop("train_loader", None)
        rank = kwargs.pop("rank", -1)
        if train_loader is not None:
            if rank == 0 or rank == -1:
                self.train_progress_bar = tqdm(
                    total=len(train_loader),
                    unit="batch",
                    desc=f"Training of epoch {epoch}/{training_config.num_epochs}",
                )

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        epoch = kwargs.pop("epoch", None)
        eval_loader = kwargs.pop("eval_loader", None)
        rank = kwargs.pop("rank", -1)
        if eval_loader is not None:
            if rank == 0 or rank == -1:
                self.eval_progress_bar = tqdm(
                    total=len(eval_loader),
                    unit="batch",
                    desc=f"Eval of epoch {epoch}/{training_config.num_epochs}",
                )

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        if self.train_progress_bar is not None:
            self.train_progress_bar.update(1)

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        if self.eval_progress_bar is not None:
            self.eval_progress_bar.update(1)

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwags):
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()

        if self.eval_progress_bar is not None:
            self.eval_progress_bar.close()


class WandbCallback(TrainingCallback):  # pragma: no cover
    """A :class:`TrainingCallback` integrating the experiment tracking tool
    `wandb` (https://wandb.ai/).

    It allows users to store their configs, monitor their trainings
    and compare runs through a graphic interface. To be able use this feature you will need:

        - a valid `wandb` account
        - the package `wandb` installed in your virtual env. If not you can install it with

        .. code-block::

            $ pip install wandb

        - to be logged in to your wandb account using

        .. code-block::

            $ wandb login

    """

    def __init__(self):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )

        else:
            import wandb

            self._wandb = wandb

    def setup(
        self,
        training_config: BaseTrainerConfig,
        model_config: BaseConfig = None,
        project_name: str = "multivae_experiment",
        entity_name: str = None,
        run_id: str = None,
        resume: Literal[
            "allow",
            "must",
        ] = "allow",
        **kwargs,
    ):
        """Setup the WandbCallback.

        Args:
            training_config (BaseTrainerConfig): The training configuration used in the run.

            model_config (BaseMultiVAEConfig): The model configuration used in the run.

            project_name (str): The name of the wandb project to use.

            entity_name (str): The name of the wandb entity to use.

            run_id (str): If resume training, the id of the existing wandb_run

            resume (Literal) : wether to log on the provided run_id. Default to 'allow'.
        """
        self.is_initialized = True

        training_config_dict = training_config.to_dict()

        self.run = self._wandb.init(
            project=project_name, entity=entity_name, id=run_id, resume=resume
        )

        if model_config is not None:
            model_config_dict = model_config.to_dict()

            self._wandb.config.update(
                {
                    "training_config": training_config_dict,
                    "model_config": model_config_dict,
                }
            )

        else:
            self._wandb.config.update({**training_config_dict})

        self._wandb.define_metric("train/global_step")
        self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)
        logs = rename_logs(logs)

        self._wandb.log({**logs, "train/global_step": global_step})

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        kwargs.pop("global_step", None)

        reconstructions = kwargs.pop("reconstructions", None)

        for cond_mod in reconstructions:
            image = self._wandb.Image(reconstructions[cond_mod])
            self._wandb.log({"recon_from_" + cond_mod: image})

    def on_save_checkpoint(self, training_config: BaseTrainerConfig, **kwargs):
        checkpoint_dir = kwargs.pop("checkpoint_dir", None)
        if checkpoint_dir is None:
            raise AttributeError(
                "wandb callback on_save_checkpoint is called without"
                "a checkpoint directory information. Please provide checkpoint_dir=.."
            )
        with open(os.path.join(checkpoint_dir, "info_checkpoint.json"), "r") as fp:
            info_dict = json.load(fp)
            info_dict["wandb_run"] = self._wandb.run.id
        with open(os.path.join(checkpoint_dir, "info_checkpoint.json"), "w") as fp:
            json.dump(info_dict, fp)

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        dir_path = kwargs.pop("dir_path", None)
        if dir_path is None:
            warnings.warn(
                "wandb callback on_save is called without"
                "a  directory information. Please provide dir_path=.."
            )
            return
        info_dict = dict(
            entity_name=self.run.entity,
            project_name=self.run.project,
            id=self.run.id,
            path=self.run.path,
        )
        with open(os.path.join(dir_path, "wandb_info.json"), "w") as fp:
            json.dump(info_dict, fp)

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.run.finish()
