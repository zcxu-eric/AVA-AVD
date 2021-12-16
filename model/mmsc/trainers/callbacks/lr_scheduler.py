# Copyright (c) Facebook, Inc. and its affiliates.
from mmsc.trainers.callbacks.base import Callback
from mmsc.utils.build import build_scheduler


class LRSchedulerCallback(Callback):
    """Callback which executes a LR scheduler. It is executed after every
    batch iteration.
    """

    def __init__(self, config, trainer):
        """
        Attr:
            config(mmsc_typings.DictConfig): Config for the callback
            trainer(Type[BaseTrainer]): Trainer object
        """
        super().__init__(config, trainer)

        self._scheduler = None
        if self.training_config.lr_scheduler is True:
            self._scheduler = build_scheduler(trainer.optimizer, self.config)

    def on_update_end(self, **kwargs):
        if self._scheduler is not None:
            self._scheduler.step()
