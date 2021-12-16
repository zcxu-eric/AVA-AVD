# Copyright (c) Facebook, Inc. and its affiliates.
"""
The processors exist in MMSC to make data processing pipelines in various
datasets as similar as possible while allowing code reuse.

The processors also help maintain proper abstractions to keep only what matters
inside the dataset's code. This allows us to keep the dataset ``__getitem__``
logic really clean and no need about maintaining opinions about data type.
Processors can work on both images and audio due to their generic structure.

To create a new processor, follow these steps:

1. Inherit the ``BaseProcessor`` class.
2. Implement ``_call`` function which takes in a dict and returns a dict with
   same keys preprocessed as well as any extra keys that need to be returned.
3. Register the processor using ``@registry.register_processor('name')`` to
   registry where 'name' will be used to refer to your processor later.

In processor's config you can specify ``preprocessor`` option to specify
different kind of preprocessors you want in your dataset.

``BaseDataset`` will init the processors and they will available inside your
dataset with same attribute name as the key name, for e.g. `text_processor` will
be available as `self.text_processor` inside your dataset. As is with every module
in MMSC, processor also accept a ``DictConfig`` with a `type` and `params`
attributes. `params` defined the custom parameters for each of the processors.
By default, processor initialization process will also init `preprocessor` attribute
which can be a processor config in itself. `preprocessor` can be then be accessed
inside the processor's functions.

Example::

    from mmsc.common.registry import registry
    from mmsc.datasets.processors import BaseProcessor

    @registry.register_processor('my_processor')
    class MyProcessor(BaseProcessor):
        def __init__(self, config, *args, **kwargs):
            return

        def __call__(self, item, *args, **kwargs):
            text = item['text']
            text = [t.strip() for t in text.split(" ")]
            return {"text": text}
"""

import collections
import copy
import logging
import os
import random
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from mmsc.common.registry import registry
from mmsc.common.typings import ProcessorConfigType
from mmsc.utils.configuration import get_mmsc_cache_dir, get_mmsc_env
from mmsc.utils.distributed import is_master, synchronize
from mmsc.utils.fileio import PathManager
from mmsc.utils.logger import log_class_usage
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


@dataclass
class BatchProcessorConfigType:
    processors: ProcessorConfigType


class BaseProcessor:
    """Every processor in MMSC needs to inherit this class for compatibility
    with MMSC. End user mainly needs to implement ``__call__`` function.

    Args:
        config (DictConfig): Config for this processor, containing `type` and
                             `params` attributes if available.

    """

    def __init__(self, config: Optional[DictConfig] = None, *args, **kwargs):

        log_class_usage("Processor", self.__class__)
        return

    def __call__(self, item: Any, *args, **kwargs) -> Any:
        """Main function of the processor. Takes in a dict and returns back
        a dict

        Args:
            item (Dict): Some item that needs to be processed.

        Returns:
            Dict: Processed dict.

        """
        return item


class Processor:
    """Wrapper class used by MMSC to initialized processor based on their
    ``type`` as passed in configuration. It retrieves the processor class
    registered in registry corresponding to the ``type`` key and initializes
    with ``params`` passed in configuration. All functions and attributes of
    the processor initialized are directly available via this class.

    Args:
        config (DictConfig): DictConfig containing ``type`` of the processor to
                             be initialized and ``params`` of that processor.

    """

    def __init__(self, config: ProcessorConfigType, *args, **kwargs):
        if "type" not in config:
            raise AttributeError(
                "Config must have 'type' attribute to specify type of processor"
            )

        processor_class = registry.get_processor_class(config.type)

        params = {}
        if "params" not in config:
            logger.warning(
                "Config doesn't have 'params' attribute to "
                "specify parameters of the processor "
                f"of type {config.type}. Setting to default {{}}"
            )
        else:
            params = config.params

        self.processor = processor_class(params, *args, **kwargs)

        self._dir_representation = dir(self)

    def __call__(self, item, *args, **kwargs):
        return self.processor(item, *args, **kwargs)

    def __getattr__(self, name):
        if "_dir_representation" in self.__dict__ and name in self._dir_representation:
            return getattr(self, name)
        elif "processor" in self.__dict__ and hasattr(self.processor, name):
            return getattr(self.processor, name)
        else:
            raise AttributeError(f"The processor {name} doesn't exist in the registry.")


class BatchProcessor(BaseProcessor):
    """BatchProcessor is an extension of normal processor which usually are
    used in cases where dataset works on full batch instead of samples.
    Such cases can be observed in the case of the iterable datasets.
    BatchProcessor if provided with processors key in the config, will
    initialize a member variable processors_dict for you which will contain
    initialization of all of the processors you specified and will need to process
    your complete batch.

    Rest it behaves in same way, expects an item and returns an item which can be
    of any type.
    """

    def __init__(self, config: BatchProcessorConfigType, *args, **kwargs):
        extra_params = {"data_dir": get_mmsc_env(key="data_dir")}
        processors_dict = config.get("processors", {})

        # Since build_processors also imports processor, import it at runtime to
        # avoid circular dependencies
        from mmsc.utils.build import build_processors

        self.processors = build_processors(processors_dict, **extra_params)

    def __call__(self, item: Any) -> Any:
        return item


@registry.register_processor("copy")
class CopyProcessor(BaseProcessor):
    """
    Copy boxes from numpy array
    """

    def __init__(self, config, *args, **kwargs):
        self.max_length = config.max_length

    def __call__(self, item):
        blob = item["blob"]
        final_blob = np.zeros((self.max_length,) + blob.shape[1:], blob.dtype)
        final_blob[: len(blob)] = blob[: len(final_blob)]

        return {"blob": torch.from_numpy(final_blob)}