import math, pdb
from typing import Optional, Iterator
import warnings

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
from omegaconf import OmegaConf
from mmsc.datasets.samplers.base_sampler import BaseSampler
from mmsc.common.registry import registry

from torch.utils.data import DataLoader
@registry.register_sampler('avaactive')
class AVAActiveSampler(BaseSampler):
    '''
    Sampler for AVA speaker or face verification
    '''
    def __init__(self, dataset: Dataset, config: OmegaConf, 
                 batch_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:

        super().__init__(dataset, config, batch_size, 
                         shuffle=shuffle, seed=seed, drop_last=drop_last)

        if not hasattr(dataset, 'idx_db'):
            raise RuntimeError("Annotation database not found")

        self.num_samples = len(dataset.idx_db)

    def __iter__(self) -> Iterator:
        
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

        indices = []  # type: ignore
        batch_indices = []
        idx_db = self.dataset.idx_db
        
        for i, _ in enumerate(idx_db):
            batch_indices.append(i)
            if len(batch_indices) % self.batch_size == 0:
                if self.shuffle:
                    idx = torch.randperm(self.batch_size, generator=g)
                    batch_indices = torch.tensor(batch_indices)[idx].tolist()
                indices.append(batch_indices)
                batch_indices = []

        indices = torch.tensor(indices)

        if self.shuffle:
            randidx = torch.randperm(len(indices), generator=g)
            indices = indices[randidx, :]
        
        num_batches = len(indices) - len(indices) % self.num_replicas
        indices = indices[:num_batches, :].view(-1, 1).tolist()

        self.total_size = len(indices)

        # subsample
        start_index = int((self.rank) / self.num_replicas * self.total_size)
        end_index = int((self.rank + 1 ) / self.num_replicas * self.total_size)
        indices = indices[start_index: end_index]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples