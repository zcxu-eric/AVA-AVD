import math, pdb
from typing import Optional, Iterator
import warnings
import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
from omegaconf import OmegaConf
from mmsc.datasets.samplers.base_sampler import BaseSampler
from mmsc.common.registry import registry


@registry.register_sampler('voxceleb')
class VoxCelebSampler(BaseSampler):
    '''
    Sampler for matching and verification
    '''
    def __init__(self, dataset: Dataset, config: OmegaConf, 
                 batch_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:

        super().__init__(dataset, config, batch_size, 
                         shuffle=shuffle, seed=seed, drop_last=drop_last)

        if not hasattr(dataset, 'num_speakers'):
            raise RuntimeError("Requires number of speakers")
        if not hasattr(dataset, 'pid_db_group'):
            raise RuntimeError("Requires speaker groups")
        self.num_instances = config.get('num_instances', float('inf'))
        self.num_samples = len(dataset.idx_db)
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and  self.num_samples % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = int(math.floor(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                self.num_samples / self.num_replicas)) # type: ignore
        else:
            self.num_samples = int(math.ceil(self.num_samples / self.num_replicas)
                                ) # type: ignore
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator:
        
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

        indices = []  # type: ignore
        pid_db_group = self.dataset.pid_db_group
        labels = list(pid_db_group.keys())
        
        for label in labels:
            group = pid_db_group[label]
            if len(group) % 2 != 0:
                k = torch.randint(0, len(group), [1], generator=g).item()
                group.append(group[k])
            numvid = len(group)
            randidx = torch.randperm(numvid, generator=g) if self.shuffle else torch.arange(numvid)
            if self.num_instances != float('inf'):
                if numvid < self.num_instances:
                    randidx = randidx.repeat(math.ceil(self.num_instances/numvid))
                indices.append(torch.tensor(group)[randidx[:self.num_instances]].view(-1, 2).tolist())
            else:
                indices.extend(torch.tensor(group)[randidx].view(-1, 2).tolist())

        if self.shuffle:
            indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)].tolist()

        batch_label = []
        batch_indices = []

        # remove dupicate speakers in one batch
        for pair in indices:
            startbatch = len(batch_label) - len(batch_label) % self.batch_size
            label = self.dataset.idx_db[pair[0]][-1]
            if label not in batch_label[startbatch:]:
                batch_label.append(label)
                batch_indices.append(pair)
        self.total_size = len(batch_indices) - len(batch_indices) % (self.num_replicas * self.batch_size)
        indices = batch_indices[:self.total_size]
        # subsample
        start_index = int((self.rank) / self.num_replicas * self.total_size)
        end_index = int((self.rank + 1 ) / self.num_replicas * self.total_size)
        indices = indices[start_index: end_index]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


@registry.register_sampler('ava')
class AVASampler(BaseSampler):
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
        if not hasattr(dataset, 'db_group'):
            raise RuntimeError("Requires speaker groups")
        self.num_instances = config.get('num_instances', float('inf'))
        self.num_samples = len(dataset.idx_db)
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and  self.num_samples % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = int(math.floor(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                self.num_samples / self.num_replicas)) # type: ignore
        else:
            self.num_samples = int(math.ceil(self.num_samples / self.num_replicas)
                                ) # type: ignore
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator:
        
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

        indices = []  # type: ignore
        db_group = self.dataset.db_group
        labels = list(db_group.keys())
        
        for label in labels:
            group = db_group[label]
            if len(group) % 2 != 0:
                k = torch.randint(0, len(group), [1], generator=g).item()
                group.append(group[k])
            numvid = len(group)
            randidx = torch.randperm(numvid, generator=g) if self.shuffle else torch.arange(numvid)
            if self.num_instances != float('inf'):
                if numvid < self.num_instances:
                    randidx = randidx.repeat(math.ceil(self.num_instances/numvid))
                indices.extend(torch.tensor(group)[randidx[:self.num_instances]].view(-1, 2).tolist())
            else:
                indices.extend(torch.tensor(group)[randidx].view(-1, 2).tolist())

        if self.shuffle:
            indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)].tolist()

        batch_label = []
        batch_indices = []

        # remove speakers from the same video in one batch
        for pair in indices:
            startbatch = len(batch_label) - len(batch_label) % self.batch_size
            label = self.dataset.idx_db[pair[0]][-1].split(':')[0]
            if '_c_0' in label:
                label = label[:-5]
            if label not in batch_label[startbatch:]:
                batch_label.append(label)
                batch_indices.append(pair)
        self.total_size = len(batch_indices) - len(batch_indices) % (self.num_replicas * self.batch_size)
        indices = batch_indices[:self.total_size]
        # subsample
        # total_size  = len(mixed_list) - len(mixed_list) % (self.batch_size * dist.get_world_size())
        start_index = int((self.rank) / self.num_replicas * self.total_size)
        end_index = int((self.rank + 1 ) / self.num_replicas * self.total_size)
        indices = indices[start_index: end_index]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples