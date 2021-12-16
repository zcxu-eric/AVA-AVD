import math, pdb
from typing import Optional, Iterator
import warnings

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
from omegaconf import OmegaConf
from mmsc.datasets.samplers.base_sampler import BaseSampler
from mmsc.common.registry import registry


@registry.register_sampler('fewshot')
class FewShotSampler(BaseSampler):
    '''
    Sampler for matching and verification
    '''
    def __init__(self, dataset: Dataset, config: OmegaConf, 
                 batch_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:

        super().__init__(dataset, config, batch_size, 
                         shuffle=shuffle, seed=seed, drop_last=drop_last)

        if not hasattr(dataset, 'pid_db_group'):
            raise RuntimeError("Requires speaker groups")
        self.num_shot = config.get('num_shot', 1) + 1
        self.num_way = config.get('num_way', 20)
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
            
            if numvid < self.num_shot:
                randidx = randidx.repeat(math.ceil(self.num_shot/numvid))
            indices.append(torch.tensor(group)[randidx[:self.num_shot]].tolist()) 

        if self.shuffle:
            indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)].tolist()

        # divide into batch
        self.total_size = len(indices) - len(indices) % self.num_way
        indices = torch.tensor(indices[:self.total_size]).view(-1, self.num_way, self.num_shot).tolist()
        
        # expand test indices
        if self.dataset.dataset_type == 'val':
            indices = torch.tensor(indices).repeat(self.num_replicas * self.batch_size, 1, 1).tolist()

        # subsample
        num_batches = len(indices) - len(indices) % (self.num_replicas * self.batch_size)
        start_index = int((self.rank) / self.num_replicas * num_batches)
        end_index = int((self.rank + 1 ) / self.num_replicas * num_batches)
        indices = indices[start_index: end_index]
        self.total_size = num_batches

        return iter(indices)

    def __len__(self) -> int:
        return self.total_size


@registry.register_sampler('fewshot_ava')
class FewShotAVASampler(FewShotSampler):
    '''
    Sampler for matching and verification
    '''
    def __init__(self, dataset: Dataset, config: OmegaConf, 
                 batch_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:

        super().__init__(dataset, config, batch_size, 
                         shuffle=shuffle, seed=seed, drop_last=drop_last)

    def __iter__(self) -> Iterator:
        
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

        indices = []
        pid_db_group = self.dataset.pid_db_group
        labels = list(pid_db_group.keys())
        videos = list(set([label[:11] for label in labels]))
        
        for clip, batch_spk in self.dataset.vid_db_group.items():
            pos = videos.index(clip[:11])
            neg = videos[:pos] + videos[pos+1:]
            neg_spk = []
            for k in neg:
                for i in range(3):
                    neg_clip = f'{k}_c_0{i}'
                    if neg_clip in self.dataset.vid_db_group:
                        neg_spk.extend(self.dataset.vid_db_group[neg_clip])
            randidx = torch.randperm(len(neg_spk), generator=g).tolist()
            batch_spk.extend([neg_spk[i] for i in randidx[:self.num_way]])
            batch = [self._sample_within_video(label, generator=g) for label in batch_spk[:self.num_way]]
            indices.append(batch)

        if self.shuffle:
            indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)].tolist()
        
        # # expand test indices
        # if self.dataset.dataset_type == 'val':
        #     indices = torch.tensor(indices).repeat(self.num_replicas * self.batch_size, 1, 1).tolist()

        # subsample
        num_batches = len(indices) - len(indices) % (self.num_replicas * self.batch_size)
        start_index = int((self.rank) / self.num_replicas * num_batches)
        end_index = int((self.rank + 1 ) / self.num_replicas * num_batches)
        indices = indices[start_index: end_index]
        self.total_size = num_batches

        return iter(indices)
    
    def _sample_within_video(self, label, generator):
        group = self.dataset.pid_db_group[label]
        if len(group) % 2 != 0:
            k = torch.randint(0, len(group), [1], generator=generator).item()
            group.append(group[k])
        numvid = len(group)
        randidx = torch.randperm(numvid, generator=generator) if self.shuffle else torch.arange(numvid)
        
        if numvid < self.num_shot:
            randidx = randidx.repeat(math.ceil(self.num_shot/numvid))
        return torch.tensor(group)[randidx[:self.num_shot]].tolist()