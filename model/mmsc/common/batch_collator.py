import torch
from mmsc.common.sample import SampleList, convert_batch_to_sample_list


class BatchCollator:
    def __init__(self, dataset_name, dataset_type):
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type

    def __call__(self, batch):
        sample_list = convert_batch_to_sample_list(batch)
        # sample_list = self._crop_video(sample_list)
        sample_list.dataset_name = self._dataset_name
        sample_list.dataset_type = self._dataset_type
        return sample_list
    
    def _crop_video(self, sample_list):
        meta = sample_list.meta
        len_keep = torch.randint(5, meta.slen[0]+1, [1])
        sample_list.frames = sample_list.frames[:, :, :len_keep, ...]
        sample_list.meta.slen = [len_keep.item()]*len(meta.slen)
        return sample_list


class FewShotBatchCollator:
    def __init__(self, dataset_name, dataset_type):
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type

    def __call__(self, batch):
        sample_list = []
        if isinstance(batch[0], list):
            for group in batch:
                sample_list.append(convert_batch_to_sample_list(group))
            sample_list = SampleList(sample_list)
        else:
            sample_list = convert_batch_to_sample_list(batch)
        sample_list.dataset_name = self._dataset_name
        sample_list.dataset_type = self._dataset_type
        return sample_list