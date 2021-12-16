import os, glob, torch
import numpy as np
from collections import defaultdict, OrderedDict
from mmsc.common.sample import Sample, SampleList
from mmsc.common.registry import registry
from mmsc.datasets.base_dataset import BaseDataset
from mmsc.utils.dataset import load_video_ava

_CONSTANTS = {
    'dataset_name': 'ava', 
    'wave_ext': 'wav', 
    'wave_folder': 'denoised_waves', 
    'img_ext': 'jpg', 
    'img_folder': 'aligned_tracklets', 
    'rttm_folder': 'rttms', 
    'database': 'database', 
    'train_list': 'split/train.list', 
    'val_list': 'split/val.list', 
    'offsets': 'split/offsets.txt'
}

class AVADataset(BaseDataset):

    def __init__(self, config, dataset_type, 
                 dataset_name=_CONSTANTS['dataset_name'], *args, **kwargs):
        super().__init__(dataset_name, config, dataset_type)
        self.config = config
        self._data_dir = config.data_dir
        self._max_frames = config.max_frame
        self._min_frames = config.min_frame
        self._step_frame = config.step_frame
        self._snippet_length = config.snippet_length
        
        if not os.path.exists(self._data_dir):
            raise RuntimeError(
                f"Data folder {self._data_dir} for AVA does not exist."
            )

        self._processors_map = []
        processor_config = config.get('processors', None)
        if processor_config is not None:
            self._init_processor_map(processor_config)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = [[indices]]
        samples = [[self._get_one_item_(index) for index in pair] for pair in indices]
        return samples
    
    def __len__(self):
        return self.num_speakers

    def _get_one_item_(self, index):
        sample = load_video_ava(self.idx_db[index], 
                                self._snippet_length, 
                                self._max_frames, 
                                crop=True)

        for processor_name in self.processor_map:
            processor = getattr(self, processor_name)
            sample = processor(sample)

        sample.targets = self._get_target(index)
        sample.meta.update(self._get_meta(index))
        return sample

    def load(self):
        if self.dataset_type in ['train', 'val']:
            self._load_dataset()
        else:
            raise RuntimeError("{} {} set not found".format(
                                _CONSTANTS['dataset_name'], 
                                self.dataset_type))
    
    def _load_dataset(self):
        if self.dataset_type == 'train':
            video_list = os.path.join(self._data_dir, 
                                        _CONSTANTS['train_list'])
        elif self.dataset_type == 'val':
            video_list = os.path.join(self._data_dir, 
                                        _CONSTANTS['val_list'])
        else:
            raise TypeError(f'invalid {self.dataset_type} dataset type')

        with open(video_list, 'r') as f:
            videos = f.readlines()
        
        offsets = os.path.join(self._data_dir, 
                               _CONSTANTS['offsets'])
        with open(offsets, 'r') as f:
            offsets = f.readlines()
        offsets = { d.split()[0]: float(d.split()[1]) for d in offsets}

        maxs = self._max_frames / 100.0
        mins = self._min_frames / 100.0
        step = self._step_frame / 100.0
        wave_folder = os.path.join(self._data_dir, _CONSTANTS['wave_folder'])

        self._idx_db = []
        self._pid_label = []
        self._vid_db_group = defaultdict(list)
        self._pid_db_group = defaultdict(list)
        
        # parse audio and face segments
        for video in videos:
            video = video.strip()
            rttm = os.path.join(self._data_dir, 
                                _CONSTANTS['rttm_folder'], 
                                f'{video}.rttm')

            with open(rttm, 'r') as f:
                lines = f.readlines()

            faces = defaultdict(self._nested_dict)
            imgs = glob.glob(os.path.join(self._data_dir, 
                              _CONSTANTS['img_folder'], 
                              video[:11], f'{video[:11]}*{video[-2:]}spk*.jpg'))
            for img in imgs:
                _, _, ts, _, spkid = os.path.basename(img)[:-4].split(':')
                faces[spkid[2:]][float(ts)] = img

            for line in lines:
                item = line.split()
                spkid = video + ':' + item[7]
                start = float(item[3])
                end = start + float(item[4])
                dbs = []
                ts = np.sort(np.array(list(faces[item[7]].keys())))
                for seg_start in np.arange(start, end, step):
                    if (end - seg_start) >= mins:
                        dura = min(maxs, end - seg_start)
                        keep = ts[np.searchsorted(ts, seg_start): np.searchsorted(ts, seg_start+dura)]
                        imgs = [faces[item[7]][k] for k in keep]
                        dbs.append((f'{wave_folder}/{video}.wav', seg_start, 
                                    dura, offsets[video], imgs, spkid))
                        self._pid_label.append(spkid)
                self._idx_db.extend(dbs)
        
        self._pid_label = sorted(list(set(self._pid_label)))
        self._pid_label = { pid: i for i, pid in enumerate(self._pid_label) }

        for i, db in enumerate(self._idx_db):
            self._pid_db_group[db[-1]].append(i)
        for k in self._pid_db_group:
            self._vid_db_group[k[:16]].append(k)
    
    def _get_target(self, index):
        pid = self.idx_db[index][-1]
        return torch.LongTensor([self.pid_label[pid]])

    def _get_meta(self, index):
        meta = {}
        return meta

    def _init_processor_map(self, config):
        '''
        init processor map from config
        '''
        for p in OrderedDict(config).keys():
            if self.dataset_type != 'train' and 'augment' in p:
                continue
            self._processors_map.append(p)
    
    def _nested_dict(self):
        return defaultdict(list)
    
    @property
    def idx_db(self):
        return self._idx_db
    
    @property
    def num_speakers(self):
        return len(self.pid_db_group.keys()) // 2
    
    @property
    def pid_label(self):
        return self._pid_label

    @property
    def pid_db_group(self):
        return self._pid_db_group
    
    @property
    def vid_db_group(self):
        return self._vid_db_group
    
    @property
    def processor_map(self):
        return self._processors_map