import os, re, torch, glob
import pandas as pd
from collections import defaultdict, OrderedDict
from mmsc.datasets.base_dataset import BaseDataset
from mmsc.utils.dataset import load_video_vc

_CONSTANTS = {
    'dataset_name': 'voxceleb1', 
    'wave_ext': 'wav', 
    'face_ext': 'jpg', 
    'wave_folder': 'wav', 
    'video_folder': 'aligned_unzippedFaces', 
    'video_list': 'train.list', 
    'test_list': 'veri_test.txt', 
    'meta_file': 'vox1_meta.csv', 
}

class VoxCeleb1Dataset(BaseDataset):

    def __init__(self, config, dataset_type, *args, **kwargs):
        super().__init__(_CONSTANTS['dataset_name'], config, dataset_type)

        self._data_dir = config.data_dir
        self._max_frames = config.max_frame
        self._snippet_length = config.snippet_length
        
        if not os.path.exists(self._data_dir):
            raise RuntimeError(
                f"Data folder {self._data_dir} for VoxCeleb1 does not exist."
            )

        self._processors_map = []
        processor_config = config.get('processors', None)
        if processor_config is not None:
            self._init_processor_map(processor_config)

    def __getitem__(self, indices):
        if self.dataset_type == 'train':
            samples = [[self._get_one_item_(index) for index in pair] for pair in indices]
        else:
            samples = [[self._get_one_item_(index) for index in self.test_indices[indices]]]
        return samples

    def __len__(self):
        if self.dataset_type == 'train':
            return self.num_speakers
        else:
            return len(self.test_indices)
    
    def _get_one_item_(self, index):
        sample = load_video_vc(self.idx_db[index], 
                                self._snippet_length, 
                                self._max_frames)

        for processor_name in self.processor_map:
            processor = getattr(self, processor_name)
            sample = processor(sample)

        sample.targets = self._get_target(index)
        sample.meta.update(self._get_meta(index))

        return sample

    def load(self):
        if self.dataset_type in ['train']:
            self._load_dataset()
        elif self.dataset_type == 'val':
            self._load_val_set()
        else:
            raise RuntimeError("{} {}".format(_CONSTANTS['dataset_name'], 
                              self.dataset_type), "set not found")

    def _load_dataset(self):
        video_path = os.path.join(self._data_dir, 
                                  _CONSTANTS['video_folder'])
        wave_path = os.path.join(self._data_dir, 
                                  _CONSTANTS['wave_folder'])
        meta_file = os.path.join(self._data_dir, 
                                  _CONSTANTS['meta_file'])

        if self.dataset_type == 'train':
            video_list = os.path.join(self._data_dir, 
                                      _CONSTANTS['video_list'])
        elif self.dataset_type == 'val':
            video_list = os.path.join(self._data_dir, 
                                      _CONSTANTS['test_list'])
        else:
            raise TypeError(f'invalid {self.dataset_type} dataset type')

        self._pid_db_group = defaultdict(list)

        df = pd.read_csv(meta_file, sep='\t')
        pid2name = {d['VoxCeleb1 ID']: d['VGGFace1 ID'] for _, d in df.iterrows()}

        with open(video_list, 'r') as f:
            lines = f.readlines()
        
        pattern = re.compile(r'.*?(id\d{5})/(.*?)/(\d{5}).wav\n')
        self._idx_db = []
        self._pid_label = []
        
        for i, line in enumerate(lines):
            if not line.strip().endswith(_CONSTANTS['wave_ext']):
                continue
            [pid, url, clip] = list(pattern.search(line).groups())

            key = '{}.{}'.format(clip, _CONSTANTS['wave_ext'])
            wave = '{}/{}/{}/{}'.format(wave_path, pid, 
                                        url, key)
            faces = '{}/{}/1.6/{}/*.{}'.format(video_path, pid2name[pid], 
                                                       url, _CONSTANTS['face_ext'])

            self._idx_db.append((wave, faces, key, pid))
            self._pid_label.append(pid)
            self._pid_db_group[pid].append(i)
        
        self._pid_label = sorted(list(set(self._pid_label)))
        self._pid_label = { pid: i for i, pid in enumerate(self._pid_label) }
    
    def _load_val_set(self):
        test_list = '{}/{}'.format(
                            self._data_dir, 
                            _CONSTANTS['test_list'])

        video_path = os.path.join(self._data_dir, 
                                  _CONSTANTS['video_folder'])
        wave_path = os.path.join(self._data_dir, 
                                  _CONSTANTS['wave_folder'])
        meta_file = os.path.join(self._data_dir, 
                                  _CONSTANTS['meta_file'])

        df = pd.read_csv(meta_file, sep='\t')
        pid2name = {d['VoxCeleb1 ID']: d['VGGFace1 ID'] for _, d in df.iterrows()}

        self._idx_db = []
        self._pid_label = []
        pattern = re.compile(r'(id\d{5})/(.*?)/(\d{5}).wav')

        with open(test_list, 'r') as f:
            test_list = f.readlines()

        for line in test_list:
            pair = line.split()
            
            for p in pair[1:]:
                [pid, url, clip] = list(pattern.search(p).groups())

                key = '{}.{}'.format(clip, _CONSTANTS['wave_ext'])
                wave = '{}/{}/{}/{}'.format(wave_path, pid, 
                                            url, key)
                faces = '{}/{}/1.6/{}/*.{}'.format(video_path, pid2name[pid], 
                                                   url, _CONSTANTS['face_ext'])
                self._idx_db.append((wave, faces, key, pid))
                self._pid_label.append(pid)
        
        self._pid_label = sorted(list(set(self._pid_label)))
        self._pid_label = { pid: i for i, pid in enumerate(self._pid_label) }
        self.test_indices = torch.arange(len(self._idx_db)).reshape(-1, 2).tolist()
    
    def _get_target(self, index):
        pid = self.idx_db[index][-1]
        return torch.LongTensor([self.pid_label[pid]])

    def _get_meta(self, index):
        meta = {'identifier':'/'.join(self.idx_db[index][0].split('/')[-3:])}
        return meta

    def _init_processor_map(self, config):
        '''
        init processor map from config
        '''
        for p in OrderedDict(config).keys():
            if self.dataset_type != 'train' and 'augment' in p:
                continue
            self._processors_map.append(p)
    
    @property
    def num_speakers(self):
        return len(self.pid_db_group.keys())

    @property
    def pid_db_group(self):
        return self._pid_db_group
    
    @property
    def idx_db(self):
        return self._idx_db
    
    @property
    def pid_label(self):
        return self._pid_label
    
    @property
    def processor_map(self):
        return self._processors_map