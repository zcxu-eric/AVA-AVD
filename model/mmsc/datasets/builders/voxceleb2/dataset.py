import os, re, torch
from collections import defaultdict, OrderedDict
from mmsc.datasets.base_dataset import BaseDataset
from mmsc.utils.dataset import load_video

_CONSTANTS = {
    'dataset_name': 'voxceleb2', 
    'wave_ext': 'wav', 
    'face_ext': 'jpg', 
    'face_folder': 'aligned_faces', 
    'video_ext': 'mp4', 
    'video_folder': 'dev/mp4', 
    'video_list': 'train.list', 
    'test_list': 'test.list'
}

class VoxCeleb2Dataset(BaseDataset):

    def __init__(self, config, dataset_type, *args, **kwargs):
        super().__init__(_CONSTANTS['dataset_name'], config, dataset_type)

        self._data_dir = config.data_dir
        self._max_frames = config.max_frame
        self._snippet_length = config.snippet_length
        
        if not os.path.exists(self._data_dir):
            raise RuntimeError(
                f"Data folder {self._data_dir} for VoxCeleb2 does not exist."
            )

        self._processors_map = []
        processor_config = config.get('processors', None)
        if processor_config is not None:
            self._init_processor_map(processor_config)

    def __getitem__(self, indices):
        samples = [[self._get_one_item_(index) for index in pair] for pair in indices]
        return samples
    
    def __len__(self):
        return self.num_speakers
    
    def _get_one_item_(self, index):
        sample = load_video(self.idx_db[index], 
                            self._snippet_length, 
                            self._max_frames, 
                            compressed=False)

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
            raise RuntimeError("{} {}".format(_CONSTANTS['dataset_name'], 
                              self.dataset_type), "set not found")

    def _load_dataset(self):
        video_path = os.path.join(self._data_dir, 
                                  _CONSTANTS['video_folder'])
        if self.dataset_type == 'train':
            video_list = os.path.join(self._data_dir, 
                                      _CONSTANTS['video_list'])
        elif self.dataset_type == 'val':
            video_list = os.path.join(self._data_dir, 
                                      _CONSTANTS['test_list'])
        else:
            raise TypeError(f'invalid {self.dataset_type} dataset type')

        self._pid_db_group = defaultdict(list)

        with open(video_list, 'r') as f:
            lines = f.readlines()
        
        pattern = re.compile(r'.*?(id\d{5})/(.*?)/(\d{5}).mp4\n')
        self._idx_db = []
        self._pid_label = []
        
        for i, line in enumerate(lines):
            if not line.strip().endswith(_CONSTANTS['video_ext']):
                continue
            [pid, url, clip] = list(pattern.search(line).groups())

            key = '{}.{}'.format(clip, _CONSTANTS['video_ext'])
            video = '{}/{}/{}/{}'.format(video_path, pid, url, key)
            faces = '{}/{}/{}/{}/{}/*.{}'.format(self._data_dir, _CONSTANTS['face_folder'], pid, 
                                                 url, clip, _CONSTANTS['face_ext'])
            self._idx_db.append((video, faces, key, pid))
            self._pid_label.append(pid)
            self._pid_db_group[pid].append(i)
        
        self._pid_label = sorted(list(set(self._pid_label)))
        self._pid_label = { pid: i for i, pid in enumerate(self._pid_label) }
    
    def _load_test_dataset(self):
        video_path = os.path.join(self._data_dir, 
                                  _CONSTANTS['video_folder'])
        test_list =  os.path.join(self._data_dir, 
                                  _CONSTANTS['test_list'])
        with open(test_list, 'r') as f:
            pairs = f.readlines()
        keep = []
        self._idx_db = []
        for p in pairs:
            keep.extend(p.split()[1:])
        keep = list(set(keep))
        for db in keep:
            pid, url, key = db.split('/')
            video = '{}/{}/{}/{}'.format(video_path, pid, url, key)
            self._idx_db.append((video, key, pid))
    
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