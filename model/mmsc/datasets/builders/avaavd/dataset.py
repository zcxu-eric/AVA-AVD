import os, glob, torch, copy
import numpy as np
from collections import defaultdict, OrderedDict
from mmsc.common.sample import Sample, SampleList
from mmsc.common.registry import registry
from mmsc.datasets.builders.ava.dataset import AVADataset, _CONSTANTS

_CONSTANTS = {
    'dataset_name': 'avaavd', 
    'lab_folder': 'labs', 
    'wave_ext': 'wav', 
    'img_ext': 'jpg', 
    'wave_folder': 'denoised_waves', 
    'img_folder': 'aligned_tracklets', 
    'val_list': 'split/val.list', 
    'test_list': 'split/test.list', 
    'offsets': 'split/offsets.txt'
}

class AVAAVDDataset(AVADataset):
    '''AVA audiovisual diarization dataset'''
    def __init__(self, config, dataset_type, *args, **kwargs):
        super().__init__(config, dataset_type, _CONSTANTS['dataset_name'])
        self._missing_rate = config.missing_rate
    
    def __len__(self):
        return len(self.idx_db)

    def load(self):
        if self.dataset_type in ['val', 'test']:
            self._load_dataset()
        else:
            raise RuntimeError("{} {} set not found".format(
                                _CONSTANTS['dataset_name'], 
                                self.dataset_type))
    
    def _load_dataset(self): 
        if self.dataset_type == 'val':
            video_list = os.path.join(self._data_dir, 
                                      _CONSTANTS['val_list'])
        elif self.dataset_type == 'test':
            video_list = os.path.join(self._data_dir, 
                                      _CONSTANTS['test_list'])
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
            vad = os.path.join(self._data_dir, 
                                _CONSTANTS['lab_folder'], 
                                f'{video}.lab')

            with open(vad, 'r') as f:
                lines = f.readlines()

            faces = defaultdict(self._nested_dict)
            imgs = glob.glob(os.path.join(self._data_dir, 
                              _CONSTANTS['img_folder'], 
                              video[:11], f'{video[:11]}*{video[-2:]}spk*.jpg'))
            for img in imgs:
                track, id, ts, _, _ = os.path.basename(img)[:-4].split(':')
                trackid = track+':'+id
                faces[float(ts)][trackid].append(img)

            for line in lines:
                item = line.split()
                start = float(item[0])
                end = float(item[1])
                dbs = []
                ts = np.sort(np.array(list(faces.keys())))
                for seg_start in np.arange(start, end, step):
                    if (end - seg_start) >= mins:
                        dura = min(maxs, end - seg_start)
                        # load associated tracklets
                        keep = ts[np.searchsorted(ts, seg_start): np.searchsorted(ts, seg_start+dura)]
                        assocfaces = defaultdict(list)
                        for t in keep:
                            face = faces[t]
                            for k,v in face.items():
                                assocfaces[k].extend(v)

                        if len(assocfaces) > 0:
                            # visible speaker
                            for _, imgs in assocfaces.items():
                                # ignore faces if tracks are too short
                                if torch.rand([1]).item() < (1 - self._missing_rate):
                                    dbs.append((f'{wave_folder}/{video}.wav', seg_start, 
                                                dura, offsets[video], imgs, video))
                                else:
                                    dbs.append((f'{wave_folder}/{video}.wav', seg_start, 
                                                dura, offsets[video], [], video))
                        else:
                            # offscreen speaker
                            dbs.append((f'{wave_folder}/{video}.wav', seg_start, 
                                        dura, offsets[video], [], video))

                self._idx_db.extend(dbs)
    
    def _get_target(self, index):
        return

    def _get_meta(self, index):
        if len(self.idx_db[index][-2]) > 0:
            track, id, ts, _, _ = os.path.basename(self.idx_db[index][-2][0])[:-4].split(':')
            trackid = track + ':' + id
        else:
            trackid = 'NA'
        meta = {
            'video': self.idx_db[index][-1], 
            'start': self.idx_db[index][1],
            'end': self.idx_db[index][1] + self.idx_db[index][2], 
            'trackid': trackid
        }
        return meta