import av, gc, os, cv2, glob
import warnings
from typing import List
import torch
import soundfile
import numpy as np
from omegaconf import DictConfig
from io import BytesIO
from torch._C import default_generator
from torch.serialization import default_restore_location
from mmsc.common.sample import Sample
from mmsc.common.registry import registry


_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 20

# remove warnings
av.logging.set_level(av.logging.ERROR)


def build_bbox_tensors(infos, max_length):
    num_bbox = min(max_length, len(infos))

    # After num_bbox, everything else should be zero
    coord_tensor = torch.zeros((max_length, 4), dtype=torch.float)
    width_tensor = torch.zeros(max_length, dtype=torch.float)
    height_tensor = torch.zeros(max_length, dtype=torch.float)
    bbox_types = ["xyxy"] * max_length

    infos = infos[:num_bbox]
    sample = Sample()

    for idx, info in enumerate(infos):
        bbox = info["bounding_box"]
        x = bbox.get("top_left_x", bbox["topLeftX"])
        y = bbox.get("top_left_y", bbox["topLeftY"])
        width = bbox["width"]
        height = bbox["height"]

        coord_tensor[idx][0] = x
        coord_tensor[idx][1] = y
        coord_tensor[idx][2] = x + width
        coord_tensor[idx][3] = y + height

        width_tensor[idx] = width
        height_tensor[idx] = height
    sample.coordinates = coord_tensor
    sample.width = width_tensor
    sample.height = height_tensor
    sample.bbox_types = bbox_types

    return sample


def build_dataset_from_multiple_imdbs(config, dataset_cls, dataset_type):
    from mmsc.datasets.concat_dataset import MMSCConcatDataset

    if dataset_type not in config.imdb_files:
        warnings.warn(
            "Dataset type {} is not present in "
            "imdb_files of dataset config. Returning None. "
            "This dataset won't be used.".format(dataset_type)
        )
        return None

    imdb_files = config["imdb_files"][dataset_type]

    datasets = []

    for imdb_idx in range(len(imdb_files)):
        dataset = dataset_cls(dataset_type, imdb_idx, config)
        datasets.append(dataset)

    dataset = MMSCConcatDataset(datasets)

    return dataset


def dataset_list_from_config(config: DictConfig) -> List[str]:
    if "datasets" not in config:
        warnings.warn("No datasets attribute present. Setting default to voxceleb1.")
        datasets = "voxceleb1"
    else:
        datasets = config.datasets

    if type(datasets) == str:
        datasets = list(map(lambda x: x.strip(), datasets.split(",")))

    return datasets


def load_wave(db, max_frames, compressed=False, pad=True, 
              crop=False, cache=None, verbose=False):
    DEFAULT_SAMPLE_RATE = 16000

    if cache is None:
        if isinstance(db, str):
            audio, sample_rate = soundfile.read(db)
        elif isinstance(db, (tuple, list)):
            if compressed:
                cwaves = np.load(db[0], allow_pickle=True)['waves'].item()
                audio, sample_rate = cwaves[db[1]]
            else:
                audio, sample_rate = soundfile.read(db[0])
        else:
            raise TypeError('Data format not supported')
    else:
        audio, sample_rate = cache['audio'], cache['sample_rate']
    
    if crop:
        if isinstance(db[3], float):
            start, dura = db[1]-db[3], db[2]
        else:
            start, dura = db[1], db[2]
        audio = audio[int(start*sample_rate): int((start+dura)*sample_rate)]

    audiosize = audio.shape[0]

    if audiosize == 0:
        raise RuntimeError('Audio length is zero, check the file')
    
    
    assert sample_rate == DEFAULT_SAMPLE_RATE, (f'inconsist sample rate: {sample_rate}'
                          ', please check')
    # Maximum audio length
    max_audio = max_frames * int(sample_rate / 100)

    if pad and audiosize < max_audio:
        shortage = max_audio - audiosize
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if audiosize >= max_audio:
        startframe = int(torch.rand([1]).item()*(audiosize-max_audio))
        audio = audio[startframe: startframe+max_audio]

    if verbose:
        startframe = startframe / sample_rate
        return audio, startframe
    else:
        return audio


def load_frames(frames, snippet_length):
    if len(frames) > 0:
        frames = np.array([cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB)
                           for i in torch.randint(0, len(frames), [snippet_length])])
    else:
        frames = np.array([], dtype=np.uint8)
    return frames


def load_video_ava(db, snippet_length, max_frames, crop=True, debug=False):
    audio = load_wave(db, max_frames, crop=crop)
    frames = load_frames(db[4], snippet_length)
    if debug:
        visualize(frames, audio)
    sample = Sample()
    sample.frames = frames
    sample.audio = audio
    sample.meta = { 'visible': len(frames) > 0 }
    return sample


def load_video_vc(db, snippet_length, max_frames, debug=False):
    audio = load_wave(db, max_frames)
    faces = glob.glob(db[1])
    frames = load_frames(faces, snippet_length)
    if debug:
        visualize(frames, audio)
    sample = Sample()
    sample.frames = frames
    sample.audio = audio
    sample.meta = { 'visible': len(frames) > 0 }
    return sample


def load_video_ami(db, snippet_length, max_frames, cache=None, crop=True, debug=False):
    audio = load_wave(db, max_frames, crop = crop, cache=cache)
    frames = load_frames(db[3], snippet_length)
    if debug:
        visualize(frames, audio)
    sample = Sample()
    sample.frames = frames
    sample.audio = audio
    sample.meta = { 'visible': len(frames) > 0 }
    return sample


def load_video(db, snippet_length, max_frames, compressed=True):

    if isinstance(db, tuple):
        if compressed:
            videos = np.load(db[0], allow_pickle=True)['videos'].item()
            reader = VideoReader(BytesIO(videos[db[1]]), snippet_length, 
                                 max_frames, db)
        else:
            reader = VideoReader(open(db[0], 'rb'), snippet_length, 
                                 max_frames, db)
        
        frames, audio, meta = reader.sample()

    sample = Sample()
    sample.frames = frames
    sample.audio = audio
    sample.meta = meta

    return sample


def visualize(video_frames, audio):
    import cv2, subprocess
    from scipy.io import wavfile
    for i, f in enumerate(video_frames):
        cv2.imwrite(f'debug/{i:04d}.jpg', cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    
    wavfile.write('debug/debug.wav', 16000, audio)
    cmd = ('ffmpeg -r 25 -i debug/%04d.jpg -i "debug/debug.wav" -c:v libx264 -c:a'+
            ' aac -pix_fmt yuv420p -crf 23 -r 25 -shortest -y debug/debug.mp4')
    subprocess.call(cmd, shell=True, stdout=False)


class VideoReader(object):
    """
    Simple wrapper around PyAV that exposes a few useful functions for
    dealing with video reading. PyAV is a pythonic binding for the ffmpeg libraries.
    Acknowledgement: Codes are borrowed from Bruno Korbar
    """
    def __init__(self, video, snippet_length, max_frames, db,  
                 frame_rate=25, decode_lossy=False, audio_resample_rate=None):
        """
        Arguments:
            video_path (str): path or byte of the video to be loaded
        """
        self.container = av.open(video)
        self.max_frames = max_frames
        self.frame_rate = frame_rate
        self.snippet_length = snippet_length
        self.resampler = None
        if audio_resample_rate is not None:
            self.resampler = av.AudioResampler(rate=audio_resample_rate)
        
        if self.container.streams.video:
            # enable multi-threaded video decoding
            if decode_lossy:
                warnings.warn('VideoReader| thread_type==AUTO can yield potential frame dropping!', RuntimeWarning)
                self.container.streams.video[0].thread_type = 'AUTO'
            self.video_stream = self.container.streams.video[0]
        else:
            self.video_stream = None
        self.db = db

    def seek(self, pts, backward=True, any_frame=False):
        stream = self.video_stream
        self.container.seek(pts, any_frame=any_frame, backward=backward, stream=stream)

    def _occasional_gc(self):
        # there are a lot of reference cycles in PyAV, so need to manually call
        # the garbage collector from time to time
        global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
            gc.collect()

    def _read_video(self, offset):
        self._occasional_gc()

        stream = self.video_stream
        def iter_frames():
            for packet in self.container.demux(stream):
                for frame in packet.decode():
                    yield frame

        pts = self.container.duration * offset
        time_ = pts / float(av.time_base)

        self.container.seek(int(pts))
        
        video_frames = []
        count = 0
        for _, frame in enumerate(iter_frames()):
            if frame.pts * frame.time_base >= time_:
                video_frames.append(frame)
                if count >= self.snippet_length - 1:
                    break
                count += 1
        return video_frames
    
    def _read_entire_audio(self):
        # read all the audio
        self._occasional_gc()
        if not self.container.streams.audio:
            with open('bad_video_list.txt') as f:
                f.write(self.db)
            raise RuntimeError('audio stream not found')
        assert self._get_audio_sample_rate() == 16000, 'sample rate must be 16000'
        self.container.seek(-1, backward=True, any_frame=False, 
                            stream=self.container.streams.audio[0])
        audio_frames = []
        for _, frame in enumerate(self.container.decode(audio=0)):
            if self.resampler is not None:
                frame = self._resample_audio_frame(frame)
            audio_frames.append(frame.to_ndarray())
        entire_audio = {
            'audio': np.concatenate(audio_frames, axis=1)[0], 
            'sample_rate': 16000
        }
        return entire_audio

    def sample(self, debug=False):

        if self.container is None:
            raise RuntimeError('video stream not found')

        entire_audio = self._read_entire_audio()

        audio, start_sec = load_wave(None, self.max_frames, 
                                     cache=entire_audio, verbose=True)

        max_sec = int(self.max_frames / 100)

        offset = torch.randint(int(start_sec * self.frame_rate), 
                        int((start_sec + max_sec) * 
                        self.frame_rate - self.snippet_length), [1]).item()

        _, _, total_num_frames = self._compute_video_stats()
        snippet_pos = offset / total_num_frames

        # video_frames = self._read_video(snippet_pos)
        # video_frames = np.array([np.uint8(f.to_rgb().to_ndarray()) for f in video_frames])
        video_frames = glob.glob(self.db[1])
        video_frames = load_frames(video_frames, self.snippet_length)
        meta = { 'visible': len(video_frames) > 0 }

        # start_idx = int(snippet_pos*entire_audio[0].shape[0] - start_sec*entire_audio[1])
        # dura = self.snippet_length / 25.0

        # if debug:
        #     visualize(video_frames, audio[start_idx: start_idx+int(dura*16000)])

        return video_frames, audio, meta

    def _compute_video_stats(self):
        if self.video_stream is None or self.container is None:
            return 0
        num_of_frames = self.container.streams.video[0].frames
        self.seek(0, backward=False)
        count = 0
        time_base = 512
        for p in self.container.decode(video=0):
            count = count + 1
            if count == 1:
                start_pts = p.pts
            elif count == 2:
                time_base = p.pts - start_pts
                break
        return start_pts, time_base, num_of_frames
    
    def _get_audio_sample_rate(self):
        return self.container.streams.audio[0].time_base.denominator