import os, glob, random, librosa, soundfile
import torch
import numpy as np
import torchvision.transforms as T
from scipy import signal
from mmsc.common.registry import registry
from mmsc.datasets.processors.processors import BaseProcessor
from omegaconf import OmegaConf
from mmsc.utils.dataset import load_wave
from mmsc.utils.fileio import PathManager

@registry.register_processor('audio_augment')
class Augmentation(BaseProcessor):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        assert OmegaConf.is_dict(config),'Invalid processor param'
        self.musan_path = config.musan_path
        self.rir_path = config.rir_path
        
        if not PathManager.exists(self.musan_path):
            raise RuntimeError(f'{self.musan_path} not exist')
        if not PathManager.exists(self.rir_path):
            raise RuntimeError(f'{self.rir_path} not exist')
        self.augmentor = AugmentWAV(self.musan_path, 
                                    self.rir_path, 
                                    config.max_frame)
    
    def __call__(self, item):
        augtype = random.randint(0,4)
        if augtype == 1:
            item.audio = self.augmentor.reverberate(item.audio)
            # item.audio = item.audio
        elif augtype == 2:
            item.audio = self.augmentor.additive_noise('music',item.audio)
        elif augtype == 3:
            item.audio = self.augmentor.additive_noise('speech',item.audio)
        elif augtype == 4:
            item.audio = self.augmentor.additive_noise('noise',item.audio)
        return item

#TODO: load augment corpus from compressed audio       
class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_frames * 160

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = load_wave(noise, self.max_frames)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = np.expand_dims(rir.astype(np.float),0)
        rir         = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(audio[None,:], rir, mode='full')[:,:self.max_audio][0]


@registry.register_processor('audio_normalize')
class AudioNormalize(BaseProcessor):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        assert OmegaConf.is_dict(config),'Invalid processor param'

        self.desired_rms = config.desired_rms
        self.eps = config.eps
    
    def __call__(self, item):
        audio = item.audio
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        elif not isinstance(audio, np.ndarray):
            raise TypeError('Invalid audio data type')
        rms = np.maximum(self.eps, np.sqrt(np.mean(audio**2)))
        item.audio = audio * (self.desired_rms / rms)
        return item


@registry.register_processor('audio_to_spectrogram')
class Spectrogram(BaseProcessor):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        assert OmegaConf.is_dict(config),'Invalid processor param'

        self.stft_frame = config.stft_frame
        self.stft_hop = config.stft_hop
        self.n_fft = config.n_fft
        
    def __call__(self, item):
        audio = item.audio
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        elif not isinstance(audio, np.ndarray):
            raise TypeError('Invalid audio data type')
        spectro = librosa.core.stft(audio, 
                                    hop_length=self.stft_hop, 
                                    n_fft=self.n_fft, 
                                    win_length=self.stft_frame, 
                                    center=True)
        real = np.expand_dims(np.real(spectro), axis=-1)
        imag = np.expand_dims(np.imag(spectro), axis=-1)
    
        spectro_two_channel = np.concatenate((real, imag), axis=-1)
        item.spec = spectro_two_channel.astype(np.float32)
        item.pop('audio')
        return item


@registry.register_processor('audio_to_tensor')
class AudioTransform(BaseProcessor):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'

    def __call__(self, item):
        if isinstance(item.audio, np.ndarray):
            item.audio = torch.FloatTensor(item.audio)
        assert isinstance(item.audio, torch.Tensor), 'invalid audio data type'
        return item
