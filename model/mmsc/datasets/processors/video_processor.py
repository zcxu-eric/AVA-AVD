import os, cv2, glob, torch, random, librosa, soundfile
import numpy as np
import torchvision.transforms as T
from mmsc.common.registry import registry
from mmsc.datasets.processors.functional import *
from mmsc.datasets.processors.processors import BaseProcessor
from omegaconf import OmegaConf
from mmsc.utils.fileio import PathManager


@registry.register_processor('face_to_tensor')
class FaceToTensor(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'

    def __call__(self, item):
        assert isinstance(item.frames, np.ndarray), 'invalid frame data type'
        frames = [torch.tensor(f).unsqueeze(0) for f in item.frames]
        frames = torch.cat(frames, dim=0)
        item.frames = video_to_normalized_float_tensor(frames)
        return item


@registry.register_processor('face_augment')
class FaceAugment(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.hflip = config.get('hflip', None)
        self.rand_rotate = config.get('rand_rotate', None)
        self.static = config.get('static', None)
        self.missing = config.get('missing', None)
        self.blur = config.get('blur', None)
        if self.blur:
            self.gaussian_blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        self.jitter = config.get('jitter', None)
        if self.jitter:
            self.color_jitter = T.ColorJitter(brightness=.5, hue=.3)
    
    def __call__(self, item):
        assert isinstance(item.frames, torch.Tensor), 'invalid frame data type'

        if self.hflip and torch.rand([1]).item() < self.hflip.prob:
            item.frames = video_hflip(item.frames)

        if self.rand_rotate and torch.rand([1]).item() < self.rand_rotate.prob:
            angle = (torch.rand([1]).item() * 2 - 1) * self.rand_rotate.max_angle
            angle = int(360 + angle) if angle < 0 else int(angle)
            item.frames = video_rotate(item.frames, angle)

        if self.static and torch.rand([1]).item() < self.static.prob:
            D = item.frames.shape[1]
            item.frames = item.frames[:, torch.randint(0, D, [1]), ...].repeat(1, D, 1, 1)
            item.meta.static = True

        if self.missing and torch.rand([1]).item() < self.missing.prob:
            item.frames = torch.zeros_like(item.frames)
            item.meta.visible = False
        
        if self.blur and item.meta.visible and  torch.rand([1]).item() < self.blur.prob:
            item.frames = self.gaussian_blur(item.frames)
        
        if self.jitter and item.meta.visible and  torch.rand([1]).item() < self.jitter.prob:
            item.frames = self.color_jitter(item.frames)

        return item


@registry.register_processor('audiovisual_augment')
class AudioVisualAugment(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.missing = config.get('missing', None)
    
    def __call__(self, item):
        assert isinstance(item.frames, torch.Tensor), 'invalid frame data type'
        # audio always exists
        if item.meta["visible"]:
            if self.missing and torch.rand([1]).item() < self.missing.prob:
                if torch.rand([1]).item() < self.missing.ratio:
                    #zero images => tensor filled with -1.0
                    item.frames = -1 * torch.ones_like(item.frames)
                else:
                    item.audio = torch.zeros_like(item.audio)
        return item


@registry.register_processor('face_resize')
class FaceResize(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'
        self.dest_size = list(config.dest_size)
    
    def __call__(self, item):
        assert isinstance(item.frames, torch.Tensor), 'invalid frame data type'
        item.frames = video_resize(item.frames, self.dest_size)
        return item


@registry.register_processor('face_normalize')
class FaceNormalize(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'
        self.mean = config.mean
        self.std = config.std

    def __call__(self, item):
        assert isinstance(item.frames, torch.Tensor), 'invalid frame data type'
        item.frames = video_normalize(item.frames, mean=self.mean, std=self.std)
        return item


@registry.register_processor('face_pad')
class FacePad(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'
        self.length = config.length
    
    def __call__(self, item):
        assert isinstance(item.frames, np.ndarray), 'invalid frame data type'
        if item.frames.shape[0] == 0 or not hasattr(item, 'frames'):
            item.frames = np.zeros((self.length, 224, 224, 3), dtype=np.uint8)
        if item.frames.shape[0] < self.length:
            pad_width = ((0, self.length-item.frames.shape[0]), (0, 0), (0, 0), (0, 0))
            item.frames = np.pad(item.frames, pad_width=pad_width, mode='edge')
        return item


@registry.register_processor('face_align')
class FaceAlign(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.gt_bbox = [np.array(config.gt_bbox)]

    def __call__(self, item):
        frames = item.frames
        landmarks = []
        for frame in frames:
            landmarks.append(self.fa.get_landmarks_from_image(frame, detected_faces=self.gt_bbox))
        return item


def visualize(item):
    import matplotlib.pyplot as plt
    for i in range(item.frames.shape[1]):
        frame = item.frames[:, i, ...]
        frame = (frame * 0.5 + 0.5) * 255
        frame = frame.numpy()
        frame = frame.astype(np.uint8).transpose(1, 2, 0)
        plt.imshow(frame)
        plt.show()
       

        