import logging, warnings
import torch
import torch.nn as nn
from mmsc.common.sample import Sample
from mmsc.utils.fileio import PathManager
from mmsc.common.registry import registry
from mmsc.models import BaseModel
from .encoders.audio import AudioEncoder
from .encoders.video import VideoEncoder
from .encoders.relation import RelationLayer
from mmsc.utils.checkpoint import load_pretrained_model


logger = logging.getLogger(__name__)


@registry.register_model('token')
class TOKENVARNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def build(self):
        '''
        build AVRNet and load checkpoint here, 
        download pretrained models if current/best checkpoint not found
        '''
        self.audio_encoder = AudioEncoder(self.config.audio)
        self.video_encoder = VideoEncoder(self.config.video)
        self.relation_layer = RelationLayer(self.config.relation)

        checkpoint = self.config.base_ckpt_path
        if PathManager.isfile(checkpoint):
            logger.info(f'Loading {checkpoint}')
            ckpt_state_dict = load_pretrained_model(checkpoint)['checkpoint']
            self.load_state_dict(ckpt_state_dict, strict=True)
        else:
            warnings.warn(f'{checkpoint} not found, use default weight')

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.config.audio.fix_layers:
            # logger.info("Freezing Mean/Var of BatchNorm2D in AudioEncoder.")
            for m in self.audio_encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        if self.config.video.fix_layers:
            # logger.info("Freezing Mean/Var of BatchNorm2D in VideoEncoder.")
            for m in self.video_encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, batch, exec=None):
        if batch['dataset_type'] == 'train':
            return self._forward_train(batch)
        else:
            return self._forward_predict(batch, exec)
    
    def _forward_train(self, batch):
        output = Sample()
        feat_audio = self.audio_encoder(batch)
        feat_video = self.video_encoder(batch)
        targets = batch.targets
        visible = torch.tensor(batch.meta.visible, 
                               dtype=torch.int64, 
                               device=targets.device)
        scores, targets = self.relation_layer(feat_video, feat_audio, 
                                              visible, targets)
        output.scores = scores
        output.targets = targets
        return output
    
    def _forward_predict(self, batch, exec):
        output = Sample()
        if exec == 'extraction':
            feat_audio = self.audio_encoder(batch)
            feat_video = self.video_encoder(batch)
            output.feat_audio = feat_audio
            output.feat_video = feat_video
            output.video = batch.meta.video
            output.start = batch.meta.start
            output.end = batch.meta.end
            output.trackid = batch.meta.trackid
            output.visible = batch.meta.visible
        elif exec == 'relation':
            output.scores = self.relation_layer.predict(batch['video'], 
                                                        batch['audio'], 
                                                        batch['task_full'])
        return output
