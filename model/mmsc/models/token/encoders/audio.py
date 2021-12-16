import logging, warnings
import torch, torchaudio
import torch.nn.functional as F
from torch import nn
from mmsc.utils.fileio import PathManager
from mmsc.utils.checkpoint import load_pretrained_model
from mmsc.utils.download import download_from_google_drive


logger = logging.getLogger(__name__)


class AudioEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.inplanes = config.num_filters[0]
        self.encoder_type = config.encoder_type
        self.n_mels = config.n_mels
        self.log_input = config.log_input
        self.fix_layers = config.fix_layers
        self.init_weight = config.init_weight

        self.instancenorm = nn.InstanceNorm1d(self.n_mels)
        self.torchfb = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
                                                     n_fft=512, 
                                                     win_length=400, 
                                                     hop_length=160, 
                                                     window_fn=torch.hamming_window, 
                                                     n_mels=self.n_mels))

        self.conv1 = nn.Conv2d(1, config.num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(config.num_filters[0])
        
        self.layer1 = self._make_layer(BasicBlock, config.num_filters[0], config.layers[0])
        self.layer2 = self._make_layer(BasicBlock, config.num_filters[1], config.layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(BasicBlock, config.num_filters[2], config.layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(BasicBlock, config.num_filters[3], config.layers[3], stride=(2, 2))
        
        self.audio_pooling = nn.AdaptiveAvgPool2d((7, 7))
        self._init_parameters()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if not PathManager.isfile(self.init_weight):
            warnings.warn(f'{self.init_weight} not found, doneloading...')
            download_from_google_drive('1S51eTZuM5cgULHztw0twO36oU-7eOQiV', './save/pretrained/backbone_audio.pth')
        
        ckpt_state_dict = load_pretrained_model(self.init_weight, init=True)
        self.load_state_dict(ckpt_state_dict)

        if self.fix_layers == 'all':
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, batch):
        x = batch.audio
        N, D, L = x.shape
        x = x.reshape(N*D, L)
        with torch.no_grad():
            x = self.torchfb(x) + 1e-6
            if self.log_input: 
                x = x.log()
            x = self.instancenorm(x).unsqueeze(1)
            
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.audio_pooling(x)
        _, C, H, W = x.shape
        x = x.reshape(N, D, C, H, W)
        return x
    

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplane, outplane, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.conv2 = nn.Conv2d(outplane, outplane, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(outplane, reduction)
        self.downsample = downsample

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    
    def __init__(self, channel, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PreEmphasis(nn.Module):

    def __init__(self, coef = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input):
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)