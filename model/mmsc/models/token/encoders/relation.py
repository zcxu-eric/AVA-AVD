import torch
from torch import nn
import torch.nn.functional as F


class RelationLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.dropout = config.get('dropout', 0)
        self.num_way = config.get('num_way', 5)
        self.layers = config.get('layers', [4, 3])
        self.num_shot = config.get('num_shot', 1) + 1
        self.num_filters = config.get('num_filters', [128, 64])

        self.inplanes = (256 + 512) * 2
        self.bna = nn.BatchNorm2d(256)
        self.layer1 = self._make_layer(Bottleneck, self.num_filters[0], self.layers[0], stride=2)
        self.layer2 = self._make_layer(Bottleneck, self.num_filters[1], self.layers[1], stride=2)
        self.fc = nn.Sequential(
            nn.Linear(256*2*2 , 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # self.attention = nn.Sequential(
        #     nn.Conv1d(config.num_filters[3] * outmap_size, 128, kernel_size=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128),
        #     nn.Conv1d(128, config.num_filters[3] * outmap_size, kernel_size=1),
        #     nn.Softmax(dim=2),
        #     )

        self.task_token = nn.Embedding(4, 1536)

        self._init_parameters()
    
    def partial_task_token(self, index):
        table = torch.ones((4, 1536), dtype=torch.float32, device=index.device)
        table[0, 0: 512] = 0
        table[0, 768: 1280] = 0
        table[1, 0: 512] = 0
        table[2, 768: 1280] = 0
        return torch.cat((table[index][..., :768], self.task_token(index)[..., :768]), dim=-1)

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
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.task_token.weight, 1)

    def forward(self, video, audio, visible, targets): 
        audio = self.batch_norm_audio(audio)
        support, query, token, label = self.divide_set(video, audio, visible, targets)
        N, Q, S, C, H, W = query.shape
        x1 = torch.cat((support, query), dim=3).reshape(N*S*Q, 2*C, H, W)
        x2 = torch.cat((query, support), dim=3).reshape(N*S*Q, 2*C, H, W)
        x = torch.cat((x1, x2), dim=0)
        token = token.reshape(2*N*S*Q, 1536, 1, 1)
        x = x * token
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x.flatten(1)).reshape(2*N, S, Q)
        return x, label
    
    def batch_norm_audio(self, audio):
        N, D, C, H, W = audio.shape
        audio = self.bna(audio.reshape(N*D, C, H, W))
        audio = audio.reshape(N, D, C, H, W)
        return audio
    
    def divide_set(self, video, audio, visible, targets):
        feat = torch.cat((video, audio), dim=2)
        N, _, C, H, W = feat.shape
        feat = feat.reshape(N, self.num_way, self.num_shot, C, H, W)
        targets = targets.reshape(N, self.num_way, self.num_shot)
        visible = visible.reshape(N, self.num_way, self.num_shot)

        support = feat[:, :, 1:, ...].reshape(N, -1, C, H, W)
        support_t = targets[:, :, 1:].reshape(N, -1)
        support_v = visible[:, :, 1:].reshape(N, -1)
        num_support = support.shape[1]

        query = feat[:, :, 0, ...]
        query_t = targets[:, :, 0]
        query_v = visible[:, :, 0]
        num_query = query.shape[1]

        support = support.unsqueeze(1).expand(-1, num_query, -1, -1, -1, -1)
        query = query.unsqueeze(2).expand(-1, -1, num_support, -1, -1, -1)

        support_t = support_t.unsqueeze(1).expand(-1, num_query, -1)
        query_t = query_t.unsqueeze(2).expand(-1, -1, num_support)
        label = torch.eq(support_t, query_t).float().repeat(2, 1, 1)

        support_v = support_v.unsqueeze(1).expand(-1, num_query, -1)
        query_v = query_v.unsqueeze(2).expand(-1, -1, num_support)
        token = torch.cat((self.task_token(2*support_v + query_v), self.task_token(2*query_v + support_v)), dim=0)

        return support, query, token, label
    
    def predict(self, video, audio, task):
        audio = self.batch_norm_audio(audio)
        feat = torch.cat((video, audio), dim=2)
        N = feat.shape[0]
        x1 = torch.cat([feat[:, 0:1, ...], feat[:, 1:2, ...]], dim=2)
        x2 = torch.cat([feat[:, 1:2, ...], feat[:, 0:1, ...]], dim=2)
        x = torch.cat([x1, x2], dim=1)
        token = torch.cat((self.task_token(task[0]).unsqueeze(1), self.task_token(task[1]).unsqueeze(1)), dim=1)
        N, S, C, H, W = x.shape
        x = x.reshape(N*S, C, H, W)
        token = token.reshape(N*S, 1536, 1, 1)
        x = x * token
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x.flatten(1))
        x = x.reshape(N, -1, 1).mean(1)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MLP(nn.Module):
    """multi-layer perceptron"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x