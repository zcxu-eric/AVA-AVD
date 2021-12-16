# Copyright (c) Facebook, Inc. and its affiliates.
"""
Losses module contains implementations for various losses used generally
in vision and audio space. One can register custom losses to be detected by
MMSC using the following example.

.. code::

   from mmsc.common.registry import registry
   from torch import nn


   @registry.register_loss("custom")
   class CustomLoss(nn.Module):
       ...

Then in your model's config you can specify ``losses`` attribute to use this loss
in the following way:

.. code::

   model_config:
       some_model:
           losses:
               - type: custom
               - params: {}
"""
import collections
import warnings
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmsc.common.sample import Sample
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from omegaconf.omegaconf import OmegaConf, DictConfig
from mmsc.common.registry import registry
from mmsc.utils.logger import log_class_usage
from omegaconf import MISSING
from torch import Tensor
from mmsc.utils.checkpoint import load_pretrained_model
from mmsc.utils.fileio import PathManager


@dataclass
class LossConfig:
    type: str = MISSING
    params: Dict[str, Any] = MISSING


class Losses(nn.Module):
    """``Losses`` acts as an abstraction for instantiating and calculating
    losses. ``BaseModel`` instantiates this class based on the `losses`
    attribute in the model's configuration `model_config`. ``loss_list``
    needs to be a list for each separate loss containing `type` and `params`
    attributes.

    Args:
        loss_list (ListConfig): Description of parameter `loss_list`.

    Example::

        # losses:
        # - type: logit_bce
        # Can also contain `params` to specify that particular loss's init params
        # - type: combined
        config = [{"type": "logit_bce"}, {"type": "combined"}]
        losses = Losses(config)

    .. note::

        Since, ``Losses`` is instantiated in the ``BaseModel``, normal end user
        mostly doesn't need to use this class.

    Attributes:
        losses: List containing instantiations of each loss
                                   passed in config
    """

    # TODO: Union types are not supported in OmegaConf.
    # Later investigate for a workaround.for
    def __init__(self, loss_list: List[Union[str, LossConfig]]):
        super().__init__()
        self.losses = nn.ModuleList()
        config = registry.get("config")
        self._evaluation_predict = False
        if config:
            self._evaluation_predict = config.get("evaluation", {}).get(
                "predict", False
            )

        for loss in loss_list:
            self.losses.append(MMSCLoss(loss))

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        """Takes in the original ``SampleList`` returned from DataLoader
        and `model_output` returned from the model and returned a Dict containing
        loss for each of the losses in `losses`.

        Args:
            sample_list (SampleList): SampleList given be the dataloader.
            model_output (Dict): Dict returned from model as output.

        Returns:
            Dict: Dictionary containing loss value for each of the loss.

        """
        output = {}
        if "targets" not in sample_list:
            if not self._evaluation_predict:
                warnings.warn(
                    "Sample list has not field 'targets', are you "
                    "sure that your DB has labels? you may have "
                    "wanted to run with evaluation.predict=true"
                )
            else:
                return output

        for loss in self.losses:
            eval_applicable = getattr(loss, 'eval_applicable', True)
            if not eval_applicable and not self.training:
                continue
            output.update(loss(sample_list, model_output))

        if not torch.jit.is_scripting():
            registry_loss_key = "{}.{}.{}".format(
                "losses", sample_list["dataset_name"], sample_list["dataset_type"]
            )
            # Register the losses to registry
            registry.register(registry_loss_key, output)

        return output


class MMSCLoss(nn.Module):
    """Internal MMSC helper and wrapper class for all Loss classes.
    It makes sure that the value returned from a Loss class is a dict and
    contain proper dataset type in keys, so that it is easy to figure out
    which one is the val loss and which one is train loss.

    For example: it will return ``{"val/vqa2/logit_bce": 27.4}``, in case
    `logit_bce` is used and SampleList is from `val` set of dataset `vqa2`.

    Args:
        params (type): Description of parameter `params`.

    .. note::

        Since, ``MMSCLoss`` is used by the ``Losses`` class, end user
        doesn't need to worry about it.
    """

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}

        is_mapping = isinstance(params, collections.abc.MutableMapping)

        if is_mapping:
            if "type" not in params:
                raise ValueError(
                    "Parameters to loss must have 'type' field to"
                    "specify type of loss to instantiate"
                )
            else:
                loss_name = params["type"]
        else:
            assert isinstance(
                params, str
            ), "loss must be a string or dictionary with 'type' key"
            loss_name = params

        self.name = loss_name

        loss_class = registry.get_loss_class(loss_name)

        log_class_usage("Loss", loss_class)

        if loss_class is None:
            raise ValueError(f"No loss named {loss_name} is registered to registry")
        # Special case of multi as it requires an array
        if loss_name.startswith("multi"):
            assert is_mapping
            self.loss_criterion = loss_class(params)
        else:
            if is_mapping:
                loss_params = params.get("params", {})
            else:
                loss_params = {}
            self.loss_criterion = loss_class(**loss_params)
        
        self.eval_applicable = loss_params.get('eval_applicable', True)

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        loss = self.loss_criterion(sample_list, model_output)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float)

        if loss.dim() == 0:
            loss = loss.view(1)

        if not torch.jit.is_scripting():
            key = "{}/{}/{}".format(
                sample_list.dataset_type, sample_list.dataset_name, self.name
            )
        else:
            key = f"{self.name}"
        return {key: loss}


@registry.register_loss("triplet")
class TripletLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.hard_rank = params.get('hard_rank', 5)
        self.hard_prob = params.get('hard_prob', 0.5)
        self.margin = params.get('margin', 0.3)
        self.loss_weight = params.get('loss_weight', 1.0)
    
    def forward(self, sample_list, model_output):
        raise NotImplementedError
    
    def triplet_loss_fn(self, x1, x2):
        assert x1.size()[1] == 2
        assert x2.size()[1] == 2

        out_anchor = F.normalize(x1[:,0,:], p=2, dim=1)
        out_positive = F.normalize(x2[:,1,:], p=2, dim=1)

        sim = -1 * (F.pairwise_distance(out_anchor.unsqueeze(-1), out_positive.unsqueeze(-1).transpose(0,2))**2)

        negidx = self.mineHardNegative(sim.detach())

        out_negative = out_positive[negidx,:]

        ## calculate distances
        pos_dist = F.pairwise_distance(out_anchor, out_positive)
        neg_dist = F.pairwise_distance(out_anchor, out_negative)

        ## loss function
        loss = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))

        return loss
    
    def mineHardNegative(self, output):
        negidx = []
        for idx, similarity in enumerate(output):
            simval, simidx = torch.sort(similarity, descending=True)
            if self.hard_rank < 0:
                ## Semi hard negative mining
                semihardidx = simidx[(similarity[idx] - self.margin < simval) &  (simval < similarity[idx])]
                if len(semihardidx) == 0:
                    negidx.append(random.choice(simidx))
                else:
                    negidx.append(random.choice(semihardidx))
            else:
                ## Rank based negative mining
                simidx = simidx[simidx!=idx]
                if random.random() < self.hard_prob:
                    negidx.append(simidx[torch.randint(min(len(simidx), self.hard_rank), [1])])
                else:
                    negidx.append(random.choice(simidx).unsqueeze(0))
        return negidx


@registry.register_loss("voice_face_triplet")
class VoiceTripletLoss(TripletLoss):
    def __init__(self, **params):
        super().__init__(**params)
        self.loss_weight = params.get('loss_weight', 1.0)
        
    def forward(self, sample_list, model_output):
        if model_output["task"] == 'audio':
            scores = F.normalize(model_output["audio"], dim=-1)
        elif model_output["task"] == 'face':
            scores = F.normalize(model_output["faces"], dim=-1)
        N, C = scores.shape
        scores = scores.view(-1, 2, C)
        loss = self.triplet_loss_fn(scores, scores) * self.loss_weight
        return loss


@registry.register_loss("aamsoftmax")
class AAMSoftmaxLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        if isinstance(params, dict):
            config = DictConfig(params)

        self.scale = config.scale
        self.margin = config.margin
        self.weight = nn.Parameter(torch.FloatTensor(
                            config.num_classes, 
                            config.in_feature),
                            requires_grad=True)
        self._init_weight()

        self.easy_margin = config.easy_margin
        self.loss_weight = config.get('loss_weight', 1.0)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)

        # make the function cos(theta+m) monotonic 
        # decreasing while theta in [0,180] degree
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

        self.loss_fn = nn.CrossEntropyLoss(reduction=config.reduction)

    def forward(self, sample_list, model_output):
        score = model_output["scores"]
        target = sample_list["targets"]
        return self._compute_loss(score, target) * self.loss_weight

    def _compute_loss(self, score, target):
        assert len(score.shape) == 2, 'score dimension must be 2'
        if len(target.shape) > 1:
            target = target.squeeze(1)
        # cos(theta)
        cosine = F.linear(F.normalize(score), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        return self.loss_fn(output, target)
    
    def _init_weight(self):

        config = registry.get('config')
        model_config = config.get('model_config')
        model_name = config.get('model')
        base_args = model_config.get(model_name)
        checkpoint = base_args.base_ckpt_path

        if False: #PathManager.isfile(checkpoint):
            ckpt_state_dict = load_pretrained_model(checkpoint)['checkpoint']
            ckpt_state_dict = { 'weight': v for k, v in ckpt_state_dict.items() if 'losses' in k}
            self.load_state_dict(ckpt_state_dict)
        else:
            nn.init.xavier_uniform_(self.weight)    


@registry.register_loss("aamsface")
class AAMSoftMaxFace(AAMSoftmaxLoss):
    def __init__(self, **params):
        super().__init__(**params)
    
    def forward(self, sample_list, model_output):
        return self._compute_loss(model_output["faces"], 
                    sample_list["targets"]) * self.loss_weight


@registry.register_loss("aamsaudio")
class AAMSoftMaxFace(AAMSoftmaxLoss):
    def __init__(self, **params):
        super().__init__(**params)
    
    def forward(self, sample_list, model_output):
        return self._compute_loss(model_output["audio"], 
                    sample_list["targets"]) * self.loss_weight


@registry.register_loss("aamsfaceaudio")
class AAMSoftMaxFace(AAMSoftmaxLoss):
    def __init__(self, **params):
        super().__init__(**params)
    
    def forward(self, sample_list, model_output):
        loss_f = self._compute_loss(model_output["faces"], 
                    sample_list["targets"]) * self.loss_weight
        loss_a = self._compute_loss(model_output["audio"], 
                    sample_list["targets"]) * self.loss_weight
        return loss_f + loss_a


@registry.register_loss("mix_aamsoftmax")
class MixAAMSoftMax(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.loss_weight = params.get('loss_weight', 1.0)
        datasets = ['voxceleb1', 'voxceleb2', 'ava']
        num_classes = [1211, 5994, 2127]
        self.fn = nn.ModuleDict()
        for i, dataset in enumerate(datasets):
            params['num_classes'] = num_classes[i]
            self.fn[dataset] = self.build_loss_fn(**params)
    
    def forward(self, sample_list, model_output):
        N, D, C = model_output["targets"].shape
        model_output["targets"] = model_output["targets"].reshape(N*D, C)
        visible = torch.FloatTensor(model_output.visible).to(model_output["targets"].device).reshape(N*D)
        dataset_name = model_output.dataset_name
        task = model_output.task
        if task == "video":
            loss = (self.fn[dataset_name]._compute_loss(model_output["faces"], 
                    model_output["targets"]) * visible).mean()
        else:
            visible = torch.ones_like(visible, dtype=torch.float32, device=visible.device)
            loss = (self.fn[dataset_name]._compute_loss(model_output["audio"], 
                    model_output["targets"]) * visible).mean()
        return loss * self.loss_weight
    
    def build_loss_fn(self, **params):
        # fn = nn.ModuleDict()
        # params['reduction'] = 'mean'
        # fn['audio'] = AAMSoftmaxLoss(**params)
        params['reduction'] = 'none'
        fn = AAMSoftmaxLoss(**params)
        return fn


@registry.register_loss("verification")
class VerificationLoss(TripletLoss):
    def __init__(self, **params):
        super().__init__(**params)
        self.loss_intra_weight = params.get('loss_intra_weight', 1.0)
        self.loss_cross_weight = params.get('loss_cross_weight', 1.0)
    
    def forward(self, sample_list, model_output):
        faces = F.normalize(model_output["faces"], dim=-1)
        audio = F.normalize(model_output["audio"], dim=-1)
        N, C = faces.shape
        faces = faces.view(-1, 2, C)
        audio = audio.view(-1, 2, C)
        assert faces.shape == audio.shape, 'face audio dimension mismatch'
        loss = 0
        loss += self.triplet_loss_fn(faces, faces) * self.loss_intra_weight # F vs F
        loss += self.triplet_loss_fn(audio, audio) * self.loss_intra_weight # A vs A
        # loss += self.triplet_loss_fn(faces, audio) * self.loss_cross_weight # F vs A
        # loss += self.triplet_loss_fn(audio, faces) * self.loss_cross_weight # A vs F
        return loss


@registry.register_loss("synchronization_triplet")
class SyncLoss(TripletLoss):
    def __init__(self, **params):
        super().__init__(**params)
        self.loss_weight = params.get('loss_weight', 1.0)
    
    def forward(self, sample_list, model_output):
        video = F.normalize(model_output["video"], dim=-1).unsqueeze(1)
        utter = F.normalize(model_output["utter"], dim=-1).unsqueeze(1)
        assert video.shape == utter.shape, 'video utterance dimension mismatch'
        pairs = torch.cat([video, utter], dim=1)
        loss = self.triplet_loss_fn(pairs, pairs)
        return loss * self.loss_weight


@registry.register_loss("synchronization")
class SyncContrastiveLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.temp = params.get('temp', 0.07)
        self.loss_weight = params.get('loss_weight', 1.0)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, sample_list, model_output):
        video = F.normalize(model_output["video"], dim=-1)
        utter = F.normalize(model_output["utter"], dim=-1)
        assert video.shape == utter.shape, 'video audio dimension mismatch'
        N, C = video.shape
        static_mask = torch.tensor(model_output["static"], device=video.device)
        keep_mask = ~static_mask
        score = torch.mm(utter, video.T) / self.temp
        target = torch.arange(0, N, dtype=torch.long).to(score.device)
        return self.loss_fn(score[keep_mask, :], target[keep_mask]) * self.loss_weight


@registry.register_loss("active")
class ActiveSpeakerLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.loss_weight = params.get('loss_weight', 1.0)
        self.loss_fn = nn.BCELoss()
    
    def forward(self, sample_list, model_output):
        video = F.normalize(model_output["video"], dim=-1)
        utter = F.normalize(model_output["utter"], dim=-1)
        assert video.shape == utter.shape, 'video audio dimension mismatch'
        N, C = video.shape
        label = ~torch.tensor(model_output["static"], device=video.device)
        label = label.float().clamp(0.0, 1.0)
        score = torch.multiply(utter, video).sum(-1).sigmoid()
        return self.loss_fn(score, label) * self.loss_weight


@registry.register_loss("mse")
class MSELoss(nn.Module):
    """Mean Squared Error loss"""
    def __init__(self, **params):
        super().__init__()
        self.loss_weight = params.get('loss_weight', 1.0)
        self.loss_fn = nn.MSELoss(reduction='sum')
    
    def forward(self, sample_list, model_output):
        score = model_output["scores"]
        target = model_output["targets"]
        bs = score.shape[0]
        return self.loss_fn(score, target) / bs * self.loss_weight


@registry.register_loss("proto")
class MSEPROLoss(nn.Module):
    """Mean Squared Error + Prototypical loss"""
    def __init__(self, **params):
        super().__init__()
        self.loss_weight = params.get('loss_weight', 1.0)
        init_w=10.0
        init_b=-5.0
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_fn  = torch.nn.CrossEntropyLoss()
    
    def forward(self, sample_list, model_output):
        embed = model_output["embed"]
        target = model_output["targets"]
        out_anchor      = torch.mean(embed[:,1:,:],1)
        out_positive    = embed[:,0,:]
        stepsize        = out_anchor.shape[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label   = torch.arange(0,stepsize, device=embed.device)
        loss   = self.loss_fn(cos_sim_matrix, label)
        return loss * self.loss_weight


@registry.register_loss("bcl")
class BCLLoss(nn.Module):
    """Ball Clustering Learning loss"""
    def __init__(self, **params):
        super().__init__()
        self.loss_weight = params.get('loss_weight', 1.0)
        self.bias = nn.Parameter(torch.tensor(0.7))
        self.loss_fn = nn.MSELoss(reduction='sum')
    
    def forward(self, sample_list, model_output):
        score = model_output["scores"]
        target = model_output["targets"]
        pos_similarity = torch.masked_select(score, torch.tensor(target, dtype=torch.bool))
        neg_similarity = torch.masked_select(score, ~torch.tensor(target, dtype=torch.bool))
        pos_loss = 0.5 * (F.relu(self.bias - pos_similarity.mean()))
        neg_loss = neg_similarity.mean()
        # Loss = 0.5 * pos_distances_sq   +   0.5 * (max(0, m - neg_distances))^2
        bcl_loss = 0.5 * (pos_loss + neg_loss)
        bs = score.shape[0]
        mse_loss = self.loss_fn(score, target) / bs
        loss = 40*bcl_loss + mse_loss
        return loss * self.loss_weight


@registry.register_loss("multi")
class MultiLoss(nn.Module):
    """A loss for combining multiple losses with weights.

    Args:
        params (List(Dict)): A list containing parameters for each different loss
                             and their weights.

    Example::

        # MultiLoss works with config like below where each loss's params and
        # weights are defined
        losses:
        - type: multi
          params:
          - type: amsfotmax
            weight: 0.3
            params: {}
          - type: verification
            weight: 0.7
            params: {}

    """

    def __init__(self, params):
        super().__init__()
        self.losses = []
        self.losses_weights = []

        self.loss_names = []

        for loss_params in params["params"]:
            self.loss_names.append(loss_params["type"])
            loss_fn = MMSCLoss(loss_params)
            loss_weight = loss_params.get("weight", {})
            self.losses.append(loss_fn)
            self.losses_weights.append(loss_weight)

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `attentions` attribute.
            model_output (Dict): Model output containing `attention_supervision`
                                 attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        loss = 0
        for idx, loss_fn in enumerate(self.losses):
            value = loss_fn(sample_list, model_output, *args, **kwargs)
            loss += self.losses_weights[idx] * list(value.values())[0]
        return loss


@registry.register_loss("logit_bce")
class LogitBinaryCrossEntropy(nn.Module):
    """Returns Binary Cross Entropy for logits.

    Attention:
        `Key`: logit_bce
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy for logits

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("bce")
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        loss = F.binary_cross_entropy(scores, targets, reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("nll_loss")
class NLLLoss(nn.Module):
    """Negative log likelikehood loss."""

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the negative log likelihood.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        _, idx = targets.max(dim=1)
        loss = F.nll_loss(scores, idx, reduction="mean")

        return loss * targets.size(1)


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


@registry.register_loss("weighted_softmax")
class WeightedSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss("softmax_kldiv")
class SoftmaxKlDivLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**params)

    def forward(self, sample_list, model_output):
        return self.loss_fn(model_output["scores"], sample_list["targets"])


@registry.register_loss("soft_label_cross_entropy")
class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean", normalize_targets=True):
        assert reduction in (
            "mean",
            "sum",
        ), "Argument `reduction` only supports `mean` and `sum`"

        super().__init__()

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.normalize_targets = normalize_targets
        self.eps = torch.finfo(torch.float32).eps

    @staticmethod
    def convert_to_one_hot(targets, n_classes):
        one_hot_targets = torch.zeros(
            (targets.size(0), n_classes), dtype=torch.long, device=targets.device
        )
        one_hot_targets.scatter_(1, targets.long().view(-1, 1), 1)
        return one_hot_targets

    def compute_loss(self, targets, scores):
        """for N examples and C classes
        - scores: N x C these are raw outputs (without softmax/sigmoid)
        - targets: N x C or N corresponding targets

        Target elements set to ignore_index contribute 0 loss.

        Samples where all entries are ignore_index do not contribute to the loss
        reduction.
        """

        assert targets.size(0) == scores.size(
            0
        ), "`targets` and `scores` should have the same batch size"

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
            mask = targets.ne(self.ignore_index).float()  # mask out `ignore_index`
        else:
            mask = targets.sum(-1, keepdim=True).ne(0).float()  # mask out zero rows

        if targets.size(1) == 1:
            targets = self.convert_to_one_hot(targets, scores.size(1))
        targets = targets.float() * mask

        if self.normalize_targets:
            targets /= self.eps + targets.sum(dim=1, keepdim=True)

        per_sample_per_target_loss = -targets * F.log_softmax(scores, dim=-1)
        per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
        loss = per_sample_loss.sum()
        # perform reduction
        if self.reduction == "mean":
            # normalize based on the number of samples with > 0 non-ignored targets
            loss /= torch.sum(torch.sum(mask, -1) > 0).clamp(min=1)
        return loss

    def forward(self, sample_list, model_output):
        return self.compute_loss(sample_list["targets"], model_output["scores"])


@registry.register_loss("contrastive_loss")
class ContrastiveLoss(nn.Module):
    """
    This is a generic contrastive loss typically used for pretraining. No modality
    assumptions are made here.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        assert (
            "embedding_1" in model_output and "embedding_2" in model_output
        ), "Embedding names must be available before loss calculation"
        embedding_1 = model_output["embedding_1"]
        embedding_2 = model_output["embedding_2"]

        mma = embedding_1 @ embedding_2.T
        labels = torch.arange(mma.shape[0], device=mma.device)
        loss1 = F.cross_entropy(mma, labels)
        loss2 = F.cross_entropy(mma.T, labels)
        return (loss1 + loss2) / 2


@registry.register_loss("cos_emb_loss")
class CosineEmbeddingLoss(nn.Module):
    """Cosine embedding loss"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, sample_list, model_output):
        targets = sample_list["targets"]
        scores = model_output["scores"]
        y = torch.ones(targets.size(0)).to(targets.device)
        loss = self.loss_fn(scores, targets, y)
        return loss