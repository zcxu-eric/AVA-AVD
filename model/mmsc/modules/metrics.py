# Copyright (c) Facebook, Inc. and its affiliates.
"""
The metrics module contains implementations of various metrics used commonly to
understand how well our models are performing. For e.g. accuracy etc.

For implementing your own metric, you need to follow these steps:

1. Create your own metric class and inherit ``BaseMetric`` class.
2. In the ``__init__`` function of your class, make sure to call
   ``super().__init__('name')`` where 'name' is the name of your metric. If
   you require any parameters in your ``__init__`` function, you can use
   keyword arguments to represent them and metric constructor will take care of
   providing them to your class from config.
3. Implement a ``calculate`` function which takes in ``SampleList`` and
   `model_output` as input and return back a float tensor/number.
4. Register your metric with a key 'name' by using decorator,
   ``@registry.register_metric('name')``.

Example::

    import torch

    from mmsc.common.registry import registry
    from mmsc.modules.metrics import BaseMetric

    @registry.register_metric("some")
    class SomeMetric(BaseMetric):
        def __init__(self, some_param=None):
            super().__init__("some")
            ....

        def calculate(self, sample_list, model_output):
            metric = torch.tensor(2, dtype=torch.float)
            return metric

Example config for above metric::

    model_config:
        pythia:
            metrics:
            - type: some
              params:
                some_param: a
"""

import collections
import warnings
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from mmsc.common.registry import registry
from mmsc.utils.logger import log_class_usage
from mmsc.modules.postprocess import Postprocessor, Postprocessor_class
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)
from torch import Tensor
from omegaconf import DictConfig


def _convert_to_one_hot(expected, output):
    # This won't get called in case of multilabel, only multiclass or binary
    # as multilabel will anyways be multi hot vector
    if output.squeeze().dim() != expected.squeeze().dim() and expected.dim() == 1:
        expected = torch.nn.functional.one_hot(
            expected.long(), num_classes=output.size(-1)
        ).float()
    return expected


class Metrics:
    """Internally used by MMSC, Metrics acts as wrapper for handling
    calculation of metrics over various metrics specified by the model in
    the config. It initializes all of the metrics and when called it runs
    calculate on each of them one by one and returns back a dict with proper
    naming back. For e.g. an example dict returned by Metrics class:
    ``{'val/vqa_accuracy': 0.3, 'val/r@1': 0.8}``

    Args:
        metric_list (ListConfig): List of DictConfigs where each DictConfig
                                        specifies name and parameters of the
                                        metrics used.
    """

    def __init__(self, metric_list):
        if not isinstance(metric_list, collections.abc.Sequence):
            metric_list = [metric_list]

        self.metrics = self._init_metrics(metric_list)

    def _init_metrics(self, metric_list):
        metrics = {}
        self.required_params = {"dataset_name", "dataset_type"}
        for metric in metric_list:
            params = {}
            dataset_names = []
            if isinstance(metric, collections.abc.Mapping):
                if "type" not in metric:
                    raise ValueError(
                        f"Metric {metric} needs to have 'type' attribute "
                        + "or should be a string"
                    )
                metric_type = key = metric.type
                params = metric.get("params", {})
                # Support cases where uses need to give custom metric name
                if "key" in metric:
                    key = metric.key

                # One key should only be used once
                if key in metrics:
                    raise RuntimeError(
                        f"Metric with type/key '{metric_type}' has been defined more "
                        + "than once in metric list."
                    )

                # a custom list of dataset where this metric will be applied
                if "datasets" in metric:
                    dataset_names = metric.datasets
            else:
                if not isinstance(metric, str):
                    raise TypeError(
                        "Metric {} has inappropriate type"
                        "'dict' or 'str' allowed".format(metric)
                    )
                metric_type = key = metric

            metric_cls = registry.get_metric_class(metric_type)
            if metric_cls is None:
                raise ValueError(
                    f"No metric named {metric_type} registered to registry"
                )

            metric_instance = metric_cls(**params)
            metric_instance.name = key
            metric_instance.set_applicable_datasets(dataset_names)

            metrics[key] = metric_instance
            self.required_params.update(metrics[key].required_params)

        return metrics

    def __call__(self, sample_list, model_output, *args, **kwargs):
        values = {}

        dataset_type = sample_list.dataset_type
        dataset_name = sample_list.dataset_name

        with torch.no_grad():
            for metric_name, metric_object in self.metrics.items():
                if not metric_object.is_dataset_applicable(dataset_name):
                    continue
                key = f"{dataset_type}/{dataset_name}/{metric_name}"
                values[key] = metric_object._calculate_with_checks(
                    sample_list, model_output, *args, **kwargs
                )

                if not isinstance(values[key], torch.Tensor):
                    values[key] = torch.tensor(values[key], dtype=torch.float)
                else:
                    values[key] = values[key].float()

                if values[key].dim() == 0:
                    values[key] = values[key].view(1)

        registry.register(
            "{}.{}.{}".format("metrics", sample_list.dataset_name, dataset_type), values
        )

        return values


class BaseMetric:
    """Base class to be inherited by all metrics registered to MMSC. See
    the description on top of the file for more information. Child class must
    implement ``calculate`` function.

    Args:
        name (str): Name of the metric.

    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.required_params = ["scores", "targets"]
        # the set of datasets where this metric will be applied
        # an empty set means it will be applied on *all* datasets
        self._dataset_names = set()
        log_class_usage("Metric", self.__class__)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Abstract method to be implemented by the child class. Takes
        in a ``SampleList`` and a dict returned by model as output and
        returns back a float tensor/number indicating value for this metric.

        Args:
            sample_list (SampleList): SampleList provided by the dataloader for the
                                current iteration.
            model_output (Dict): Output dict from the model for the current
                                 SampleList

        Returns:
            torch.Tensor|float: Value of the metric.

        """
        # Override in your child class
        raise NotImplementedError("'calculate' must be implemented in the child class")

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def _calculate_with_checks(self, *args, **kwargs):
        value = self.calculate(*args, **kwargs)
        return value

    def set_applicable_datasets(self, dataset_names):
        self._dataset_names = set(dataset_names)

    def is_dataset_applicable(self, dataset_name):
        return len(self._dataset_names) == 0 or dataset_name in self._dataset_names


@registry.register_metric("fewshot_acc")
class FewShotAcc(BaseMetric):
    """Metric for fewshot classification accuracy
    """
    def __init__(self, **params):
        super().__init__('fewshot_acc')
    
    def calculate(self, sample_list, model_output):
        scores = model_output['scores']
        targets = model_output['targets']
        correct = (scores.argmax(2) == targets.argmax(2))
        acc = correct.sum() / (correct.shape[0] * correct.shape[1])
        return acc


@registry.register_metric("der")
class DiarizationErrorRate(BaseMetric):
    """Metric for calculating diarization error rate
    """
    def __init__(self, **params):
        super().__init__('der')
        if isinstance(params, dict):
            self.config = DictConfig(params)
        self.required_params = ['feat_audio', 'feat_video', 'video', 
                                'start', 'end', 'visible', 'trackid']

    def calculate(self, sample_list, model_output):
        dataset_type = model_output.dataset_type
        postprocessor = Postprocessor(self.config)
        der = postprocessor(model_output, dataset_type)
        der = torch.tensor([der], device=model_output.model.device)
        del postprocessor, model_output
        torch.cuda.empty_cache()
        return der


@registry.register_metric("cder")
class DiarizationErrorRate(BaseMetric):
    """Metric for calculating diarization error rate
    """
    def __init__(self, **params):
        super().__init__('cder')
        if isinstance(params, dict):
            self.config = DictConfig(params)
        self.required_params = ['feat_audio', 'feat_video', 'video', 
                                'start', 'end', 'visible']

    def calculate(self, sample_list, model_output):
        dataset_type = model_output.dataset_type
        postprocessor = Postprocessor_class(self.config)
        der = postprocessor(model_output, dataset_type)
        der = torch.tensor([der], device=model_output.model.device)
        del postprocessor, model_output
        torch.cuda.empty_cache()
        return der


@registry.register_metric("eer")
class EqualErrorRate(BaseMetric):
    """Metric for calculating equal error rate
    """
    def __init__(self, **params):
        super().__init__('eer')
        if isinstance(params, dict):
            self.config = DictConfig(params)
            
        self.normalize = self.config.normalize

        if self.config.testlist.endswith('npy'):
            test_list = np.load(self.config.testlist, allow_pickle=True)
            self.test_list = [[l, ':'.join(map(str, ref)), ':'.join(map(str, com))] 
                               for l, ref, com in test_list]
        else:
            with open(self.config.testlist, 'r') as f:
                pairs = f.readlines()
            self.test_list = [p.split() for p in pairs]
    
    def _calculate(self, sample_list, model_output):
        scores = []
        labels = []
        device = model_output.scores.device
        embeddings = collections.defaultdict()
        for embed, uid in zip(model_output.scores, model_output.identifier):
            embeddings[uid] = embed
        
        for line in self.test_list:
            label, ref, com = line
            ref_embed = embeddings[ref].unsqueeze(0)
            com_embed = embeddings[com].unsqueeze(0)
            
            if self.normalize:
                ref_embed = F.normalize(ref_embed, dim=-1)
                com_embed = F.normalize(com_embed, dim=-1)
            
            dist = F.pairwise_distance(ref_embed, com_embed).detach().cpu().numpy()
            score = -1 * np.mean(dist)

            scores.append(score)
            labels.append(int(label))

        return self._compute_error_rate(scores, labels, device)
    
    def _compute_error_rate(self, scores, labels, device, target_fa=[1, 0.1], target_fr=None):

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        tunedThreshold = []

        if target_fr:
            for tfr in target_fr:
                idx = np.nanargmin(np.absolute((tfr - fnr)))
                tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
        
        for tfa in target_fa:
            idx = np.nanargmin(np.absolute((tfa - fpr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
        
        idxE = np.nanargmin(np.absolute((fnr - fpr)))
        eer  = max(fpr[idxE],fnr[idxE])*100.0
        eer = torch.tensor([eer], device=device)
        return eer


@registry.register_metric("voice_eer")
class VoiceEER(EqualErrorRate):
    """Metric for calculating AVA voice equal error rate
    """
    def __init__(self, **params):
        super().__init__(**params)
        self.required_params = ['audio', 'identifier']

    def calculate(self, sample_list, model_output):
        model_output.scores = model_output.audio
        return self._calculate(model_output, model_output)


@registry.register_metric("face_eer")
class FaceEER(EqualErrorRate):
    """Metric for calculating AVA face equal error rate
    """
    def __init__(self, **params):
        super().__init__(**params)
        self.required_params = ['faces', 'identifier']

    def calculate(self, sample_list, model_output):
        model_output.scores = model_output.faces
        return self._calculate(model_output, model_output)


@registry.register_metric("pre_eer")
class VoiceEER(EqualErrorRate):
    """Metric for calculating equal error rate using sim scores
    """
    def __init__(self, **params):
        super().__init__(**params)
        self.required_params = ['scores']

    def calculate(self, sample_list, model_output):
        device = model_output.scores.device
        scores = model_output.scores.cpu().numpy()
        labels = [int(p[0]) for p in self.test_list]
        return self._compute_error_rate(scores, labels, device)


@registry.register_metric("sync_veri_loss")
class ValidationLoss(BaseMetric):
    """Metric for collect sync and veri val loss
    """
    def __init__(self, **params):
        super().__init__('sync_veri_loss')
    
    def calculate(self, sample_list, model_output):
        loss_sync_avg = model_output['losses']['val/voxceleb/synchronization']
        loss_veri_avg = model_output['losses']['val/voxceleb/verification']
        return loss_sync_avg + loss_veri_avg


@registry.register_metric("sync_acc")
class SynchronizationAcc(BaseMetric):
    """Metric for synchronization accuracy
    """
    def __init__(self, **params):
        super().__init__('sync_acc')
    
    def calculate(self, sample_list, model_output):
        video = F.normalize(model_output["video"], dim=-1)
        utter = F.normalize(model_output["utter"], dim=-1)
        assert video.shape == utter.shape, 'video audio dimension mismatch'
        N, C = video.shape
        similarity = torch.mm(video, utter.T) / self.temp
        target = torch.arange(0, N, dtype=torch.long).to(similarity.device)
        acc = (similarity == target).sum() / N * 100.0
        return acc


@registry.register_metric("veri_acc")
class VerificationAcc(BaseMetric):
    """Metric for collect sync and veri val loss
    """
    def __init__(self, **params):
        super().__init__('sync_veri_loss')
    
    def calculate(self, sample_list, model_output):
        loss_sync_avg = model_output['losses']['val/voxceleb/synchronization']
        loss_veri_avg = model_output['losses']['val/voxceleb/verification']
        return loss_sync_avg + loss_veri_avg


@registry.register_metric("accuracy")
class Accuracy(BaseMetric):
    """Metric for calculating accuracy.

    **Key:** ``accuracy``
    """

    def __init__(self, score_key="scores", target_key="targets", topk=1):
        super().__init__("accuracy")
        self.score_key = score_key
        self.target_key = target_key
        self.topk = topk

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: accuracy.

        """
        output = model_output[self.score_key]
        batch_size = output.shape[0]
        expected = sample_list[self.target_key]

        assert (
            output.dim() <= 2
        ), "Output from model shouldn't have more than dim 2 for accuracy"
        assert (
            expected.dim() <= 2
        ), "Expected target shouldn't have more than dim 2 for accuracy"

        if output.dim() == 2:
            output = output.topk(self.topk, 1, True, True)[1].t().squeeze()

        # If more than 1
        # If last dim is 1, we directly have class indices
        if expected.dim() == 2 and expected.size(-1) != 1:
            expected = expected.topk(self.topk, 1, True, True)[1].t().squeeze()

        correct = (expected == output.squeeze()).sum().float()
        return correct / batch_size


@registry.register_metric("topk_accuracy")
class TopKAccuracy(Accuracy):
    def __init__(self, score_key: str, k: int):
        super().__init__(score_key=score_key, topk=k)


class RecallAtK(BaseMetric):
    def __init__(self, name="recall@k"):
        super().__init__(name)

    def score_to_ranks(self, scores):
        # sort in descending order - largest score gets highest rank
        sorted_ranks, ranked_idx = scores.sort(1, descending=True)

        # convert from ranked_idx to ranks
        ranks = ranked_idx.clone().fill_(0)
        for i in range(ranked_idx.size(0)):
            for j in range(100):
                ranks[i][ranked_idx[i][j]] = j
        ranks += 1
        return ranks

    def get_gt_ranks(self, ranks, ans_ind):
        _, ans_ind = ans_ind.max(dim=1)
        ans_ind = ans_ind.view(-1)
        gt_ranks = torch.LongTensor(ans_ind.size(0))

        for i in range(ans_ind.size(0)):
            gt_ranks[i] = int(ranks[i, ans_ind[i].long()])
        return gt_ranks

    def process_ranks(self, ranks):
        num_opts = 100

        # none of the values should be 0, there is gt in options
        if torch.sum(ranks.le(0)) > 0:
            num_zero = torch.sum(ranks.le(0))
            warnings.warn(f"Some of ranks are zero: {num_zero}")
            ranks = ranks[ranks.gt(0)]

        # rank should not exceed the number of options
        if torch.sum(ranks.ge(num_opts + 1)) > 0:
            num_ge = torch.sum(ranks.ge(num_opts + 1))
            warnings.warn(f"Some of ranks > 100: {num_ge}")
            ranks = ranks[ranks.le(num_opts + 1)]
        return ranks

    def get_ranks(self, sample_list, model_output, *args, **kwargs):
        output = model_output["scores"]
        expected = sample_list["targets"]

        ranks = self.score_to_ranks(output)
        gt_ranks = self.get_gt_ranks(ranks, expected)

        ranks = self.process_ranks(gt_ranks)
        return ranks.float()

    def calculate(self, sample_list, model_output, k, *args, **kwargs):
        ranks = self.get_ranks(sample_list, model_output)
        recall = float(torch.sum(torch.le(ranks, k))) / ranks.size(0)
        return recall


@registry.register_metric("r@1")
class RecallAt1(RecallAtK):
    """
    Calculate Recall@1 which specifies how many time the chosen candidate
    was rank 1.

    **Key**: ``r@1``.
    """

    def __init__(self):
        super().__init__("r@1")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall@1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Recall@1

        """
        return super().calculate(sample_list, model_output, k=1)


@registry.register_metric("r@5")
class RecallAt5(RecallAtK):
    """
    Calculate Recall@5 which specifies how many time the chosen candidate
    was among first 5 rank.

    **Key**: ``r@5``.
    """

    def __init__(self):
        super().__init__("r@5")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall@5 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Recall@5

        """
        return super().calculate(sample_list, model_output, k=5)


@registry.register_metric("r@10")
class RecallAt10(RecallAtK):
    """
    Calculate Recall@10 which specifies how many time the chosen candidate
    was among first 10 ranks.

    **Key**: ``r@10``.
    """

    def __init__(self):
        super().__init__("r@10")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall@10 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Recall@10

        """
        return super().calculate(sample_list, model_output, k=10)


@registry.register_metric("mean_r")
class MeanRank(RecallAtK):
    """
    Calculate MeanRank which specifies what was the average rank of the chosen
    candidate.

    **Key**: ``mean_r``.
    """

    def __init__(self):
        super().__init__("mean_r")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Mean Rank and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: mean rank

        """
        ranks = self.get_ranks(sample_list, model_output)
        return torch.mean(ranks)


@registry.register_metric("mean_rr")
class MeanReciprocalRank(RecallAtK):
    """
    Calculate reciprocal of mean rank..

    **Key**: ``mean_rr``.
    """

    def __init__(self):
        super().__init__("mean_rr")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Mean Reciprocal Rank and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Mean Reciprocal Rank

        """
        ranks = self.get_ranks(sample_list, model_output)
        return torch.mean(ranks.reciprocal())


@registry.register_metric("f1")
class F1(BaseMetric):
    """Metric for calculating F1. Can be used with type and params
    argument for customization. params will be directly passed to sklearn
    f1 function.
    **Key:** ``f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("f1")
        self._multilabel = kwargs.pop("multilabel", False)
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate f1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: f1.
        """
        scores = model_output["scores"]
        expected = sample_list["targets"]

        if self._multilabel:
            output = torch.sigmoid(scores)
            output = torch.round(output)
            expected = _convert_to_one_hot(expected, output)
        else:
            # Multiclass, or binary case
            output = scores.argmax(dim=-1)
            if expected.dim() != 1:
                # Probably one-hot, convert back to class indices array
                expected = expected.argmax(dim=-1)

        value = f1_score(expected.cpu(), output.cpu(), **self._sk_kwargs)

        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("macro_f1")
class MacroF1(F1):
    """Metric for calculating Macro F1.

    **Key:** ``macro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "macro_f1"


@registry.register_metric("micro_f1")
class MicroF1(F1):
    """Metric for calculating Micro F1.

    **Key:** ``micro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "micro_f1"


@registry.register_metric("binary_f1")
class BinaryF1(F1):
    """Metric for calculating Binary F1.

    **Key:** ``binary_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", labels=[1], **kwargs)
        self.name = "binary_f1"


@registry.register_metric("multilabel_f1")
class MultiLabelF1(F1):
    """Metric for calculating Multilabel F1.

    **Key:** ``multilabel_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(multilabel=True, **kwargs)
        self.name = "multilabel_f1"


@registry.register_metric("multilabel_micro_f1")
class MultiLabelMicroF1(MultiLabelF1):
    """Metric for calculating Multilabel Micro F1.

    **Key:** ``multilabel_micro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "multilabel_micro_f1"


@registry.register_metric("multilabel_macro_f1")
class MultiLabelMacroF1(MultiLabelF1):
    """Metric for calculating Multilabel Macro F1.

    **Key:** ``multilabel_macro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "multilabel_macro_f1"


@registry.register_metric("roc_auc")
class ROC_AUC(BaseMetric):
    """Metric for calculating ROC_AUC.
    See more details at `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ # noqa

    **Note**: ROC_AUC is not defined when expected tensor only contains one
    label. Make sure you have both labels always or use it on full val only

    **Key:** ``roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("roc_auc")
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate ROC_AUC and returns it back. The function performs softmax
        on the logits provided and then calculated the ROC_AUC.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: ROC_AUC.

        """

        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)
        expected = sample_list["targets"]
        expected = _convert_to_one_hot(expected, output)
        value = roc_auc_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("micro_roc_auc")
class MicroROC_AUC(ROC_AUC):
    """Metric for calculating Micro ROC_AUC.

    **Key:** ``micro_roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "micro_roc_auc"


@registry.register_metric("macro_roc_auc")
class MacroROC_AUC(ROC_AUC):
    """Metric for calculating Macro ROC_AUC.

    **Key:** ``macro_roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "macro_roc_auc"


@registry.register_metric("ap")
class AveragePrecision(BaseMetric):
    """Metric for calculating Average Precision.
    See more details at `sklearn.metrics.average_precision_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score>`_ # noqa
    If you are looking for binary case, please take a look at binary_ap
    **Key:** ``ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("ap")
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate AP and returns it back. The function performs softmax
        on the logits provided and then calculated the AP.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: AP.

        """

        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)
        expected = sample_list["targets"]
        expected = _convert_to_one_hot(expected, output)
        value = average_precision_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("binary_ap")
class BinaryAP(AveragePrecision):
    """Metric for calculating Binary Average Precision.
    See more details at `sklearn.metrics.average_precision_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score>`_ # noqa
    **Key:** ``binary_ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.name = "binary_ap"

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Binary AP and returns it back. The function performs softmax
        on the logits provided and then calculated the binary AP.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: AP.

        """

        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)
        # Take the score for positive (1) label
        output = output[:, 1]
        expected = sample_list["targets"]

        # One hot format -> Labels
        if expected.dim() == 2:
            expected = expected.argmax(dim=1)

        value = average_precision_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("micro_ap")
class MicroAP(AveragePrecision):
    """Metric for calculating Micro Average Precision.

    **Key:** ``micro_ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "micro_ap"


@registry.register_metric("macro_ap")
class MacroAP(AveragePrecision):
    """Metric for calculating Macro Average Precision.

    **Key:** ``macro_ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "macro_ap"


@registry.register_metric("r@pk")
class RecallAtPrecisionK(BaseMetric):
    """Metric for calculating recall when precision is above a
    particular threshold. Use `p_threshold` param to specify the
    precision threshold i.e. k. Accepts precision in both 0-1
    and 1-100 format.

    **Key:** ``r@pk``
    """

    def __init__(self, p_threshold, *args, **kwargs):
        """Initialization function recall @ precision k

        Args:
            p_threshold (float): Precision threshold
        """
        super().__init__(name="r@pk")
        self.name = "r@pk"
        self.p_threshold = p_threshold if p_threshold < 1 else p_threshold / 100

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall at precision k and returns it back. The function
        performs softmax on the logits provided and then calculated the metric.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: Recall @ precision k.

        """
        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)[:, 1]
        expected = sample_list["targets"]

        # One hot format -> Labels
        if expected.dim() == 2:
            expected = expected.argmax(dim=1)

        precision, recall, thresh = precision_recall_curve(expected.cpu(), output.cpu())

        try:
            value, _ = max(
                (r, p) for p, r in zip(precision, recall) if p >= self.p_threshold
            )
        except ValueError:
            value = 0

        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("r@k_retrieval")
class RecallAtK_ret(BaseMetric):
    def __init__(self, name="recall@k"):
        super().__init__(name)

    def _get_RatK_multi(
        self, correlations: Tensor, labels: Tensor, k: int, factor: int
    ):
        _, top_k_ids = torch.topk(correlations, k, dim=1)
        hits = (
            torch.logical_and(
                labels[:, None] <= top_k_ids, top_k_ids < labels[:, None] + factor
            )
            .long()
            .max(dim=1)[0]
        )
        return hits

    def calculate(
        self,
        sample_list: Dict[str, Tensor],
        model_output: Dict[str, Tensor],
        k: int,
        flip=False,
        *args,
        **kwargs,
    ):
        # calculate image to text retrieval recalls
        # correlations shape is either BxB or Bx(5B)
        # when flip=True, calculate text to image
        image_embeddings = model_output["scores"]
        text_embeddings = model_output["targets"]

        correlations = image_embeddings @ text_embeddings.t()  # B x B or Bx5B
        assert correlations.shape[1] % correlations.shape[0] == 0
        batch_size = correlations.shape[0]
        factor = correlations.shape[1] // correlations.shape[0]
        labels = torch.arange(batch_size, device=image_embeddings.device) * factor
        if flip:
            correlations = correlations.t()  # 5B x B
            labels = torch.arange(batch_size, device=image_embeddings.device)
            labels = labels[:, None].expand(-1, factor).flatten()
            factor = 1
        hits = self._get_RatK_multi(correlations, labels, k, factor)
        ratk = hits.sum().float() / hits.shape[0]
        return ratk


@registry.register_metric("r@1_retrieval")
class RecallAt1_ret(RecallAtK_ret):
    def __init__(self):
        super().__init__("r@1")

    def calculate(
        self,
        sample_list: Dict[str, Tensor],
        model_output: Dict[str, Tensor],
        *args,
        **kwargs,
    ):
        ratk = super().calculate(sample_list, model_output, 1)
        return ratk


@registry.register_metric("r@1_rev_retrieval")
class RecallAt1_rev_ret(RecallAtK_ret):
    def __init__(self):
        super().__init__("r@1_rev")

    def calculate(
        self,
        sample_list: Dict[str, Tensor],
        model_output: Dict[str, Tensor],
        *args,
        **kwargs,
    ):
        ratk = super().calculate(sample_list, model_output, 1, flip=True)
        return ratk


@registry.register_metric("r@5_retrieval")
class RecallAt5_ret(RecallAtK_ret):
    def __init__(self):
        super().__init__("r@5")

    def calculate(
        self,
        sample_list: Dict[str, Tensor],
        model_output: Dict[str, Tensor],
        *args,
        **kwargs,
    ):
        ratk = super().calculate(sample_list, model_output, 5)
        return ratk


@registry.register_metric("r@5_rev_retrieval")
class RecallAt5_rev_ret(RecallAtK_ret):
    def __init__(self):
        super().__init__("r@5_rev")

    def calculate(
        self,
        sample_list: Dict[str, Tensor],
        model_output: Dict[str, Tensor],
        *args,
        **kwargs,
    ):
        ratk = super().calculate(sample_list, model_output, 5, flip=True)
        return ratk


@registry.register_metric("r@10_retrieval")
class RecallAt10_ret(RecallAtK_ret):
    def __init__(self):
        super().__init__("r@10")

    def calculate(
        self,
        sample_list: Dict[str, Tensor],
        model_output: Dict[str, Tensor],
        *args,
        **kwargs,
    ):
        ratk = super().calculate(sample_list, model_output, 10)
        return ratk


@registry.register_metric("r@10_rev_retrieval")
class RecallAt10_rev_ret(RecallAtK_ret):
    def __init__(self):
        super().__init__("r@10_rev")

    def calculate(
        self,
        sample_list: Dict[str, Tensor],
        model_output: Dict[str, Tensor],
        *args,
        **kwargs,
    ):
        ratk = super().calculate(sample_list, model_output, 10, flip=True)
        return ratk