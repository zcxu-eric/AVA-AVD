# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
import logging
from abc import ABC
from typing import Any, Dict, Tuple, Type

import torch
import tqdm
from caffe2.python.timeout_guard import CompleteInTimeOrDie
from mmsc.common.meter import Meter
from mmsc.common.report import Report
from mmsc.common.sample import Sample, to_device
from mmsc.utils.distributed import gather_tensor, is_master


logger = logging.getLogger(__name__)


class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, dataset_type: str, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        use_cpu = self.config.evaluation.get("use_cpu", False)
        loaded_batches = 0
        skipped_batches = 0

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()
            while reporter.next_dataset(flush_report=False):
                dataloader = reporter.get_dataloader()
                combined_report = None
                predict = 'avd' in reporter.current_datamodule.dataset_name
                if self._can_use_tqdm(dataloader):
                    dataloader = tqdm.tqdm(dataloader, disable=disable_tqdm)
                for batch in dataloader:
                    # Do not timeout quickly on first batch, as workers might start at
                    # very different times.
                    with CompleteInTimeOrDie(600 if loaded_batches else 3600 * 24):
                        loaded_batches += 1
                        prepared_batch = reporter.prepare_batch(batch)
                        prepared_batch = to_device(prepared_batch, self.device)
                        if not validate_batch_sizes(prepared_batch.get_batch_size()):
                            logger.info("Skip batch due to uneven batch sizes.")
                            skipped_batches += 1
                            continue
                        if predict:
                            model_output = self.model(prepared_batch, exec='extraction')
                        else:
                            model_output = self.model(prepared_batch)
                        report = Report(prepared_batch, model_output)
                        reporter.add_to_report(report, self.model)
                        report.detach()

                        meter.update_from_report(report, should_update_loss=not predict)

                        moved_report = report
                        # Move to CPU for metrics calculation later if needed
                        # Explicitly use `non_blocking=False` as this can cause
                        # race conditions in next accumulate
                        if use_cpu:
                            moved_report = report.copy().to("cpu", non_blocking=False)

                        # accumulate necessary params for metric calculation
                        if combined_report is None:
                            # make a copy of report since `reporter.add_to_report` will
                            # change some of the report keys later
                            combined_report = moved_report.copy()
                        else:
                            if predict:
                                combined_report._accumulate_tensor_fields(
                                    moved_report, self.metrics.required_params
                                )
                            else:
                                combined_report.accumulate_tensor_fields_and_loss(
                                    moved_report, self.metrics.required_params
                                )
                            combined_report.batch_size += moved_report.batch_size

                        # Each node generates a separate copy of predict JSON from the
                        # report, which will be used to evaluate dataset-level metrics
                        # (such as mAP in object detection or CIDEr in image captioning)
                        # Since `reporter.add_to_report` changes report keys,
                        # (e.g scores) do this after
                        # `combined_report.accumulate_tensor_fields_and_loss`
                        if "__prediction_report__" in self.metrics.required_params:
                            # Still need to use original report here on GPU/TPU since
                            # it will be gathered
                            reporter.add_to_report(report, self.model)

                        if single_batch is True:
                            break

                logger.info(f"Finished forward. Loaded {loaded_batches}")
                logger.info(f" -- skipped {skipped_batches} batches.")

                reporter.postprocess_dataset_report()
                assert (
                    combined_report is not None
                ), "Please check if your validation set is empty!"
                # add prediction_report is used for set-level metrics
                combined_report.prediction_report = reporter.report

                logger.info('Computing metrics')

                if predict:
                    combined_report.model = self.model
                    sample_list = Sample()
                    sample_list.dataset_type = combined_report.dataset_type
                    sample_list.dataset_name = combined_report.dataset_name
                    combined_report.metrics = self.metrics(sample_list, combined_report)
                else:
                    combined_report.metrics = self.metrics(combined_report, combined_report)

                # Since update_meter will reduce the metrics over GPUs, we need to
                # move them back to GPU but we will only move metrics and losses
                # which are needed by update_meter to avoid OOM
                # Furthermore, do it in a non_blocking way to avoid any issues
                # in device to host or host to device transfer
                if use_cpu:
                    combined_report = combined_report.to(
                        self.device, fields=["metrics", "losses"], non_blocking=False
                    )

                meter.update_from_report(combined_report, should_update_loss=False)

            # enable train mode again
            self.model.train()

        return combined_report, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        use_cpu = self.config.evaluation.get("use_cpu", False)
        skipped_batches = 0
        loaded_batches = 0
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()
                combined_report = None
                predict = 'avd' in reporter.current_datamodule.dataset_name
                extract = 'profile' in reporter.current_datamodule.dataset_name
                if self._can_use_tqdm(dataloader):
                    dataloader = tqdm.tqdm(dataloader)
                for batch in dataloader:
                    # Do not timeout quickly on first batch, as workers might start at
                    # very different times.
                    with CompleteInTimeOrDie(600 if loaded_batches else 3600 * 24):
                        prepared_batch = reporter.prepare_batch(batch)
                        prepared_batch = to_device(prepared_batch, self.device)
                        loaded_batches += 1
                        if not validate_batch_sizes(prepared_batch.get_batch_size()):
                            logger.info("Skip batch due to unequal batch sizes.")
                            skipped_batches += 1
                            continue
                        with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                            if predict or extract:
                                model_output = self.model(prepared_batch, exec='extraction')
                            else:
                                model_output = self.model(prepared_batch)
                        report = Report(prepared_batch, model_output)
                        reporter.add_to_report(report, self.model)
                        report.detach()

                        moved_report = report
                        # Move to CPU for metrics calculation later if needed
                        # Explicitly use `non_blocking=False` as this can cause
                        # race conditions in next accumulate
                        if use_cpu:
                            moved_report = report.copy().to("cpu", non_blocking=False)

                        # accumulate necessary params for metric calculation
                        if combined_report is None:
                            # make a copy of report since `reporter.add_to_report` will
                            # change some of the report keys later
                            combined_report = moved_report.copy()
                        else:
                            combined_report._accumulate_tensor_fields(
                                moved_report, self.metrics.required_params
                            )
                            combined_report.batch_size += moved_report.batch_size

                reporter.postprocess_dataset_report()

            logger.info(f"Finished predicting. Loaded {loaded_batches}")
            logger.info(f" -- skipped {skipped_batches} batches.")

            if predict:
                logger.info('Computing metrics')
                combined_report.model = self.model
                sample_list = Sample()
                sample_list.dataset_type = combined_report.dataset_type
                sample_list.dataset_name = combined_report.dataset_name
                combined_report.metrics = self.metrics(sample_list, combined_report)
            else:
                speaker_feat = defaultdict(list)
                for feat_video, feat_audio, video in zip(combined_report.feat_video, combined_report.feat_audio, combined_report.video):
                    speaker_feat[video[0]].append((feat_video.cpu(), feat_audio.cpu()))
                for v in speaker_feat:
                    torch.save(speaker_feat[v], f'{self.config.dataset_config.amiprofile.feat_path}/{v}.pkl')
            # enable train mode again
            self.model.train()

    def _can_use_tqdm(self, dataloader: torch.utils.data.DataLoader):
        """
        Checks whether tqdm can be gracefully used with a dataloader
        1) should have `__len__` property defined
        2) calling len(x) should not throw errors.
        """
        use_tqdm = hasattr(dataloader, "__len__")

        try:
            _ = len(dataloader)
        except (AttributeError, TypeError, NotImplementedError):
            use_tqdm = False
        return use_tqdm


def validate_batch_sizes(my_batch_size: int) -> bool:
    """
    Validates all workers got the same batch size.
    """
    batch_size_tensor = torch.IntTensor([my_batch_size])
    if torch.cuda.is_available():
        batch_size_tensor = batch_size_tensor.cuda()
    all_batch_sizes = gather_tensor(batch_size_tensor)
    for j, oth_batch_size in enumerate(all_batch_sizes.data):
        if oth_batch_size != my_batch_size:
            logger.error(f"Node {j} batch {oth_batch_size} != {my_batch_size}")
            return False
    return True
