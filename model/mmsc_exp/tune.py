#!/usr/bin/env python3 -u
import torch, logging
from mmsc.utils.options import options
from mmsc.utils.env import setup_imports
from mmsc.utils.configuration import Configuration, get_global_config
from mmsc.modules.postprocess import Postprocessor
from mmsc.utils.build import build_config
from mmsc.utils.diarization import metrics
from mmsc.common.registry import registry
from mmsc.utils.logger import setup_logger, setup_very_basic_config


setup_very_basic_config()


def main():
    setup_imports()
    parser = options.get_parser()
    args = parser.parse_args()
    configuration = Configuration(args)
    configuration.args = args
    config = build_config(configuration)

    setup_logger(
        color=config.training.colored_logs, disable=config.training.should_not_log
    )
    logger = logging.getLogger("mmsc_exp.finetune")
    # Log args for debugging purposes
    logger.info(configuration.args)

    postprocessor = Postprocessor(config.evaluation.metrics[0].params)
    run_type = get_global_config('run_type')
    path = f'{config.env.save_dir}/rttms/intermidiate_{run_type}.pkl'
    intermidiate = torch.load(path)

    ###################
    # print('======================== sim merge')
    # simvar = torch.load(path)['similarities']
    ###################
    similarities = intermidiate['similarities']
    # a_similarities = intermidiate['audio_similarities']
    # v_similarities = intermidiate['video_similarities']
    starts = intermidiate['starts']
    ends = intermidiate['ends']
    # masks = intermidiate['masks']

    # diarize
    for video in similarities:
        # # audio
        # similarity = a_similarities[video]

        # # relation
        # similarity = simvar[video].numpy()

        # # audio + video
        similarity = similarities[video]

        # # audio + relation
        # similarity = 0.5*a_similarities[video] + 0.5*simvar[video].numpy()

        # # video + relation
        # similarity = simvar[video].numpy()
        # similarity[masks[video]] = 0.5*similarity[masks[video]] + 0.5*v_similarities[video][masks[video]]

        # # audio + video + relation
        # similarity = 0.5*a_similarities[video] + 0.5*simvar[video].numpy()
        # similarity[masks[video]] = 0.5*similarity[masks[video]] + 0.5*v_similarities[video][masks[video]]

        start = starts[video]
        end = ends[video]
        labels = postprocessor.cluster.fit_predict(similarity)
        postprocessor.process(labels, video, start, end, True)

    overall_ref_path = f'{postprocessor.config.save_dir}/ref.rttm'
    overall_sys_path = f'{postprocessor.config.save_dir}/sys.rttm'
    with open(overall_ref_path, 'w+') as f:
        f.writelines(postprocessor.ref_rttm)
    with open(overall_sys_path, 'w+') as f:
        f.writelines(postprocessor.sys_rttm)
    overall_results = metrics([overall_sys_path], [overall_ref_path])[1]
    overall_der = overall_results.der
    overall_ms = overall_results.ms
    overall_fa = overall_results.fa
    overall_es = overall_results.es

    global_config = get_global_config()
    model_name = global_config.model
    dataset_name = global_config.datasets
    missing_rate = global_config.get('dataset_config').get(dataset_name).get('missing_rate')
    checkpoint = global_config.get('model_config').get(model_name).get('base_ckpt_path')
    logger.info(f'threshold: {config.evaluation.metrics[0].params.threshold}, overall der: {overall_der}, missing rate: {missing_rate}, checkpoint: {checkpoint}')
    logger.info(f'miss: {overall_ms}, false alarm: {overall_fa}, error speaker: {overall_es}')

if __name__ == '__main__':
    main()