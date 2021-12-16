from mmsc.utils.fileio import PathManager
from mmsc.utils.general import get_batch_size
import torch
import os, glob, shutil, logging
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from mmsc.modules.cluster import AVDCluster
from mmsc.utils.diarization import metrics
from mmsc.common.registry import registry
from mmsc.common.meter import SmoothedValue
from mmsc.utils.configuration import get_global_config


logger = logging.getLogger(__name__)


class Base_Postprocessor():
    def __init__(self, config):
        self.config = config
        self.report_key = ['der', 'ms', 'fa', 'es']
        self.ground_truth = config.ground_truth
        self.save_dir = f'{config.save_dir}/rttms'
        self.meters = { k: SmoothedValue() for k in self.report_key }
        self.ref_rttm = []
        self.sys_rttm = []
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    def __call__(self):
        raise NotImplementedError

    def process(self, labels, video, starts, ends, verbose):
        starts, ends, labels = merge_frames(starts, ends, labels)

        sys_path = self.dump_turns(video, starts, ends, labels)
        ref_path = sys_path.replace(self.save_dir, self.ground_truth)
        der = metrics([sys_path], [ref_path])
        self.meters['der'].update(der[1].der)
        self.meters['ms'].update(der[1].ms)
        self.meters['fa'].update(der[1].fa)
        self.meters['es'].update(der[1].es)

        with open(ref_path, 'r') as f:
            self.ref_rttm += f.readlines()
        with open(sys_path, 'r') as f:
            self.sys_rttm += f.readlines()

        if verbose:
            logger.info('video: {}\t DER: {:.2f}%\t mean DER: {:.2f}%\t '
                        'MS: {:.2f}%\t mean MS: {:.2f}%\t '
                        'FA: {:.2f}%\t mean FA: {:.2f}%\t '
                        'ES: {:.2f}%\t mean ES: {:.2f}%'.format(video, 
                        self.meters['der'].get_latest(), self.meters['der'].global_avg, 
                        self.meters['ms'].get_latest(), self.meters['ms'].global_avg,  
                        self.meters['fa'].get_latest(), self.meters['fa'].global_avg,  
                        self.meters['es'].get_latest(), self.meters['es'].global_avg))
    
    def dump_turns(self, video, starts, ends, labels):
        rec = []
        for label, start, end in zip(labels, starts, ends):
            if end - start < 0.01:
                continue
            rec += [f'SPEAKER {video} 1 {start:.6f} {end-start:.6f} <NA> <NA> {label} <NA> <NA>\n']
        pred_file = '{}/{}.rttm'.format(self.save_dir, video)
        if os.path.exists(pred_file):
            os.remove(pred_file)
        with open(pred_file, 'w+') as f:
            f.writelines(rec)
        return pred_file


class Postprocessor(Base_Postprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.relation = config.relation
        self.cluster = AVDCluster(self.config.threshold)
    
    def __call__(self, output, dataset_type):
        run_type = get_global_config('run_type')
        should_print = run_type != 'train'

        # compute similarities using relation layer
        intermidiate = compute_similarity(output, 
                                    relation = self.relation,
                                    verbose = should_print)

        # save metrics for fine tuning
        torch.save(intermidiate, f'{self.save_dir}/intermidiate_{run_type}.pkl')

        # diarize
        if run_type != 'train':
            threshold = [self.config.threshold]
        else:
            threshold = np.arange(
                        self.config.min_thres,
                        self.config.max_thres, 
                        self.config.step
                        )
        
        best_der = float('inf')
        best_t = 0
        for t in threshold:
            self.cluster = AVDCluster(t)
            self.meters = { k: SmoothedValue() for k in self.report_key }
            self.ref_rttm = []
            self.sys_rttm = []
            for video in intermidiate['similarities']:
                similarity = intermidiate['similarities'][video]
                start = intermidiate['starts'][video]
                end = intermidiate['ends'][video]
                labels = self.cluster.fit_predict(similarity)
                self.process(labels, video, start, end, should_print)

            overall_ref_path = f'{self.config.save_dir}/ref.rttm'
            overall_sys_path = f'{self.config.save_dir}/sys.rttm'
            with open(overall_ref_path, 'w+') as f:
                f.writelines(self.ref_rttm)
            with open(overall_sys_path, 'w+') as f:
                f.writelines(self.sys_rttm)
            overall_der = metrics([overall_sys_path], [overall_ref_path])[1].der

            if overall_der < best_der:
                best_der = overall_der
                best_t = t

        if run_type != 'train':    
            global_config = get_global_config()
            model_name = global_config.model
            dataset_name = global_config.datasets
            missing_rate = global_config.get('dataset_config').get(dataset_name).get('missing_rate')
            checkpoint = global_config.get('model_config').get(model_name).get('base_ckpt_path')
            logger.info(f'threshold: {self.config.threshold}, overall_der: {overall_der}, missing rate: {missing_rate}, checkpoint: {checkpoint}')
        else:
            logger.info(f'threshold: {best_t}')

        der_record = {
            'best_der': best_der,
            'threshold': best_t
        }
        registry.register("best_der_info", der_record)

        return best_der


def merge_frames(starts, ends, labels):
    """ Labeled segments defined as start and end times are compacted in such a way that
    adjacent or overlapping segments with the same label are merged. Overlapping
    segments with different labels are further adjusted not to overlap (the boundary
    is set in the middle of the original overlap).
    """
    # sort
    inds = np.argsort(starts)
    starts, ends, labels = starts[inds], ends[inds], labels[inds]
    # Merge neighbouring (or overlaping) segments with the same label
    adjacent_or_overlap = np.logical_or(np.isclose(ends[:-1], starts[1:]), ends[:-1] > starts[1:])
    to_split = np.nonzero(np.logical_or(~adjacent_or_overlap, labels[1:] != labels[:-1]))[0]
    starts  = starts[np.r_[0, to_split+1]]
    ends    = ends[np.r_[to_split, -1]]
    labels  = labels[np.r_[0, to_split+1]]

    overlaping = np.nonzero(starts[1:]<ends[:-1])[0]
    ends[overlaping] = starts[overlaping+1] = (ends[overlaping]+starts[overlaping+1]) / 2.0

    return starts, ends, labels


def compute_similarity(output, relation=True, verbose=True):
    logger.info('start diarization postprocessing...')
    similarities = {}
    if not relation:
        audio_similarities = {}
        video_similarities = {}
        masks = {}
    starts = {}
    ends = {}
    feat_by_video = defaultdict(list)
    for feat_v, feat_a, video, start, end, visible, tackid in zip(
        output.feat_video, output.feat_audio, output.video, 
        output.start, output.end, output.visible, output.trackid
    ):
        feat_by_video[video[0]].append((feat_v, feat_a, start[0], end[0], visible[0], tackid[0]))
    
    if relation:
        batch_size = 256
        model = output.model
        model.eval()
    del output

    from tqdm import tqdm
    for k in tqdm(list(feat_by_video.keys()), disable=not verbose):
        v = feat_by_video[k]
        D = len(v)
        if relation:
            batch = []
            similarity = torch.diag_embed(torch.ones([D]))
            for i in range(D):
                for j in range(i+1, D):
                    batch.append((torch.cat((v[i][0], v[j][0])), 
                                torch.cat((v[i][1], v[j][1])),
                                i,j, v[i][4], v[j][4]))
                    if len(batch) == batch_size:
                        process_one_batch(batch, model, similarity)
                        batch = []
            if len(batch) > 0:
                process_one_batch(batch, model, similarity)
        else:
            faces_embedding = torch.cat([d[0].cpu().unsqueeze(0) for d in v], dim=0)
            audio_embedding = torch.cat([d[1].cpu().unsqueeze(0) for d in v], dim=0)
            faces_embedding = F.normalize(faces_embedding, dim=-1).numpy()
            audio_embedding = F.normalize(audio_embedding, dim=-1).numpy()
            visible_mask = np.array([d[4] for d in v])[None, :]
            visible_mask = np.logical_and(visible_mask, visible_mask.T)
            similarity = audio_embedding.dot(audio_embedding.T)
            similarity_v = faces_embedding.dot(faces_embedding.T)
            similarity[visible_mask] = 0.5 * (similarity[visible_mask] + similarity_v[visible_mask])
        del feat_by_video[k]
        similarities[k] = similarity
        if not relation:
            audio_similarities[k] = audio_embedding.dot(audio_embedding.T)
            video_similarities[k] = similarity_v
            masks[k] = visible_mask
        starts[k] = np.array([record[2] for record in v])
        ends[k] = np.array([record[3] for record in v])
    if relation:
        intermidiate = {
                'similarities': similarities, 
                'starts': starts,
                'ends': ends
            }
    else:
        intermidiate = {
            'similarities': similarities, 
            'starts': starts,
            'ends': ends,
            'audio_similarities': audio_similarities, 
            'video_similarities': video_similarities,
            'masks': masks
        }
    return intermidiate


def process_one_batch(batch, model, similarity):
    video = torch.cat([p[0].unsqueeze(0) for p in batch], dim=0)
    audio = torch.cat([p[1].unsqueeze(0) for p in batch], dim=0)
    task = torch.tensor([torch.tensor(int(p[4])+int(p[5]), 
                         dtype=torch.int64) for p in batch], device=video.device)
    task1 = torch.tensor([torch.tensor(2*int(p[4])+int(p[5]), 
                         dtype=torch.int64) for p in batch], device=video.device)
    task2 = torch.tensor([torch.tensor(2*int(p[5])+int(p[4]), 
                         dtype=torch.int64) for p in batch], device=video.device)
    prepared_batch = {
        'dataset_name': 'avaavd', 
        'dataset_type': 'val', 
        'video': video,
        'audio': audio,
        'task': task, 
        'task_full': [task1, task2]
    }
    scores = model(prepared_batch, exec='relation')['scores'].cpu()
    for p, score in zip(batch, scores):
        similarity[p[2], p[3]] = similarity[p[3], p[2]] = score.cpu()


class Postprocessor_class(Base_Postprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.feat_path = config.feat_path

    def __call__(self, output, dataset_type):
        intermidiate = compute_similarity_c(output, self.feat_path)

        for video in intermidiate['preds']:
            labels = intermidiate['preds'][video]
            start = intermidiate['starts'][video]
            end = intermidiate['ends'][video]
            self.process(labels, video, start, end, verbose=True)

        overall_ref_path = f'{self.config.save_dir}/ref.rttm'
        overall_sys_path = f'{self.config.save_dir}/sys.rttm'
        with open(overall_ref_path, 'w+') as f:
            f.writelines(self.ref_rttm)
        with open(overall_sys_path, 'w+') as f:
            f.writelines(self.sys_rttm)
        overall_results = metrics([overall_sys_path], [overall_ref_path])[1]
        overall_der = overall_results.der
        overall_ms = overall_results.ms
        overall_fa = overall_results.fa
        overall_es = overall_results.es
        logger.info(f'overall der: {overall_der} miss: {overall_ms}, false alarm: {overall_fa}, error speaker: {overall_es}')
        return overall_der
    

def compute_similarity_c(output, feat_path, relation=True, verbose=True):
    logger.info('start diarization (classification) postprocessing...')
    preds = {}
    starts = {}
    ends = {}
    feat_by_video = defaultdict(list)

    for feat_v, feat_a, video, start, end, visible in zip(
        output.feat_video, output.feat_audio, output.video, 
        output.start, output.end, output.visible
    ):
        feat_by_video[video[0]].append((feat_v, feat_a, start[0], end[0], visible[0]))

    model = output.model
    model.eval()
    del output

    from tqdm import tqdm
    for k in tqdm(list(feat_by_video.keys()), disable=not verbose):
        v = feat_by_video[k]
        D = len(v)
        speaker_feat = []
        look_up_table = []
        feats = glob.glob(f'{feat_path}/{k}*.pkl')
        for feat in feats:
            rec = torch.load(feat)
            spk = os.path.basename(feat[:-4]).split('_')[1]
            speaker_feat.extend(rec)
            look_up_table.extend([spk]*len(rec))
        look_up_table = np.array(look_up_table)
        batch = []
        pred = []
        for i in range(D):
            for sf in speaker_feat:
                batch.append((torch.cat((v[i][0], sf[0].cuda())), 
                              torch.cat((v[i][1], sf[1].cuda())), 
                              v[i][4], True))
            pred.append(process_one_batch_c(batch, model, look_up_table))
            batch = []

        del feat_by_video[k]
        preds[k] = np.array(pred)
        starts[k] = np.array([record[2] for record in v])
        ends[k] = np.array([record[3] for record in v])

    intermidiate = {
            'preds': preds, 
            'starts': starts,
            'ends': ends
        }
    return intermidiate


def process_one_batch_c(batch, model, look_up_table):
    video = torch.cat([p[0].unsqueeze(0) for p in batch], dim=0)
    audio = torch.cat([p[1].unsqueeze(0) for p in batch], dim=0)
    task = torch.tensor([torch.tensor(int(p[2])+int(p[3]), 
                         dtype=torch.int64) for p in batch], device=video.device)
    task1 = torch.tensor([torch.tensor(2*int(p[2])+int(p[3]), 
                         dtype=torch.int64) for p in batch], device=video.device)
    task2 = torch.tensor([torch.tensor(2*int(p[3])+int(p[2]), 
                         dtype=torch.int64) for p in batch], device=video.device)
    prepared_batch = {
        'dataset_name': 'avaavd', 
        'dataset_type': 'val', 
        'video': video,
        'audio': audio,
        'task': task, 
        'task_full': [task1, task2]
    }
    scores = model(prepared_batch, exec='relation')['scores'].cpu()
    scores_by_id = defaultdict(list)
    for spk, score in zip(look_up_table, scores.numpy()):
        # if score[0] < 0.14:
        #     score[0] = 0
        scores_by_id[spk].append(score[0])
    for spk in scores_by_id:
        scores_by_id[spk] = np.mean(scores_by_id[spk])
    k = np.array(list(scores_by_id.keys()))
    v = np.array(list(scores_by_id.values()))
    return k[np.argmax(v)]
