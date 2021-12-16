import subprocess
import sys
sys.path.append('.')
import os, cv2, glob, shutil
from torch.utils import data
from tqdm import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.io import wavfile
from collections import defaultdict
from dataset.lib.face_aligner import face_aligner


def parallel_process(fn, n=2):
    videos = glob.glob('dataset/videos/*')
    video_split = []
    for i in range(n):
        video_split.append(videos[i:len(videos):n])
    pool = Pool(n)
    pool.map(fn, video_split)
    pool.close()
    pool.join()


def crop_align_face(videos):
    faligner = face_aligner()
    colnames = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2',
                'entity_box_y2','label','entity_id', 'spkid']

    for video in videos:
        items = video.split('/')
        uid = items[-1].split('.')[0]
        tracks = f'dataset/tracks/{uid}-activespeaker.csv'
        if not os.path.exists(tracks):
            continue
        print(f'{video} tracklet not found')
        df = pd.read_csv(tracks, engine='python', header=None, names=colnames)
        df = df[df['spkid'].str.contains('spk')]

        save_dir = f'dataset/aligned_tracklets/{uid}'
        os.makedirs(save_dir, exist_ok=True)

        V = cv2.VideoCapture(video)

        for _, row in df.iterrows():
            
            V.set(cv2.CAP_PROP_POS_MSEC, row['frame_timestamp'] * 1e3)
            # Load frame and get dimensions
            _, frame = V.read()
            if frame is None:
                print(video, row['frame_timestamp'], 'original image not loaded, skipping')
                continue
            h = np.size(frame, 0)
            w = np.size(frame, 1)
            u_frame_id = '{}:{:.2f}'.format(row['entity_id'], row['frame_timestamp'])

            # Crop face
            x1 = int(row['entity_box_x1'] * w)
            y1 = int(row['entity_box_y1'] * h)
            x2 = int(row['entity_box_x2'] * w)
            y2 = int(row['entity_box_y2'] * h)
            face_crop = frame[y1: y2, x1: x2, :]
            face_crop = cv2.resize(face_crop, (224, 224))
            aligned_face_crop = faligner.align_face(face_crop)

            active = int(row['label'] == 'SPEAKING_AUDIBLE')
            spkid = row['spkid']

            if aligned_face_crop is not None:          
                cv2.imwrite(f'{save_dir}/{u_frame_id}:{active}:{spkid}.jpg', aligned_face_crop)
            else:
                cv2.imwrite(f'{save_dir}/{u_frame_id}:{active}:{spkid}.jpg', face_crop)


def split_waves(videos):

    os.makedirs('dataset/.waves', exist_ok=True)
    os.makedirs('dataset/waves', exist_ok=True)
    offsets = []
    for video in videos:
        items = video.split('/')
        uid = items[-1].split('.')[0]
        print(uid)
        cmd = f'ffmpeg -y -i {video} -qscale:a 0 -ac 1 -vn -threads 6 -ar 16000 dataset/.waves/{uid}.wav -loglevel panic'
        subprocess.call(cmd, shell=True)

        rttms = glob.glob(f'dataset/rttms/{uid}*.rttm')
        for rttm in sorted(rttms):
            mins = 1e9
            maxs = -1
            uid = rttm.split('/')[-1].split('.')[0]
            with open(rttm, 'r') as f:
                lines = f.readlines()
            for line in lines:
                items = line.split()
                start = float(items[3])
                end = start + float(items[4])
                if start < mins:
                    mins = start
                if end > maxs:
                    maxs = end

            sample_rate, wave = wavfile.read(f'dataset/.waves/{uid[:-5]}.wav')
            assert sample_rate == 16000
            wave = wave[int(mins*sample_rate):int(maxs*sample_rate)]
            wavfile.write(f'dataset/waves/{uid}.wav', sample_rate, wave)
            offsets.append('{} {}\n'.format(os.path.basename(rttm)[:-5], mins))
            
    with open(f'dataset/split/offsets.txt', 'w+') as f:
        f.writelines(offsets)
    shutil.rmtree(f'dataset/.waves')
                

if __name__ == "__main__":
    # extract frame, crop and align faces
    parallel_process(crop_align_face, n=1)

    # extract audio stream and crop
    parallel_process(split_waves, n=1)