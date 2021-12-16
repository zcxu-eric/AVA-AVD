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


def parallel_process(fn, data_path, n=2):
    videos = sorted(glob.glob(f'{data_path}/id*/*'))
    video_split = []
    for i in range(n):
        video_split.append(videos[i:len(videos):n])
    pool = Pool(n)
    pool.map(fn, video_split)
    pool.close()
    pool.join()


def align_clip_vc1(videos):
    faligner = face_aligner()
    for video in videos:
        os.makedirs(video.replace('unzippedFaces', 'aligned_unzippedFaces'), exist_ok=True)
        imgs = glob.glob(f'{video}/*.jpg')
        for img in imgs:
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            aimg = faligner.align_face(image)
            if aimg is not None:          
                cv2.imwrite(img.replace('unzippedFaces', 'aligned_unzippedFaces'), cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB))


def align_clip_vc2(videos):
    faligner = face_aligner()
    for video in videos:
        clips = glob.glob(f'{video}/*.mp4')
        for clip in clips:
            try:
                save_dir = clip[:-4].replace('dev/mp4', 'aligned_faces')
                os.makedirs(save_dir, exist_ok=True)
                cmd = f'ffmpeg -i {clip} -r 1 {save_dir}/%05d.jpg'
                subprocess.call(cmd, shell=True, stdout=False)
                for img in glob.glob(f'{save_dir}/*.jpg'):
                    image = cv2.imread(img)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    aimg = faligner.align_face(image)
                    if aimg is not None:          
                        cv2.imwrite(img, cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB))
            except:
                print(video)
                pass


if __name__ == '__main__':
    parallel_process(align_clip_vc1, 'dataset/voxceleb/voxceleb1/unzippedFaces', n=4)
    parallel_process(align_clip_vc2, 'dataset/voxceleb/voxceleb2/dev/mp4', n=4)