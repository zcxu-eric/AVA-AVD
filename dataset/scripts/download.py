import os, subprocess

def download_videos(save_dir):
    print('downloading videos...')
    with open('dataset/split/video.list', 'r') as f:
        videos = f.readlines()

    for i, video in enumerate(videos):
        print(f'downloading {video}[{i+1}]/[{len(videos)}]')
        cmd = f'wget -P {save_dir} https://s3.amazonaws.com/ava-dataset/trainval/{video.strip()}'
        subprocess.call(cmd, shell=True)

def download_annotations():
    print('downloading annotaitons...')
    cmd = f'gdown --id 18kjJJbebBg7e8umI6HoGE4_tI3OWufzA'
    subprocess.call(cmd, shell=True)
    cmd = f'tar -xvf annotations.tar.gz -C dataset/'
    subprocess.call(cmd, shell=True)
    os.remove('annotations.tar.gz')

if __name__ == '__main__':
    os.makedirs('dataset/videos', exist_ok=True)
    download_videos('dataset/videos')
    download_annotations()