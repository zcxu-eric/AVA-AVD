# AVA-AVD Dataset
AVA-AVD is built on top of [AVA-Active Speaker](https://research.google.com/ava/index.html).

## Download AVA-AVD
We select 117 high-quality videos from AVA dataset and annotate diarization labels for each clip. You can either download videos following the instructions in [ava-dataset](https://github.com/cvdfoundation/ava-dataset) or run the following scripts:
```
python dataset/scripts/download.py
```
The default folder is `dataset`. If you save data in another directory, please create symbolic links in `dataset`.

The structure of dataset folder should be similar to:
* dataset/
    * labs/
        * -FaXLcSFjUI_c_01.lab
        * -FaXLcSFjUI_c_02.lab
        * ...
    * rttms/
        * -FaXLcSFjUI_c_01.rttm
        * -FaXLcSFjUI_c_02.rttm
        * ...
    * split/
        * test.list
        * train.list
        * ...
    * tracks/
        * -FaXLcSFjUI-activespeaker.csv
        * -IELREHX_js-activespeaker.csv
        * ...
    * videos/
        * 0f39OWEqJ24.mp4
        * ...

## Diarization Preprocessing

Following the next instructions if you want to run experiments of our [AVR-Net](AVRNet/README.md):

We apply retinaface to align faces. First, clone [insightface](https://github.com/deepinsight/insightface) into `dataset/third_party/`.

Then, install the librabry following their instructions for [Retina face](https://github.com/deepinsight/insightface/tree/master/detection/retinaface). Download the `retinaface-R50` checkpoint into `dataset/third_party/insightface/detection/retinaface`.

Run the following script to process the videos:
```
export PYTHONPATH=./dataset/third_party/insightface/detection/retinaface
python dataset/scripts/preprocessing.py
```

[denoising DIHARD18](https://github.com/staplesinLA/denoising_DIHARD18) is used for audio denoising, please clone this repo into `dataset/third_party/`, install models and dependencies following their instructions. Modify the path in `run_eval.sh` as:

```
WAV_DIR=../../waves/  # Directory of WAV files (16 kHz, 16 bit) to enhance.
SE_WAV_DIR=../../denoised_waves  # Output directory for enhanced WAV.
```

Then run the denoising model in `denoising_DIHARD18` directory:
```
bash run_eval.sh
```

### VoxCeleb
We also train AVR-Net on voxceleb datsets. Please download [voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) from the official website.

Since voxceleb1 does not provide cropped face tracks, we directly download the face images from [CMBiometrics](http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/zippedFaces.tar.gz).

We adopt a similar pipeline to align the faces:
```
python dataset/scripts/voxpreprocess.py
```