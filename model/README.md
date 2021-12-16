## AVR-Net

This codebase is developed based on [mmf](https://github.com/facebookresearch/mmf), a modular framework for multi-modal research.

### Data preparation
Please refer to [AVA-AVD](https://github.com/zcxu-eric/AVA-AVD/tree/main/dataset).

### Training
```
export PYTHONPATH=./model \
CUDA_VISIBLE_DEVICES=0 \
python model/mmsc_exp/run.py \
       config=projects/token/mix.yaml \
       datasets=ava,voxceleb1,voxceleb2,avaavd \
       model=token \
       run_type=train \
       training.batch_size=8 \
       training.num_workers=4 \
       env.save_dir=./save/token/output
```

### Testing
Download the [model](https://drive.google.com/file/d/1JNeYSlU--U8NY7luGWNIc8yQVWlCR7w2/view?usp=sharing) trained on AVA-AVD, VoxCeleb 1&2. Modify the checkpoint path in `projects/token/avaavd.yaml`.
```
export PYTHONPATH=./model \
CUDA_VISIBLE_DEVICES=0 \
python model/mmsc_exp/predict.py \
       config=projects/token/avaavd.yaml \
       datasets=avaavd \
       model=token \
       run_type=test \
       env.save_dir=save/token/avaavd
```
