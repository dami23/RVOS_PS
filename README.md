# PRVOS

This is the implementation of the paper "Uncertainty-aware Adaptive Pseudo-Labeling for Referring Video Object Segmentation" which is submitted to ICANN 2023.

## Prerequisites
- Python 3.7
- Pytorch 1.10
- torchvision 0.11.2
- CUDA 11.4
- numpy 1.19.2
## Dataset Requirements
### Refer-Youtube-VOS
1. Download the dataset from the competition's website in [Codalab](https://competitions.codalab.org/competitions/29139#participate-get_data).
2. We can get two .json file meta_expression.json and meta.json
3. Run this code to integrate them into a new .json file: `python tools/data/merge_meta_json.py`
   The ready-made .json are stored in `Annotations/refer-youtube-vos/new_train.json`
   The structure of the new json file is as follow:
```text
videos
    ├── 003234408d
    │   └── objects
    │       ├── 1
    │       │   ├── category : 'penguin'
    │       │   ├── frames : {"00000", "00005"...}
    │       │   └── expressions : {"a penguin is on the left in the front with many others on the hill", ...}
    │       ├── 2
    │       │   ├── category : 'penguin'
    │       │   ├── frames : {"00000", "00005"...}
    │       │   └── expressions : {"a penguin is on the left in the front with many others on the hill", ...}
    │       ├──...
    ├── ...
```
4. The dataset of Refer-youtube-vos is recommended to be organized as following:
```text
refer_youtube_vos/ 
    ├── train/
    │   ├── JPEGImages/
    │   │   └── */ (video folders)
    │   │       └── *.jpg (frame image files) 
    │   ├── Annotations/
    │   │   └── */ (video folders)
    │   │       └── *.png (mask annotation files) 
    │   │
    │   └── new_train.json (text annotations)
    │
    └── valid/
        ├── JPEGImages/
        │    └── */ (video folders)
        │        └── *.jpg (frame image files)
        ├── Annotations/
        │   └── */ (video folders)
        │       └── *.png (mask annotation files)
        │
        └── new_valid.json (text annotations)
```
### Ref-DAVIS-2017
1. The dataset and the language queries used in the experiments can be found in DAVIS 2017 website [DAVIS](https://davischallenge.org/davis2017/code.html)
2. For convenience, we convert the data structure of DAVIS-2017 dataset into youtube-vos-like as following:
```text
videos
    ├── bear
    │   └── objects
    │       ├── 1
    │       │   ├── category : 'bear'
    │       │   ├── frames : {"00000", "00005"...}
    │       │   └── expressions : {"a brown bear", "a brown bear moving", "a bear", "a brown bear"}
    │       ├──...
    ├── ...
```

3. The dataset of Refer-youtube-vos is recommended to be organized as following:
```text
Ref-DAVIS/ 
    ├── train/
    │   ├── JPEGImages/
    │   │   └── */ (video folders)
    │   │       └── *.jpg (frame image files) 
    │   ├── Annotations/
    │   │   └── */ (video folders)
    │   │       └── *.png (mask annotation files) 
    │   │
    │   └── new_train.json (text annotations)
    │
    └── valid/
        ├── JPEGImages/
        │    └── */ (video folders)
        │        └── *.jpg (frame image files)
        ├── Annotations/
        │   └── */ (video folders)
        │       └── *.png (mask annotation files)
        │
        └── new_valid.json (text annotations)
```
# How to Run the Code
1. First, clone this repo to your local machine using:
`git clone https://github.com/dami23/RVOS_PS.git`
2. Environment configuration as shown in Prerequisites

## Training
For training the models with the different datasets, the command is the following:
1. Training on Refer-Youtube-VOS/train: `python ./Baseline/pretrain.py --mode train_yv --splits train --dataset refer-yv-2019`
## Evaluation
1. The following command evaluates our model on the public validation subset of Refer-YouTube-VOS dataset. Since annotations are not publicly available for this subset, our code generates a zip file with the predicted masks:
`python ./Baseline/pretrain.py --eval --mode eval_yv --splits valid --dataset refer-yv-2019 --test_dataset refer-yv-2019 --checkpoint ./checkpoint/refer-yv-2019/model/e0020.pth --epoch 20`
2. The following command evaluates our model on the public validation subset of DAVIS-2017 dataset.
- Pre-training only: `python ./Baseline/pretrain.py --eval --mode eval_davis --splits valid --dataset ref-davis --test_dataset ref-davis --checkpoint ./checkpoint/ref-davis/model/e0020.pth --epoch 20`
- Finetuned: `python ./Baseline/pretrain.py --eval --mode eval_davis --splits valid --dataset ref-davis --test_dataset ref-davis --checkpoint ./checkpoint/ref-davis/model/finetuned.pth --epoch 21`
