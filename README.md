# IA-GM

This repository contains PyTorch implementation of our AAAI 2021 paper.

## Requirements

- pytorch 1.1+ (with GPU support)
- ninja-build
- tensorboardX
- scipy
- easydict
- pyyaml

## Datasets

- VOC2011 dataset
    1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/VOC2011``
    1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like ``data/PascalVOC/annotations``
    1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``

- Cars and Motorbikes dataset

- CMU House/Hotel dataset
    

## Training

Run training and evaluation

``python train_xxx.py --cfg path/to/your/yaml``

- For VOC2011 dataset: ``python train_ia.py --cfg model_ia.yaml``

- For Cars and Motorbikes dataset: ``python train_ia.py --cfg biia_pac.yaml``

- For CMU House/Hotel dataset: ``python train_unsup.py --cfg unsup_cmu.yaml``

## Evaluation

Run evaluation on epoch ``k``

``python eval_xxx.py --cfg path/to/your/yaml --epoch k`` 


## Citation

## References
