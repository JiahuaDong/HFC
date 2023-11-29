# Our paper

# Installation

# Requirements
- torch>=1.7.0
- torchvision>=0.8.1
- timm==0.6.5
- continuum>=1.0.27
- numpy
- scikit-learn

# Datasets
## CIFAR100
You don't need to do anything before running the experiments on CIFAR100 dataset.
## ImageNet100
Refer to [ImageNet100_Split](https://github.com/arthurdouillard/incremental_learning.pytorch/tree/master/imagenet_split)
## ImageNet1000
Data preparation: download and extract ImageNet images from http://image-net.org/. The directory structure should be
```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

# Experiments
You should pretrain the backbone ViT-Tiny refer to https://github.com/facebookresearch/mae, then give the the path of model in option.py(--model_path).
## Training 
For exampler, if you wangt to run CILformer on CIFAR100 in the 10 steps setting:

Modify the path of dataset in './scripts/cifar/task10.sh'.

sh scripts/cifar/task10.sh
## Results
The results of CILformer will be  written in './traning_log'.

# Acknowledgement
Thanks for the great code base from https://github.com/DRSAD/iCaRL and https://github.com/arthurdouillard/dytox.

# Citation
