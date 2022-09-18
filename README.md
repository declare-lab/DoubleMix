# DoubleMix
This repository contains the official implementation code of the paper [DoubleMix: Simple Interpolation-Based Data Augmentation for Text Classification](https://arxiv.org/pdf/2209.05297.pdf), accepted at COLING 2022.


## Usage

1. Check the datasets. Training sets of SNLI and MultiNLI can be found in [this link](https://drive.google.com/drive/folders/1zPsFKZ6cdIyGKNNaiPfsMU2XzoFk1WnU?usp=sharing). Place them under the folder ```dataset/snli``` and ```dataset/multinli```. We implemented the augmentation methods in DoubleMix using files under ```src/augment``` folder.

2. Set up the environment
```
conda create -n doublemix python==3.8
conda activate doublemix
cd DoubleMix/
pip3 install -r requirements.txt
```

3. Run DoubleMix
```
cd src/
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset [dataset] --aug 1
```

## Contact
Should you have any question, feel free to contact the author through [chchenhui1996@gmail.com](chchenhui1996@gmail.com).

