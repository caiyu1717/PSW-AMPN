# PSW-AMPN
**Abstract**—Open set recognition (OSR) aims to correctly classify known samples while identifying samples from unknown classes. Existing prototype-based methods may struggle to represent intra-class diversity with a single prototype, which can reduce their ability to distinguish unknown samples. PSW-AMPN uses a Swin Transformer as the feature extractor and introduces attention-based multi-prototype learning to model diverse class distributions. The method is evaluated on the OCT image dataset for open set recognition.

# Requirements
First, install Python 3.9. Then:

```shell
# create an environment
conda create -n PSW-AMPN python=3.9
conda activate PSW-AMPN

# Install following packages
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 pandas==2.0.3 scikit-learn==1.3.0 matplotlib==3.7.1 Pillow==11.3.0
```

# Training & Evaluation
To train and evaluate the open set recognition model on the OCT image dataset, run this command:

```train
python osr11.py --dataset <DATASET> --gpu <GPU> --lr <learning rate> --dataroot <Dataroot>
```
