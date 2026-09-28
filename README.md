# PSW-AMPN
**Abstract**—Existing prototype learning methods for open-set recognition (OSR) often use a fixed number of prototypes to represent each class, which struggle to model the inherent intra class variations widely existing in practical scenarios. This limitation is particularly pronounced in medical image applications, where high intra-class heterogeneity makes accurate OSR challenging. To address this issue, we propose a novel framework called Adaptive Multi-Prototype Network with Pretrained Swin Transformer (PSW-AMPN) for OSR on medical images. Specifically, a sparse gated attention module is devised to compute attention scores based on prototype–sample relations, thereby adaptively suppressing redundant sub-prototypes and selectively activating discriminative ones for each class in an end to-end optimization process. By jointly optimizing classification loss and the regularization for open space risk based on multiple prototypes, PSW-AMPN effectively captures complex intra-class structures and enhances class discriminability. Furthermore, PSW-AMPN uses a Pretrained Swin Transformer and a lightweight projector as feature extractor to effectively capture both local and global features. Extensive experiments demonstrate that our approach significantly outperforms existing baselines and achieves state-of-the-art performance on multiple medical image OSR tasks.

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
