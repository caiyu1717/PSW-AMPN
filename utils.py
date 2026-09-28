import os
import sys
import errno
import os.path as osp
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    """
    Write console output to external text file.
    
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def save_networks(networks, result_dir, name='', loss='', criterion=None):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    torch.save(weights, filename)
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        torch.save(weights, filename)

def load_networks(networks, result_dir, name='', loss='', criterion=None):
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    networks.load_state_dict(torch.load(filename))
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        a = torch.load(filename)
        criterion.load_state_dict(torch.load(filename), strict=False)

    return networks, criterion


def print_trainable_params(model, filepath=None):
   
    lines = []
    lines.append("=" * 80)
    lines.append("所有可训练参数列表:")
    lines.append("=" * 80)

    for name, param in model.named_parameters():
        if param.requires_grad:
            lines.append(f"{name:80} | shape: {str(param.shape):20} | dtype: {param.dtype}")

    lines.append("=" * 80)

    # 打印到控制台
    print("\n".join(lines))

    # 保存到文件
    if filepath:
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")  # 添加换行符


def print_all_params(model, filepath=None):
    
    lines = []
    trainable_count = 0
    total_count = 0

    lines.append("=" * 80)
    lines.append("所有参数列表:")
    lines.append("=" * 80)

    for name, param in model.named_parameters():
        total_count += 1
        if param.requires_grad:
            trainable_count += 1
        lines.append(
            f"{name:80} | shape: {str(param.shape):20} | dtype: {param.dtype} | trainable: {param.requires_grad}")

    lines.append("=" * 80)
    lines.append(f"总参数数量: {total_count}")
    lines.append(f"可训练参数数量: {trainable_count}")
    lines.append(f"冻结参数数量: {total_count - trainable_count}")
    lines.append("=" * 80)

    # 打印到控制台
    print("\n".join(lines))

    # 保存到文件
    if filepath:
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")  # 添加换行符

