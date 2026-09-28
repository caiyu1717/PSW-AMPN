import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, KMNIST

import numpy as np
from PIL import Image
import medmnist
from utils import mkdir_if_missing


class CIFAR10Subset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = torch.tensor(img).permute(2, 0, 1) / 255.0  # 转换为 PyTorch 格式
        if self.transform:
            img = self.transform(img)
        return img, target

class CIFAR10(object):
    def __init__(self, **options):

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = 64
        data_root = '/media/admin1/cxy/PCLAE-AMPN/data/cifar10'

        pin_memory = True

        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)

        # 获取CIFAR-10的图像和标签数据
        full_data = trainset.data  # numpy array of shape (50000, 32, 32, 3)
        full_targets = trainset.targets  # list of length 50000

        # 创建筛选后的数据和标签列表
        filtered_data = []
        filtered_targets = []
        num_samples_per_class = 40

        # 按类别筛选
        for class_id in range(10):
            class_indices = [i for i, label in enumerate(full_targets) if label == class_id]
            selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
            filtered_data.extend(full_data[selected_indices])
            filtered_targets.extend([class_id] * num_samples_per_class)

        # 转换为 numpy 数组以创建自定义数据集
        filtered_data = np.array(filtered_data)
        filtered_targets = np.array(filtered_targets)

        filtered_trainset = CIFAR10Subset(filtered_data, filtered_targets, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(
            filtered_trainset, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader
        self.data = trainloader.dataset.data

class BloodMNIST(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            #transforms.Resize(32, padding=4),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        #data_root = options['dataroot']
        data_root = '/media/admin1/cxy/PCLAE-AMPN/data/bloodmnist'
        pin_memory = True if options['use_gpu'] else False

        # 加载训练集
        trainset = medmnist.BloodMNIST(root=data_root, split='train', download=True, transform=transform_train)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        # 加载测试集
        testset = medmnist.BloodMNIST(root=data_root, split='test', download=True, transform=transform)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 8


class SVHN(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'svhn')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.SVHN(root=data_root, split='Atest', download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader

class KVASIR(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.Resize(32, padding=4),

            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        #data_root = options['dataroot']
        data_root = os.path.join(options['dataroot'], 'kvasir_capsule')
        pin_memory = True if options['use_gpu'] else False

     
        trainset = ImageFolder(root=os.path.join(data_root, 'train'), transform=transform_train)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        testset = ImageFolder(root=os.path.join(data_root, 'val'), transform=transform)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 14


class HYPER(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            #transforms.Resize(32, padding=4),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),

            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize(224,224),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        #data_root = options['dataroot']
        data_root = os.path.join(options['dataroot'], 'hyper-kvasir')
        pin_memory = True if options['use_gpu'] else False

     
        trainset = ImageFolder(root=os.path.join(data_root, 'train'), transform=transform_train)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        testset = ImageFolder(root=os.path.join(data_root, 'val'), transform=transform)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 23


class ISIC2018(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.Resize(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        #data_root = options['dataroot']
        data_root = os.path.join(options['dataroot'], 'isic2018')
        pin_memory = True if options['use_gpu'] else False

       
        trainset = ImageFolder(root=os.path.join(data_root, 'train'), transform=transform_train)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

      
        testset = ImageFolder(root=os.path.join(data_root, 'val'), transform=transform)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 7

class ISIC2019(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.Resize(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        #data_root = options['dataroot']
        data_root = os.path.join(options['dataroot'], 'isic2019')
        pin_memory = True if options['use_gpu'] else False

      
        trainset = ImageFolder(root=os.path.join(data_root, 'train'), transform=transform_train)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

     
        testset = ImageFolder(root=os.path.join(data_root, 'val'), transform=transform)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 9

class OCT(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.Resize(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        #data_root = options['dataroot']
        data_root = os.path.join(options['dataroot'], 'oct11')
        pin_memory = True if options['use_gpu'] else False

        trainset = ImageFolder(root=os.path.join(data_root, 'train'), transform=transform_train)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

     
        testset = ImageFolder(root=os.path.join(data_root, 'val'), transform=transform)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 8


class CUB200Dataset(Dataset):
    def __init__(self, root, train=True, img_size=224, transform=None):
        """
        CUB-200-2011 数据集加载器

        Args:
            root (str): CUB-200-2011 数据集的根目录
            train (bool): 是否加载训练集（True 加载训练集，False 加载测试集）
            img_size (int): 输入图像大小，默认 224x224（适用于 Swin Transformer）
        """
        self.root = root
        self.img_size = img_size

      
        images_file = os.path.join(root, "images.txt")
        labels_file = os.path.join(root, "image_class_labels.txt")
        split_file = os.path.join(root, "train_test_split.txt")

      
        images = pd.read_csv(images_file, sep=" ", header=None, names=["id", "path"])
        labels = pd.read_csv(labels_file, sep=" ", header=None, names=["id", "label"])
        split = pd.read_csv(split_file, sep=" ", header=None, names=["id", "is_train"])

       
        data = images.merge(labels, on="id").merge(split, on="id")

       
        self.data = data[data["is_train"] == int(train)]
        self.image_paths = self.data["path"].values
        self.labels = self.data["label"].values - 1  # 标签从 0 开始

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.image_paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


__factory = {
    'cifar10': CIFAR10,
    'svhn':SVHN,
    'kvasir_capsule':KVASIR,
    'hyper-kvasir':HYPER,
    'isic2018':ISIC2018,
    'isic2019':ISIC2019,
    'oct11':OCT,
    'bloodmnist':BloodMNIST

}

def create(name, **options):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](**options)


