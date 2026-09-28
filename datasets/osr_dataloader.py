import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, ImageFolder, CIFAR100
from torch.utils.data import WeightedRandomSampler
try:
    from medmnist import BloodMNIST
except ImportError:
    class BloodMNIST(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            raise ImportError("medmnist is required only when using the bloodmnist dataset")

import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS


class ImageFolderAllowEmpty(ImageFolder):

    def make_dataset(
        self,
        directory,
        class_to_idx,
        extensions=None,
        is_valid_file=None,
    ):
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)

        if extensions is not None:
            def is_valid_file(x):
                return x.lower().endswith(extensions)

        instances = []

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)

            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

        return instances

class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                a = targets[i]
                b = known.index(a)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR(object):
    def __init__(self, known, dataroot='./data/cifar10', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=False, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = CIFAR10_Filter(root=dataroot, train=False, download=False, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=False, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class BloodMNIST_Filter(BloodMNIST):
  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
   
        self.actual_length = self.imgs.shape[0]

    def __len__(self):
      
        return self.actual_length

    def __Filter__(self, known):
        
        targets = [self[i][1][0] for i in range(len(self))]
        targets = np.array(targets)

      
        mask = np.isin(targets, known)
        self.imgs = self.imgs[mask]
        self.labels = np.array([known.index(t) for t in targets[mask]])

        self.actual_length = self.imgs.shape[0]



class BloodMNIST_OSR(object):
    def __init__(self, known, dataroot='./data/bloodmnist', use_gpu=True, num_workers=8, batch_size=128,
                 img_size=224):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 8))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

      
        trainset = BloodMNIST_Filter(root=dataroot, split='train', download=False, transform=train_transform)
        print('All Train Data:', len(trainset))

        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

  
        testset = BloodMNIST_Filter(root=dataroot, split='test', download=False, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

      
        outset = BloodMNIST_Filter(root=dataroot, split='test', download=False, transform=transform)
        print(f"完整测试集样本数：{len(outset)}")
        outset.__Filter__(known=self.unknown)
        print(f"未知类样本数：{len(outset)}")

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))
class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """
    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)

class SVHN_OSR(object):
    def __init__(self, known, dataroot='./data/svhn', use_gpu=True, num_workers=2, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train', download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = SVHN_Filter(root=dataroot, split='Atest', download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='Atest', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """

    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot='./data/tiny_imagenet', use_gpu=True, num_workers=0, batch_size=128,
                 img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        class_to_idx = trainset.class_to_idx

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # Load validation labels from val_annotations.txt  准备验证集标签
        val_labels = self._load_val_annotations(
            os.path.join(dataroot, 'tiny-imagenet-200', 'val', 'val_annotations.txt'), class_to_idx)

        # Filter Atest and out-of-set data  构建已知类的验证集（test_loader）
        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform=transform)
        testset.samples = [(path, val_labels[path.split('/')[-1]]) for path, _ in testset.samples]
        testset.imgs = testset.samples
        testset.targets = [val_labels[path.split('/')[-1]] for path, _ in testset.samples]
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                      pin_memory=pin_memory)
        #构建未知类的验证集（out_loader）
        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform=transform)
        outset.samples = [(path, val_labels[path.split('/')[-1]]) for path, _ in outset.samples]
        outset.imgs = outset.samples
        outset.targets = [val_labels[path.split('/')[-1]] for path, _ in outset.samples]
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=pin_memory)

        print('Train:', len(trainset), 'Test:', len(testset), 'Out:', len(outset))
        print('All Test:', (len(testset) + len(outset)))

    def _load_val_annotations(self, annotations_path, class_to_idx):
        val_labels = {}
        with open(annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                class_name = parts[1]
                if class_name in class_to_idx:
                    val_labels[img_name] = class_to_idx[class_name]
        return val_labels

class Kvasir_Filter(ImageFolder):
    """Custom Dataset."""

    def __Filter__(self, known):
        datas, targets = self.samples, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class Kvasir_OSR(object):
    def __init__(self, known, dataroot='./data/kvasir_capsule', use_gpu=True, num_workers=8, batch_size=128,
                 img_size=224):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 14))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = Kvasir_Filter(root=os.path.join(dataroot, 'train'), transform=train_transform)
        print('All Train Data:', len(trainset))

        print("类别映射 class_to_idx：", trainset.class_to_idx)
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

      
        testset = Kvasir_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

       
        outset = Kvasir_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class hyper_Filter(ImageFolderAllowEmpty):
    """Custom Dataset."""

    def __Filter__(self, known):
        datas, targets = self.samples, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class hyper_OSR(object):
    def __init__(self, known, dataroot='./data/hyper-kvasir', use_gpu=True, num_workers=8, batch_size=128,
                 img_size=224):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 23))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

   
        trainset = hyper_Filter(root=os.path.join(dataroot, 'train'), transform=train_transform)
        print('All Train Data:', len(trainset))

        print("类别映射 ：", trainset.class_to_idx)
        trainset.__Filter__(known=self.known)


        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = hyper_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    
        outset = hyper_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class ISIC2018_Filter(ImageFolder):
    """Custom Dataset."""

    def __Filter__(self, known):
        datas, targets = self.samples, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class ISIC2018_OSR(object):
    def __init__(self, known, dataroot='./data/isic2018', use_gpu=True, num_workers=8, batch_size=128,
                 img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 7))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

      
        trainset = ISIC2018_Filter(root=os.path.join(dataroot, 'train'), transform=train_transform)
        print('All Train Data:', len(trainset))

        print("类别映射 class_to_idx：", trainset.class_to_idx)
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

   
        testset = ISIC2018_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

  
        outset = ISIC2018_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class ISIC2019_Filter(ImageFolder):
    """Custom Dataset."""

    def __Filter__(self, known):
        datas, targets = self.samples, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class ISIC2019_OSR(object):
    def __init__(self, known, dataroot='./data/isic2019', use_gpu=True, num_workers=8, batch_size=128,
                 img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 9))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

       
        trainset = ISIC2019_Filter(root=os.path.join(dataroot, 'train'), transform=train_transform)
        print('All Train Data:', len(trainset))

        print("类别映射 class_to_idx：", trainset.class_to_idx)
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = ISIC2019_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )


        outset = ISIC2019_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class OCT_Filter(ImageFolder):
    """Custom Dataset."""

    def __Filter__(self, known):
        datas, targets = self.samples, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class OCT_OSR(object):
    def __init__(self, known, dataroot='./data/oct11', use_gpu=True, num_workers=8, batch_size=128,
                 img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 8))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

      
        trainset = OCT_Filter(root=os.path.join(dataroot, 'train'), transform=train_transform)
        print('All Train Data:', len(trainset))

        print("类别映射 class_to_idx：", trainset.class_to_idx)
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = OCT_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    
        outset = OCT_Filter(root=os.path.join(dataroot, 'val'), transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class CIFAR100_Filter(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR100_OSR(object):

    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=0, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))
