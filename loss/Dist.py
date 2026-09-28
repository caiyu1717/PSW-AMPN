import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# torch.set_printoptions(sci_mode=False, precision=4)

class Dist(nn.Module):
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        """if init == 'random':
            all_centers = 0.1 * torch.randn(num_classes * num_centers, feat_dim)
            self.centers = nn.ParameterList([
                nn.Parameter(all_centers[i * num_centers:(i + 1) * num_centers])
                for i in range(num_classes)
            ])
        else:
            self.centers = nn.ParameterList([
                nn.Parameter(torch.zeros(num_centers, feat_dim))
                for _ in range(num_classes)
            ])"""
        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes, num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes, num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        device = features.device  
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            dist_list = []
            for cls in range(self.num_classes):
                if center is None:
                    class_centers = self.centers[cls].to(device)
                else:
                    class_centers = center[cls].to(device)
                c_2 = torch.sum(torch.pow(class_centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(class_centers, 1, 0)) + torch.transpose(c_2, 1,
                                                                                                                0)
                dist = dist / float(features.shape[1])
                dist = torch.mean(dist, dim=1)  # Reduce over num_centers
                dist_list.append(dist)
            dist = torch.stack(dist_list, dim=1)  # Shape: (batch_size, num_classes, num_centers)

        else:
            dist_list = []
            for cls in range(self.num_classes):
                if center is None:
                    class_centers = self.centers[cls].to(device)
                else:
                    class_centers = center[cls].to(device)
                dist = features.matmul(class_centers.t())  # Compute similarity
                dist = torch.mean(dist, dim=1)  # Reduce over num_centers
                dist_list.append(dist)
            dist = torch.stack(dist_list, dim=1)  # Shape: (batch_size, num_classes, num_centers)

        return dist
