import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch import nn

from utils import AverageMeter
from sklearn.metrics import silhouette_score

def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()
    loss_all = 0

    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, _ = net(data, True)

            logits, loss = criterion(x, labels)

            # loss = loss + options["lamda"] * rec_loss

            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))
        #merge_prototypes(options['num_classes'], criterion.center_points, 0.7)

        # Step 5: 更新原型（调用自适应软KMeans）
        #new_centers = criterion.soft_kmeans_adaptive(x, labels)
        #criterion.center_points = nn.ParameterList([nn.Parameter(c) for c in new_centers])

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))

        loss_all += losses.avg


    return loss_all