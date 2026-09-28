import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist1 import AttentionDist


class Loss1(nn.Module):
    def __init__(self, **options):
        super(Loss1, self).__init__()
        # 基础参数
        self.num_classes = options['num_classes']
        self.num_centers = options['num_centers']  
        self.feat_dim = options['feat_dim']
        self.temp = options['temp']
        self.weight_pl = options['weight_pl']
        self.Dist = AttentionDist(
            num_classes=options['num_classes'],
            num_centers=options['num_centers'],
            feat_dim=options['feat_dim'],
        )
        self.center_points = self.Dist.centers  # [num_classes * num_centers, feat_dim]
        self.radius = nn.Parameter(torch.Tensor(self.num_classes))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

        
        #self.weight_inter = options.get('weight_inter', 0.1)

    def forward(self, x, labels=None, train=False):
        
        dist_l2, proto_M, proto_mask, _ = self.Dist(x, label=labels, train=train)  
        dist_dot, _, _, _ = self.Dist(x, metric='dot', label=labels, train=train)  
        '''print(dist_l2.shape)  # [batch_size, num_classes]
        print(dist_dot.shape)
        print(proto_M.shape)  # [num_classes, feat_dim]'''

        logits = -(dist_l2 - dist_dot) #

        if train:
            
            loss_cls = F.cross_entropy(logits / self.temp, labels.long())

           
            loss_r_list = []
            for label in labels.unique():  
                mask = (labels == label)  
                x_label = x[mask]  
                centers_label = proto_M[label]  

                center_batch = centers_label.unsqueeze(0)
                center_batch = center_batch.expand(x_label.size(0), -1, -1)
                x_expanded = x_label.unsqueeze(1)
                x_expanded = x_expanded.expand(-1, center_batch.size(1), -1)
                _dis_known = (x_expanded - center_batch).pow(2).mean(2).mean(1)

                target = torch.ones(_dis_known.size()).cuda()
                radius = self.radius[label].expand(x_label.size(0)) 
                loss_r = self.margin_loss(radius, _dis_known, target)
                loss_r_list.append(loss_r)
            #loss_r = loss_r / total
            loss_r_total = torch.mean(torch.stack(loss_r_list))

            loss = loss_cls + self.weight_pl * loss_r_total
            return logits, loss
        else:
            return logits, 0
