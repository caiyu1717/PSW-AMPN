import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Attention_Dist import AttentionDist


class ATMPLoss(nn.Module):
    def __init__(self, **options):
        super(ATMPLoss, self).__init__()
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

        self.attn_sparsity = options.get('attn_sparsity', 0.01)

        
        self.weight_inter = options.get('weight_inter', 0.1)
        self.inter_margin = options.get('inter_margin', 1.5)

        self.debug_inter = options.get('debug_inter', False)
        self.debug_print_freq = options.get('debug_print_freq', 100)
        self._debug_step = 0


    '''
    def inter_class_proto_loss(self):
        #centers = self.center_points.view(self.num_classes, self.num_centers, self.feat_dim)  # [C, K, D]
        centers = self.Dist.centers
        loss_inter = 0
        count = 0
        margin = self.inter_margin
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
               
                dist_mat = torch.cdist(centers[i], centers[j], p=2)  # [K, K]
                min_dist = dist_mat.min()
                loss_inter += F.relu(margin - min_dist).pow(2)
                count += 1
        
        return loss_inter/ max(count, 1)
        '''
    
    def inter_class_proto_loss(self, proto_M, proto_mask):
        loss_inter = 0.0
        count = 0
        margin = self.inter_margin

        for i in range(self.num_classes):
            centers_i = proto_M[i][proto_mask[i]]
            if centers_i.shape[0] == 0:
               continue

            for j in range(i + 1, self.num_classes):
                centers_j = proto_M[j][proto_mask[j]]
                if centers_j.shape[0] == 0:
                   continue

                centers_i_norm = F.normalize(centers_i, p=2, dim=1)
                centers_j_norm = F.normalize(centers_j, p=2, dim=1)

                sim = torch.matmul(centers_i_norm, centers_j_norm.t())

                #loss_inter = loss_inter + sim.mean()
                #count += 1

                loss_inter = loss_inter + sim.sum()
                count += sim.numel()

                #dist_mat = torch.cdist(centers_i, centers_j, p=2)
                #min_dist = dist_mat.min()
                #loss_inter = loss_inter + F.relu(margin - min_dist).pow(2)
                #count += 1

        if count == 0:
           return proto_M.sum() * 0.0

        return loss_inter / count


    def forward(self, x, labels=None, train=False):

        dist_l2, proto_M, proto_mask, attn_weight = self.Dist(x, label=labels, train=train)  
        dist_dot, _, _, _ = self.Dist(x, metric='dot', label=labels, train=train)  
        logits = -(dist_l2 - dist_dot) 
        
       
        if train:
            
            loss_cls = F.cross_entropy(logits / self.temp, labels.long())

          
            #center_batch = proto_M[labels, :]
            #loss_r = F.mse_loss(x, center_batch) / 2
            #loss_r = 0.0
            loss_r_list = []
            total = 0  # 
            for label in labels.unique(): 
                mask = (labels == label)  
                x_label = x[mask]  
                centers_label = proto_M[label]
                class_mask = proto_mask[label]
                valid_centers = centers_label[class_mask]  
                #class_mask = proto_mask[label].float().unsqueeze(1)  
                #valid_centers = centers_label * class_mask  

                center_batch = valid_centers.unsqueeze(0)
                center_batch = center_batch.expand(x_label.size(0), -1, -1)
                x_expanded = x_label.unsqueeze(1)
                x_expanded = x_expanded.expand(-1, center_batch.size(1), -1)
                _dis_known = (x_expanded - center_batch).pow(2).mean(2).mean(1)
                target = torch.ones(_dis_known.size()).cuda()
                radius = self.radius[label].expand(x_label.size(0))
                loss_r = self.margin_loss(radius, _dis_known, target)
                #loss_r += F.mse_loss(x_expanded, center_batch) / 2
                #total += x_label.size(0)
                loss_r_list.append(loss_r)
            #loss_r = loss_r / total
            loss_r_total = torch.mean(torch.stack(loss_r_list))

            
            loss_attn_sparse = self.attn_sparsity * attn_weight.abs().mean()

           
            #loss_inter = self.inter_class_proto_loss()
            loss_inter = self.inter_class_proto_loss(proto_M, proto_mask)
          


            loss = loss_cls + self.weight_pl * loss_r_total + loss_attn_sparse + self.weight_inter * loss_inter
            return logits, loss
        else:
            return logits, 0
