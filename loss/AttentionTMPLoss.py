import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.AttentionDist import AttentionDist


class KL_Divergence_Loss(nn.Module):
    def __init__(self):
        super(KL_Divergence_Loss, self).__init__()

    def forward(self, logits1, logits2):
       
        probs1 = nn.Sigmoid()(logits1)
        probs2 = nn.Sigmoid()(logits2)

        loss = nn.KLDivLoss(reduction='batchmean')(torch.log(probs1), probs2)

        return loss

class AttentionTMPLoss(nn.Module):
    def __init__(self, **options):
        super(AttentionTMPLoss, self).__init__()
        # 基础参数
        self.num_classes = options['num_classes']
        self.num_centers = options['num_centers'] 
        self.feat_dim = options['feat_dim']
        self.temp = options['temp']
        self.weight_pl = options['weight_pl']
        #self.log_beta = nn.Parameter(torch.tensor(math.log(options['beta'])))  
        self.beta = options['beta']  
        self.training = options['train']
        self.Dist = AttentionDist(
            num_classes=options['num_classes'],
            num_centers=options['num_centers'],
            feat_dim=options['feat_dim'],
        )
        self.center_points = self.Dist.centers  # [num_classes * num_centers, feat_dim]

      
        self.attn_sparsity = options.get('attn_sparsity', 0.01)

        self.ERloss_function = KL_Divergence_Loss()


    def forward(self, x, labels=None):
        # === TMP距离计算 ===
        dist_l2, proto_M, proto_mask, attn_weight = self.Dist(x, label=labels)  # [B, num_active]
        dist_dot, _, _, _ = self.Dist(x, metric='dot', label=labels)
        logits = -(dist_l2 - dist_dot)

        if labels is None:
            return logits, 0  

        
        
        loss_cls = F.cross_entropy(logits / self.temp, labels.long())

        loss_r = 0.0
        total = 0  
        for label in labels.unique():  
            mask = (labels == label) 
            x_label = x[mask] 
            centers_label = proto_M[label]
            class_mask = proto_mask[label]
            valid_centers = centers_label[class_mask]  # (k', d)
            #class_mask = proto_mask[label].float().unsqueeze(1)  # (k, 1)
            #valid_centers = centers_label * class_mask  
            center_batch = valid_centers.unsqueeze(0)
            center_batch = center_batch.expand(x_label.size(0), -1, -1)
            x_expanded = x_label.unsqueeze(1)
            x_expanded = x_expanded.expand(-1, center_batch.size(1), -1)
            loss_r += F.mse_loss(x_expanded, center_batch) / 2
            total += x_label.size(0)
        loss_r = loss_r / total

      
        loss_attn_sparse = self.attn_sparsity * attn_weight.abs().mean()

        loss = loss_cls + self.weight_pl * loss_r + loss_attn_sparse

        return logits, loss