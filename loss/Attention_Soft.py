import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.Sparsemax import Sparsemax



class KL_Divergence_Loss(nn.Module):
    def __init__(self):
        super(KL_Divergence_Loss, self).__init__()

    def forward(self, logits1, logits2):
        # 计算Softmax
        probs1 = nn.Sigmoid()(logits1)
        probs2 = nn.Sigmoid()(logits2)

        # 计算KL散度损失
        loss = nn.KLDivLoss(reduction='batchmean')(torch.log(probs1), probs2)

        return loss


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_proto=10, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D),
                            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)   #
        self.attention_b = nn.Sequential(*self.attention_b)   

    def forward(self, proto, x):
        a = self.attention_a(proto)  # p * size[2] 
        b = self.attention_b(x)  # n * size[2] 
        #a = F.normalize(self.attention_a(proto), dim=-1)
        #b = F.normalize(self.attention_b(x), dim=-1)
        proto_A = torch.mm(a, b.T)  # p * n  
        proto_A = torch.transpose(proto_A, 0, 1)  # n * p   
        return proto_A


class AttentionDist(nn.Module):
    def __init__(self, num_classes=14, num_centers=1, feat_dim=2, init='random'):
        super(AttentionDist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers
       
        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes, num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes, num_centers, self.feat_dim))
            self.centers.data.fill_(0)
      
        self.attention_net = Attn_Net_Gated(L=self.feat_dim, D=self.feat_dim, dropout=True, n_proto=self.num_centers,
                                            n_classes=1)
       
        self.proto_M = torch.zeros(self.num_classes, self.num_centers, self.feat_dim).cuda()
        self.proto_mask = torch.zeros(self.num_classes, self.num_centers).bool().cuda()
        self.attn_weight = torch.zeros(self.num_classes, self.num_centers).cuda()

    def forward(self, features, center=None, metric='l2', label=None, train=False):
        if train:
            proto_M_tmp = torch.zeros(self.num_classes, self.num_centers, self.feat_dim).cuda() 
            proto_mask_tmp = torch.zeros(self.num_classes, self.num_centers).bool().cuda()
            attn_weight_tmp = torch.zeros(self.num_classes, self.num_centers).cuda()
           
            for c in label.unique(): 
                idx = (label == c)
                
                x_c = features[idx]  # n_c × d  

                p_c = self.centers[c]  # k × d 

                 
                proto_A = self.attention_net(p_c, x_c)  # (n_c, k)

                # proto_score, _ = torch.max(proto_A, dim=0)  
                # proto_score = proto_score / torch.max(proto_score)
                proto_score = torch.mean(proto_A, dim=0)
                #T = 20 
                #proto_score = F.softmax(proto_score / T, dim=0)
                proto_score = F.softmax(proto_score , dim=0) 
                #sparsemax = Sparsemax(dim=0)
                #proto_score = sparsemax(proto_score)
                """
                avg_proto_A = torch.mean(proto_A, dim=0)  # [k]
                selected_prototypes = avg_proto_A > torch.mean(avg_proto_A)  
                if selected_prototypes.sum() == 0:
                    selected_prototypes[torch.argmax(avg_proto_A)] = True  

                
                selected_proto_A = proto_A[:, selected_prototypes]  # (n_c, selected_k)

                
                proto_score_selected, _ = torch.max(selected_proto_A, dim=0)  
                proto_score_selected = proto_score_selected / torch.max(proto_score_selected)
                proto_score_selected = F.softmax(proto_score_selected, dim=0)  # [1, selected_k]

              
                attn_weight[c, selected_prototypes] = proto_score_selected

             
                proto_M[c] = torch.mm(proto_score_selected.unsqueeze(0), selected_p_c)  # [1, d] → [d]
                """


                # 3) keep all candidate sub-prototypes
                mask = torch.ones_like(proto_score, dtype=torch.bool)  # [K]
                selected_p = p_c  # [K, D]
              
                attn_weight_tmp[c] = proto_score  
               
                k_prime = selected_p.shape[0]
              

                proto_M_tmp[c, :k_prime] = selected_p 
                proto_mask_tmp[c, :k_prime] = 1   
               
          
            self.proto_M = proto_M_tmp  
            self.proto_mask = proto_mask_tmp
            self.attn_weight = attn_weight_tmp

            #if metric == 'l2':
                #dist = self._compute_l2_dist(features, proto_M_tmp, proto_mask_tmp)
                # dist = self.compute_l2_dist(features, proto_M)
            #else:
                #dist = self._compute_dot_dist(features, proto_M_tmp, proto_mask_tmp)
                # dist = self.compute_dot_dist(features, proto_M)

            #print(f"[Train] proto_M.shape: {proto_M_tmp.shape}, dist.shape: {dist.shape}")

            if metric == 'l2':
               dist = self._compute_l2_dist_weighted(
               features,
               proto_M_tmp,
               proto_mask_tmp,
               attn_weight_tmp
            )
            else:
               dist = self._compute_dot_dist(features, proto_M_tmp, proto_mask_tmp)

            return dist, proto_M_tmp, proto_mask_tmp, attn_weight_tmp
        else:

            if metric == 'l2':
               dist = self._compute_l2_dist_weighted(
               features,
               self.proto_M,
               self.proto_mask,
               self.attn_weight
            )
            else:
               dist = self._compute_dot_dist(features, self.proto_M, self.proto_mask)
            """
            if metric == 'l2':
                dist = self._compute_l2_dist(features, self.proto_M, self.proto_mask)
                # dist = self.compute_l2_dist(features, proto_M)
            else:
                dist = self._compute_dot_dist(features, self.proto_M, self.proto_mask)
                # dist = self.compute_dot_dist(features, proto_M)
            #print(f"[Test] proto_M.shape: {self.proto_M.shape}, dist.shape: {dist.shape}")
            """
            return dist, self.proto_M, self.proto_mask, self.attn_weight
            


    def _compute_l2_dist(self, features, center, mask):
        device = features.device  
        f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True) #[32,1]
        dist_list = []
        for cls in range(self.num_classes):
            class_centers = center[cls].to(device)  #[10,128]
            # class_mask = mask[cls].float().unsqueeze(1).to(device)   # (k, 1)
            # valid_centers = class_centers * class_mask  
            class_mask = mask[cls].to(device)  # (num_centers, )=[10,]
           
            valid_centers = class_centers[class_mask]  
            if valid_centers.shape[0] == 0:
              
                dist = torch.full((features.shape[0],), 1e6, device=device)
            else:
                c_2 = torch.sum(torch.pow(valid_centers, 2), dim=1, keepdim=True) #[5,1]
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(valid_centers, 1, 0)) + torch.transpose(c_2, 1,0) #[32,5]
                dist = dist / float(features.shape[1]) 
                dist = torch.mean(dist, dim=1)  # Reduce over num_centers 
            dist_list.append(dist)

        # Stack all class distances into one tensor
        dist = torch.stack(dist_list, dim=1)  # Shape: (batch_size, num_classes, num_centers)
        return dist

    def _compute_l2_dist_weighted(self, features, center, mask, attn_weight):
        device = features.device
        f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        dist_list = []

        for cls in range(self.num_classes):
            class_centers = center[cls].to(device)
            class_mask = mask[cls].to(device)

            valid_centers = class_centers[class_mask]  # [K', D]
            valid_weight = attn_weight[cls].to(device)[class_mask]  # [K']

            if valid_centers.shape[0] == 0:
                dist = torch.full((features.shape[0],), 1e6, device=device)
            else:
                valid_weight = valid_weight / (valid_weight.sum() + 1e-8)

                c_2 = torch.sum(torch.pow(valid_centers, 2), dim=1, keepdim=True)  # [K', 1]

                dist_each = (
                    f_2
                    - 2 * torch.matmul(features, valid_centers.t())
                    + c_2.t()
                )  # [B, K']

                dist_each = dist_each / float(features.shape[1])

                dist = torch.sum(
                    dist_each * valid_weight.unsqueeze(0),
                    dim=1
                )  # [B]

            dist_list.append(dist)

        dist = torch.stack(dist_list, dim=1)  # [B, C]
        return dist

    def compute_l2_dist(self, features, center): 
        f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
        dist = f_2 - 2 * torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)  # (n, p)
        dist = dist / float(features.shape[1])
        return dist

    def _compute_dot_dist(self, features, center, mask):
        device = features.device 
        dist_list = []
        for cls in range(self.num_classes):
            class_centers = center[cls].to(device)
            class_mask = mask[cls].to(device)
            valid_centers = class_centers[class_mask]  # (k', d)
            # class_mask = mask[cls].float().unsqueeze(1).to(device)  # (k, 1)
            # valid_centers = class_centers * class_mask  
            if valid_centers.shape[0] == 0:
                dist = torch.full((features.shape[0],), -1e6, device=device)  
            else:
                dist = features.matmul(valid_centers.t())  # Compute similarity
                dist = torch.mean(dist, dim=1)  # Reduce over num_centers
            dist_list.append(dist)
        # Stack all class distances into one tensor
        dist = torch.stack(dist_list, dim=1)  # Shape: (batch_size, num_classes, num_centers)
        return dist

    def compute_dot_dist(self, features, center):
        dist = features.matmul(center.t())
        return dist