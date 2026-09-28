import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.Sparsemax import Sparsemax


# torch.set_printoptions(sci_mode=False, precision=4)

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
    def __init__(self, L=1024, D=256, dropout=False, n_proto = 10, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D),
                            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(n_proto, n_classes)  # 加softmax

    def forward(self, proto, x):
        a = self.attention_a(proto)  # p * size[2]
        b = self.attention_b(x)  # n * size[2]
        proto_A = torch.mm(a, b.T)  # p * n
        proto_A = torch.transpose(proto_A, 0, 1)  # n * p
        A = self.attention_c(proto_A)  # n x n_classes
        return proto_A, A

class AttentionDist(nn.Module):
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, init='random'):
        super(AttentionDist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes, num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes, num_centers, self.feat_dim))
            self.centers.data.fill_(0)

        self.attention_net = Attn_Net_Gated(L=self.feat_dim, D=self.feat_dim, dropout=True, n_proto = self.num_centers, n_classes = 1)
        self.ERloss_function = KL_Divergence_Loss()


    def forward(self, features, center=None, metric='l2', label=None):
        center = self.centers
     
        #proto_M = torch.zeros(self.num_classes, self.feat_dim).cuda()
        M = torch.zeros(self.num_classes, self.feat_dim).cuda()
        attn_weight = torch.zeros(self.num_classes, self.num_centers).cuda()
        proto_M = torch.zeros(self.num_classes, self.num_centers, self.feat_dim).cuda()
        proto_mask = torch.zeros(self.num_classes, self.num_centers).bool().cuda()

        for c in label.unique(): 
            idx = (label == c)

            x_c = features[idx]  # n_c × d
            p_c = center[c]  # k × d 

           
            proto_A, A_raw = self.attention_net(p_c, x_c)  # (n_c, k)

            A_raw = torch.transpose(A_raw, 1, 0)  # c * N
            A = F.softmax(A_raw, dim=1)  # softmax over N
            M[c] = torch.mm(A, x_c)  

            #proto_score, _ = torch.max(proto_A, dim=0)  # 
            #proto_score = proto_score / torch.max(proto_score)
            proto_score = torch.mean(proto_A, dim=0)
            proto_score = F.softmax(proto_score, dim=0)
            #sparsemax = Sparsemax(dim=0)
            #proto_score = sparsemax(proto_score)


            mean_score = proto_score.mean()
            std_score = proto_score.std()
            threshold = 0.5 * mean_score + 0.1 * std_score  
            mask = proto_score > threshold
            selected_p = p_c[mask]  # m × d

            attn_weight[c] = proto_score

           
            #proto_M[c] = torch.mm(proto_score.unsqueeze(0), p_c)  # [1, d] → [d]
        
            k_prime = selected_p.shape[0]
          
            proto_M[c, :k_prime] = selected_p
            proto_mask[c, :k_prime] = 1

        if metric == 'l2':
            dist = self._compute_l2_dist(features, proto_M, proto_mask)
            #dist = self.compute_l2_dist(features, proto_M)
        else:
            dist = self._compute_dot_dist(features, proto_M, proto_mask)
            #dist = self.compute_dot_dist(features, proto_M)
        return dist, proto_M, proto_mask, attn_weight

    def _compute_l2_dist(self, features, center, mask):
        device = features.device  
        f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        dist_list = []
        for cls in range(self.num_classes):
            class_centers = center[cls].to(device)
            #class_mask = mask[cls].float().unsqueeze(1).to(device)  
            #valid_centers = class_centers * class_mask  
            class_mask = mask[cls].to(device)  # (num_centers, )
           
            valid_centers = class_centers[class_mask]  # (k', d)
            if valid_centers.shape[0] == 0:
                
                dist = torch.full((features.shape[0],), 1e6, device=device)
            else:
                c_2 = torch.sum(torch.pow(valid_centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(valid_centers, 1, 0)) + torch.transpose(c_2, 1, 0)
                dist = dist / float(features.shape[1])
                dist = torch.mean(dist, dim=1)  # Reduce over num_centers
            dist_list.append(dist)

        # Stack all class distances into one tensor
        dist = torch.stack(dist_list, dim=1)  # Shape: (batch_size, num_classes, num_centers)
        return dist

    def compute_l2_dist(self, features, center):
        f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
        dist = f_2 - 2 * torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)# (n, p)
        dist = dist / float(features.shape[1])
        return dist

    def _compute_dot_dist(self, features, center, mask):
        device = features.device 
        dist_list = []
        for cls in range(self.num_classes):
            class_centers = center[cls].to(device)
            class_mask = mask[cls].to(device)
            valid_centers = class_centers[class_mask]  # (k', d)
            #class_mask = mask[cls].float().unsqueeze(1).to(device)  # (k, 1)
            #valid_centers = class_centers * class_mask 
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