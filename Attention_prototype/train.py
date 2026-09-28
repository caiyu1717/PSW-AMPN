import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
#from density.DenPeakcode import DenPeakCluster, ChooseCenter_N_Cluster
from utils import AverageMeter
from torch.autograd import Variable


def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()
    loss_all = 0

    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        scaler = torch.cuda.amp.GradScaler()
        #scaler = torch.amp.GradScaler('cuda')


        with torch.cuda.amp.autocast():
            optimizer.zero_grad()
            x, y = net(data, True)
            logits, loss = criterion(x, labels, train=True)

            #loss.backward()
            #print(criterion.center_points.grid)
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))

        loss_all += losses.avg

    return loss_all



def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG,
             trainloader, epoch=None, **options):
    print('train with confusing samples')
    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    netD.train()
    netG.train()
    torch.cuda.empty_cache()

    loss_all, real_label, fake_label = 0, 1, 0

    for batch_idx, (data, labels,_) in enumerate(trainloader):

        batch_size = data.size(0)

        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            device = data.device
        else:
            device = torch.device('cpu')

        gan_target = torch.full((batch_size,), 0.0, device=device)

        ######################################
        # Step 1: Update Discriminator (D)
        ######################################
        noise = torch.randn(batch_size, options['nz'], options['ns'], options['ns'], device=data.device)
        fake = netG(noise).detach()  # ❗阻断反向传播给 G

        optimizerD.zero_grad()

        # real data
        gan_target_real = torch.full((batch_size,), real_label, device=data.device,dtype=torch.float)

        out_real = netD(data)
        loss_real = criterionD(out_real, gan_target_real)

        # fake data
        gan_target_fake = torch.full((batch_size,), fake_label, device=data.device,dtype=torch.float)
        out_fake = netD(fake)
        loss_fake = criterionD(out_fake, gan_target_fake)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        #torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=5)  # 裁剪 D 梯度

        """  total_norm = 0
        for p in net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f'Grad norm net: {total_norm}')"""

        optimizerD.step()
        lossesD.update(loss_D.item(), batch_size)

        ######################################
        # Step 2: Update Generator (G)
        ######################################
        noise = torch.randn(batch_size, options['nz'], options['ns'], options['ns'], device=data.device)
        fake = netG(noise)

        optimizerG.zero_grad()
        gan_target.fill_(real_label)  # G wants D to believe fake is real

        out_D = netD(fake)
        loss_adv = criterionD(out_D, gan_target)

        # pass fake through classifier
        fake_resized = F.interpolate(fake.detach().clone(), size=(224, 224), mode='bilinear', align_corners=False)
        x_fake, _ = net(fake_resized, True)
        loss_feat = criterion.fake_loss(x_fake).mean()

        loss_G = loss_adv + options['beta'] * loss_feat
        loss_G.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=5)  # 裁剪 G 梯度

        optimizerG.step()
        lossesG.update(loss_G.item(), batch_size)

        ######################################
        # Step 3: Update Classifier (net)
        ######################################
        optimizer.zero_grad()

        # 真实图像分类损失
        x_real, _ = net(data, True)
        _, loss = criterion(x_real, labels=labels, train=True)

        # fake 分类 KL损失
        noise2 = torch.randn(batch_size, options['nz'], options['ns'], options['ns'], device=data.device)
        fake2 = netG(noise2).detach()
        fake2_resized = F.interpolate(fake2, size=(224, 224), mode='bilinear', align_corners=False)
        x_fake2, _ = net(fake2_resized, True)
        loss_kl = criterion.fake_loss(x_fake2).mean()

        total_loss = loss + options['beta'] * loss_kl
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)

        optimizer.step()
        losses.update(total_loss.item(), batch_size)

        ######################################
        # Logging
        ######################################
        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})".format(
                batch_idx + 1, len(trainloader),
                losses.val, losses.avg,
                lossesG.val, lossesG.avg,
                lossesD.val, lossesD.avg))

        loss_all += losses.avg

    """import gc

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.grad_fn is not None:
                print(f"Tensor with grad_fn still in memory: {type(obj)}, size: {obj.size()}")
        except:
            pass"""

    return loss_all


