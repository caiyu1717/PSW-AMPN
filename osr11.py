import os

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import argparse
import datetime
import time
import pandas as pd
import importlib
import torch
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from Attention_prototype.test import test
from Attention_prototype.train import train
from datasets.osr_dataloader import CIFAR10_OSR, SVHN_OSR, Tiny_ImageNet_OSR, Kvasir_OSR,hyper_OSR,ISIC2018_OSR,OCT_OSR,BloodMNIST_OSR
from models.swin import SwinTransformer
from utils import save_networks, load_networks

from utils import print_all_params,print_trainable_params

parser = argparse.ArgumentParser("Training")

import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

# Dataset
parser.add_argument('--dataset', type=str, default='oct11', help="kvasir_capsule|hyper-kvasir | svhn | cifar10 | isic2018| oct11 | bloodmnist")
parser.add_argument('--dataroot', type=str, default='/8T/WJZZY/cxy/PCLAE-AMPN/data')
parser.add_argument('--outf', type=str, default='/8T/WJZZY/cxy/PCLAE-AMPN/log_results')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')

# optimization
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1, help="temp")
parser.add_argument('--num_centers', type=int, default=10)
# model
parser.add_argument('--weight-pl', type=float, default=0.3, help="weight for center loss")
parser.add_argument('--model', type=str, default='swin')

# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use_gpu', action='store_true', default=True)
parser.add_argument('--save-dir', type=str, default='/8T/WJZZY/cxy/PCLAE-AMPN/log_result')
parser.add_argument('--loss', type=str, default='ATMPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--train', action='store_true', help="Training", default=True)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
parser.add_argument('--class', type=str, default='oct11_ATMPLoss_swin')
parser.add_argument('--total_classes', type=int, default=10,help="tiny_imagenet 200")

def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']

    if options['eval']:
        print("-------  -------->Evaluating<---------------")
    else:
        print("--------------->Training<-----------------")

    if options['use_gpu']:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'kvasir_capsule' in options['dataset']:
        Data = Kvasir_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                                 img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader

    elif 'hyper-kvasir' in options['dataset']:
        Data = hyper_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                                 img_size=options['img_size'])

        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader

    elif 'isic2018' in options['dataset']:
        Data = ISIC2018_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                                 img_size=options['img_size'])

        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader

    elif 'oct11' in options['dataset']:
        Data = OCT_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                                 img_size=options['img_size'])
        

        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader

    elif 'bloodmnist' in options['dataset']:
        Data = BloodMNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                                 img_size=options['img_size'])

        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader


    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
   
    #model_path = "/media/admin1/yhy/ARPL_muti/swin_base_patch4_window7_224.pth"  
    model_path = "/8T/WJZZY/cxy/PCLAE-AMPN/swin_tiny_patch4_window7_224_22k.pth"
    checkpoint = torch.load(model_path)
  
    checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if 'head' not in k}

    #net = SwinTransformer(num_classes=options['num_classes'], embed_dim=128, img_size=224, window_size=7,
    #                      num_heads=[4, 8, 16, 32], depths=[2, 2, 18, 2])  
    net = SwinTransformer(num_classes=options['num_classes'], embed_dim=96, img_size=224, window_size=7,
                          num_heads=[3, 6, 12, 24], depths=[2, 2, 6, 2])  

    net.load_state_dict(checkpoint['model'], strict=False)
    #for name, param in net.named_parameters():
    #    if 'head' in name:  
    #        param.requires_grad = False
    for name, param in net.named_parameters():

        if (
                'layers.3' in name or
                'head' in name or
                'projector' in name or
                 name == 'norm.weight' or
                 name == 'norm.bias'
        ):
            param.requires_grad = True

        else:
            param.requires_grad = False

    '''for name, param in net.named_parameters():
        param.requires_grad = True'''  
        #print(f"Trainable: {name}, shape={param.shape}")

    feat_dim = 128

    # Loss
    options.update(
        {
            'feat_dim': feat_dim
        }
    )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)
    #print(criterion.Dist.proto_M.grid)
    #for name, param in criterion.named_parameters():
    #    print(f"{name}: requires_grad={param.requires_grad}")

    if options['use_gpu']:
        net = net.cuda()
        criterion = criterion.cuda()

    # 打印参数信息
    #print_all_params(net, "all_params.txt")
    #print_trainable_params(net, "trainable_params.txt")

    model_path = os.path.join(options['outf'], 'models',options['dataset'],dir_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #file_name = '{}_{}_{}'.format(options['model'], options['loss'], options['item'])
    file_name = 'seed{}'.format(options['seed'])

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)

        #ckpt_dir = "/media/admin1/cxy/PCLAE-AMPN/log_results/models/hyper-kvasir/swin_seed1_0115/checkpoints"
        #net.load_state_dict(torch.load(os.path.join(ckpt_dir, "seed1_best.pth")), strict=True)
        #criterion.load_state_dict(torch.load(os.path.join(ckpt_dir, "seed1_best_criterion.pth")), strict=False)

        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

        return results

    params_list = [{'params': net.parameters()},  
                {'params': criterion.parameters()}]  

    
    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60, 80, 90])

    start_time = time.time()
    best_performance = 0
    results_best = dict()
    patience = 20 
    early_stopping_counter = 0
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))


            
            if results['OSCR'] > best_performance:
                best_performance = results['OSCR']
                results_best = results
                print("Best Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_best['ACC'],
                                                                                             results_best['AUROC'],
                                                                                             results_best['OSCR']))

                save_networks(net, model_path, file_name,criterion=criterion)

                """early_stopping_counter = 0
                print(f"==================New best model saved=========================")
                print(
                   "Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
                                                                                     results['OSCR']))
            else:
               
                early_stopping_counter += 1

                
            if early_stopping_counter >= patience:
               print("Early stopping triggered.")
               break"""


        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


    return results_best

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])

    options['debug_inter'] = True
    options['debug_print_freq'] = 50
    
    img_size = 224
    results = dict()

    
    from split import splits_2020 as splits

    print(len(splits[options['dataset']]))
    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']]) - i - 1]
        #if 'tiny_imagenet' in options['dataset']:
            #unknown = list(set(list(range(0, 200))) - set(known))
        #else:
            #unknown = list(set(list(range(0, 10))) - set(known))
        total_classes = options.get('total_classes', 10)
        unknown = list(set(range(total_classes)) - set(known))

        print('--------------------------------------------')
        print('selected known classes:', len(known))
        print('selected unknown classes:', len(unknown))

        options.update(
            {
                'item': i,
                'known': known,
                'unknown': unknown,
                'img_size': img_size
            }
        )

        now = datetime.datetime.now().strftime("%m%d")
        dir_name = 'k=10_{}_seed{}_{}'.format(options['model'],  options['seed'],now)
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = '{}_{}.csv'.format(options['dataset'], 'ATMPLoss_swin')

        res = main_worker(options)
        res['unknown'] = unknown
        res['known'] = known
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))
